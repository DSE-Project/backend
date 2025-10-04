import httpx
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import asyncio
import calendar
from services.database_service import db_service
from services.forecast_service_1m import predict_1m, initialize_1m_service
from services.shared_fred_date_service import shared_fred_date_service
from schemas.forecast_schema_1m import InputFeatures1M, CurrentMonthData1M, ForecastResponse1M

logger = logging.getLogger(__name__)

# FRED API Configuration
FRED_API_KEY = os.getenv("FRED_API_KEY")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series IDs mapping
SERIES_IDS = {
    "fedfunds": "FEDFUNDS",
    "TB3MS": "TB3MS",
    "TB6MS": "TB6MS",
    "TB1YR": "TB1YR",
    "USTPU": "USTPU",
    "USGOOD": "USGOOD",
    "SRVPRD": "SRVPRD",
    "USCONS": "USCONS",
    "MANEMP": "MANEMP",
    "USWTRADE": "USWTRADE",
    "USTRADE": "USTRADE",
    "USINFO": "USINFO",
    "UNRATE": "UNRATE",
    "UNEMPLOY": "UNEMPLOY",
    "CPIFOOD": "CUSR0000SAF11",
    "CPIMEDICARE": "CPIMEDSL",
    "CPIRENT": "CUUR0000SEHA",
    "CPIAPP": "CPIAPPSL",
    "GDP": "GDP",
    "REALGDP": "GDPC1",
    "PCEPI": "PCEPI",
    "PSAVERT": "PSAVERT",
    "PSTAX": "W055RC1Q027SBEA",
    "COMREAL": "BOGZ1FL075035503Q",
    "COMLOAN": "H8B1023NCBCMG",
    "SECURITYBANK": "H8B1002NCBCMG",
    "PPIACO": "PPIACO",
    "M1SL": "M1SL",
    "M2SL": "M2SL",
    "recession": "USREC"
}

def is_last_5_days_of_month() -> bool:
    """Check if today is within the last 5 days of the current month"""
    today = datetime.now()
    # Get the last day of the current month
    last_day = calendar.monthrange(today.year, today.month)[1]
    # Calculate how many days until end of month
    days_until_end = last_day - today.day
    return days_until_end < 5

def forward_fill_nulls_from_previous_row(latest_row: Dict[str, Any], previous_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Forward fill NULL/None values in the latest row with values from the previous row.
    This handles the case where the scheduler creates incomplete records with NaN values.
    """
    if not latest_row or not previous_row:
        return latest_row
    
    filled_row = latest_row.copy()
    null_count = 0
    filled_count = 0
    
    for key, value in latest_row.items():
        if key == 'observation_date':  # Don't forward fill dates
            continue
            
        if value is None or (isinstance(value, float) and pd.isna(value)):
            null_count += 1
            if key in previous_row and previous_row[key] is not None:
                if not (isinstance(previous_row[key], float) and pd.isna(previous_row[key])):
                    filled_row[key] = previous_row[key]
                    filled_count += 1
                    logger.info(f"Forward filled {key}: {previous_row[key]}")
    
    if null_count > 0:
        logger.info(f"Forward filled {filled_count}/{null_count} NULL values from previous row")
    
    return filled_row

async def fetch_latest_observation(series_id: str, timeout: int = 30, max_retries: int = 3) -> Dict[str, Any]:
    """Fetch latest observation for a specific series from FRED API with retry logic"""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1
    }
    
    for attempt in range(max_retries):
        try:
            # Add timeout and retry configuration
            timeout_config = httpx.Timeout(timeout)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                logger.info(f"Attempting to fetch {series_id} (attempt {attempt + 1}/{max_retries})")
                response = await client.get(BASE_URL, params=params)
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"Successfully fetched {series_id}")
                return data
                
        except httpx.TimeoutException as e:
            logger.warning(f"Timeout for {series_id} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)  # Wait 1 second before retry
            else:
                raise Exception(f"Timeout after {max_retries} attempts: {e}")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for {series_id} on attempt {attempt + 1}: Status {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Rate limited, waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)
                if attempt < max_retries - 1:
                    continue
            raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
            
        except httpx.RequestError as e:
            logger.error(f"Request error for {series_id} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                raise Exception(f"Request error after {max_retries} attempts: {e}")
                
        except Exception as e:
            logger.error(f"Unexpected error for {series_id} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                raise Exception(f"Unexpected error after {max_retries} attempts: {e}")

async def get_fred_latest_date() -> Optional[str]:
    """Get the latest observation date from FRED API using shared service"""
    try:
        logger.info("Getting latest FRED date using shared service")
        return await shared_fred_date_service.get_latest_fred_date()
    except Exception as e:
        logger.error(f"Failed to get latest FRED date from shared service: {type(e).__name__}: {str(e)}")
        return None

def get_database_latest_date() -> Optional[str]:
    """Get the latest observation date from database"""
    try:
        logger.info("Fetching latest database date")
        # Get the latest record from historical_data_1m table
        response = db_service.supabase.table('historical_data_1m')\
            .select("observation_date")\
            .order('observation_date', desc=True)\
            .limit(1)\
            .execute()
        
        if response.data and len(response.data) > 0:
            latest_date = response.data[0]["observation_date"]
            # Convert to FRED format (YYYY-MM-DD)
            db_date = pd.to_datetime(latest_date).strftime('%Y-%m-%d')
            logger.info(f"Latest database date: {db_date}")
            return db_date
        else:
            logger.info("No data found in database")
            return None
    except Exception as e:
        logger.error(f"Failed to fetch latest database date: {type(e).__name__}: {str(e)}")
        return None

async def fetch_all_fred_data() -> Optional[Dict[str, Any]]:
    """Fetch all latest observations from FRED API with better error handling"""
    try:
        logger.info("Starting to fetch all FRED data")
        results = {}
        failed_series = []
        
        # Add delay between requests to avoid rate limiting
        for i, (name, series_id) in enumerate(SERIES_IDS.items()):
            try:
                # Add small delay between requests
                if i > 0:
                    await asyncio.sleep(0.1)  # 100ms delay
                
                logger.info(f"Fetching {name} ({series_id}) - {i+1}/{len(SERIES_IDS)}")
                data = await fetch_latest_observation(series_id, timeout=20)
                
                if data and "observations" in data and len(data["observations"]) > 0:
                    obs = data["observations"][0]
                    results[name] = {
                        "date": obs["date"],
                        "value": float(obs["value"]) if obs["value"] != "." else None
                    }
                    logger.info(f"Successfully fetched {name}: {obs['value']}")
                else:
                    logger.warning(f"No data found for {name} ({series_id})")
                    results[name] = None
                    failed_series.append(name)
                    
            except Exception as e:
                logger.error(f"Failed to fetch {name} ({series_id}): {type(e).__name__}: {str(e)}")
                results[name] = None
                failed_series.append(name)
        
        if failed_series:
            logger.warning(f"Failed to fetch {len(failed_series)} series: {failed_series}")
        
        logger.info(f"Completed fetching FRED data. Success: {len(results) - len(failed_series)}/{len(results)}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to fetch FRED data: {type(e).__name__}: {str(e)}")
        return None

def insert_fred_data_to_database(fred_data: Dict[str, Any]) -> bool:
    """Insert new FRED data into the database"""
    try:
        # Prepare data for insertion
        observation_date = None
        new_row = {}
        
        for name, data in fred_data.items():
            if data and data["value"] is not None:
                if observation_date is None:
                    observation_date = data["date"]
                
                # Handle data type conversion properly
                if name == "recession":
                    # Recession should be integer (0 or 1)
                    new_row[name] = int(float(data["value"]))
                else:
                    # All other fields are DECIMAL
                    new_row[name] = float(data["value"])
            else:
                # Handle missing values with proper data types
                if name == "recession":
                    new_row[name] = 0  # Integer 0 for recession
                else:
                    new_row[name] = 0.0  # Float 0.0 for DECIMAL fields
                logger.warning(f"Missing value for {name}, using default")
        
        if observation_date:
            new_row["observation_date"] = observation_date
            
            # Log the data being inserted for debugging
            logger.info(f"Inserting data: {new_row}")
            
            # Insert into database
            response = db_service.supabase.table('historical_data_1m')\
                .insert(new_row)\
                .execute()
            
            if response.data:
                logger.info(f"Successfully inserted new data for {observation_date}")
                return True
            else:
                logger.error(f"Failed to insert data: {response}")
                return False
        else:
            logger.error("No valid observation date found in FRED data")
            return False
            
    except Exception as e:
        logger.error(f"Failed to insert FRED data into database: {type(e).__name__}: {str(e)}")
        return False

def get_latest_database_row() -> Optional[Dict[str, Any]]:
    """
    Get the latest row from the database with NULL handling.
    Fetches the last 2 rows and forward fills NULL values from the previous row.
    """
    try:
        response = db_service.supabase.table('historical_data_1m')\
            .select("*")\
            .order('observation_date', desc=True)\
            .limit(2)\
            .execute()
        
        if response.data and len(response.data) > 0:
            latest_row = response.data[0]
            
            # If we have a previous row, use it to forward fill NULLs
            if len(response.data) > 1:
                previous_row = response.data[1]
                latest_row = forward_fill_nulls_from_previous_row(latest_row, previous_row)
                logger.info("Applied forward filling for NULL values using previous row")
            else:
                logger.info("Only one row available, no forward filling possible")
            
            return latest_row
        return None
    except Exception as e:
        logger.error(f"Failed to get latest database row: {type(e).__name__}: {str(e)}")
        return None

def convert_to_input_features(data_row: Dict[str, Any]) -> InputFeatures1M:
    """Convert database row or FRED data to InputFeatures1M format"""
    try:
        # Format observation_date
        obs_date = pd.to_datetime(data_row["observation_date"]).strftime('%m/%d/%Y')
        
        # Helper function to safely convert values
        def safe_float(value, default=0.0):
            try:
                if value is None:
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def safe_int(value, default=0):
            try:
                if value is None:
                    return default
                return int(float(value))
            except (ValueError, TypeError):
                return default
        
        current_data = CurrentMonthData1M(
            observation_date=obs_date,
            fedfunds=safe_float(data_row.get("fedfunds")),
            TB3MS=safe_float(data_row.get("TB3MS")),
            TB6MS=safe_float(data_row.get("TB6MS")),
            TB1YR=safe_float(data_row.get("TB1YR")),
            USTPU=safe_float(data_row.get("USTPU")),
            USGOOD=safe_float(data_row.get("USGOOD")),
            SRVPRD=safe_float(data_row.get("SRVPRD")),
            USCONS=safe_float(data_row.get("USCONS")),
            MANEMP=safe_float(data_row.get("MANEMP")),
            USWTRADE=safe_float(data_row.get("USWTRADE")),
            USTRADE=safe_float(data_row.get("USTRADE")),
            USINFO=safe_float(data_row.get("USINFO")),
            UNRATE=safe_float(data_row.get("UNRATE")),
            UNEMPLOY=safe_float(data_row.get("UNEMPLOY")),
            CPIFOOD=safe_float(data_row.get("CPIFOOD")),
            CPIMEDICARE=safe_float(data_row.get("CPIMEDICARE")),
            CPIRENT=safe_float(data_row.get("CPIRENT")),
            CPIAPP=safe_float(data_row.get("CPIAPP")),
            GDP=safe_float(data_row.get("GDP")),
            REALGDP=safe_float(data_row.get("REALGDP")),
            PCEPI=safe_float(data_row.get("PCEPI")),
            PSAVERT=safe_float(data_row.get("PSAVERT")),
            PSTAX=safe_float(data_row.get("PSTAX")),
            COMREAL=safe_float(data_row.get("COMREAL")),
            COMLOAN=safe_float(data_row.get("COMLOAN")),
            SECURITYBANK=safe_float(data_row.get("SECURITYBANK")),
            PPIACO=safe_float(data_row.get("PPIACO")),
            M1SL=safe_float(data_row.get("M1SL")),
            M2SL=safe_float(data_row.get("M2SL")),
            recession=safe_int(data_row.get("recession"))
        )
        
        return InputFeatures1M(
            current_month_data=current_data,
            use_historical_data=True,
            historical_data_source="database"
        )
    except Exception as e:
        logger.error(f"Failed to convert data to InputFeatures1M: {type(e).__name__}: {str(e)}")
        raise

async def get_latest_prediction_1m() -> ForecastResponse1M:
    """
    Main function that handles the complete flow:
    1. Check FRED for latest date
    2. Compare with database
    3. Fetch data accordingly
    4. Make prediction
    """
    try:
        logger.info("Starting 1M prediction process")
        
        # Initialize service if needed
        if not initialize_1m_service():
            raise RuntimeError("Failed to initialize 1M forecasting service")
        
        # Step 1: Get latest date from FRED
        logger.info("Step 1: Getting latest FRED date")
        fred_latest_date = await get_fred_latest_date()
        if not fred_latest_date:
            # Fallback to database if FRED fails
            logger.warning("Could not fetch latest date from FRED API, using database fallback")
            db_latest_date = get_database_latest_date()
            if not db_latest_date:
                raise RuntimeError("Could not fetch date from FRED or database")
            
            # Use database data as fallback
            logger.info("Using database data as fallback due to FRED API failure")
            latest_row = get_latest_database_row()
            if not latest_row:
                raise RuntimeError("Failed to get latest row from database")
            
            features = convert_to_input_features(latest_row)
            prediction_result = predict_1m(features)
            
            # Add metadata
            prediction_result.feature_importance = {
                "data_source": "database_fallback",
                "fred_date": "unavailable",
                "db_date": db_latest_date,
                "note": "Used database fallback due to FRED API failure"
            }
            return prediction_result
        
        # Step 2: Get latest date from database
        logger.info("Step 2: Getting latest database date")
        db_latest_date = get_database_latest_date()
        
        logger.info(f"FRED latest: {fred_latest_date}, DB latest: {db_latest_date}")
        
        # Step 3: Determine data source and fetch accordingly
        if db_latest_date and fred_latest_date == db_latest_date:
            # ✅ OPTIMAL PATH - Use database data (fast, scheduler working correctly)
            logger.info("✅ Using existing database data (dates match) - scheduler working correctly")
            latest_row = get_latest_database_row()
            if not latest_row:
                raise RuntimeError("Failed to get latest row from database")
            
            # Convert to input features
            features = convert_to_input_features(latest_row)
            
        else:
            # Check if we should fetch new data (month-end fallback mechanism)
            should_fetch = is_last_5_days_of_month()
            
            if should_fetch:
                # ⚠️ FALLBACK PATH - Fetch from FRED (slower, scheduler may need attention)
                logger.warning(f"⚠️ Fetching new data from FRED API - dates don't match (FRED: {fred_latest_date}, DB: {db_latest_date})")
                logger.info("Running month-end fallback: fetching fresh data from FRED API")
                fred_data = await fetch_all_fred_data()
            else:
                # Use existing database data even if dates don't match (not month-end)
                logger.info(f"Not in month-end period (last 5 days). Using existing database data even though dates don't match.")
                logger.info(f"FRED date: {fred_latest_date}, DB date: {db_latest_date}")
                latest_row = get_latest_database_row()
                if not latest_row:
                    raise RuntimeError("Failed to get latest row from database")
                
                features = convert_to_input_features(latest_row)
                prediction_result = predict_1m(features)
                prediction_result.feature_importance = {
                    "data_source": "database_scheduled",
                    "fred_date": fred_latest_date,
                    "db_date": db_latest_date,
                    "note": "Used database data (not month-end period)"
                }
                return prediction_result
            if not fred_data:
                # Fallback to database if available
                if db_latest_date:
                    logger.warning("FRED fetch failed, falling back to database data")
                    latest_row = get_latest_database_row()
                    if latest_row:
                        features = convert_to_input_features(latest_row)
                        prediction_result = predict_1m(features)
                        prediction_result.feature_importance = {
                            "data_source": "database_fallback",
                            "fred_date": fred_latest_date,
                            "db_date": db_latest_date,
                            "note": "Fell back to database due to FRED data fetch failure"
                        }
                        return prediction_result
                        
                raise RuntimeError("Failed to fetch data from FRED API and no database fallback available")
            
            # Insert new data into database
            if not insert_fred_data_to_database(fred_data):
                logger.warning("Failed to insert new data into database, continuing with prediction")
            
            # Convert FRED data to the format we need
            data_row = {}
            observation_date = None
            for name, data in fred_data.items():
                if data and data["value"] is not None:
                    if observation_date is None:
                        observation_date = data["date"]
                    
                    # Handle data type conversion
                    if name == "recession":
                        data_row[name] = int(float(data["value"]))
                    else:
                        data_row[name] = float(data["value"])
                else:
                    # Handle missing values with proper data types
                    if name == "recession":
                        data_row[name] = 0
                    else:
                        data_row[name] = 0.0
            
            data_row["observation_date"] = observation_date
            features = convert_to_input_features(data_row)
        
        # Step 4: Make prediction
        logger.info("Step 4: Making prediction")
        prediction_result = predict_1m(features)
        
        # Add metadata about data source
        if prediction_result.feature_importance:
            prediction_result.feature_importance["data_source"] = "database" if db_latest_date == fred_latest_date else "fred_api"
            prediction_result.feature_importance["fred_date"] = fred_latest_date
            prediction_result.feature_importance["db_date"] = db_latest_date
        else:
            prediction_result.feature_importance = {
                "data_source": "database" if db_latest_date == fred_latest_date else "fred_api",
                "fred_date": fred_latest_date,
                "db_date": db_latest_date
            }
        
        logger.info("Successfully completed 1M prediction")
        return prediction_result
        
    except Exception as e:
        logger.error(f"Failed to get latest prediction: {type(e).__name__}: {str(e)}")
        raise RuntimeError(f"Prediction failed: {e}")