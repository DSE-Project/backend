import httpx
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import calendar
from services.database_service import db_service
from services.forecast_service_3m import predict_3m, initialize_3m_service
from services.shared_fred_date_service import shared_fred_date_service
from schemas.forecast_schema_3m import InputFeatures3M, CurrentMonthData3M, ForecastResponse3M

logger = logging.getLogger(__name__)

# FRED API Configuration
FRED_API_KEY = os.getenv("FRED_API_KEY")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series IDs mapping for 3M model
SERIES_IDS_3M = {
    "ICSA": "ICSA",  # Weekly series - needs special handling
    "CPIMEDICARE": "CPIMEDSL",
    "USWTRADE": "USWTRADE",
    "BBKMLEIX": "BBKMLEIX",
    "COMLOAN": "H8B1023NCBCMG",
    "UMCSENT": "UMCSENT",
    "MANEMP": "MANEMP",
    "fedfunds": "FEDFUNDS",
    "PSTAX": "W055RC1Q027SBEA",
    "USCONS": "USCONS",
    "USGOOD": "USGOOD",
    "USINFO": "USINFO",
    "CPIAPP": "CPIAPPSL",
    "CSUSHPISA": "CSUSHPISA",
    "SECURITYBANK": "H8B1002NCBCMG",
    "SRVPRD": "SRVPRD",
    "INDPRO": "INDPRO",
    "TB6MS": "TB6MS",
    "UNEMPLOY": "UNEMPLOY",
    "USTPU": "USTPU",
    "recession": "USREC"
}

# Define which series are weekly (need monthly averaging)
WEEKLY_SERIES = {"ICSA"}

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

async def fetch_latest_observation_3m(series_id: str) -> Dict[str, Any]:
    """Fetch latest observation for a specific series from FRED API"""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

async def fetch_monthly_average_for_weekly_series(series_id: str, target_date: str) -> Optional[float]:
    """
    Fetch all weekly observations for a specific month and calculate the average
    target_date should be in format 'YYYY-MM-DD'
    """
    try:
        # Parse the target date to get year and month
        target_dt = pd.to_datetime(target_date)
        year = target_dt.year
        month = target_dt.month
        
        # Calculate the first and last day of the month
        first_day = f"{year}-{month:02d}-01"
        if month == 12:
            last_day = f"{year + 1}-01-01"
        else:
            last_day = f"{year}-{month + 1:02d}-01"
        
        logger.info(f"Fetching weekly data for {series_id} from {first_day} to {last_day}")
        
        # Fetch all observations for the month
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": first_day,
            "observation_end": last_day,
            "sort_order": "asc"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        if data and "observations" in data and len(data["observations"]) > 0:
            # Extract valid values (exclude "." values)
            values = []
            dates = []
            
            for obs in data["observations"]:
                if obs["value"] != ".":
                    try:
                        obs_date = pd.to_datetime(obs["date"])
                        # Only include observations that are actually in the target month
                        if obs_date.year == year and obs_date.month == month:
                            values.append(float(obs["value"]))
                            dates.append(obs["date"])
                    except (ValueError, TypeError):
                        continue
            
            if values:
                monthly_avg = sum(values) / len(values)
                logger.info(f"Calculated monthly average for {series_id} ({year}-{month:02d}): {monthly_avg:.2f} from {len(values)} observations")
                logger.info(f"Weekly values: {values}")
                logger.info(f"Dates used: {dates}")
                return monthly_avg
            else:
                logger.warning(f"No valid weekly observations found for {series_id} in {year}-{month:02d}")
                return None
        else:
            logger.warning(f"No observations returned for {series_id} in {year}-{month:02d}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch monthly average for weekly series {series_id}: {e}")
        return None

async def get_fred_latest_date_3m() -> Optional[str]:
    """Get the latest observation date from FRED API using shared service"""
    try:
        logger.info("Getting latest FRED date for 3M using shared service")
        return await shared_fred_date_service.get_latest_fred_date()
    except Exception as e:
        logger.error(f"Failed to get latest FRED date for 3M from shared service: {e}")
        return None

def get_database_latest_date_3m() -> Optional[str]:
    """Get the latest observation date from database"""
    try:
        # Get the latest record from historical_data_3m table
        response = db_service.supabase.table('historical_data_3m')\
            .select("observation_date")\
            .order('observation_date', desc=True)\
            .limit(1)\
            .execute()
        
        if response.data and len(response.data) > 0:
            latest_date = response.data[0]["observation_date"]
            # Convert to FRED format (YYYY-MM-DD)
            db_date = pd.to_datetime(latest_date).strftime('%Y-%m-%d')
            logger.info(f"Latest database date for 3M: {db_date}")
            return db_date
        return None
    except Exception as e:
        logger.error(f"Failed to fetch latest database date for 3M: {e}")
        return None

async def fetch_all_fred_data_3m() -> Optional[Dict[str, Any]]:
    """Fetch all latest observations from FRED API for 3M model"""
    try:
        results = {}
        reference_date = None  # Will be set by the first monthly series (like fedfunds)
        
        # First, fetch monthly series to establish the reference date
        for name, series_id in SERIES_IDS_3M.items():
            if name not in WEEKLY_SERIES:  # Skip weekly series in first pass
                try:
                    data = await fetch_latest_observation_3m(series_id)
                    if data and "observations" in data and len(data["observations"]) > 0:
                        obs = data["observations"][0]
                        if reference_date is None:
                            reference_date = obs["date"]  # Set reference date from first monthly series
                        
                        results[name] = {
                            "date": obs["date"],
                            "value": float(obs["value"]) if obs["value"] != "." else None
                        }
                    else:
                        logger.warning(f"No data found for {name} ({series_id})")
                        results[name] = None
                except Exception as e:
                    logger.error(f"Failed to fetch {name} ({series_id}): {e}")
                    results[name] = None
        
        # Now handle weekly series using the reference date
        if reference_date:
            logger.info(f"Using reference date {reference_date} for weekly series calculations")
            
            for name in WEEKLY_SERIES:
                if name in SERIES_IDS_3M:
                    series_id = SERIES_IDS_3M[name]
                    try:
                        # Calculate monthly average for the weekly series
                        monthly_avg = await fetch_monthly_average_for_weekly_series(series_id, reference_date)
                        
                        if monthly_avg is not None:
                            results[name] = {
                                "date": reference_date,  # Use the same date as monthly series
                                "value": monthly_avg
                            }
                            logger.info(f"Successfully calculated monthly average for {name}: {monthly_avg}")
                        else:
                            logger.warning(f"Could not calculate monthly average for {name}, using default value")
                            results[name] = {
                                "date": reference_date,
                                "value": 0.0  # Default value if calculation fails
                            }
                    except Exception as e:
                        logger.error(f"Failed to fetch monthly average for {name} ({series_id}): {e}")
                        results[name] = {
                            "date": reference_date,
                            "value": 0.0  # Default value on error
                        }
        else:
            logger.error("No reference date available for weekly series calculations")
            # Handle weekly series with default values
            for name in WEEKLY_SERIES:
                results[name] = None
        
        return results
    except Exception as e:
        logger.error(f"Failed to fetch FRED data for 3M: {e}")
        return None

def insert_fred_data_to_database_3m(fred_data: Dict[str, Any]) -> bool:
    """Insert new FRED data into the database for 3M model"""
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
                    # All other fields are DECIMAL/FLOAT
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
            logger.info(f"Inserting 3M data: {new_row}")
            
            # Insert into database
            response = db_service.supabase.table('historical_data_3m')\
                .insert(new_row)\
                .execute()
            
            if response.data:
                logger.info(f"Successfully inserted new 3M data for {observation_date}")
                return True
            else:
                logger.error(f"Failed to insert 3M data: {response}")
                return False
        else:
            logger.error("No valid observation date found in FRED data for 3M")
            return False
            
    except Exception as e:
        logger.error(f"Failed to insert FRED data into database for 3M: {e}")
        return False

def get_latest_database_row_3m() -> Optional[Dict[str, Any]]:
    """
    Get the latest row from the database for 3M with NULL handling.
    Fetches the last 2 rows and forward fills NULL values from the previous row.
    """
    try:
        response = db_service.supabase.table('historical_data_3m')\
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
        logger.error(f"Failed to get latest database row for 3M: {e}")
        return None

def convert_to_input_features_3m(data_row: Dict[str, Any]) -> InputFeatures3M:
    """Convert database row or FRED data to InputFeatures3M format"""
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
        
        current_data = CurrentMonthData3M(
            observation_date=obs_date,
            ICSA=safe_float(data_row.get("ICSA")),
            CPIMEDICARE=safe_float(data_row.get("CPIMEDICARE")),
            USWTRADE=safe_float(data_row.get("USWTRADE")),
            BBKMLEIX=safe_float(data_row.get("BBKMLEIX")),
            COMLOAN=safe_float(data_row.get("COMLOAN")),
            UMCSENT=safe_float(data_row.get("UMCSENT")),
            MANEMP=safe_float(data_row.get("MANEMP")),
            fedfunds=safe_float(data_row.get("fedfunds")),
            PSTAX=safe_float(data_row.get("PSTAX")),
            USCONS=safe_float(data_row.get("USCONS")),
            USGOOD=safe_float(data_row.get("USGOOD")),
            USINFO=safe_float(data_row.get("USINFO")),
            CPIAPP=safe_float(data_row.get("CPIAPP")),
            CSUSHPISA=safe_float(data_row.get("CSUSHPISA")),
            SECURITYBANK=safe_float(data_row.get("SECURITYBANK")),
            SRVPRD=safe_float(data_row.get("SRVPRD")),
            INDPRO=safe_float(data_row.get("INDPRO")),
            TB6MS=safe_float(data_row.get("TB6MS")),
            UNEMPLOY=safe_float(data_row.get("UNEMPLOY")),
            USTPU=safe_float(data_row.get("USTPU")),
            recession=safe_int(data_row.get("recession"))
        )
        
        return InputFeatures3M(
            current_month_data=current_data,
            use_historical_data=True,
            historical_data_source="database"
        )
    except Exception as e:
        logger.error(f"Failed to convert data to InputFeatures3M: {e}")
        raise

async def get_latest_prediction_3m() -> ForecastResponse3M:
    """
    Main function that handles the complete flow for 3M:
    1. Check FRED for latest date
    2. Compare with database
    3. Fetch data accordingly (including weekly series averaging)
    4. Make prediction
    """
    try:
        # Initialize service if needed
        if not initialize_3m_service():
            raise RuntimeError("Failed to initialize 3M forecasting service")
        
        # Step 1: Get latest date from FRED
        fred_latest_date = await get_fred_latest_date_3m()
        if not fred_latest_date:
            raise RuntimeError("Could not fetch latest date from FRED API for 3M")
        
        # Step 2: Get latest date from database
        db_latest_date = get_database_latest_date_3m()
        
        logger.info(f"3M - FRED latest: {fred_latest_date}, DB latest: {db_latest_date}")
        
        # Step 3: Determine data source and fetch accordingly
        if db_latest_date and fred_latest_date == db_latest_date:
            # Dates match - use database data
            logger.info("Using existing database data for 3M")
            latest_row = get_latest_database_row_3m()
            if not latest_row:
                raise RuntimeError("Failed to get latest row from database for 3M")
            
            # Convert to input features
            features = convert_to_input_features_3m(latest_row)
            
        else:
            # Check if we should fetch new data (month-end fallback mechanism)
            should_fetch = is_last_5_days_of_month()
            
            if should_fetch:
                # Dates don't match or no DB data - fetch from FRED
                logger.info("Running month-end fallback: Fetching new data from FRED API for 3M (including weekly series averaging)")
                fred_data = await fetch_all_fred_data_3m()
            else:
                # Use existing database data even if dates don't match (not month-end)
                logger.info(f"Not in month-end period (last 5 days). Using existing database data for 3M even though dates don't match.")
                logger.info(f"FRED date: {fred_latest_date}, DB date: {db_latest_date}")
                latest_row = get_latest_database_row_3m()
                if not latest_row:
                    raise RuntimeError("Failed to get latest row from database for 3M")
                
                features = convert_to_input_features_3m(latest_row)
                prediction_result = predict_3m(features)
                prediction_result.feature_importance = {
                    "data_source": "database_scheduled",
                    "fred_date": fred_latest_date,
                    "db_date": db_latest_date,
                    "note": "Used database data (not month-end period)"
                }
                return prediction_result
            if not fred_data:
                raise RuntimeError("Failed to fetch data from FRED API for 3M")
            
            # Insert new data into database
            if not insert_fred_data_to_database_3m(fred_data):
                logger.warning("Failed to insert new data into database for 3M, continuing with prediction")
            
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
            features = convert_to_input_features_3m(data_row)
        
        # Step 4: Make prediction
        prediction_result = predict_3m(features)
        
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
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Failed to get latest prediction for 3M: {e}")
        raise RuntimeError(f"3M Prediction failed: {e}")