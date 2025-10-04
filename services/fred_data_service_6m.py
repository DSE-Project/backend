import httpx
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import calendar
from services.database_service import db_service
from services.forecast_service_6m import predict_6m, initialize_6m_service
from services.shared_fred_date_service import shared_fred_date_service
from schemas.forecast_schema_6m import InputFeatures6M, CurrentMonthData6M, ForecastResponse6M

logger = logging.getLogger(__name__)

# FRED API Configuration
FRED_API_KEY = os.getenv("FRED_API_KEY")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series IDs mapping for 6M model
SERIES_IDS_6M = {
    "PSTAX": "W055RC1Q027SBEA",
    "USWTRADE": "USWTRADE",
    "MANEMP": "MANEMP",
    "CPIAPP": "CPIAPPSL",
    "CSUSHPISA": "CSUSHPISA",
    "ICSA": "ICSA",  
    "fedfunds": "FEDFUNDS",
    "BBKMLEIX": "BBKMLEIX",
    "TB3MS": "TB3MS",
    "USINFO": "USINFO",
    "PPIACO": "PPIACO",
    "CPIMEDICARE": "CPIMEDSL",
    "UNEMPLOY": "UNEMPLOY",
    "TB1YR": "TB1YR",
    "USGOOD": "USGOOD",
    "CPIFOOD": "CUSR0000SAF11",
    "UMCSENT": "UMCSENT",
    "SRVPRD": "SRVPRD",
    "GDP": "GDP",
    "INDPRO": "INDPRO",
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

async def fetch_latest_observation_6m(series_id: str) -> Dict[str, Any]:
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

async def fetch_monthly_average_for_weekly_series_6m(series_id: str, target_date: str) -> Optional[float]:
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
        
        logger.info(f"Fetching weekly data for 6M {series_id} from {first_day} to {last_day}")
        
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
                logger.info(f"Calculated monthly average for 6M {series_id} ({year}-{month:02d}): {monthly_avg:.2f} from {len(values)} observations")
                logger.info(f"Weekly values: {values}")
                logger.info(f"Dates used: {dates}")
                return monthly_avg
            else:
                logger.warning(f"No valid weekly observations found for 6M {series_id} in {year}-{month:02d}")
                return None
        else:
            logger.warning(f"No observations returned for 6M {series_id} in {year}-{month:02d}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch monthly average for weekly series 6M {series_id}: {e}")
        return None

async def get_fred_latest_date_6m() -> Optional[str]:
    """Get the latest observation date from FRED API using shared service"""
    try:
        logger.info("Getting latest FRED date for 6M using shared service")
        return await shared_fred_date_service.get_latest_fred_date()
    except Exception as e:
        logger.error(f"Failed to get latest FRED date for 6M from shared service: {e}")
        return None

def get_database_latest_date_6m() -> Optional[str]:
    """Get the latest observation date from database"""
    try:
        # Get the latest record from historical_data_6m table
        response = db_service.supabase.table('historical_data_6m')\
            .select("observation_date")\
            .order('observation_date', desc=True)\
            .limit(1)\
            .execute()
        
        if response.data and len(response.data) > 0:
            latest_date = response.data[0]["observation_date"]
            # Convert to FRED format (YYYY-MM-DD)
            db_date = pd.to_datetime(latest_date).strftime('%Y-%m-%d')
            logger.info(f"Latest database date for 6M: {db_date}")
            return db_date
        return None
    except Exception as e:
        logger.error(f"Failed to fetch latest database date for 6M: {e}")
        return None

async def fetch_all_fred_data_6m() -> Optional[Dict[str, Any]]:
    """Fetch all latest observations from FRED API for 6M model"""
    try:
        results = {}
        all_dates = []  # Collect all dates to find the latest one
        
        # First, fetch monthly series to collect all their dates
        for name, series_id in SERIES_IDS_6M.items():
            if name not in WEEKLY_SERIES:  # Skip weekly series in first pass
                try:
                    data = await fetch_latest_observation_6m(series_id)
                    if data and "observations" in data and len(data["observations"]) > 0:
                        obs = data["observations"][0]
                        obs_date = obs["date"]
                        all_dates.append(obs_date)
                        
                        results[name] = {
                            "date": obs_date,
                            "value": float(obs["value"]) if obs["value"] != "." else None
                        }
                        logger.info(f"6M {name}: {obs_date} = {obs['value']}")
                    else:
                        logger.warning(f"No data found for 6M {name} ({series_id})")
                        results[name] = None
                except Exception as e:
                    logger.error(f"Failed to fetch 6M {name} ({series_id}): {e}")
                    results[name] = None
        
        # Find the latest date among all monthly series
        if all_dates:
            # Convert to datetime objects for proper comparison
            date_objects = [pd.to_datetime(date) for date in all_dates]
            latest_date_obj = max(date_objects)
            reference_date = latest_date_obj.strftime('%Y-%m-%d')
            
            logger.info(f"All monthly dates for 6M: {sorted(all_dates)}")
            logger.info(f"Using LATEST date as reference for 6M: {reference_date}")
            
            # Update all monthly series to use the reference date
            # (fetch data for the reference date for series that don't have it)
            updated_results = {}
            for name, series_id in SERIES_IDS_6M.items():
                if name not in WEEKLY_SERIES:
                    if results[name] and results[name]["date"] == reference_date:
                        # Already have data for the reference date
                        updated_results[name] = results[name]
                    else:
                        # Need to fetch data for the reference date
                        try:
                            # Fetch data for the specific reference month
                            ref_year = latest_date_obj.year
                            ref_month = latest_date_obj.month
                            
                            # Get data for the reference month
                            month_data = await fetch_data_for_specific_month_6m(series_id, ref_year, ref_month)
                            
                            if month_data:
                                updated_results[name] = {
                                    "date": reference_date,
                                    "value": month_data
                                }
                                logger.info(f"6M {name}: Updated to {reference_date} = {month_data}")
                            else:
                                logger.warning(f"No data found for 6M {name} in {ref_year}-{ref_month:02d}, using default")
                                updated_results[name] = {
                                    "date": reference_date,
                                    "value": 0.0
                                }
                        except Exception as e:
                            logger.error(f"Failed to fetch 6M {name} for reference date: {e}")
                            updated_results[name] = {
                                "date": reference_date,
                                "value": 0.0
                            }
            
            results = updated_results
            
            # Now handle weekly series using the reference date
            logger.info(f"Using reference date {reference_date} for 6M weekly series calculations")
            
            for name in WEEKLY_SERIES:
                if name in SERIES_IDS_6M:
                    series_id = SERIES_IDS_6M[name]
                    try:
                        # Calculate monthly average for the weekly series
                        monthly_avg = await fetch_monthly_average_for_weekly_series_6m(series_id, reference_date)
                        
                        if monthly_avg is not None:
                            results[name] = {
                                "date": reference_date,  # Use the same date as monthly series
                                "value": monthly_avg
                            }
                            logger.info(f"Successfully calculated monthly average for 6M {name}: {monthly_avg}")
                        else:
                            logger.warning(f"Could not calculate monthly average for 6M {name}, using default value")
                            results[name] = {
                                "date": reference_date,
                                "value": 0.0  # Default value if calculation fails
                            }
                    except Exception as e:
                        logger.error(f"Failed to fetch monthly average for 6M {name} ({series_id}): {e}")
                        results[name] = {
                            "date": reference_date,
                            "value": 0.0  # Default value on error
                        }
        else:
            logger.error("No valid dates found from any monthly series for 6M")
            return None
        
        return results
    except Exception as e:
        logger.error(f"Failed to fetch FRED data for 6M: {e}")
        return None

async def fetch_data_for_specific_month_6m(series_id: str, year: int, month: int) -> Optional[float]:
    """Fetch data for a specific month and year"""
    try:
        # Calculate the date range for the specific month
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"
        
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "sort_order": "desc",
            "limit": 1
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        if data and "observations" in data and len(data["observations"]) > 0:
            obs = data["observations"][0]
            if obs["value"] != ".":
                return float(obs["value"])
        
        return None
    except Exception as e:
        logger.error(f"Failed to fetch data for {series_id} in {year}-{month:02d}: {e}")
        return None

async def get_fred_data_and_update_database_6m() -> Optional[Dict[str, Any]]:
    """Fetch FRED data and update the database for 6M model"""
    try:
        # Step 1: Fetch all latest observations from FRED API
        fred_data = await fetch_all_fred_data_6m()
        if not fred_data:
            raise RuntimeError("Failed to fetch data from FRED API for 6M")
        
        # Step 2: Insert new data into database
        if not insert_fred_data_to_database_6m(fred_data):
            logger.warning("Failed to insert new data into database for 6M, continuing with prediction")
        
        return fred_data
    except Exception as e:
        logger.error(f"Failed to fetch FRED data and update database for 6M: {e}")
        return None

def insert_fred_data_to_database_6m(fred_data: Dict[str, Any]) -> bool:
    """Insert new FRED data into the database for 6M model"""
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
                logger.warning(f"Missing value for 6M {name}, using default")
        
        if observation_date:
            new_row["observation_date"] = observation_date
            
            # Log the data being inserted for debugging
            logger.info(f"Inserting 6M data: {new_row}")
            
            # Insert into database
            response = db_service.supabase.table('historical_data_6m')\
                .insert(new_row)\
                .execute()
            
            if response.data:
                logger.info(f"Successfully inserted new 6M data for {observation_date}")
                return True
            else:
                logger.error(f"Failed to insert 6M data: {response}")
                return False
        else:
            logger.error("No valid observation date found in FRED data for 6M")
            return False
            
    except Exception as e:
        logger.error(f"Failed to insert FRED data into database for 6M: {e}")
        return False

def get_latest_database_row_6m() -> Optional[Dict[str, Any]]:
    """
    Get the latest row from the database for 6M with NULL handling.
    Fetches the last 2 rows and forward fills NULL values from the previous row.
    """
    try:
        response = db_service.supabase.table('historical_data_6m')\
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
        logger.error(f"Failed to get latest database row for 6M: {e}")
        return None

def convert_to_input_features_6m(data_row: Dict[str, Any]) -> InputFeatures6M:
    """Convert database row or FRED data to InputFeatures6M format"""
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
        
        current_data = CurrentMonthData6M(
            observation_date=obs_date,
            PSTAX=safe_float(data_row.get("PSTAX")),
            USWTRADE=safe_float(data_row.get("USWTRADE")),
            MANEMP=safe_float(data_row.get("MANEMP")),
            CPIAPP=safe_float(data_row.get("CPIAPP")),
            CSUSHPISA=safe_float(data_row.get("CSUSHPISA")),
            ICSA=safe_float(data_row.get("ICSA")),
            fedfunds=safe_float(data_row.get("fedfunds")),
            BBKMLEIX=safe_float(data_row.get("BBKMLEIX")),
            TB3MS=safe_float(data_row.get("TB3MS")),
            USINFO=safe_float(data_row.get("USINFO")),
            PPIACO=safe_float(data_row.get("PPIACO")),
            CPIMEDICARE=safe_float(data_row.get("CPIMEDICARE")),
            UNEMPLOY=safe_float(data_row.get("UNEMPLOY")),
            TB1YR=safe_float(data_row.get("TB1YR")),
            USGOOD=safe_float(data_row.get("USGOOD")),
            CPIFOOD=safe_float(data_row.get("CPIFOOD")),
            UMCSENT=safe_float(data_row.get("UMCSENT")),
            SRVPRD=safe_float(data_row.get("SRVPRD")),
            GDP=safe_float(data_row.get("GDP")),
            INDPRO=safe_float(data_row.get("INDPRO")),
            recession=safe_int(data_row.get("recession"))
        )
        
        return InputFeatures6M(
            current_month_data=current_data,
            use_historical_data=True,
            historical_data_source="database"
        )
    except Exception as e:
        logger.error(f"Failed to convert data to InputFeatures6M: {e}")
        raise

async def get_latest_prediction_6m() -> ForecastResponse6M:
    """
    Main function that handles the complete flow for 6M:
    1. Check FRED for latest date
    2. Compare with database
    3. Fetch data accordingly (including weekly series averaging)
    4. Make prediction
    """
    try:
        # Initialize service if needed
        if not initialize_6m_service():
            raise RuntimeError("Failed to initialize 6M forecasting service")
        
        # Step 1: Get latest date from FRED
        fred_latest_date = await get_fred_latest_date_6m()
        if not fred_latest_date:
            raise RuntimeError("Could not fetch latest date from FRED API for 6M")
        
        # Step 2: Get latest date from database
        db_latest_date = get_database_latest_date_6m()
        
        logger.info(f"6M - FRED latest: {fred_latest_date}, DB latest: {db_latest_date}")
        
        # Step 3: Determine data source and fetch accordingly
        if db_latest_date and fred_latest_date == db_latest_date:
            # Dates match - use database data
            logger.info("Using existing database data for 6M")
            latest_row = get_latest_database_row_6m()
            if not latest_row:
                raise RuntimeError("Failed to get latest row from database for 6M")
            
            # Convert to input features
            features = convert_to_input_features_6m(latest_row)
            
        else:
            # Check if we should fetch new data (month-end fallback mechanism)
            should_fetch = is_last_5_days_of_month()
            
            if should_fetch:
                # Dates don't match or no DB data - fetch from FRED
                logger.info("Running month-end fallback: Fetching new data from FRED API for 6M (including weekly series averaging)")
                fred_data = await fetch_all_fred_data_6m()
            else:
                # Use existing database data even if dates don't match (not month-end)
                logger.info(f"Not in month-end period (last 5 days). Using existing database data for 6M even though dates don't match.")
                logger.info(f"FRED date: {fred_latest_date}, DB date: {db_latest_date}")
                latest_row = get_latest_database_row_6m()
                if not latest_row:
                    raise RuntimeError("Failed to get latest row from database for 6M")
                
                features = convert_to_input_features_6m(latest_row)
                prediction_result = predict_6m(features)
                prediction_result.feature_importance = {
                    "data_source": "database_scheduled",
                    "fred_date": fred_latest_date,
                    "db_date": db_latest_date,
                    "note": "Used database data (not month-end period)"
                }
                return prediction_result
            if not fred_data:
                raise RuntimeError("Failed to fetch data from FRED API for 6M")
            
            # Insert new data into database
            if not insert_fred_data_to_database_6m(fred_data):
                logger.warning("Failed to insert new data into database for 6M, continuing with prediction")
            
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
            features = convert_to_input_features_6m(data_row)
        
        # Step 4: Make prediction
        prediction_result = predict_6m(features)
        
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
        logger.error(f"Failed to get latest prediction for 6M: {e}")
        raise RuntimeError(f"6M Prediction failed: {e}")