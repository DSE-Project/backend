import httpx
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import logging
from services.database_service import db_service
from services.forecast_service_3m import predict_3m, initialize_3m_service
from schemas.forecast_schema_3m import InputFeatures3M, CurrentMonthData3M, ForecastResponse3M

logger = logging.getLogger(__name__)

# FRED API Configuration
FRED_API_KEY = "ef123a07a5f12077a0144db1f8cabf0d"
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series IDs mapping for 3M model
SERIES_IDS_3M = {
    "ICSA": "ICSA",
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

async def get_fred_latest_date_3m() -> Optional[str]:
    """Get the latest observation date from FRED API using FEDFUNDS series"""
    try:
        data = await fetch_latest_observation_3m(SERIES_IDS_3M["fedfunds"])
        if data and "observations" in data and len(data["observations"]) > 0:
            latest_date = data["observations"][0]["date"]
            logger.info(f"Latest FRED date for 3M: {latest_date}")
            return latest_date
        return None
    except Exception as e:
        logger.error(f"Failed to fetch latest FRED date for 3M: {e}")
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
        for name, series_id in SERIES_IDS_3M.items():
            try:
                data = await fetch_latest_observation_3m(series_id)
                if data and "observations" in data and len(data["observations"]) > 0:
                    obs = data["observations"][0]
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
    """Get the latest row from the database for 3M"""
    try:
        response = db_service.supabase.table('historical_data_3m')\
            .select("*")\
            .order('observation_date', desc=True)\
            .limit(1)\
            .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
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
    3. Fetch data accordingly
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
            # Dates don't match or no DB data - fetch from FRED
            logger.info("Fetching new data from FRED API for 3M")
            fred_data = await fetch_all_fred_data_3m()
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