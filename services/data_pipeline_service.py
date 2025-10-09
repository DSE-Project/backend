import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
import pandas as pd
import numpy as np
from fredapi import Fred
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import local services
from services.database_service import db_service
from services.model_training_service import ModelTrainingService

logger = logging.getLogger(__name__)

class DataPipelineService:
    """
    Automated data pipeline for fetching new economic data from FRED,
    storing it in the database, and triggering model retraining when needed.
    """
    
    def __init__(self):
        """Initialize the data pipeline service"""
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            raise ValueError("FRED_API_KEY environment variable is required")
        
        self.fred = Fred(api_key=self.fred_api_key)
        self.model_trainer = ModelTrainingService()
        
        # FRED series mapping based on your actual data columns
        # This includes all unique series from your 1m, 3m, and 6m models
        self.fred_series_mapping = {
            # Interest rates
            'fedfunds': 'FEDFUNDS',      # Federal Funds Rate
            'TB3MS': 'TB3MS',            # 3-Month Treasury Rate
            'TB6MS': 'TB6MS',            # 6-Month Treasury Rate
            'TB1YR': 'TB1YR',            # 1-Year Treasury Rate
            
            # Employment indicators
            'UNRATE': 'UNRATE',          # Unemployment Rate
            'UNEMPLOY': 'UNEMPLOY',      # Unemployment Level
            'MANEMP': 'MANEMP',          # Manufacturing Employment
            'ICSA': 'ICSA',              # Initial Claims
            
            # Economic production
            'GDP': 'GDP',                # Gross Domestic Product
            'REALGDP': 'GDPC1',          # Real GDP
            'INDPRO': 'INDPRO',          # Industrial Production Index
            
            # Price indices
            'PPIACO': 'PPIACO',          # Producer Price Index
            'CPIFOOD': 'CPIFOOD',        # CPI Food
            'CPIMEDICARE': 'CPIMEDSL',   # CPI Medical Care
            'CPIRENT': 'CPIRENT',        # CPI Rent
            'CPIAPP': 'CPIAPPSL',        # CPI Apparel
            'PCEPI': 'PCEPI',            # PCE Price Index
            
            # Trade and industry sectors
            'USTPU': 'USTPU',            # Trade, Transportation & Utilities Employment
            'USGOOD': 'USGOOD',          # Goods Producing Employment
            'SRVPRD': 'SRVPRD',          # Service Providing Employment
            'USCONS': 'USCONS',          # Construction Employment
            'USWTRADE': 'USWTRADE',      # Wholesale Trade Employment
            'USTRADE': 'USTRADE',        # Retail Trade Employment
            'USINFO': 'USINFO',         # Information Employment
            
            # Financial indicators
            'PSAVERT': 'PSAVERT',        # Personal Saving Rate
            'PSTAX': 'PSTAX',            # Personal Tax Rate
            'COMREAL': 'COMREAL',        # Commercial Real Estate Loans
            'COMLOAN': 'COMLOAN',        # Commercial Loans
            'SECURITYBANK': 'TOTRESNS', # Bank Securities (using Total Reserves)
            'M1SL': 'M1SL',              # M1 Money Stock
            'M2SL': 'M2SL',              # M2 Money Stock
            
            # Housing and sentiment
            'CSUSHPISA': 'CSUSHPISA',    # Case-Shiller Home Price Index
            'UMCSENT': 'UMCSENT',        # Consumer Sentiment
            'BBKMLEIX': 'DEXUSEU',       # USD/EUR Exchange Rate (proxy for BBKMLEIX)
        }
        
        # Tables to update for different forecasting periods
        self.data_tables = ['historical_data_1m', 'historical_data_3m', 'historical_data_6m']
        
        # Minimum data points required before triggering retrain
        self.min_new_records_for_retrain = 5
        
        logger.info("DataPipelineService initialized successfully")
    
    async def run_pipeline(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Main pipeline execution method
        
        Args:
            force_retrain: If True, force model retraining regardless of new data amount
            
        Returns:
            Dict containing pipeline execution results
        """
        pipeline_start = datetime.now()
        results = {
            'pipeline_start': pipeline_start.isoformat(),
            'status': 'running',
            'new_data_fetched': False,
            'data_stored': False,
            'models_retrained': False,
            'errors': [],
            'summary': {}
        }
        
        try:
            logger.info("🚀 Starting data pipeline execution...")
            
            # Step 1: Check for new data availability
            new_data_available = await self.check_for_new_data()
            results['new_data_check'] = new_data_available
            
            if new_data_available or force_retrain:
                logger.info("📡 New data detected or retrain forced - proceeding with pipeline...")
                
                # Step 2: Fetch new data from FRED
                new_data, fetch_summary = await self.fetch_new_data()
                results['new_data_fetched'] = len(new_data) > 0
                results['fetch_summary'] = fetch_summary
                
                if len(new_data) > 0:
                    # Step 3: Validate and clean the data
                    validated_data = self.validate_and_clean_data(new_data)
                    results['validated_records'] = len(validated_data)
                    
                    # Step 4: Store data in database
                    storage_results = await self.store_data_in_database(validated_data)
                    results['data_stored'] = storage_results['success']
                    results['storage_summary'] = storage_results
                    
                    # Step 5: Check if retraining is needed
                    should_retrain = (
                        force_retrain or 
                        storage_results.get('new_records_count', 0) >= self.min_new_records_for_retrain
                    )
                    
                    if should_retrain:
                        # Step 6: Trigger model retraining
                        retrain_results = await self.retrain_models()
                        results['models_retrained'] = retrain_results['success']
                        results['retrain_summary'] = retrain_results
                    else:
                        logger.info(f"⏳ Skipping retrain - only {storage_results.get('new_records_count', 0)} new records")
                        results['retrain_summary'] = {'skipped': True, 'reason': 'insufficient_new_data'}
                else:
                    logger.info("📊 No new data to process")
                    results['retrain_summary'] = {'skipped': True, 'reason': 'no_new_data'}
            else:
                logger.info("✅ No new data available - pipeline complete")
                results['retrain_summary'] = {'skipped': True, 'reason': 'no_new_data_available'}
            
            results['status'] = 'completed'
            results['pipeline_duration'] = (datetime.now() - pipeline_start).total_seconds()
            
            # Log pipeline completion
            await self.log_pipeline_execution(results)
            
            logger.info(f"🎉 Pipeline completed successfully in {results['pipeline_duration']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            results['status'] = 'failed'
            results['errors'].append(error_msg)
            results['pipeline_duration'] = (datetime.now() - pipeline_start).total_seconds()
            
            # Log pipeline failure
            await self.handle_pipeline_failure(e, results)
            
            raise
        
        return results
    
    def get_model_required_features(self) -> Dict[str, List[str]]:
        """
        Get the required features for each model based on your actual data structure
        
        Returns:
            Dict mapping model periods to their required feature lists
        """
        return {
            '1m': [
                'fedfunds', 'TB3MS', 'TB6MS', 'TB1YR', 'USTPU', 'USGOOD', 'SRVPRD', 'USCONS', 
                'MANEMP', 'USWTRADE', 'USTRADE', 'USINFO', 'UNRATE', 'UNEMPLOY', 'CPIFOOD', 
                'CPIMEDICARE', 'CPIRENT', 'CPIAPP', 'GDP', 'REALGDP', 'PCEPI', 'PSAVERT', 
                'PSTAX', 'COMREAL', 'COMLOAN', 'SECURITYBANK', 'PPIACO', 'M1SL', 'M2SL'
            ],
            '3m': [
                'ICSA', 'CPIMEDICARE', 'USWTRADE', 'BBKMLEIX', 'COMLOAN', 'UMCSENT', 'MANEMP', 
                'fedfunds', 'PSTAX', 'USCONS', 'USGOOD', 'USINFO', 'CPIAPP', 'CSUSHPISA', 
                'SECURITYBANK', 'SRVPRD', 'INDPRO', 'TB6MS', 'UNEMPLOY', 'USTPU'
            ],
            '6m': [
                'PSTAX', 'USWTRADE', 'MANEMP', 'CPIAPP', 'CSUSHPISA', 'ICSA', 'fedfunds', 
                'BBKMLEIX', 'TB3MS', 'USINFO', 'PPIACO', 'CPIMEDICARE', 'UNEMPLOY', 'TB1YR', 
                'USGOOD', 'CPIFOOD', 'UMCSENT', 'SRVPRD', 'GDP', 'INDPRO'
            ]
        }
    
    def get_all_required_series(self) -> Set[str]:
        """
        Get all unique FRED series required across all models
        
        Returns:
            Set of all unique FRED series IDs needed
        """
        model_features = self.get_model_required_features()
        all_features = set()
        
        for period_features in model_features.values():
            all_features.update(period_features)
        
        # Map feature names to FRED series IDs
        required_series = set()
        for feature in all_features:
            if feature in self.fred_series_mapping:
                required_series.add(self.fred_series_mapping[feature])
            else:
                logger.warning(f"⚠️ No FRED series mapping found for feature: {feature}")
        
        return required_series
    
    async def check_for_new_data(self) -> bool:
        """
        Check if new data is available from FRED by comparing
        the latest data timestamps with our database records
        
        Returns:
            bool: True if new data is available
        """
        try:
            logger.info("🔍 Checking for new data availability...")
            
            # Get the latest data timestamp from our database
            latest_db_date = await self.get_latest_database_timestamp()
            
            if not latest_db_date:
                logger.info("📊 No existing data found - will fetch initial dataset")
                return True
            
            # Check a few key indicators for updates
            key_indicators = ['UNRATE', 'CPIAUCSL', 'GDP']
            
            for series_id in key_indicators:
                try:
                    # Get the latest data point from FRED
                    latest_fred_data = self.fred.get_series(
                        series_id, 
                        limit=1,
                        sort_order='desc'
                    )
                    
                    if not latest_fred_data.empty:
                        latest_fred_date = latest_fred_data.index[0].date()
                        
                        if latest_fred_date > latest_db_date:
                            logger.info(f"📈 New data found for {series_id}: {latest_fred_date} > {latest_db_date}")
                            return True
                            
                except Exception as e:
                    logger.warning(f"⚠️ Error checking {series_id}: {str(e)}")
                    continue
            
            logger.info("✅ No new data detected")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking for new data: {str(e)}")
            # Assume new data is available if check fails
            return True
    
    async def get_latest_database_timestamp(self) -> Optional[datetime.date]:
        """Get the latest observation date from our database"""
        try:
            # Check the 1m table for the latest date (all tables should have similar dates)
            df = db_service.load_historical_data('historical_data_1m')
            
            if df is not None and not df.empty:
                # The index should be observation_date
                latest_date = df.index.max().date()
                logger.info(f"📅 Latest database timestamp: {latest_date}")
                return latest_date
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting latest database timestamp: {str(e)}")
            return None
    
    async def fetch_new_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch new economic data from FRED API
        
        Returns:
            Tuple of (DataFrame with new data, summary dict)
        """
        logger.info("📡 Fetching new data from FRED API...")
        
        # Calculate date range for fetching
        end_date = datetime.now().date()
        
        # Get latest DB date, or fetch last 2 years if no data exists
        latest_db_date = await self.get_latest_database_timestamp()
        if latest_db_date:
            start_date = latest_db_date - timedelta(days=30)  # Fetch with some overlap
        else:
            start_date = end_date - timedelta(days=730)  # Fetch last 2 years for initial load
        
        all_data = {}
        fetch_summary = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'series_fetched': 0,
            'series_failed': 0,
            'total_records': 0,
            'series_details': {}
        }
        
        logger.info(f"📊 Fetching data from {start_date} to {end_date}")
        
        # Get all required series across all models
        required_series = self.get_all_required_series()
        logger.info(f"📊 Total series to fetch: {len(required_series)}")
        
        # Fetch only the series we need (use feature name as key, FRED ID as value)
        series_to_fetch = {k: v for k, v in self.fred_series_mapping.items() if v in required_series}
        
        for indicator_name, series_id in series_to_fetch.items():
            try:
                logger.info(f"🔄 Fetching {indicator_name} ({series_id})...")
                
                # Fetch the series data
                series_data = self.fred.get_series(
                    series_id,
                    start=start_date,
                    end=end_date
                )
                
                if not series_data.empty:
                    # Remove any NaN values
                    series_data = series_data.dropna()
                    
                    all_data[indicator_name] = series_data
                    fetch_summary['series_fetched'] += 1
                    fetch_summary['total_records'] += len(series_data)
                    fetch_summary['series_details'][indicator_name] = {
                        'records': len(series_data),
                        'latest_date': series_data.index.max().strftime('%Y-%m-%d'),
                        'latest_value': float(series_data.iloc[-1])
                    }
                    
                    logger.info(f"✅ {indicator_name}: {len(series_data)} records fetched")
                else:
                    logger.warning(f"⚠️ No data returned for {indicator_name} ({series_id})")
                    fetch_summary['series_failed'] += 1
                
                # Add small delay to be nice to the API
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error fetching {indicator_name} ({series_id}): {str(e)}")
                fetch_summary['series_failed'] += 1
                fetch_summary['series_details'][indicator_name] = {'error': str(e)}
        
        # Combine all series into a single DataFrame
        if all_data:
            combined_df = pd.DataFrame(all_data)
            combined_df.index.name = 'observation_date'
            
            logger.info(f"📊 Total data fetched: {len(combined_df)} rows, {len(combined_df.columns)} columns")
            return combined_df, fetch_summary
        else:
            logger.warning("⚠️ No data was successfully fetched")
            return pd.DataFrame(), fetch_summary
    
    def validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the fetched data
        
        Args:
            data: Raw data from FRED API
            
        Returns:
            Cleaned and validated DataFrame
        """
        logger.info("🧹 Validating and cleaning data...")
        
        if data.empty:
            logger.warning("⚠️ No data to validate")
            return data
        
        original_shape = data.shape
        
        # Remove rows where all values are NaN
        data_cleaned = data.dropna(how='all')
        
        # Forward fill missing values (common in economic data)
        data_cleaned = data_cleaned.fillna(method='ffill')
        
        # Remove any remaining NaN values
        data_cleaned = data_cleaned.dropna()
        
        # Ensure index is datetime
        if not isinstance(data_cleaned.index, pd.DatetimeIndex):
            data_cleaned.index = pd.to_datetime(data_cleaned.index)
        
        # Sort by date
        data_cleaned = data_cleaned.sort_index()
        
        # Remove duplicates
        data_cleaned = data_cleaned[~data_cleaned.index.duplicated(keep='last')]
        
        logger.info(f"✅ Data validation complete: {original_shape} -> {data_cleaned.shape}")
        
        return data_cleaned
    
    async def store_data_in_database(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Store the validated data in all relevant database tables
        
        Args:
            data: Cleaned DataFrame to store
            
        Returns:
            Dict with storage results
        """
        logger.info("💾 Storing data in database...")
        
        storage_results = {
            'success': False,
            'tables_updated': [],
            'new_records_count': 0,
            'errors': [],
            'details': {}
        }
        
        if data.empty:
            storage_results['success'] = True
            logger.info("✅ No data to store")
            return storage_results
        
        try:
            # Store data in each table
            for table_name in self.data_tables:
                try:
                    # Get existing data to avoid duplicates
                    existing_data = db_service.load_historical_data(table_name)
                    
                    if existing_data is not None and not existing_data.empty:
                        # Find new records (dates not in existing data)
                        existing_dates = set(existing_data.index.date)
                        new_records = data[~data.index.date.isin(existing_dates)]
                    else:
                        new_records = data.copy()
                    
                    if not new_records.empty:
                        # Prepare data for insertion
                        records_to_insert = []
                        for date, row in new_records.iterrows():
                            record = {
                                'observation_date': date.strftime('%Y-%m-%d'),
                                **{col: float(val) if pd.notna(val) else None 
                                   for col, val in row.items()}
                            }
                            records_to_insert.append(record)
                        
                        # Insert into Supabase
                        if db_service.supabase:
                            response = db_service.supabase.table(table_name).insert(records_to_insert).execute()
                            
                            if response.data:
                                inserted_count = len(response.data)
                                storage_results['tables_updated'].append(table_name)
                                storage_results['new_records_count'] += inserted_count
                                storage_results['details'][table_name] = {
                                    'new_records': inserted_count,
                                    'date_range': f"{new_records.index.min().date()} to {new_records.index.max().date()}"
                                }
                                
                                logger.info(f"✅ {table_name}: {inserted_count} new records inserted")
                            else:
                                error_msg = f"No data returned after insertion to {table_name}"
                                storage_results['errors'].append(error_msg)
                                logger.error(f"❌ {error_msg}")
                        else:
                            error_msg = f"Database connection not available for {table_name}"
                            storage_results['errors'].append(error_msg)
                            logger.error(f"❌ {error_msg}")
                    else:
                        logger.info(f"ℹ️ {table_name}: No new records to insert")
                        storage_results['details'][table_name] = {'new_records': 0}
                
                except Exception as e:
                    error_msg = f"Error storing data in {table_name}: {str(e)}"
                    storage_results['errors'].append(error_msg)
                    logger.error(f"❌ {error_msg}")
            
            storage_results['success'] = len(storage_results['errors']) == 0
            
            if storage_results['success']:
                logger.info(f"✅ Data storage complete: {storage_results['new_records_count']} total new records")
            else:
                logger.error(f"❌ Data storage completed with errors: {storage_results['errors']}")
            
        except Exception as e:
            error_msg = f"Critical error during data storage: {str(e)}"
            storage_results['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        return storage_results
    
    def validate_data_for_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that the fetched data has the required columns for each model
        
        Args:
            data: DataFrame with fetched economic indicators
            
        Returns:
            Dict with validation results for each model
        """
        model_features = self.get_model_required_features()
        validation_results = {}
        
        available_columns = set(data.columns)
        
        for period, required_features in model_features.items():
            missing_features = [f for f in required_features if f not in available_columns]
            available_features = [f for f in required_features if f in available_columns]
            
            validation_results[period] = {
                'total_required': len(required_features),
                'available_count': len(available_features),
                'missing_count': len(missing_features),
                'missing_features': missing_features,
                'coverage_percent': (len(available_features) / len(required_features)) * 100,
                'can_retrain': len(missing_features) == 0
            }
            
            logger.info(f"📊 {period} model validation: {len(available_features)}/{len(required_features)} features available ({validation_results[period]['coverage_percent']:.1f}%)")
            
            if missing_features:
                logger.warning(f"⚠️ {period} model missing features: {missing_features}")
        
        return validation_results
    
    async def retrain_models(self) -> Dict[str, Any]:
        """
        Trigger retraining of all forecasting models
        
        Returns:
            Dict with retraining results
        """
        logger.info("🤖 Starting model retraining...")
        
        retrain_results = {
            'success': False,
            'models_retrained': [],
            'errors': [],
            'details': {}
        }
        
        try:
            # First validate that we have the required data for each model
            # Load some recent data to validate
            sample_data = db_service.load_historical_data('historical_data_1m')
            if sample_data is not None and not sample_data.empty:
                validation_results = self.validate_data_for_models(sample_data)
                retrain_results['validation'] = validation_results
            
            # Retrain all models
            for period in ['1m', '3m', '6m']:
                try:
                    logger.info(f"🔄 Retraining {period} model...")
                    
                    # Check if we have sufficient data for this model
                    if 'validation' in retrain_results:
                        model_validation = retrain_results['validation'].get(period, {})
                        if not model_validation.get('can_retrain', True):
                            logger.warning(f"⚠️ Skipping {period} model retrain due to missing features: {model_validation.get('missing_features', [])}")
                            retrain_results['errors'].append(f"{period} model skipped - missing required features")
                            continue
                    
                    result = await self.model_trainer.retrain_model(period)
                    
                    if result['success']:
                        retrain_results['models_retrained'].append(period)
                        retrain_results['details'][period] = result
                        logger.info(f"✅ {period} model retrained successfully")
                    else:
                        error_msg = f"Failed to retrain {period} model: {result.get('error', 'Unknown error')}"
                        retrain_results['errors'].append(error_msg)
                        logger.error(f"❌ {error_msg}")
                
                except Exception as e:
                    error_msg = f"Error retraining {period} model: {str(e)}"
                    retrain_results['errors'].append(error_msg)
                    logger.error(f"❌ {error_msg}")
            
            retrain_results['success'] = len(retrain_results['models_retrained']) > 0
            
            if retrain_results['success']:
                logger.info(f"✅ Model retraining complete: {len(retrain_results['models_retrained'])} models updated")
            else:
                logger.error("❌ All model retraining attempts failed")
            
        except Exception as e:
            error_msg = f"Critical error during model retraining: {str(e)}"
            retrain_results['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        return retrain_results
    
    async def log_pipeline_execution(self, results: Dict[str, Any]):
        """Log pipeline execution results"""
        try:
            # You can implement logging to database or file here
            logger.info(f"📝 Pipeline execution logged: {results['status']}")
        except Exception as e:
            logger.error(f"❌ Error logging pipeline execution: {str(e)}")
    
    async def handle_pipeline_failure(self, error: Exception, results: Dict[str, Any]):
        """Handle pipeline failure - could send alerts, etc."""
        try:
            logger.error(f"🚨 Pipeline failure handled: {str(error)}")
            # You can implement alerting here (email, Slack, etc.)
        except Exception as e:
            logger.error(f"❌ Error handling pipeline failure: {str(e)}")

# Initialize the service
pipeline_service = DataPipelineService()