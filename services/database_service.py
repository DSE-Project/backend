import os
import pandas as pd
from supabase import create_client, Client
from typing import Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.supabase: Optional[Client] = None
        self._initialize_connection()
        
        # Column mapping from lowercase (DB) to original case (model expects)
        self.column_mapping = {
            'tb3ms': 'TB3MS',
            'fedfunds': 'fedfunds',  # Already lowercase
            'tb6ms': 'TB6MS', 
            'tb1yr': 'TB1YR',
            'usgood': 'USGOOD',
            'ustpu': 'USTPU',
            'srvprd': 'SRVPRD',
            'uscons': 'USCONS',
            'manemp': 'MANEMP',
            'uswtrade': 'USWTRADE',
            'ustrade': 'USTRADE',
            'usinfo': 'USINFO',
            'unrate': 'UNRATE',
            'unemploy': 'UNEMPLOY',
            'cpifood': 'CPIFOOD',
            'cpimedicare': 'CPIMEDICARE',
            'cpirent': 'CPIRENT',
            'cpiapp': 'CPIAPP',
            'gdp': 'GDP',
            'realgdp': 'REALGDP',
            'pcepi': 'PCEPI',
            'psavert': 'PSAVERT',
            'pstax': 'PSTAX',
            'comreal': 'COMREAL',
            'comloan': 'COMLOAN',
            'securitybank': 'SECURITYBANK',
            'ppiaco': 'PPIACO',
            'm1sl': 'M1SL',
            'm2sl': 'M2SL',
            'csushpisa': 'CSUSHPISA',
            'icsa': 'ICSA',
            'bbkmleix': 'BBKMLEIX',
            'umcsent': 'UMCSENT',
            'indpro': 'INDPRO',
            'recession': 'recession'
        }
    
    def _initialize_connection(self):
        """Initialize Supabase connection"""
        try:
            # Get Supabase credentials from environment
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_ANON_KEY')
            
            if not supabase_url or not supabase_key:
                raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY environment variables must be set")
            
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("✅ Supabase connection initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase connection: {e}")
            raise
    
    def _map_columns_to_original_case(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map lowercase column names back to original case"""
        df_mapped = df.copy()
        
        # Create reverse mapping for columns that exist in the dataframe
        existing_mappings = {col: self.column_mapping.get(col, col) 
                           for col in df.columns if col in self.column_mapping}
        
        df_mapped = df_mapped.rename(columns=existing_mappings)
        
        logger.info(f"Mapped {len(existing_mappings)} columns to original case")
        return df_mapped
    
    def load_historical_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Load historical data from Supabase table"""
        try:
            # Query all data from the table, ordered by observation_date
            response = self.supabase.table(table_name).select("*").order('observation_date').execute()
            
            if response.data:
                # Convert to DataFrame
                df = pd.DataFrame(response.data)
                
                # Map column names to original case
                df = self._map_columns_to_original_case(df)
                
                # Set observation_date as index and parse as datetime
                df['observation_date'] = pd.to_datetime(df['observation_date'])
                df.set_index('observation_date', inplace=True)
                
                logger.info(f"✅ Loaded {len(df)} records from {table_name}")
                return df
            else:
                logger.warning(f"⚠️ No data found in table {table_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load data from {table_name}: {e}")
            return None
    
    def load_data_with_filter(self, table_name: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Load historical data with date filtering"""
        try:
            query = self.supabase.table(table_name).select("*")
            
            if start_date:
                query = query.gte('observation_date', start_date)
            if end_date:
                query = query.lte('observation_date', end_date)
                
            response = query.order('observation_date').execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                
                # Map column names to original case
                df = self._map_columns_to_original_case(df)
                
                df['observation_date'] = pd.to_datetime(df['observation_date'])
                df.set_index('observation_date', inplace=True)
                
                logger.info(f"✅ Loaded {len(df)} filtered records from {table_name}")
                return df
            else:
                logger.warning(f"⚠️ No data found in table {table_name} with given filters")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load filtered data from {table_name}: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test Supabase connection"""
        try:
            # Try a simple query to test connection
            response = self.supabase.table('historical_data_1m').select("observation_date").limit(1).execute()
            print(response)
            return True
        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {e}")
            return False

# Global database service instance
db_service = DatabaseService()

