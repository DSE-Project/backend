import os
import pandas as pd
from supabase import create_client, Client
from typing import Optional
import logging
from dotenv import load_dotenv
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
import time
from colorama import Fore, Style, init

# Initialize colorama (for Windows compatibility)
init(autoreset=True)


# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.supabase: Optional[Client] = None
        self._initialize_connection()
        
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
    #-----------------------------------------------------------------------------------
    def validate_dataframe(self, df: pd.DataFrame, table_name: str) -> Optional[pd.DataFrame]:
        """Validate dataframe using Pandera + extra data quality tests + pytest-style printing"""
        start_time = time.time()  # Track execution time
        
        try:
            # ----- Define schema (basic type & range checks) -----
            schema = DataFrameSchema({
                "GDP": Column(float, nullable=False),
                "fedfunds": Column(float, checks=Check.in_range(0, 20)),
                "REALGDP": Column(float, nullable=False),
                "recession": Column(int, checks=Check.isin([0, 1]))
                # Add more numeric columns and their checks as needed
            })

            # Run Pandera validation
            validated_df = schema.validate(df, lazy=True)

            # ----- Extra quality checks -----
            missing_values = df.isnull().sum()
            duplicates = df.duplicated().sum()
            passed, failed = 0, 0

            print(f"\n{Fore.CYAN}============================= DATA VALIDATION ============================={Style.RESET_ALL}")
            print(f"Table: {table_name}\n")

            # --- Missing values check for all columns ---
            missing_columns = missing_values[missing_values > 0].index.tolist()

            if not missing_columns:
                print(f"{Fore.GREEN}PASSED:{Style.RESET_ALL} No missing values in any column ✅")
                passed += 1
            else:
                print(f"{Fore.RED}FAILED:{Style.RESET_ALL} Missing values found in columns ❌: {missing_columns}")
                failed += 1


            # --- Duplicate row check ---
            if duplicates > 0:
                print(f"{Fore.RED}FAILED:{Style.RESET_ALL} Found {duplicates} duplicate rows ❌")
                failed += 1
            else:
                print(f"{Fore.GREEN}PASSED:{Style.RESET_ALL} No duplicate rows ✅")
                passed += 1


            
            # --- Summary ---
            duration = time.time() - start_time
            print(f"\n{Fore.CYAN}---------------------------- Test Result ----------------------------------{Style.RESET_ALL}")
            if failed == 0:
                print(f"{Fore.GREEN}=== {passed} passed, {failed} failed in {duration:.2f}s ==={Style.RESET_ALL}\n")
            else:
                print(f"{Fore.RED}=== {passed} passed, {failed} failed in {duration:.2f}s ==={Style.RESET_ALL}\n")
            print(f"{Fore.CYAN}============================================================================{Style.RESET_ALL}")

            logger.info(f"✅ Data validation passed for {table_name}")
            return validated_df

        except pa.errors.SchemaErrors as e:
            duration = time.time() - start_time
            print(f"\n{Fore.CYAN}============================= DATA VALIDATION ============================={Style.RESET_ALL}")
            print(f"Table: {table_name}\n")
            print(f"{Fore.RED}Schema validation FAILED ❌{Style.RESET_ALL}\n")
            print(e.failure_cases.to_string(index=False))
            print(f"\n{Fore.RED}=== 0 passed, 1 failed in {duration:.2f}s ==={Style.RESET_ALL}\n")
            print(f"{Fore.CYAN}============================================================================{Style.RESET_ALL}")

            logger.warning(f"⚠️ Data validation failed for {table_name}")
            logger.warning(e.failure_cases)
            return None

    #---------------------------------------------------------------------------------------
    def load_historical_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Load historical data from Supabase table"""
        try:
            # Query all data from the table, ordered by observation_date
            response = self.supabase.table(table_name).select("*").order('observation_date').execute()
            
            if response.data:
                # Convert to DataFrame
                df = pd.DataFrame(response.data)
                
                
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
        
    def load_last_n_rows(self, table_name: str, n: int = 2) -> Optional[pd.DataFrame]:
        try:
            # Query last n rows ordered descending
            response = (
                self.supabase
                .table(table_name)
                .select("*")
                .order("observation_date", desc=True)
                .limit(n)
                .execute()
            )

            if response.data:
                df = pd.DataFrame(response.data)
                df['observation_date'] = pd.to_datetime(df['observation_date'])
                df.set_index('observation_date', inplace=True)

                # Sort ascending so the last rows are at the bottom
                df = df.sort_index()
                logger.info(f"✅ Loaded last {n} rows from {table_name}")
                return df
            else:
                logger.warning(f"⚠️ No data found in {table_name}")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to load last {n} rows from {table_name}: {e}")
            return None


# Global database service instance
db_service = DatabaseService()

