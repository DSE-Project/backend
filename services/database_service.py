import os
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.engine = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            # Get database URL from environment
            database_url = os.getenv('SUPABASE_DATABASE_URL')
            if not database_url:
                raise ValueError("SUPABASE_DATABASE_URL environment variable not set")
            
            self.engine = create_engine(database_url)
            logger.info("✅ Database connection initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connection: {e}")
            raise
    
    def load_historical_data(self, table_name: str, date_column: str = 'observation_date') -> Optional[pd.DataFrame]:
        """Load historical data from database table"""
        try:
            query = f"""
            SELECT * FROM {table_name} 
            ORDER BY {date_column}
            """
            
            df = pd.read_sql_query(
                query, 
                self.engine, 
                index_col=date_column,
                parse_dates=[date_column]
            )
            
            logger.info(f"✅ Loaded {len(df)} records from {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data from {table_name}: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False

# Global database service instance
db_service = DatabaseService()