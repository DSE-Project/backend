#!/usr/bin/env python3
"""
Test manual trigger of the enhanced FRED data scheduler
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.fred_data_scheduler import fred_scheduler
from services.database_service import db_service

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_manual_trigger():
    """
    Test manual trigger of enhanced scheduler functionality
    """
    print("🚀 Testing Manual Enhanced FRED Data Scheduler Trigger")
    print("=" * 60)
    
    try:
        # Test with 1M timeframe first
        timeframe = '1m'
        print(f"\n📊 Testing enhanced update for {timeframe} timeframe")
        
        # Get initial database state
        table_name = fred_scheduler.series_mappings[timeframe]['table']
        initial_records = await fred_scheduler.get_database_last_12_months(table_name)
        print(f"  Initial database records: {len(initial_records)}")
        
        if initial_records:
            latest_date = initial_records[0]['observation_date']
            print(f"  Latest record date: {latest_date}")
            
            # Show some sample values from latest record
            sample_values = {}
            for key, value in initial_records[0].items():
                if key not in ['observation_date'] and value is not None:
                    sample_values[key] = value
                    if len(sample_values) >= 3:  # Show first 3 non-null values
                        break
            print(f"  Sample current values: {sample_values}")
        
        # Run the enhanced update process
        print(f"\n🔄 Running enhanced update process for {timeframe}...")
        
        updates_made, records_created = await fred_scheduler.check_and_update_timeframe_data(timeframe)
        
        print(f"\n✅ Enhanced update completed:")
        print(f"  Records updated: {updates_made}")
        print(f"  Records created: {records_created}")
        
        # Get final database state
        final_records = await fred_scheduler.get_database_last_12_months(table_name)
        print(f"  Final database records: {len(final_records)}")
        
        if final_records:
            final_latest_date = final_records[0]['observation_date']
            print(f"  Latest record date after update: {final_latest_date}")
            
            # Show null counts to demonstrate NaN handling
            if final_records:
                null_counts = {}
                for record in final_records[:3]:  # Check first 3 records
                    record_date = record['observation_date']
                    null_count = sum(1 for k, v in record.items() if k != 'observation_date' and v is None)
                    total_fields = len(record) - 1  # -1 for observation_date
                    null_counts[record_date] = f"{null_count}/{total_fields}"
                
                print(f"  NaN/null counts by date: {null_counts}")
        
        # Show scheduler statistics
        stats = fred_scheduler.get_stats()
        print(f"\n📈 Scheduler Statistics:")
        print(f"  Total runs: {stats['statistics']['total_runs']}")
        print(f"  Successful runs: {stats['statistics']['successful_runs']}")
        print(f"  Last run: {stats['statistics']['last_run']}")
        
        print(f"\n✅ Manual trigger test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Manual trigger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🧪 Enhanced FRED Data Scheduler Manual Trigger Test")
    print(f"⏰ Test started at: {datetime.now()}")
    
    success = await test_manual_trigger()
    
    if success:
        print("\n🎉 Manual trigger test passed!")
        return 0
    else:
        print("\n❌ Manual trigger test failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)