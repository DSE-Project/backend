"""
Test script for the FRED Data Scheduler

This script demonstrates how to use the FRED Data Scheduler programmatically
and provides examples of manual triggering and monitoring.
"""

import asyncio
import json
from datetime import datetime
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.fred_data_scheduler import fred_scheduler
from config.scheduler_config import validate_config, get_scheduler_summary

async def test_scheduler():
    """Test the FRED Data Scheduler functionality"""
    
    print("🧪 FRED Data Scheduler Test Suite")
    print("=" * 50)
    
    # 1. Configuration Validation
    print("\n1. Configuration Validation")
    print("-" * 30)
    issues = validate_config()
    if issues:
        print("❌ Configuration Issues:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n⚠️  Please fix configuration issues before running the scheduler")
        return
    else:
        print("✅ Configuration is valid")
    
    # 2. Configuration Summary
    print("\n2. Configuration Summary")
    print("-" * 30)
    config_summary = get_scheduler_summary()
    print(json.dumps(config_summary, indent=2))
    
    # 3. Start Scheduler
    print("\n3. Starting Scheduler")
    print("-" * 30)
    try:
        await fred_scheduler.start_scheduler()
        print("✅ Scheduler started successfully")
    except Exception as e:
        print(f"❌ Failed to start scheduler: {e}")
        return
    
    # 4. Check Initial Status
    print("\n4. Initial Status")
    print("-" * 30)
    stats = fred_scheduler.get_stats()
    print(f"Scheduler Running: {stats['scheduler_running']}")
    print(f"Next Run: {stats['next_scheduled_run']}")
    
    # 5. Get Job Status
    print("\n5. Scheduled Jobs")
    print("-" * 30)
    jobs = fred_scheduler.get_job_status()
    for job in jobs:
        print(f"• {job['name']} ({job['id']})")
        print(f"  Next Run: {job['next_run_time']}")
        print(f"  Trigger: {job['trigger']}")
    
    # 6. Test Manual Update (1m only to keep it quick)
    print("\n6. Testing Manual Update (1m data)")
    print("-" * 30)
    print("🚀 Triggering manual update for 1m data...")
    
    try:
        start_time = datetime.now()
        result = await fred_scheduler.trigger_immediate_update('1m')
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Manual update completed in {duration:.1f}s")
        print(f"Records Updated: {result.get('updated', 0)}")
        print(f"Records Created: {result.get('created', 0)}")
        
    except Exception as e:
        print(f"❌ Manual update failed: {e}")
    
    # 7. Final Statistics
    print("\n7. Final Statistics")
    print("-" * 30)
    final_stats = fred_scheduler.get_stats()
    print(f"Total Runs: {final_stats['statistics']['total_runs']}")
    print(f"Successful Runs: {final_stats['statistics']['successful_runs']}")
    print(f"Failed Runs: {final_stats['statistics']['failed_runs']}")
    print(f"Records Updated: {final_stats['statistics']['records_updated']}")
    print(f"Records Created: {final_stats['statistics']['records_created']}")
    
    # 8. Stop Scheduler
    print("\n8. Stopping Scheduler")
    print("-" * 30)
    try:
        await fred_scheduler.stop_scheduler()
        print("✅ Scheduler stopped successfully")
    except Exception as e:
        print(f"❌ Failed to stop scheduler: {e}")
    
    print("\n🎉 Test completed!")

async def test_data_comparison():
    """Test the data comparison logic"""
    
    print("\n🔍 Testing Data Comparison Logic")
    print("=" * 50)
    
    # Test with sample data
    sample_fred_data = {
        'fedfunds': {'date': '2024-12-01', 'value': 4.40},
        'UNRATE': {'date': '2024-12-01', 'value': 4.0},
        'TB3MS': {'date': '2024-12-01', 'value': 4.22}
    }
    
    try:
        # Get actual database date for comparison
        db_date = await fred_scheduler.get_database_latest_date('historical_data_1m')
        print(f"Latest DB Date: {db_date}")
        
        if db_date:
            analysis = await fred_scheduler.analyze_data_updates(
                sample_fred_data, db_date, 'historical_data_1m'
            )
            
            print("\nData Analysis Results:")
            print(f"Value Updates: {len(analysis['value_updates'])}")
            print(f"New Data: {len(analysis['new_data'])}")
            print(f"No Changes: {len(analysis['no_changes'])}")
            
            if analysis['value_updates']:
                print("\nValue Updates Required:")
                for series, update_info in analysis['value_updates'].items():
                    print(f"  • {series}: {update_info['old_value']} → {update_info['new_value']}")
            
            if analysis['new_data']:
                print("\nNew Data Available:")
                for series, data_info in analysis['new_data'].items():
                    print(f"  • {series}: {data_info['date']} = {data_info['value']}")
        
    except Exception as e:
        print(f"❌ Data comparison test failed: {e}")

if __name__ == "__main__":
    print("Select test to run:")
    print("1. Full Scheduler Test")
    print("2. Data Comparison Test")
    print("3. Both Tests")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    async def run_tests():
        if choice in ['1', '3']:
            await test_scheduler()
        
        if choice in ['2', '3']:
            await test_data_comparison()
    
    # Run the selected tests
    asyncio.run(run_tests())