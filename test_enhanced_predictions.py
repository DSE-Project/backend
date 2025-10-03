#!/usr/bin/env python3
"""
Test script to verify the enhanced forecast services with NULL handling and month-end fallback logic
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta

# Add the parent directory to the path to import from services
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.fred_data_service_1m import (
    get_latest_database_row,
    is_last_5_days_of_month,
    forward_fill_nulls_from_previous_row,
    get_latest_prediction_1m
)
from services.fred_data_service_3m import (
    get_latest_database_row_3m,
    get_latest_prediction_3m
)
from services.fred_data_service_6m import (
    get_latest_database_row_6m,
    get_latest_prediction_6m
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_month_end_detection():
    """Test the month-end detection logic"""
    print("🔍 Testing month-end detection...")
    
    is_month_end = is_last_5_days_of_month()
    today = datetime.now()
    print(f"Today: {today.strftime('%Y-%m-%d')}")
    print(f"Is last 5 days of month: {is_month_end}")
    
    # Also test for a specific date (for demonstration)
    import calendar
    last_day = calendar.monthrange(today.year, today.month)[1]
    days_until_end = last_day - today.day
    print(f"Days until end of month: {days_until_end}")
    
    return is_month_end

def test_forward_fill_logic():
    """Test the forward fill logic"""
    print("\n🔄 Testing forward fill logic...")
    
    # Create sample data with NULLs
    latest_row = {
        'observation_date': '2025-09-01',
        'fedfunds': None,  # NULL value
        'TB3MS': 4.22,
        'GDP': None,      # NULL value
        'UNEMPLOY': 6600
    }
    
    previous_row = {
        'observation_date': '2025-08-01',
        'fedfunds': 4.40,  # This should fill the NULL
        'TB3MS': 4.18,
        'GDP': 25000.0,   # This should fill the NULL
        'UNEMPLOY': 6550
    }
    
    print(f"Latest row (with NULLs): {latest_row}")
    print(f"Previous row: {previous_row}")
    
    filled_row = forward_fill_nulls_from_previous_row(latest_row, previous_row)
    print(f"Filled row: {filled_row}")
    
    # Verify the logic worked
    assert filled_row['fedfunds'] == 4.40, "fedfunds should be forward filled"
    assert filled_row['GDP'] == 25000.0, "GDP should be forward filled"
    assert filled_row['TB3MS'] == 4.22, "TB3MS should remain unchanged"
    assert filled_row['UNEMPLOY'] == 6600, "UNEMPLOY should remain unchanged"
    
    print("✅ Forward fill logic test passed!")
    return True

def test_database_row_fetching():
    """Test database row fetching with NULL handling"""
    print("\n📊 Testing database row fetching...")
    
    try:
        # Test 1M
        print("Testing 1M database row fetching...")
        row_1m = get_latest_database_row()
        if row_1m:
            print(f"✅ 1M latest row date: {row_1m.get('observation_date')}")
            # Count NULL values
            null_count = sum(1 for v in row_1m.values() if v is None)
            print(f"   NULL values in row: {null_count}")
        else:
            print("❌ No 1M data found")
        
        # Test 3M
        print("Testing 3M database row fetching...")
        row_3m = get_latest_database_row_3m()
        if row_3m:
            print(f"✅ 3M latest row date: {row_3m.get('observation_date')}")
            # Count NULL values
            null_count = sum(1 for v in row_3m.values() if v is None)
            print(f"   NULL values in row: {null_count}")
        else:
            print("❌ No 3M data found")
        
        # Test 6M
        print("Testing 6M database row fetching...")
        row_6m = get_latest_database_row_6m()
        if row_6m:
            print(f"✅ 6M latest row date: {row_6m.get('observation_date')}")
            # Count NULL values
            null_count = sum(1 for v in row_6m.values() if v is None)
            print(f"   NULL values in row: {null_count}")
        else:
            print("❌ No 6M data found")
            
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

async def test_prediction_flow():
    """Test the complete prediction flow"""
    print("\n🎯 Testing prediction flow...")
    
    try:
        # Test 1M prediction
        print("Testing 1M prediction...")
        result_1m = await get_latest_prediction_1m()
        print(f"✅ 1M prediction: {result_1m.prob_1m:.4f}")
        print(f"   Data source: {result_1m.feature_importance.get('data_source', 'unknown') if result_1m.feature_importance else 'unknown'}")
        
        # Test 3M prediction
        print("Testing 3M prediction...")
        result_3m = await get_latest_prediction_3m()
        print(f"✅ 3M prediction: {result_3m.prob_3m:.4f}")
        print(f"   Data source: {result_3m.feature_importance.get('data_source', 'unknown') if result_3m.feature_importance else 'unknown'}")
        
        # Test 6M prediction
        print("Testing 6M prediction...")
        result_6m = await get_latest_prediction_6m()
        print(f"✅ 6M prediction: {result_6m.prob_6m:.4f}")
        print(f"   Data source: {result_6m.feature_importance.get('data_source', 'unknown') if result_6m.feature_importance else 'unknown'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting enhanced forecast service tests...\n")
    
    # Test 1: Month-end detection
    test_month_end_detection()
    
    # Test 2: Forward fill logic
    test_forward_fill_logic()
    
    # Test 3: Database row fetching
    test_database_row_fetching()
    
    # Test 4: Complete prediction flow
    await test_prediction_flow()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())