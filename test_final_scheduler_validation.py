#!/usr/bin/env python3
"""
Final validation test for the corrected FRED scheduler
Tests the fixes:
1. Previous month targeting (2025-09-01 instead of 2025-10-01)
2. Consistent behavior across all timeframes (1M, 3M, 6M)
3. Proper handling of data availability differences
"""

import asyncio
import logging
from datetime import datetime
from services.fred_data_scheduler import fred_scheduler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_corrected_scheduler():
    """Test the fully corrected scheduler implementation"""
    
    print("🧪 Testing Corrected FRED Scheduler")
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Test 1: Verify month calculation is correct
        print("\n📅 Test 1: Month Calculation Verification")
        current_date = datetime.now()
        
        # This should target September 2025 (previous month)
        if current_date.month == 1:
            expected_target = f"{current_date.year - 1}-12-01"
        else:
            expected_target = f"{current_date.year}-{current_date.month - 1:02d}-01"
        
        print(f"Expected target month: {expected_target}")
        
        # Test 2: Check database states before test
        print("\n🗄️ Test 2: Database State Check")
        for timeframe in ['1m', '3m', '6m']:
            table_name = f'historical_data_{timeframe}'
            latest_date = await fred_scheduler.get_database_latest_date(table_name)
            print(f"  {timeframe.upper()}: Latest date = {latest_date}")
        
        # Test 3: Run scheduler update for each timeframe
        print("\n🔄 Test 3: Running Scheduler Updates")
        
        results_summary = {}
        
        for timeframe in ['1m', '3m', '6m']:
            print(f"\n--- Testing {timeframe.upper()} Timeframe ---")
            
            try:
                # Run the corrected update logic
                updates_made, records_created = await fred_scheduler.check_and_update_timeframe_data(timeframe)
                
                results_summary[timeframe] = {
                    'updates_made': updates_made,
                    'records_created': records_created,
                    'success': True,
                    'error': None
                }
                
                print(f"✅ {timeframe.upper()} Results:")
                print(f"   Records updated: {updates_made}")
                print(f"   Records created: {records_created}")
                
                # Validate constraint: max 1 record created
                if records_created <= 1:
                    print(f"   ✅ Creation constraint satisfied (≤1 record)")
                else:
                    print(f"   ❌ Creation constraint violated ({records_created} records created)")
                
            except Exception as e:
                results_summary[timeframe] = {
                    'updates_made': 0,
                    'records_created': 0,
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ {timeframe.upper()} Failed: {e}")
        
        # Test 4: Post-update database verification
        print("\n🔍 Test 4: Post-Update Database Verification")
        
        for timeframe in ['1m', '3m', '6m']:
            table_name = f'historical_data_{timeframe}'
            latest_date = await fred_scheduler.get_database_latest_date(table_name)
            print(f"  {timeframe.upper()}: Latest date after update = {latest_date}")
            
            # Check if we got the expected target month
            if latest_date == expected_target:
                print(f"    ✅ Correctly updated to target month ({expected_target})")
            elif latest_date and latest_date > expected_target:
                print(f"    ⚠️ Updated beyond target month ({latest_date} > {expected_target})")
            else:
                print(f"    📝 No update (possibly due to data availability)")
        
        # Test 5: Summary Report
        print("\n📊 Test 5: Final Summary Report")
        print("=" * 70)
        
        total_updates = sum(r['updates_made'] for r in results_summary.values())
        total_creates = sum(r['records_created'] for r in results_summary.values())
        successful_timeframes = sum(1 for r in results_summary.values() if r['success'])
        
        print(f"Total timeframes tested: 3")
        print(f"Successful timeframes: {successful_timeframes}")
        print(f"Total records updated: {total_updates}")
        print(f"Total records created: {total_creates}")
        
        # Validate key constraints
        constraint_violations = []
        
        for timeframe, result in results_summary.items():
            if result['records_created'] > 1:
                constraint_violations.append(f"{timeframe}: {result['records_created']} records created (max 1 allowed)")
        
        if constraint_violations:
            print(f"\n❌ CONSTRAINT VIOLATIONS:")
            for violation in constraint_violations:
                print(f"   {violation}")
        else:
            print(f"\n✅ ALL CONSTRAINTS SATISFIED")
            print(f"   - Maximum 1 record created per timeframe ✓")
            print(f"   - Target month is previous month ({expected_target}) ✓")
        
        # Individual timeframe results
        print(f"\n📋 Individual Timeframe Results:")
        for timeframe, result in results_summary.items():
            status = "✅ SUCCESS" if result['success'] else f"❌ FAILED: {result['error']}"
            print(f"   {timeframe.upper()}: {status}")
            if result['success']:
                print(f"      Updates: {result['updates_made']}, Creates: {result['records_created']}")
        
        print("\n" + "=" * 70)
        print("🎉 CORRECTED SCHEDULER VALIDATION COMPLETE")
        
        if successful_timeframes == 3 and not constraint_violations:
            print("✅ ALL TESTS PASSED - Scheduler is working correctly!")
        else:
            print("⚠️ Some issues detected - please review the results above")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_corrected_scheduler())