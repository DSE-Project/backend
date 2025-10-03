#!/usr/bin/env python3
"""
Investigation script to understand why 3M scheduler behaves differently
"""

import asyncio
import logging
from datetime import datetime
from services.fred_data_scheduler import fred_scheduler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def investigate_3m_issue():
    """Investigate why 3M doesn't create records like 1M and 6M"""
    
    print("🔍 Investigating 3M scheduler behavior...")
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    try:
        # Test series IDs loading for each timeframe
        for timeframe in ['1m', '3m', '6m']:
            print(f"\n📊 Testing {timeframe.upper()} timeframe:")
            
            if timeframe == '1m':
                from services.fred_data_service_1m import SERIES_IDS as SERIES_1M
                series_ids = SERIES_1M
            elif timeframe == '3m':
                from services.fred_data_service_3m import SERIES_IDS_3M
                series_ids = SERIES_IDS_3M
            elif timeframe == '6m':
                try:
                    from services.fred_data_service_6m import SERIES_IDS_6M
                    series_ids = SERIES_IDS_6M
                except ImportError:
                    print(f"  ⚠️  Could not import 6M series IDs")
                    series_ids = {}
            
            print(f"  Series count: {len(series_ids)}")
            print(f"  Series names: {list(series_ids.keys())[:5]}...")  # Show first 5
            
            # Check database latest dates
            table_name = f'historical_data_{timeframe}'
            db_records = await fred_scheduler.get_database_last_12_months(table_name)
            if db_records:
                latest_date = db_records[0]['observation_date'] if db_records else None
                print(f"  Latest DB date: {latest_date}")
                print(f"  DB records count: {len(db_records)}")
            else:
                print(f"  ❌ No database records found for {table_name}")
        
        print("\n" + "=" * 60)
        print("🧪 Testing FRED data fetching for each timeframe:")
        
        # Test FRED data fetching for each timeframe
        for timeframe in ['1m', '3m', '6m']:
            print(f"\n🌐 Fetching FRED data for {timeframe.upper()}:")
            
            try:
                config = fred_scheduler.series_mappings[timeframe]
                series_ids = config['series_ids']
                
                # Test fetching first 3 series only (to avoid rate limits)
                test_series = dict(list(series_ids.items())[:3])
                print(f"  Testing with {len(test_series)} series: {list(test_series.keys())}")
                
                fred_data = await fred_scheduler.fetch_12_months_fred_data_correct(test_series)
                
                if fred_data:
                    print(f"  ✅ Successfully fetched FRED data")
                    
                    # Check data availability
                    for series_name, observations in fred_data.items():
                        if observations:
                            print(f"    {series_name}: {len(observations)} observations")
                            # Show most recent observation
                            recent_obs = observations[0] if observations else None
                            if recent_obs:
                                print(f"      Most recent: {recent_obs.get('date', 'N/A')} = {recent_obs.get('value', 'N/A')}")
                        else:
                            print(f"    {series_name}: ❌ No observations")
                else:
                    print(f"  ❌ Failed to fetch FRED data for {timeframe}")
                    
            except Exception as e:
                print(f"  ❌ Error testing {timeframe}: {e}")
        
        print("\n" + "=" * 60)
        print("📝 Testing month calculation logic:")
        
        # Test the month calculation to see if there are issues
        current_date = datetime.now()
        print(f"Current date: {current_date}")
        
        # Show what the corrected logic should produce
        if current_date.month == 1:
            start_month_date = datetime(current_date.year - 1, 12, 1)
        else:
            start_month_date = datetime(current_date.year, current_date.month - 1, 1)
        
        print(f"Previous month (target): {start_month_date.strftime('%Y-%m-01')}")
        
        # Show the 12 months that should be processed
        from datetime import timedelta
        months_to_check = []
        for i in range(12):
            month_date = start_month_date - timedelta(days=32*i)
            month_date = month_date.replace(day=1)
            months_to_check.append(month_date.strftime('%Y-%m-01'))
        
        months_to_check.reverse()
        print(f"Months to process (oldest to newest):")
        for i, month in enumerate(months_to_check):
            marker = "👈 TARGET" if i == len(months_to_check) - 1 else ""
            print(f"  {month} {marker}")
    
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(investigate_3m_issue())