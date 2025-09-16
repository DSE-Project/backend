"""
Test script to verify date range filtering works correctly for Economic Charts Service
"""

import asyncio
import os
from services.economic_charts_service import EconomicChartsService
from datetime import datetime

async def test_date_range_filtering():
    """Test the date range filtering functionality"""
    
    print("🧪 Testing Economic Charts Service Date Range Filtering")
    print("=" * 60)
    
    # Check if FRED API key is set
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key or fred_api_key == 'your_fred_api_key_here':
        print("❌ FRED_API_KEY not set. Cannot test with real data.")
        return False
    
    # Initialize service
    service = EconomicChartsService()
    
    test_periods = ["6m", "12m", "24m", "all"]
    
    for period in test_periods:
        try:
            print(f"\n📊 Testing period: {period}")
            print("-" * 30)
            
            # Fetch data for this period
            data = await service.get_historical_data(period=period, indicators=["gdp", "cpi"])
            
            # Check results
            dates = data['dates']
            metadata = data['metadata']
            
            print(f"Period requested: {period}")
            print(f"Total data points: {metadata['total_points']}")
            print(f"Date range: {metadata['start_date']} to {metadata['end_date']}")
            
            if dates:
                start_date = datetime.strptime(dates[0], '%Y-%m-%d')
                end_date = datetime.strptime(dates[-1], '%Y-%m-%d')
                actual_days = (end_date - start_date).days
                print(f"Actual date span: {actual_days} days")
                
                # Verify date ranges
                if period == "6m":
                    expected_max_days = 180
                    if actual_days > expected_max_days + 30:  # Allow some buffer
                        print(f"⚠️  Warning: 6m period has {actual_days} days (expected ~{expected_max_days})")
                    else:
                        print("✅ 6m date range looks correct")
                
                elif period == "12m":
                    expected_max_days = 365
                    if actual_days > expected_max_days + 30:  # Allow some buffer
                        print(f"⚠️  Warning: 12m period has {actual_days} days (expected ~{expected_max_days})")
                    else:
                        print("✅ 12m date range looks correct")
                
                elif period == "24m":
                    expected_max_days = 730
                    if actual_days > expected_max_days + 30:  # Allow some buffer
                        print(f"⚠️  Warning: 24m period has {actual_days} days (expected ~{expected_max_days})")
                    else:
                        print("✅ 24m date range looks correct")
                
                elif period == "all":
                    # For "all", we expect data going back to around 1947-1960s
                    if start_date.year > 1970:
                        print(f"⚠️  Warning: 'all' period starts in {start_date.year} (expected 1940s-1960s)")
                    else:
                        print(f"✅ 'all' period starts in {start_date.year} - looks correct")
                
                # Show some sample indicators
                if 'indicators' in data:
                    gdp_data = data['indicators'].get('gdp', {})
                    if gdp_data:
                        gdp_values = gdp_data.get('values', [])
                        if gdp_values:
                            print(f"GDP: First = {gdp_values[0]:.2f}, Last = {gdp_values[-1]:.2f}")
            else:
                print("❌ No dates returned")
            
        except Exception as e:
            print(f"❌ Error testing period {period}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("📋 Date Range Filter Test Summary:")
    print("- 6m, 12m, 24m should show recent data within the specified timeframe")
    print("- 'all' should show historical data from 1940s-1960s to present")
    print("- Each period should have different date ranges")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_date_range_filtering())
