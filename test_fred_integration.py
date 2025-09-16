"""
Test script for the Economic Charts Service with FRED API integration
"""

import asyncio
import os
from services.economic_charts_service import EconomicChartsService

async def test_economic_service():
    """Test the economic charts service"""
    
    # Check if FRED API key is set
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key or fred_api_key == 'your_fred_api_key_here':
        print("⚠️  FRED_API_KEY not set. Service will use fallback sample data.")
        print("To use real FRED data:")
        print("1. Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Create a .env file (copy from .env.example)")
        print("3. Set FRED_API_KEY=your_actual_api_key in the .env file")
        print()
    else:
        print("✅ FRED_API_KEY found. Will fetch real economic data.")
        print()
    
    # Initialize service
    service = EconomicChartsService()
    
    try:
        # Test getting historical data
        print("📈 Fetching 12-month historical data...")
        data = await service.get_historical_data(period="12m")
        
        print(f"Data source: {data['metadata']['source']}")
        print(f"Period: {data['metadata']['period']}")
        print(f"Total data points: {data['metadata']['total_points']}")
        print(f"Date range: {data['metadata']['start_date']} to {data['metadata']['end_date']}")
        print()
        
        # Show available indicators
        print("Available indicators:")
        for indicator, info in data['indicators'].items():
            current_value = info.get('current_value')
            change = info.get('change_from_previous', {})
            print(f"  {info['name']} ({info['unit']}): {current_value:.2f} "
                  f"({change.get('percentage', 0):+.2f}%)")
        
        print()
        
        # Test getting summary statistics
        print("📊 Fetching summary statistics...")
        stats = await service.get_summary_statistics()
        
        print(f"Statistics source: {stats['metadata']['source']}")
        print(f"Indicators analyzed: {stats['metadata']['indicators_count']}")
        print(f"Data points: {stats['metadata']['data_points']}")
        print()
        
        # Show some correlation data
        if 'correlation_matrix' in stats:
            print("Sample correlations (GDP with other indicators):")
            correlations = stats['correlation_matrix'].get('gdp', {})
            for indicator, correlation in correlations.items():
                if indicator != 'gdp' and correlation is not None:
                    print(f"  GDP vs {indicator.upper()}: {correlation:.3f}")
        
        print("\n✅ Service test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing service: {str(e)}")
        if "FRED_API_KEY" in str(e):
            print("This error suggests the FRED API key is not properly configured.")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_economic_service())
