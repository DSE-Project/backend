"""
Priority System Test Script
Tests the request priority system by making concurrent requests
"""
import asyncio
import aiohttp
import time
from typing import List, Dict

async def make_request(session: aiohttp.ClientSession, url: str, name: str) -> Dict:
    """Make a single request and measure timing"""
    start_time = time.time()
    try:
        async with session.get(url) as response:
            duration = time.time() - start_time
            headers = dict(response.headers)
            
            return {
                "name": name,
                "url": url,
                "status": response.status,
                "duration": duration,
                "priority": headers.get("x-request-priority", "UNKNOWN"),
                "request_id": headers.get("x-request-id", "UNKNOWN"),
                "server_duration": headers.get("x-request-duration", "UNKNOWN")
            }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "name": name,
            "url": url,
            "status": "ERROR",
            "duration": duration,
            "error": str(e)
        }

async def test_priority_system():
    """Test the priority system with concurrent requests"""
    base_url = "http://localhost:8000"
    
    # Define test requests in order of expected priority
    test_requests = [
        # HIGH PRIORITY - Forecast predictions (should be fastest)
        (f"{base_url}/api/v1/forecast/predict/1m", "1M Prediction (HIGH)"),
        (f"{base_url}/api/v1/forecast/predict/3m", "3M Prediction (HIGH)"),
        (f"{base_url}/api/v1/forecast/predict/6m", "6M Prediction (HIGH)"),
        
        # MEDIUM PRIORITY - Other forecast endpoints
        (f"{base_url}/api/v1/forecast/status/1m", "1M Status (MEDIUM)"),
        (f"{base_url}/api/v1/forecast/cache/stats", "Cache Stats (MEDIUM)"),
        
        # LOW PRIORITY - Charts and indicators (should be slower)
        (f"{base_url}/api/v1/macro-indicators", "Macro Indicators (LOW)"),
        (f"{base_url}/api/v1/economic-charts/summary-stats", "Chart Stats (LOW)"),
        
        # CRITICAL - Health (should be fastest)
        (f"{base_url}/health", "Health Check (CRITICAL)"),
    ]
    
    print("🚀 Starting Priority System Test")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        # Make all requests concurrently
        start_time = time.time()
        tasks = [
            make_request(session, url, name) 
            for url, name in test_requests
        ]
        
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Sort results by completion time (duration)
        results.sort(key=lambda x: x.get('duration', float('inf')))
        
        print(f"📊 Test Results (Total time: {total_time:.2f}s)")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            status = result.get('status', 'ERROR')
            duration = result.get('duration', 0)
            priority = result.get('priority', 'UNKNOWN')
            server_duration = result.get('server_duration', 'N/A')
            
            print(f"{i:2d}. {result['name']:<25} | "
                  f"Status: {status:<3} | "
                  f"Priority: {priority:<8} | "
                  f"Client: {duration:.3f}s | "
                  f"Server: {server_duration}s")
        
        print("-" * 60)
        
        # Analyze results by priority
        priority_groups = {}
        for result in results:
            priority = result.get('priority', 'UNKNOWN')
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(result['duration'])
        
        print("📈 Average Response Times by Priority:")
        for priority in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if priority in priority_groups:
                avg_time = sum(priority_groups[priority]) / len(priority_groups[priority])
                print(f"   {priority:<8}: {avg_time:.3f}s (avg)")
        
        print("\n✅ Priority system test completed!")

if __name__ == "__main__":
    asyncio.run(test_priority_system())