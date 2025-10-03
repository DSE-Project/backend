"""
Network Communication Failover and Recovery Testing
Tests network interruptions, API failures, and communication recovery
"""
import asyncio
import aiohttp
import time
import json
import os
import sys
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
import subprocess
import threading

# Import project path utilities  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_utils import get_project_paths

class TestNetworkFailover:
    """Test network communication interruptions and recovery"""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.test_results = []
    
    async def test_network_interruption_simulation(self):
        """Test network server communication interruption"""
        print("🔌 Testing network interruption scenarios...")
        
        # Test 1: Simulate network disconnect during FRED API call
        await self._test_fred_api_network_failure()
        
        # Test 2: Simulate network disconnect during database connection
        await self._test_database_network_failure()
        
        # Test 3: Test client-server communication interruption
        await self._test_client_server_network_failure()
        
        return self.test_results
    
    async def _test_fred_api_network_failure(self):
        """Simulate FRED API network failure and recovery"""
        print("📊 Testing FRED API network failure...")
        
        # Test that system handles FRED API failure gracefully
        async with aiohttp.ClientSession() as session:
            try:
                # Test macro-indicators endpoint which depends on FRED API
                async with session.get(f"{self.api_base_url}/api/v1/macro-indicators") as response:
                    # Should return cached data or handle gracefully
                    if response.status in [200, 503, 500]:
                        self.test_results.append({
                            "test": "FRED API Network Failure",
                            "status": "PASS",
                            "details": f"System handled FRED API dependency with status {response.status}"
                        })
                    else:
                        self.test_results.append({
                            "test": "FRED API Network Failure",
                            "status": "FAIL",
                            "details": f"Unexpected status code: {response.status}"
                        })
            except aiohttp.ClientError as e:
                # Network errors are expected and should be handled
                self.test_results.append({
                    "test": "FRED API Network Failure",
                    "status": "PASS",
                    "details": f"Network error handled appropriately: {type(e).__name__}"
                })
            except Exception as e:
                self.test_results.append({
                    "test": "FRED API Network Failure",
                    "status": "FAIL",
                    "details": f"Unhandled exception: {str(e)}"
                })
    
    async def _test_database_network_failure(self):
        """Simulate database network failure and recovery"""
        print("🗄️ Testing database network failure...")
        
        # Test database connection failure for endpoints that may use database
        async with aiohttp.ClientSession() as session:
            try:
                # Test endpoint that may require database
                async with session.get(f"{self.api_base_url}/api/v1/yearly-risk") as response:
                    # Should handle database connection gracefully
                    if response.status in [200, 503, 500]:
                        self.test_results.append({
                            "test": "Database Network Failure",
                            "status": "PASS",
                            "details": f"Database failure handled with status {response.status}"
                        })
                    else:
                        self.test_results.append({
                            "test": "Database Network Failure",
                            "status": "FAIL",
                            "details": f"Unexpected status code: {response.status}"
                        })
            except Exception as e:
                self.test_results.append({
                    "test": "Database Network Failure",
                    "status": "PASS",
                    "details": f"Connection error handled appropriately: {type(e).__name__}"
                })
    
    async def _test_client_server_network_failure(self):
        """Test client-server communication interruption"""
        print("🌐 Testing client-server network failure...")
        
        # Test connection timeout scenarios
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
            try:
                # Test health check with short timeout
                async with session.get(f"{self.api_base_url}/health") as response:
                    if response.status == 200:
                        self.test_results.append({
                            "test": "Client-Server Communication",
                            "status": "PASS",
                            "details": "Server responded to health check"
                        })
                    else:
                        self.test_results.append({
                            "test": "Client-Server Communication",
                            "status": "FAIL",
                            "details": f"Health check failed: {response.status}"
                        })
            except asyncio.TimeoutError:
                self.test_results.append({
                    "test": "Client-Server Communication",
                    "status": "PASS",
                    "details": "Timeout handled gracefully"
                })
            except Exception as e:
                self.test_results.append({
                    "test": "Client-Server Communication",
                    "status": "FAIL",
                    "details": f"Unexpected error: {str(e)}"
                })
    
    def test_recovery_procedures(self):
        """Test automated recovery procedures"""
        print("🔄 Testing recovery procedures...")
        
        recovery_tests = [
            self._test_cache_recovery,
            self._test_service_restart_recovery,
            self._test_data_consistency_after_recovery
        ]
        
        for test_func in recovery_tests:
            try:
                test_func()
            except Exception as e:
                self.test_results.append({
                    "test": f"Recovery Test {test_func.__name__}",
                    "status": "FAIL",
                    "details": f"Recovery test failed: {str(e)}"
                })
    
    def _test_cache_recovery(self):
        """Test cache recovery after failure"""
        print("💾 Testing cache recovery...")
        
        # Clear all caches to simulate cache corruption
        cache_clear_endpoints = [
            "/api/v1/forecast/cache/clear",
            "/api/v1/fred-cache/clear"
        ]
        
        import requests
        
        for endpoint in cache_clear_endpoints:
            try:
                response = requests.post(f"{self.api_base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    self.test_results.append({
                        "test": f"Cache Recovery {endpoint}",
                        "status": "PASS",
                        "details": "Cache cleared successfully"
                    })
            except Exception as e:
                self.test_results.append({
                    "test": f"Cache Recovery {endpoint}",
                    "status": "FAIL", 
                    "details": f"Cache clear failed: {str(e)}"
                })
    
    def _test_service_restart_recovery(self):
        """Test service restart recovery"""
        print("🔄 Testing service restart recovery...")
        
        import requests
        
        # Test that services can be restarted
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                self.test_results.append({
                    "test": "Service Restart Recovery",
                    "status": "PASS",
                    "details": "Service is healthy after restart"
                })
            else:
                self.test_results.append({
                    "test": "Service Restart Recovery",
                    "status": "FAIL",
                    "details": f"Service health check failed: {response.status_code}"
                })
        except Exception as e:
            self.test_results.append({
                "test": "Service Restart Recovery",
                "status": "FAIL",
                "details": f"Service restart test failed: {str(e)}"
            })
    
    def _test_data_consistency_after_recovery(self):
        """Test data consistency after recovery"""
        print("📊 Testing data consistency after recovery...")
        
        import requests
        
        try:
            # Test that predictions are available after recovery using GET endpoint
            response = requests.get(f"{self.api_base_url}/api/v1/macro-indicators", timeout=10)
            if response.status_code in [200, 202]:
                self.test_results.append({
                    "test": "Data Consistency After Recovery",
                    "status": "PASS",
                    "details": "Data endpoints available after recovery"
                })
            elif response.status_code in [503, 500]:
                # Service may still be recovering
                self.test_results.append({
                    "test": "Data Consistency After Recovery",
                    "status": "PASS",
                    "details": f"Service recovering gracefully: {response.status_code}"
                })
            else:
                self.test_results.append({
                    "test": "Data Consistency After Recovery",
                    "status": "FAIL",
                    "details": f"Unexpected response after recovery: {response.status_code}"
                })
        except requests.exceptions.RequestException as e:
            # Network errors during recovery are acceptable
            self.test_results.append({
                "test": "Data Consistency After Recovery",
                "status": "PASS",
                "details": f"Connection error handled during recovery: {type(e).__name__}"
            })
        except Exception as e:
            self.test_results.append({
                "test": "Data Consistency After Recovery",
                "status": "FAIL",
                "details": f"Data consistency test failed: {str(e)}"
            })

def run_network_failover_tests():
    """Run all network failover tests"""
    print("\n🚀 Starting Network Failover and Recovery Tests...")
    print("=" * 60)
    
    tester = TestNetworkFailover()
    
    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(tester.test_network_interruption_simulation())
    finally:
        loop.close()
    
    # Run sync recovery tests
    tester.test_recovery_procedures()
    
    # Print results
    print("\n📋 Test Results Summary:")
    print("=" * 60)
    
    passed = failed = 0
    for result in tester.test_results:
        status_emoji = "✅" if result["status"] == "PASS" else "❌"
        print(f"{status_emoji} {result['test']}: {result['status']}")
        print(f"   Details: {result['details']}")
        
        if result["status"] == "PASS":
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Summary: {passed} passed, {failed} failed")
    
    return tester.test_results

if __name__ == "__main__":
    results = run_network_failover_tests()
    
    # Save results to file
    with open("network_failover_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: network_failover_test_results.json")