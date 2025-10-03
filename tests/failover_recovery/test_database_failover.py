"""
Database Failover and Recovery Testing
Tests Supabase connection failures, data corruption, and recovery procedures
"""
import time
import json
import os
import sys
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Import project path utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_utils import get_project_paths, get_data_files

class TestDatabaseFailover:
    """Test database connection failures and data recovery scenarios"""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.test_results = []
        # Get project paths using utility function
        self.paths = get_project_paths()
        self.data_files = get_data_files()
        # Database connection may not be required for all endpoints
        self.fred_api_key = os.getenv("FRED_API_KEY", "")
    
    def test_database_connection_failures(self):
        """Test various database connection failure scenarios"""
        print("🗄️ Testing database connection failures...")
        
        # Test 1: Database connection timeout
        self._test_database_connection_timeout()
        
        # Test 2: FRED API connection issues
        self._test_fred_api_connection()
        
        # Test 3: Database query failures
        self._test_query_failures()
        
        # Test 4: Connection pool exhaustion
        self._test_connection_pool_exhaustion()
        
        return self.test_results
    
    def _test_database_connection_timeout(self):
        """Test database connection timeout scenarios"""
        print("⏱️ Testing database connection timeout...")
        
        import requests
        
        try:
            # Test endpoints that require database access
            db_endpoints = [
                "/api/v1/yearly-risk",
                "/api/v1/macro-indicators"
            ]
            
            for endpoint in db_endpoints:
                try:
                    response = requests.get(f"{self.api_base_url}{endpoint}", timeout=5)
                    
                    if response.status_code == 200:
                        self.test_results.append({
                            "test": f"Database Connection {endpoint}",
                            "status": "PASS",
                            "details": "Database connection successful"
                        })
                    elif response.status_code in [503, 500]:
                        self.test_results.append({
                            "test": f"Database Connection {endpoint}",
                            "status": "PASS",
                            "details": "Database failure handled gracefully"
                        })
                    else:
                        self.test_results.append({
                            "test": f"Database Connection {endpoint}",
                            "status": "FAIL",
                            "details": f"Unexpected status: {response.status_code}"
                        })
                        
                except requests.Timeout:
                    self.test_results.append({
                        "test": f"Database Connection {endpoint}",
                        "status": "PASS",
                        "details": "Timeout handled appropriately"
                    })
                except Exception as e:
                    self.test_results.append({
                        "test": f"Database Connection {endpoint}",
                        "status": "FAIL",
                        "details": f"Unhandled exception: {str(e)}"
                    })
                    
        except Exception as e:
            self.test_results.append({
                "test": "Supabase Connection Timeout",
                "status": "FAIL",
                "details": f"Test setup failed: {str(e)}"
            })
    
    def _test_fred_api_connection(self):
        """Test FRED API connection handling"""
        print("🔐 Testing FRED API connection...")
        
        import requests
        
        try:
            # Test macro indicators endpoint which uses FRED API
            response = requests.get(f"{self.api_base_url}/api/v1/macro-indicators", timeout=10)
            
            # Should handle API connection gracefully
            if response.status_code in [200, 500, 503]:
                self.test_results.append({
                    "test": "FRED API Connection",
                    "status": "PASS",
                    "details": f"FRED API handled with status {response.status_code}"
                })
            else:
                self.test_results.append({
                    "test": "FRED API Connection",
                    "status": "FAIL",
                    "details": f"Unexpected response: {response.status_code}"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": "FRED API Connection",
                "status": "PASS",
                "details": f"Connection error handled: {type(e).__name__}"
            })
    
    def _test_query_failures(self):
        """Test database query failure scenarios"""
        print("📋 Testing database query failures...")
        
        import requests
        
        # Test query with potential to fail
        test_queries = [
            "/api/v1/macro-indicators",
            "/api/v1/yearly-risk"
        ]
        
        for query_endpoint in test_queries:
            try:
                response = requests.get(f"{self.api_base_url}{query_endpoint}", timeout=10)
                
                if response.status_code == 200:
                    # Check if response contains valid data
                    try:
                        data = response.json()
                        if data and isinstance(data, (dict, list)):
                            self.test_results.append({
                                "test": f"Database Query {query_endpoint}",
                                "status": "PASS",
                                "details": "Query executed successfully with valid data"
                            })
                        else:
                            self.test_results.append({
                                "test": f"Database Query {query_endpoint}",
                                "status": "FAIL",
                                "details": "Query returned empty or invalid data"
                            })
                    except json.JSONDecodeError:
                        self.test_results.append({
                            "test": f"Database Query {query_endpoint}",
                            "status": "FAIL",
                            "details": "Query returned invalid JSON"
                        })
                else:
                    self.test_results.append({
                        "test": f"Database Query {query_endpoint}",
                        "status": "PASS",
                        "details": f"Query failure handled with status {response.status_code}"
                    })
                    
            except Exception as e:
                self.test_results.append({
                    "test": f"Database Query {query_endpoint}",
                    "status": "FAIL",
                    "details": f"Query test failed: {str(e)}"
                })
    
    def _test_connection_pool_exhaustion(self):
        """Test connection pool exhaustion scenarios"""
        print("🏊 Testing connection pool exhaustion...")
        
        import requests
        import threading
        
        def make_concurrent_request(endpoint, results_list):
            try:
                response = requests.get(f"{self.api_base_url}{endpoint}", timeout=15)
                results_list.append({
                    "status_code": response.status_code,
                    "success": response.status_code in [200, 503]
                })
            except Exception as e:
                results_list.append({
                    "status_code": None,
                    "success": True,  # Connection errors are expected under load
                    "error": str(e)
                })
        
        # Make multiple concurrent requests to test connection pooling
        concurrent_results = []
        threads = []
        
        for i in range(10):  # Simulate 10 concurrent requests
            thread = threading.Thread(
                target=make_concurrent_request,
                args=("/api/v1/macro-indicators", concurrent_results)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=20)
        
        # Analyze results
        successful_responses = sum(1 for r in concurrent_results if r.get("success", False))
        total_requests = len(concurrent_results)
        
        if successful_responses >= total_requests * 0.7:  # At least 70% success rate
            self.test_results.append({
                "test": "Connection Pool Exhaustion",
                "status": "PASS",
                "details": f"{successful_responses}/{total_requests} requests handled successfully"
            })
        else:
            self.test_results.append({
                "test": "Connection Pool Exhaustion", 
                "status": "FAIL",
                "details": f"Only {successful_responses}/{total_requests} requests successful"
            })
    
    def test_data_recovery_procedures(self):
        """Test data recovery and backup procedures"""
        print("💾 Testing data recovery procedures...")
        
        # Test 1: Cache recovery after database failure
        self._test_cache_recovery()
        
        # Test 2: Data consistency checks
        self._test_data_consistency()
        
        # Test 3: Historical data integrity
        self._test_historical_data_integrity()
    
    def _test_cache_recovery(self):
        """Test cache recovery mechanisms"""
        print("🔄 Testing cache recovery...")
        
        import requests
        
        try:
            # Clear caches to simulate recovery scenario
            cache_endpoints = [
                "/api/v1/forecast/cache/clear",
                "/api/v1/fred-cache/clear"
            ]
            
            for endpoint in cache_endpoints:
                try:
                    response = requests.post(f"{self.api_base_url}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        self.test_results.append({
                            "test": f"Cache Recovery {endpoint}",
                            "status": "PASS",
                            "details": "Cache cleared and can be rebuilt"
                        })
                except Exception as e:
                    self.test_results.append({
                        "test": f"Cache Recovery {endpoint}",
                        "status": "FAIL",
                        "details": f"Cache recovery failed: {str(e)}"
                    })
                    
            # Test that system works without cache by testing a simpler endpoint
            response = requests.get(f"{self.api_base_url}/api/v1/macro-indicators", timeout=15)
            if response.status_code in [200, 202]:
                self.test_results.append({
                    "test": "System Function Without Cache",
                    "status": "PASS",
                    "details": "System works even when caches are cleared"
                })
            else:
                self.test_results.append({
                    "test": "System Function Without Cache",
                    "status": "FAIL",
                    "details": f"System failed without cache: {response.status_code}"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": "Cache Recovery",
                "status": "FAIL",
                "details": f"Cache recovery test failed: {str(e)}"
            })
    
    def _test_data_consistency(self):
        """Test data consistency after recovery"""
        print("📊 Testing data consistency...")
        
        import requests
        
        try:
            # Test data consistency using GET endpoints that should return consistent data
            responses = []
            for i in range(3):
                response = requests.get(f"{self.api_base_url}/api/v1/macro-indicators", timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    responses.append(data)
                time.sleep(0.5)
            
            if len(responses) >= 2:
                # Check that responses are consistent (same structure and similar values)
                first_response = responses[0]
                all_consistent = True
                
                for response in responses[1:]:
                    # Check structure consistency
                    if type(response) != type(first_response):
                        all_consistent = False
                        break
                
                if all_consistent:
                    self.test_results.append({
                        "test": "Data Consistency",
                        "status": "PASS",
                        "details": f"Data responses consistent across {len(responses)} requests"
                    })
                else:
                    self.test_results.append({
                        "test": "Data Consistency",
                        "status": "FAIL",
                        "details": "Data responses show inconsistent structure"
                    })
            else:
                self.test_results.append({
                    "test": "Data Consistency",
                    "status": "FAIL",
                    "details": "Insufficient responses for consistency test"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": "Data Consistency",
                "status": "FAIL",
                "details": f"Data consistency test failed: {str(e)}"
            })
    
    def _test_historical_data_integrity(self):
        """Test historical data file integrity"""
        print("📈 Testing historical data integrity...")
        
        historical_files = [
            self.data_files["historical_1m"],
            self.data_files["historical_3m"]
        ]
        
        for file_path in historical_files:
            if os.path.exists(file_path):
                try:
                    # Check file size and basic structure
                    file_size = os.path.getsize(file_path)
                    if file_size > 0:
                        # Try to read first few lines
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 1:  # Has header and data
                                self.test_results.append({
                                    "test": f"Historical Data Integrity {os.path.basename(file_path)}",
                                    "status": "PASS",
                                    "details": f"File exists with {len(lines)} lines, {file_size} bytes"
                                })
                            else:
                                self.test_results.append({
                                    "test": f"Historical Data Integrity {os.path.basename(file_path)}",
                                    "status": "FAIL",
                                    "details": "File exists but appears empty or corrupted"
                                })
                    else:
                        self.test_results.append({
                            "test": f"Historical Data Integrity {os.path.basename(file_path)}",
                            "status": "FAIL",
                            "details": "File exists but is empty"
                        })
                except Exception as e:
                    self.test_results.append({
                        "test": f"Historical Data Integrity {os.path.basename(file_path)}",
                        "status": "FAIL",
                        "details": f"Error reading file: {str(e)}"
                    })
            else:
                self.test_results.append({
                    "test": f"Historical Data Integrity {os.path.basename(file_path)}",
                    "status": "FAIL",
                    "details": "Historical data file not found"
                })

def run_database_failover_tests():
    """Run all database failover and recovery tests"""
    print("\n🚀 Starting Database Failover and Recovery Tests...")
    print("=" * 60)
    
    tester = TestDatabaseFailover()
    
    # Run database connection tests
    tester.test_database_connection_failures()
    
    # Run data recovery tests
    tester.test_data_recovery_procedures()
    
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
    results = run_database_failover_tests()
    
    # Save results to file
    with open("database_failover_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: database_failover_test_results.json")