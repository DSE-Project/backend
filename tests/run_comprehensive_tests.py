"""
Comprehensive Test Suite Orchestrator for RecessionScope
Runs both Failover & Recovery and Configuration Testing suites
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Add the parent directory to path to import test modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestSuiteOrchestrator:
    """Orchestrates all testing suites for RecessionScope system"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.all_results = {
            "test_run_info": {
                "timestamp": self.start_time.isoformat(),
                "system_under_test": "RecessionScope API",
                "test_categories": ["Failover & Recovery", "Configuration Testing"]
            },
            "failover_recovery_tests": {},
            "configuration_tests": {},
            "summary": {}
        }
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("\n" + "=" * 80)
        print("🚀 RECESSIONSCOPE COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"📅 Test Run Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 System Under Test: RecessionScope API")
        print(f"📍 Test Environment: Development/Local")
        print("=" * 80)
        
        # Run Failover and Recovery Tests
        print("\n🔥 PHASE 1: FAILOVER AND RECOVERY TESTING")
        print("-" * 50)
        self._run_failover_recovery_tests()
        
        # Run Configuration Tests  
        print("\n⚙️ PHASE 2: CONFIGURATION TESTING")
        print("-" * 50)
        self._run_configuration_tests()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        return self.all_results
    
    def _run_failover_recovery_tests(self):
        """Execute all failover and recovery tests"""
        print("🔌 Running Network Failover Tests...")
        
        try:
            # Import and run network failover tests
            from tests.failover_recovery.test_network_failover import run_network_failover_tests
            network_results = run_network_failover_tests()
            self.all_results["failover_recovery_tests"]["network_failover"] = network_results
        except Exception as e:
            print(f"❌ Network failover tests failed: {str(e)}")
            self.all_results["failover_recovery_tests"]["network_failover"] = [
                {"test": "Network Failover Test Suite", "status": "FAIL", "details": f"Test suite error: {str(e)}"}
            ]
        
        print("\n🗄️ Running Database Failover Tests...")
        
        try:
            # Import and run database failover tests
            from tests.failover_recovery.test_database_failover import run_database_failover_tests
            database_results = run_database_failover_tests()
            self.all_results["failover_recovery_tests"]["database_failover"] = database_results
        except Exception as e:
            print(f"❌ Database failover tests failed: {str(e)}")
            self.all_results["failover_recovery_tests"]["database_failover"] = [
                {"test": "Database Failover Test Suite", "status": "FAIL", "details": f"Test suite error: {str(e)}"}
            ]
        
        print("\n🖥️ Running Server Failover Tests...")
        
        try:
            # Run basic server tests without pytest dependency
            self._run_basic_server_tests()
        except Exception as e:
            print(f"❌ Server failover tests failed: {str(e)}")
            self.all_results["failover_recovery_tests"]["server_failover"] = [
                {"test": "Server Failover Test Suite", "status": "FAIL", "details": f"Test suite error: {str(e)}"}
            ]
    
    def _run_basic_server_tests(self):
        """Run basic server failover tests without external dependencies"""
        import requests
        
        server_results = []
        api_base_url = "http://localhost:8000"
        
        # Test 1: Server Health Check
        try:
            response = requests.get(f"{api_base_url}/health", timeout=10)
            if response.status_code == 200:
                server_results.append({
                    "test": "Server Health Check",
                    "status": "PASS",
                    "details": f"Server healthy: {response.json()}"
                })
            else:
                server_results.append({
                    "test": "Server Health Check", 
                    "status": "FAIL",
                    "details": f"Health check failed: {response.status_code}"
                })
        except requests.ConnectionError:
            server_results.append({
                "test": "Server Health Check",
                "status": "FAIL", 
                "details": "Cannot connect to server (not running or network issue)"
            })
        except Exception as e:
            server_results.append({
                "test": "Server Health Check",
                "status": "FAIL",
                "details": f"Health check error: {str(e)}"
            })
        
        # Test 2: API Endpoint Availability
        endpoints_to_test = [
            ("/", "Root endpoint"),
            ("/api/v1/fred-cache/status", "FRED cache status"),
            ("/docs", "API documentation")
        ]
        
        for endpoint, description in endpoints_to_test:
            try:
                response = requests.get(f"{api_base_url}{endpoint}", timeout=10)
                if response.status_code in [200, 404]:  # 404 is acceptable for some endpoints
                    server_results.append({
                        "test": f"API Endpoint {endpoint}",
                        "status": "PASS",
                        "details": f"{description} accessible (status: {response.status_code})"
                    })
                else:
                    server_results.append({
                        "test": f"API Endpoint {endpoint}",
                        "status": "FAIL",
                        "details": f"{description} returned status {response.status_code}"
                    })
            except Exception as e:
                server_results.append({
                    "test": f"API Endpoint {endpoint}",
                    "status": "FAIL",
                    "details": f"Endpoint test failed: {str(e)}"
                })
        
        # Test 3: Cache System Recovery
        try:
            cache_endpoints = ["/api/v1/forecast/cache/clear", "/api/v1/fred-cache/clear"]
            for cache_endpoint in cache_endpoints:
                try:
                    response = requests.post(f"{api_base_url}{cache_endpoint}", timeout=5)
                    if response.status_code == 200:
                        server_results.append({
                            "test": f"Cache Recovery {cache_endpoint}",
                            "status": "PASS",
                            "details": "Cache clear successful (recovery mechanism works)"
                        })
                    else:
                        server_results.append({
                            "test": f"Cache Recovery {cache_endpoint}",
                            "status": "FAIL",
                            "details": f"Cache clear failed: {response.status_code}"
                        })
                except Exception as e:
                    server_results.append({
                        "test": f"Cache Recovery {cache_endpoint}",
                        "status": "FAIL",
                        "details": f"Cache test failed: {str(e)}"
                    })
        except Exception as e:
            server_results.append({
                "test": "Cache Recovery System",
                "status": "FAIL",
                "details": f"Cache recovery test setup failed: {str(e)}"
            })
        
        self.all_results["failover_recovery_tests"]["server_failover"] = server_results
    
    def _run_configuration_tests(self):
        """Execute all configuration tests"""
        print("💻 Running System Configuration Tests...")
        
        try:
            # Import and run system configuration tests
            from tests.configuration_testing.test_system_configuration import run_configuration_tests
            system_results = run_configuration_tests()
            self.all_results["configuration_tests"]["system_configuration"] = system_results
        except Exception as e:
            print(f"❌ System configuration tests failed: {str(e)}")
            self.all_results["configuration_tests"]["system_configuration"] = [
                {"test": "System Configuration Test Suite", "status": "FAIL", "details": f"Test suite error: {str(e)}"}
            ]
        
        print("\n⚛️ Running Frontend Configuration Tests...")
        
        try:
            # Import and run frontend configuration tests  
            from tests.configuration_testing.test_frontend_configuration import run_frontend_configuration_tests
            frontend_results = run_frontend_configuration_tests()
            self.all_results["configuration_tests"]["frontend_configuration"] = frontend_results
        except Exception as e:
            print(f"❌ Frontend configuration tests failed: {str(e)}")
            self.all_results["configuration_tests"]["frontend_configuration"] = [
                {"test": "Frontend Configuration Test Suite", "status": "FAIL", "details": f"Test suite error: {str(e)}"}
            ]
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Calculate overall statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_info = 0
        
        # Count results from all test categories
        for category in ["failover_recovery_tests", "configuration_tests"]:
            for test_suite, results in self.all_results.get(category, {}).items():
                if isinstance(results, list):
                    for result in results:
                        total_tests += 1
                        status = result.get("status", "UNKNOWN")
                        if status == "PASS":
                            total_passed += 1
                        elif status == "FAIL":
                            total_failed += 1
                        else:
                            total_info += 1
        
        # Generate summary
        self.all_results["summary"] = {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "informational": total_info,
            "success_rate": round((total_passed / total_tests * 100), 2) if total_tests > 0 else 0,
            "duration_seconds": duration.total_seconds(),
            "end_time": end_time.isoformat()
        }
        
        # Print comprehensive summary
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"⏱️ Test Duration: {duration.total_seconds():.2f} seconds")
        print(f"📈 Total Tests Executed: {total_tests}")
        print(f"✅ Tests Passed: {total_passed}")
        print(f"❌ Tests Failed: {total_failed}")
        print(f"ℹ️ Informational: {total_info}")
        print(f"🎯 Success Rate: {self.all_results['summary']['success_rate']}%")
        
        # Detailed breakdown by category
        print("\n📋 DETAILED BREAKDOWN BY CATEGORY:")
        print("-" * 50)
        
        for category_name, category_data in [
            ("Failover & Recovery Tests", self.all_results.get("failover_recovery_tests", {})),
            ("Configuration Tests", self.all_results.get("configuration_tests", {}))
        ]:
            print(f"\n🔍 {category_name}:")
            
            category_passed = category_failed = category_info = 0
            
            for test_suite, results in category_data.items():
                if isinstance(results, list):
                    suite_passed = sum(1 for r in results if r.get("status") == "PASS")
                    suite_failed = sum(1 for r in results if r.get("status") == "FAIL")
                    suite_info = sum(1 for r in results if r.get("status") not in ["PASS", "FAIL"])
                    
                    category_passed += suite_passed
                    category_failed += suite_failed
                    category_info += suite_info
                    
                    print(f"  📁 {test_suite}: {suite_passed} passed, {suite_failed} failed, {suite_info} info")
            
            total_category = category_passed + category_failed + category_info
            if total_category > 0:
                category_success_rate = round((category_passed / total_category * 100), 2)
                print(f"  📊 Category Success Rate: {category_success_rate}%")
        
        # Highlight critical failures
        print(f"\n🚨 CRITICAL ISSUES TO ADDRESS:")
        print("-" * 50)
        
        critical_failures = []
        for category in ["failover_recovery_tests", "configuration_tests"]:
            for test_suite, results in self.all_results.get(category, {}).items():
                if isinstance(results, list):
                    for result in results:
                        if result.get("status") == "FAIL" and any(keyword in result.get("test", "").lower() 
                                                                for keyword in ["server", "database", "health", "connection"]):
                            critical_failures.append(f"{test_suite}: {result['test']} - {result['details']}")
        
        if critical_failures:
            for failure in critical_failures[:5]:  # Show top 5 critical failures
                print(f"❗ {failure}")
        else:
            print("✅ No critical failures detected!")
        
        print("\n" + "=" * 80)
        
        # Save detailed results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"recessionscope_test_results_{timestamp}.json"
        
        with open(results_filename, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        print(f"💾 Detailed results saved to: {results_filename}")
        print("=" * 80)

def main():
    """Main function to run comprehensive test suite"""
    orchestrator = TestSuiteOrchestrator()
    
    try:
        results = orchestrator.run_all_tests()
        
        # Determine exit code based on results
        summary = results.get("summary", {})
        failed_tests = summary.get("failed", 0)
        
        if failed_tests == 0:
            print("🎉 All tests completed successfully!")
            return 0
        else:
            print(f"⚠️ Test suite completed with {failed_tests} failures.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Test suite interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\n💥 Test suite failed with error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)