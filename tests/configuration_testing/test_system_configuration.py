"""
Configuration Testing for RecessionScope API
Tests different hardware/software configurations, browser compatibility, and deployment environments
"""
import os
import sys
import platform
import subprocess
import json
import time
from typing import Dict, Any, List
import importlib.util

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from backend directory
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(project_dir, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not available, continue without it

# Import project path utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_utils import get_project_paths, get_model_files, get_data_files

class TestSystemConfiguration:
    """Test system configuration compatibility and requirements"""
    
    def __init__(self):
        self.test_results = []
        self.system_info = self._gather_system_info()
        self.api_base_url = "http://localhost:8000"
        # Get project paths using utility function
        self.paths = get_project_paths()
        self.model_files = get_model_files()
        self.data_files = get_data_files()
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system configuration information"""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "architecture": platform.architecture(),
            "node_version": self._get_node_version(),
            "available_memory": self._get_available_memory(),
            "cpu_count": os.cpu_count()
        }
    
    def _get_node_version(self) -> str:
        """Get Node.js version if available"""
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else "Not installed"
        except:
            return "Not available"
    
    def _get_available_memory(self) -> str:
        """Get available system memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return f"{memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available"
        except ImportError:
            return "Memory info unavailable"
    
    def test_python_environment_configurations(self):
        """Test Python environment and dependency configurations"""
        print("🐍 Testing Python environment configurations...")
        
        # Test 1: Python version compatibility
        self._test_python_version_compatibility()
        
        # Test 2: Required package availability
        self._test_required_packages()
        
        # Test 3: ML model dependencies
        self._test_ml_dependencies()
        
        # Test 4: Environment variables
        self._test_environment_variables()
        
        return self.test_results
    
    def _test_python_version_compatibility(self):
        """Test Python version compatibility"""
        print("🔍 Testing Python version compatibility...")
        
        python_version = sys.version_info
        min_version = (3, 8)  # Minimum required Python version
        recommended_version = (3, 11)  # Recommended Python version
        
        if python_version >= recommended_version:
            self.test_results.append({
                "test": "Python Version Compatibility",
                "status": "PASS",
                "details": f"Python {python_version.major}.{python_version.minor} is recommended version"
            })
        elif python_version >= min_version:
            self.test_results.append({
                "test": "Python Version Compatibility",
                "status": "PASS",
                "details": f"Python {python_version.major}.{python_version.minor} meets minimum requirements"
            })
        else:
            self.test_results.append({
                "test": "Python Version Compatibility",
                "status": "FAIL",
                "details": f"Python {python_version.major}.{python_version.minor} is below minimum requirement (3.8+)"
            })
    
    def _test_required_packages(self):
        """Test required Python packages"""
        print("📦 Testing required packages...")
        
        # Get project paths for virtual environment detection
        project_paths = get_project_paths()
        backend_path = project_paths.get("backend")
        
        # Based on actual requirements.txt
        required_packages = [
            "fastapi",
            "uvicorn", 
            "tensorflow",
            "keras", 
            "pandas",
            "numpy",
            "sklearn",  # scikit-learn is imported as sklearn
            "shap",
            "supabase",
            "pydantic",
            "dotenv",   # python-dotenv is imported as dotenv
            "requests",
            "apscheduler"  # For FRED data scheduling
        ]
        
        for package in required_packages:
            try:
                # First try to find in current environment
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    # Try to import to verify it works
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'Unknown version')
                    self.test_results.append({
                        "test": f"Package {package}",
                        "status": "PASS",
                        "details": f"Version: {version}"
                    })
                else:
                    # Check if package exists in virtual environment
                    import subprocess
                    try:
                        venv_python = os.path.join(backend_path, "venv", "bin", "python")
                        if os.path.exists(venv_python):
                            result = subprocess.run([venv_python, "-c", f"import {package}; print(getattr({package}, '__version__', 'Unknown'))"], 
                                                  capture_output=True, text=True, timeout=10)
                            if result.returncode == 0:
                                version = result.stdout.strip()
                                self.test_results.append({
                                    "test": f"Package {package}",
                                    "status": "PASS",
                                    "details": f"Version: {version} (from venv)"
                                })
                            else:
                                self.test_results.append({
                                    "test": f"Package {package}",
                                    "status": "FAIL",
                                    "details": "Package not found"
                                })
                        else:
                            self.test_results.append({
                                "test": f"Package {package}",
                                "status": "FAIL",
                                "details": "Package not found"
                            })
                    except Exception:
                        self.test_results.append({
                            "test": f"Package {package}",
                            "status": "FAIL",
                            "details": "Package not found"
                        })
            except ImportError as e:
                self.test_results.append({
                    "test": f"Package {package}",
                    "status": "FAIL",
                    "details": f"Import error: {str(e)}"
                })
            except Exception as e:
                self.test_results.append({
                    "test": f"Package {package}",
                    "status": "FAIL",
                    "details": f"Unexpected error: {str(e)}"
                })
    
    def _test_ml_dependencies(self):
        """Test ML model dependencies and GPU support"""
        print("🤖 Testing ML dependencies...")
        
        # Test TensorFlow GPU support
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                self.test_results.append({
                    "test": "TensorFlow GPU Support",
                    "status": "PASS",
                    "details": f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}"
                })
            else:
                self.test_results.append({
                    "test": "TensorFlow GPU Support",
                    "status": "PASS",
                    "details": "CPU-only mode (acceptable for development)"
                })
        except Exception as e:
            self.test_results.append({
                "test": "TensorFlow GPU Support",
                "status": "FAIL",
                "details": f"TensorFlow error: {str(e)}"
            })
        
        # Test model file accessibility using utility functions
        model_files = [self.model_files["model_1m"], self.model_files["model_3m"]]
        scaler_files = [self.model_files["scaler_1m"], self.model_files["scaler_3m"]]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                file_size = os.path.getsize(model_file) // (1024 * 1024)  # Size in MB
                self.test_results.append({
                    "test": f"Model File {os.path.basename(model_file)}",
                    "status": "PASS",
                    "details": f"File exists, size: {file_size} MB"
                })
            else:
                    self.test_results.append({
                        "test": f"Model File {os.path.basename(model_file)}",
                        "status": "FAIL",
                        "details": "Model file not found"
                    })
        
        # Test scaler files
        for scaler_file in scaler_files:
            if os.path.exists(scaler_file):
                file_size = os.path.getsize(scaler_file) // 1024  # Size in KB
                self.test_results.append({
                    "test": f"Scaler File {os.path.basename(scaler_file)}",
                    "status": "PASS",
                    "details": f"Scaler file exists, size: {file_size} KB"
                })
            else:
                self.test_results.append({
                    "test": f"Scaler File {os.path.basename(scaler_file)}",
                    "status": "FAIL",
                    "details": "Scaler file not found"
                })
    
    def _test_environment_variables(self):
        """Test required environment variables"""
        print("🌍 Testing environment variables...")
        
        # Based on actual .env.example - only FRED_API_KEY is required
        required_env_vars = [
            ("FRED_API_KEY", "FRED API key for economic data")
        ]
        
        optional_env_vars = [
            ("SUPABASE_URL", "Supabase database URL (if using Supabase)"),
            ("SUPABASE_ANON_KEY", "Supabase anonymous key (if using Supabase)")
        ]
        
        # Test required variables
        for var_name, description in required_env_vars:
            value = os.getenv(var_name)
            if value:
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                self.test_results.append({
                    "test": f"Environment Variable {var_name}",
                    "status": "PASS",
                    "details": f"{description} configured (value: {masked_value})"
                })
            else:
                self.test_results.append({
                    "test": f"Environment Variable {var_name}",
                    "status": "FAIL",
                    "details": f"{description} not configured"
                })
        
        # Test optional variables
        for var_name, description in optional_env_vars:
            value = os.getenv(var_name)
            if value:
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                self.test_results.append({
                    "test": f"Optional Environment Variable {var_name}",
                    "status": "PASS",
                    "details": f"{description} configured (value: {masked_value})"
                })
            else:
                self.test_results.append({
                    "test": f"Optional Environment Variable {var_name}",
                    "status": "INFO",
                    "details": f"{description} not configured (optional)"
                })
    
    def test_server_configurations(self):
        """Test FastAPI server configuration variations"""
        print("🖥️ Testing server configurations...")
        
        # Test 1: Server startup with different configurations
        self._test_server_startup_configurations()
        
        # Test 2: API endpoint accessibility
        self._test_api_endpoint_accessibility()
        
        # Test 3: CORS configuration
        self._test_cors_configuration()
        
        # Test 4: Performance under different loads
        self._test_performance_configurations()
    
    def _test_server_startup_configurations(self):
        """Test server startup with different configurations"""
        print("🚀 Testing server startup configurations...")
        
        import requests
        
        # Test health endpoint
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.test_results.append({
                    "test": "Server Startup Configuration",
                    "status": "PASS",
                    "details": f"Server healthy: {data}"
                })
            else:
                self.test_results.append({
                    "test": "Server Startup Configuration",
                    "status": "FAIL",
                    "details": f"Health check failed: {response.status_code}"
                })
        except requests.ConnectionError:
            self.test_results.append({
                "test": "Server Startup Configuration",
                "status": "FAIL",
                "details": "Cannot connect to server - ensure server is running on localhost:8000"
            })
        except Exception as e:
            self.test_results.append({
                "test": "Server Startup Configuration",
                "status": "FAIL",
                "details": f"Server test failed: {str(e)}"
            })
    
    def _test_api_endpoint_accessibility(self):
        """Test API endpoint accessibility"""
        print("🔗 Testing API endpoint accessibility...")
        
        import requests
        
        # Based on actual API endpoints from main.py
        api_endpoints = [
            ("/", "Root endpoint"),
            ("/docs", "API documentation"),
            ("/api/v1/forecast/predict/1m", "1-month prediction endpoint"),
            ("/api/v1/macro-indicators", "Macro indicators endpoint"),
            ("/api/v1/fred-cache/status", "FRED cache status"),
            ("/api/v1/yearly-risk", "Yearly risk endpoint"),
            ("/api/v1/economic-charts/historical-data", "Economic charts endpoint")
        ]
        
        for endpoint, description in api_endpoints:
            try:
                if endpoint == "/api/v1/forecast/predict/1m":
                    # For POST endpoints, just check that the route exists by sending minimal valid data
                    # 422 (Unprocessable Entity) is expected for invalid/missing data, which means endpoint exists
                    response = requests.post(f"{self.api_base_url}{endpoint}", 
                                           json={"current_month_data": {}}, timeout=15)
                    # Accept 405, 422, 400 as indicators that endpoint exists but needs proper data
                    if response.status_code in [200, 422, 400, 202, 405]:
                        self.test_results.append({
                            "test": f"API Endpoint {endpoint}",
                            "status": "PASS",
                            "details": f"{description} accessible (status: {response.status_code})"
                        })
                    else:
                        self.test_results.append({
                            "test": f"API Endpoint {endpoint}",
                            "status": "FAIL",
                            "details": f"{description} returned status {response.status_code}"
                        })
                else:
                    # GET endpoint with longer timeout for heavy endpoints
                    timeout_val = 30 if "economic-charts" in endpoint else 10
                    response = requests.get(f"{self.api_base_url}{endpoint}", 
                                          timeout=timeout_val)
                    
                    if response.status_code in [200, 202]:
                        self.test_results.append({
                            "test": f"API Endpoint {endpoint}",
                            "status": "PASS",
                            "details": f"{description} accessible (status: {response.status_code})"
                        })
                    else:
                        self.test_results.append({
                            "test": f"API Endpoint {endpoint}",
                            "status": "FAIL",
                            "details": f"{description} returned status {response.status_code}"
                        })
                    
            except Exception as e:
                self.test_results.append({
                    "test": f"API Endpoint {endpoint}",
                    "status": "FAIL",
                    "details": f"Endpoint test failed: {str(e)}"
                })
    
    def _test_cors_configuration(self):
        """Test CORS configuration"""
        print("🌐 Testing CORS configuration...")
        
        import requests
        
        # Test CORS headers
        try:
            headers = {
                'Origin': 'http://localhost:3000',
                'Access-Control-Request-Method': 'GET',
                'Access-Control-Request-Headers': 'Content-Type'
            }
            
            # OPTIONS preflight request
            response = requests.options(f"{self.api_base_url}/api/v1/macro-indicators", 
                                      headers=headers, timeout=5)
            
            if response.status_code in [200, 204]:
                cors_headers = {
                    'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                    'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                    'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
                }
                
                self.test_results.append({
                    "test": "CORS Configuration",
                    "status": "PASS",
                    "details": f"CORS headers configured: {cors_headers}"
                })
            else:
                self.test_results.append({
                    "test": "CORS Configuration",
                    "status": "FAIL",
                    "details": f"CORS preflight failed: {response.status_code}"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": "CORS Configuration",
                "status": "FAIL",
                "details": f"CORS test failed: {str(e)}"
            })
    
    def _test_performance_configurations(self):
        """Test performance under different configurations"""
        print("⚡ Testing performance configurations...")
        
        import requests
        import threading
        import time
        
        # Test concurrent request handling
        def make_request(endpoint, results_list):
            start_time = time.time()
            try:
                response = requests.get(f"{self.api_base_url}{endpoint}", timeout=15)
                end_time = time.time()
                results_list.append({
                    "success": response.status_code == 200,
                    "response_time": end_time - start_time,
                    "status_code": response.status_code
                })
            except Exception as e:
                end_time = time.time()
                results_list.append({
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": str(e)
                })
        
        # Test concurrent requests
        concurrent_results = []
        threads = []
        
        for i in range(5):  # 5 concurrent requests
            thread = threading.Thread(
                target=make_request,
                args=("/api/v1/macro-indicators", concurrent_results)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=20)
        
        # Analyze performance results
        successful_requests = [r for r in concurrent_results if r.get("success", False)]
        total_requests = len(concurrent_results)
        
        if len(successful_requests) >= total_requests * 0.8:  # 80% success rate
            avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
            self.test_results.append({
                "test": "Performance Configuration - Concurrent Requests",
                "status": "PASS",
                "details": f"{len(successful_requests)}/{total_requests} successful, avg response time: {avg_response_time:.2f}s"
            })
        else:
            self.test_results.append({
                "test": "Performance Configuration - Concurrent Requests",
                "status": "FAIL",
                "details": f"Only {len(successful_requests)}/{total_requests} requests successful"
            })
    
    def test_deployment_configurations(self):
        """Test different deployment configuration scenarios"""
        print("🚀 Testing deployment configurations...")
        
        # Test 1: File system permissions
        self._test_file_system_permissions()
        
        # Test 2: Port availability
        self._test_port_configuration()
        
        # Test 3: Resource utilization
        self._test_resource_utilization()
    
    def _test_file_system_permissions(self):
        """Test file system permissions for required directories"""
        print("📁 Testing file system permissions...")
        
        required_directories = [
            self.paths["ml_models_path"],
            self.paths["data_path"], 
            self.paths["utils_path"]
        ]
        
        for directory in required_directories:
            if os.path.exists(directory):
                # Test read permissions
                if os.access(directory, os.R_OK):
                    # Test write permissions (for cache files)
                    if os.access(directory, os.W_OK):
                        self.test_results.append({
                            "test": f"File System Permissions {os.path.basename(directory)}",
                            "status": "PASS",
                            "details": "Read and write permissions available"
                        })
                    else:
                        self.test_results.append({
                            "test": f"File System Permissions {os.path.basename(directory)}",
                            "status": "FAIL",
                            "details": "Write permission denied"
                        })
                else:
                    self.test_results.append({
                        "test": f"File System Permissions {os.path.basename(directory)}",
                        "status": "FAIL",
                        "details": "Read permission denied"
                    })
            else:
                self.test_results.append({
                    "test": f"File System Permissions {os.path.basename(directory)}",
                    "status": "FAIL",
                    "details": "Directory does not exist"
                })
    
    def _test_port_configuration(self):
        """Test port availability and configuration"""
        print("🔌 Testing port configuration...")
        
        import socket
        
        # Test if port 8000 is available (or in use by our service)
        try:
            # Try to connect to our service
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', 8000))
            sock.close()
            
            if result == 0:
                # Port is in use, check if it's our service
                import requests
                try:
                    response = requests.get("http://localhost:8000/health", timeout=5)
                    if response.status_code == 200:
                        self.test_results.append({
                            "test": "Port Configuration",
                            "status": "PASS",
                            "details": "Port 8000 in use by RecessionScope service"
                        })
                    else:
                        self.test_results.append({
                            "test": "Port Configuration",
                            "status": "FAIL",
                            "details": "Port 8000 in use by unknown service"
                        })
                except:
                    self.test_results.append({
                        "test": "Port Configuration",
                        "status": "FAIL",
                        "details": "Port 8000 in use but service not responding"
                    })
            else:
                self.test_results.append({
                    "test": "Port Configuration",
                    "status": "INFO",
                    "details": "Port 8000 available (service not running)"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": "Port Configuration",
                "status": "FAIL",
                "details": f"Port test failed: {str(e)}"
            })
    
    def _test_resource_utilization(self):
        """Test system resource utilization"""
        print("💾 Testing resource utilization...")
        
        # Test available memory
        memory_info = self.system_info.get("available_memory", "Unknown")
        cpu_count = self.system_info.get("cpu_count", 0)
        
        # Basic resource checks
        if cpu_count >= 2:
            self.test_results.append({
                "test": "CPU Configuration",
                "status": "PASS",
                "details": f"{cpu_count} CPU cores available (minimum: 2)"
            })
        else:
            self.test_results.append({
                "test": "CPU Configuration",
                "status": "FAIL",
                "details": f"Only {cpu_count} CPU cores available (minimum: 2)"
            })
        
        # Check disk space for model files
        try:
            import shutil
            ml_models_path = self.paths["ml_models_path"]
            if os.path.exists(ml_models_path):
                total, used, free = shutil.disk_usage(ml_models_path)
                free_gb = free // (1024**3)
                
                if free_gb >= 1:  # At least 1GB free space
                    self.test_results.append({
                        "test": "Disk Space Configuration",
                        "status": "PASS",
                        "details": f"{free_gb} GB free space available"
                    })
                else:
                    self.test_results.append({
                        "test": "Disk Space Configuration",
                        "status": "FAIL",
                        "details": f"Only {free_gb} GB free space (minimum: 1GB)"
                    })
        except Exception as e:
            self.test_results.append({
                "test": "Disk Space Configuration",
                "status": "FAIL",
                "details": f"Could not check disk space: {str(e)}"
            })

def run_configuration_tests():
    """Run all configuration tests"""
    print("\n🚀 Starting Configuration Testing...")
    print("=" * 60)
    
    tester = TestSystemConfiguration()
    
    # Print system information
    print("\n💻 System Information:")
    for key, value in tester.system_info.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    
    # Run configuration tests
    tester.test_python_environment_configurations()
    tester.test_server_configurations()
    tester.test_deployment_configurations()
    
    # Print results
    print("\n📋 Configuration Test Results:")
    print("=" * 60)
    
    passed = failed = info = 0
    for result in tester.test_results:
        if result["status"] == "PASS":
            status_emoji = "✅"
            passed += 1
        elif result["status"] == "FAIL":
            status_emoji = "❌"
            failed += 1
        else:
            status_emoji = "ℹ️"
            info += 1
            
        print(f"{status_emoji} {result['test']}: {result['status']}")
        print(f"   Details: {result['details']}")
    
    print(f"\n📊 Summary: {passed} passed, {failed} failed, {info} informational")
    
    return tester.test_results

if __name__ == "__main__":
    results = run_configuration_tests()
    
    # Save results to file
    with open("configuration_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: configuration_test_results.json")