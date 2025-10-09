"""
Pytest-based System Configuration Tests
"""
import pytest
import os
import sys
import platform
import subprocess
import importlib.util
from fastapi.testclient import TestClient

@pytest.mark.config
class TestSystemConfiguration:
    """Test system configuration compatibility"""

    def test_python_version_compatibility(self):
        """Test Python version compatibility"""
        python_version = sys.version_info
        min_version = (3, 8)
        
        assert python_version >= min_version, f"Python {min_version[0]}.{min_version[1]}+ required"

    def test_required_packages_availability(self):
        """Test that all required packages are available"""
        required_packages = [
            'fastapi', 'uvicorn', 'pandas', 'numpy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        assert len(missing_packages) == 0, f"Missing packages: {missing_packages}"

    def test_environment_variables_configuration(self):
        """Test required environment variables"""
        # In test environment, these might be mocked
        if not os.getenv('TESTING'):
            assert os.getenv('FRED_API_KEY'), "FRED_API_KEY not configured"

    def test_platform_compatibility(self):
        """Test platform compatibility"""
        supported_platforms = ['Linux', 'Darwin', 'Windows']
        current_platform = platform.system()
        
        assert current_platform in supported_platforms, f"Unsupported platform: {current_platform}"

    def test_api_endpoints_availability(self, client: TestClient):
        """Test that API endpoints are available"""
        endpoints = [
            ("/", "GET"),
            ("/api/v1/yearly-risk", "GET"),
        ]
        
        for endpoint, method in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            
            # Endpoint should be available (not 404)
            assert response.status_code != 404, f"Endpoint {endpoint} not found"
