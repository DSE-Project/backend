"""
Pytest-based Frontend Configuration Tests
"""
import pytest
import os
import subprocess
from fastapi.testclient import TestClient

@pytest.mark.config
class TestFrontendConfiguration:
    """Test frontend configuration"""

    def test_node_js_installation(self):
        """Test Node.js installation and version"""
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                major_version = int(version.replace('v', '').split('.')[0])
                min_version = 16
                assert major_version >= min_version, f"Node.js {min_version}+ required, got {version}"
            else:
                pytest.fail("Node.js not installed or not working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Node.js not found in PATH")

    def test_npm_installation(self):
        """Test npm installation and version"""
        try:
            result = subprocess.run(['npm', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                assert len(version) > 0
            else:
                pytest.fail("npm not installed or not working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("npm not found in PATH")

    def test_frontend_project_structure(self):
        """Test frontend project structure"""
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        project_root = os.path.dirname(backend_dir)
        frontend_dir = os.path.join(project_root, "frontend")
        
        if not os.path.exists(frontend_dir):
            pytest.skip("Frontend directory not found")
        
        essential_files = ["package.json", "index.html"]
        missing_files = []
        
        for filename in essential_files:
            filepath = os.path.join(frontend_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
        
        assert len(missing_files) == 0, f"Missing frontend files: {missing_files}"

    def test_api_integration_configuration(self, client: TestClient):
        """Test API integration configuration"""
        api_endpoints = ["/api/v1/yearly-risk"]
        
        for endpoint in api_endpoints:
            response = client.get(endpoint)
            assert response.status_code != 404, f"API endpoint {endpoint} not found"
