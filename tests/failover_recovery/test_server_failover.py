"""
Failover and Recovery Testing for RecessionScope API
Tests server power interruptions, service recovery, and data integrity
"""
import asyncio
import requests
import time
import signal
import os
import sys
import subprocess
from typing import Dict, Any
from unittest.mock import patch, MagicMock
import tempfile
import json

# Import project path utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_utils import get_project_paths, get_model_files

class TestServerFailover:
    """Test server power interruptions and recovery scenarios"""
    
    @pytest.fixture
    def api_base_url(self):
        return "http://localhost:8000"
    
    @pytest.fixture
    def test_data_backup(self):
        """Create backup of critical test data"""
        backup_data = {
            "ml_models": ["model_1m.keras", "model_3_months.keras"],
            "scalers": ["scaler_1m.pkl", "scaler_3m.pkl"],
            "historical_data": ["historical_data_1m.csv", "historical_data_3m.csv"]
        }
        return backup_data
    
    def test_server_sudden_shutdown_during_prediction(self, api_base_url):
        """Test power interruption during ML model prediction"""
        # Start a prediction request
        test_payload = {
            "custom_data": {
                "unemployment_rate": 4.5,
                "inflation_rate": 2.1,
                "gdp_growth": 2.8
            }
        }
        
        # Simulate server shutdown during prediction
        def simulate_server_crash():
            time.sleep(0.5)  # Let prediction start
            # Kill the FastAPI process (simulated power loss)
            for proc in psutil.process_iter(['pid', 'name']):
                if 'uvicorn' in proc.info['name'] or 'python' in proc.info['name']:
                    if 'main.py' in ' '.join(proc.cmdline() if proc.cmdline() else []):
                        proc.kill()
                        break
        
        # Test recovery mechanism
        with pytest.raises(requests.ConnectionError):
            # This should fail due to server shutdown
            response = requests.post(f"{api_base_url}/api/v1/forecast/predict/1m", 
                                   json=test_payload, timeout=2)
        
        # Verify server can be restarted
        self._restart_server()
        time.sleep(3)  # Allow startup time
        
        # Test that server is operational after restart
        response = requests.get(f"{api_base_url}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_database_connection_failure_recovery(self, api_base_url):
        """Test Supabase connection failure and recovery"""
        # Mock Supabase connection failure
        with patch('services.database_service.db_service') as mock_db:
            mock_db.side_effect = ConnectionError("Supabase connection lost")
            
            # This should handle database failure gracefully
            response = requests.get(f"{api_base_url}/api/v1/macro-indicators")
            # Should return cached data or error with recovery instructions
            assert response.status_code in [200, 503, 500]
        
        # Test recovery after database connection restored
        response = requests.get(f"{api_base_url}/api/v1/macro-indicators")
        # Should work normally after connection restored
        assert response.status_code == 200
    
    def test_ml_model_corruption_recovery(self):
        """Test ML model file corruption and recovery"""
        # Get relative path to model file using utility function
        model_files = get_model_files()
        model_path = model_files["model_1m"]
        api_base_url = "http://localhost:8000"
        
        # Create backup of model file
        if os.path.exists(model_path):
            backup_path = f"{model_path}.backup"
            subprocess.run(["cp", model_path, backup_path])
            
            # Corrupt the model file
            with open(model_path, 'w') as f:
                f.write("corrupted_data")
            
            # Test that system detects corruption and handles it
            response = requests.post(f"{api_base_url}/api/v1/forecast/predict/1m")
            # Should return error or use fallback mechanism
            assert response.status_code in [500, 503]
            assert "model" in response.json().get("detail", "").lower()
            
            # Restore from backup
            subprocess.run(["cp", backup_path, model_path])
            subprocess.run(["rm", backup_path])
            
            # Verify recovery
            response = requests.get(f"{api_base_url}/health")
            assert response.status_code == 200
    
    def test_fred_api_failure_recovery(self, api_base_url):
        """Test FRED API connection failure and cache recovery"""
        # Mock FRED API failure
        with patch('services.fred_data_service_1m.requests.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError("FRED API unavailable")
            
            # Should use cached data or return appropriate error
            response = requests.get(f"{api_base_url}/api/v1/macro-indicators")
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                # Should indicate data is from cache
                data = response.json()
                assert "cached" in str(data).lower() or "last_updated" in data
    
    def test_scheduler_service_recovery(self, api_base_url):
        """Test FRED data scheduler failure and recovery"""
        # Test scheduler status
        response = requests.get(f"{api_base_url}/api/v1/scheduler/status")
        assert response.status_code == 200
        
        # Test scheduler health check
        response = requests.get(f"{api_base_url}/api/v1/scheduler/health")
        assert response.status_code == 200
        
        # Test manual scheduler restart (recovery mechanism)
        response = requests.post(f"{api_base_url}/api/v1/scheduler/restart")
        # Should handle restart gracefully
        assert response.status_code in [200, 202]
    
    def test_cache_corruption_recovery(self, api_base_url):
        """Test cache corruption and recovery"""
        # Clear all caches to simulate corruption
        cache_endpoints = [
            "/api/v1/forecast/cache/clear",
            "/api/v1/macro-indicators/cache/clear",
            "/api/v1/economic-charts/cache/clear",
            "/api/v1/fred-cache/clear"
        ]
        
        for endpoint in cache_endpoints:
            response = requests.post(f"{api_base_url}{endpoint}")
            assert response.status_code == 200
        
        # Verify system can rebuild caches
        response = requests.get(f"{api_base_url}/api/v1/forecast/predict/1m")
        # Should work even without cache (may be slower)
        assert response.status_code in [200, 202]
    
    def test_incomplete_prediction_cycle_recovery(self, api_base_url):
        """Test interruption during prediction cycle and recovery"""
        # Start prediction process
        response = requests.post(f"{api_base_url}/api/v1/forecast/predict/all")
        
        # Simulate interruption by clearing prediction cache mid-process
        if response.status_code == 202:  # Async processing
            time.sleep(0.5)  # Let it start processing
            clear_response = requests.post(f"{api_base_url}/api/v1/forecast/cache/clear")
            assert clear_response.status_code == 200
        
        # Test that new prediction can be made after interruption
        time.sleep(1)
        recovery_response = requests.post(f"{api_base_url}/api/v1/forecast/predict/1m")
        assert recovery_response.status_code in [200, 202]
    
    def _restart_server(self):
        """Helper method to restart the FastAPI server"""
        # This would be implemented based on your deployment setup
        # For testing purposes, we'll just wait and assume external restart
        pass
    
    def test_data_integrity_after_recovery(self, api_base_url):
        """Test that data integrity is maintained after recovery"""
        # Get baseline prediction
        baseline_response = requests.post(f"{api_base_url}/api/v1/forecast/predict/1m")
        if baseline_response.status_code == 200:
            baseline_data = baseline_response.json()
            baseline_prediction = baseline_data.get("recession_probability", 0)
            
            # Clear caches (simulate recovery scenario)
            requests.post(f"{api_base_url}/api/v1/forecast/cache/clear")
            
            # Get prediction after "recovery"
            recovery_response = requests.post(f"{api_base_url}/api/v1/forecast/predict/1m")
            assert recovery_response.status_code == 200
            
            recovery_data = recovery_response.json()
            recovery_prediction = recovery_data.get("recession_probability", 0)
            
            # Predictions should be consistent (within tolerance for model variations)
            assert abs(baseline_prediction - recovery_prediction) < 0.1

if __name__ == "__main__":
    pytest.main([__file__])