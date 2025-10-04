"""
Pytest configuration and shared fixtures for RecessionScope tests
"""
import os
import sys
import pytest
import asyncio
from typing import Generator
from fastapi.testclient import TestClient

# Add the backend directory to the path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

# Import the FastAPI app
from main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """
    Create a test client for the FastAPI application.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Base URL for API tests."""
    return "http://testserver"


@pytest.fixture
def sample_forecast_data():
    """Sample forecast data for testing."""
    return {
        "timeframe": "1m",
        "historical_months": 12,
        "forecast_months": 6
    }


@pytest.fixture
def sample_simulation_data():
    """Sample simulation data for testing."""
    return {
        "timeframe": "1m",
        "scenario": "optimistic",
        "parameters": {
            "gdp_growth": 0.02,
            "unemployment_rate": 0.05,
            "inflation_rate": 0.025
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """
    Set up test environment variables.
    This fixture runs automatically for all tests.
    """
    # Set test environment variables
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    
    # Mock external API keys if not present
    if not os.getenv("FRED_API_KEY"):
        monkeypatch.setenv("FRED_API_KEY", "test_fred_api_key")
    
    if not os.getenv("SUPABASE_URL"):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    
    if not os.getenv("SUPABASE_KEY"):
        monkeypatch.setenv("SUPABASE_KEY", "test_supabase_key")


@pytest.fixture
def mock_fred_api_response():
    """Mock FRED API response for testing."""
    return {
        "observations": [
            {
                "date": "2023-01-01",
                "value": "3.5"
            },
            {
                "date": "2023-02-01", 
                "value": "3.7"
            }
        ]
    }


@pytest.fixture
def mock_database_response():
    """Mock database response for testing."""
    return {
        "data": [
            {
                "id": 1,
                "date": "2023-01-01",
                "recession_probability": 0.15,
                "risk_level": "Low"
            }
        ],
        "status": "success"
    }


# Test data directories
@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return os.path.join(backend_dir, "tests", "test_data")


@pytest.fixture(scope="session") 
def model_dir():
    """Path to ML models directory."""
    return os.path.join(backend_dir, "ml_models")


@pytest.fixture(scope="session")
def data_dir():
    """Path to data directory."""
    return os.path.join(backend_dir, "data")