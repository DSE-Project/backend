import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import requests
from dataclasses import dataclass
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class MacroIndicator:
    """Data class for macroeconomic indicators"""
    name: str
    symbol: str
    current_value: float
    previous_value: Optional[float]
    change_value: Optional[float]
    change_percent: Optional[float]
    unit: str
    last_updated: datetime
    data_source: str

class MacroIndicatorsService:
    """Service for fetching and managing macroeconomic indicators"""
    
    def __init__(self):
        # Get API key from environment variable with fallback
        self.fred_api_key = os.getenv("FRED_API_KEY", "a7322a04237938ac0ac0a89311b7ae3a")
        self.fred_base_url = "https://api.stlouisfed.org/fred"
        
        if not self.fred_api_key or self.fred_api_key == "demo_key":
            logger.warning("FRED API key not properly configured. Please set FRED_API_KEY environment variable.")
        
        # Alternative series IDs as fallbacks
        self.fallback_series = {
            "INDPRO": ["MANEMP", "CEU3000000001"],  # Industrial Production alternatives
            "UMCSENT": ["CSCICP03USM665S"],  # Consumer confidence alternatives
        }
        
        # FRED series IDs for key indicators
        self.indicators_config = {
            "unemployment_rate": {
                "fred_id": "UNRATE",
                "name": "Unemployment Rate",
                "unit": "%",
                "symbol": "UNRATE"
            },
            "cpi_inflation": {
                "fred_id": "CPIAUCSL",
                "name": "CPI Inflation (YoY)",
                "unit": "%",
                "symbol": "CPI"
            },
            "yield_spread_10y_2y": {
                "fred_id": "T10Y2Y",
                "name": "10Y-2Y Treasury Spread",
                "unit": "%",
                "symbol": "T10Y2Y"
            },
            "fed_funds_rate": {
                "fred_id": "FEDFUNDS",
                "name": "Federal Funds Rate",
                "unit": "%",
                "symbol": "FEDFUNDS"
            },
            "ism_pmi": {
                "fred_id": "INDPRO",
                "name": "Industrial Production Index",
                "unit": "Index",
                "symbol": "INDPRO"
            },
            "consumer_confidence": {
                "fred_id": "UMCSENT",
                "name": "Consumer Sentiment Index",
                "unit": "Index",
                "symbol": "UMCSENT"
            }
        }
    
    async def get_all_indicators(self) -> Dict[str, Any]:
        """Fetch all macroeconomic indicators"""
        try:
            # Fetch real data from FRED API
            real_data = await self._fetch_all_fred_data()
            
            return {
                "indicators": real_data,
                "last_updated": datetime.now().isoformat(),
                "data_sources": ["Federal Reserve Economic Data (FRED)", "Bureau of Labor Statistics", "Institute for Supply Management", "Conference Board"]
            }
        
        except Exception as e:
            logger.error(f"Error fetching indicators: {str(e)}")
            # Fallback to mock data if API fails
            logger.info("Falling back to mock data due to API error")
            mock_data = await self._get_mock_indicators()
            return {
                "indicators": mock_data,
                "last_updated": datetime.now().isoformat(),
                "data_sources": ["Mock Data (API Unavailable)"]
            }
    
    async def _fetch_all_fred_data(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all indicators from FRED API"""
        indicators_data = {}
        
        for key, config in self.indicators_config.items():
            try:
                # Try to fetch the latest 2 observations for comparison
                fred_data = await self._fetch_from_fred_with_fallback(config["fred_id"], limit=2)
                
                if fred_data and "observations" in fred_data and len(fred_data["observations"]) > 0:
                    processed_data = self._process_fred_data(fred_data, config)
                    indicators_data[key] = processed_data
                else:
                    # If no data available, use a default structure
                    logger.warning(f"No data available for {config['name']} ({config['fred_id']})")
                    indicators_data[key] = self._create_fallback_indicator(config)
                    
            except Exception as e:
                logger.error(f"Error fetching {config['name']}: {str(e)}")
                indicators_data[key] = self._create_fallback_indicator(config)
        
        return indicators_data
    
    async def _fetch_from_fred_with_fallback(self, series_id: str, limit: int = 2) -> Dict[str, Any]:
        """Fetch from FRED with fallback series if primary fails"""
        try:
            # Try primary series first
            return await self._fetch_from_fred(series_id, limit)
        except Exception as e:
            logger.warning(f"Primary series {series_id} failed: {str(e)}")
            
            # Try fallback series if available
            if series_id in self.fallback_series:
                for fallback_id in self.fallback_series[series_id]:
                    try:
                        logger.info(f"Trying fallback series {fallback_id} for {series_id}")
                        return await self._fetch_from_fred(fallback_id, limit)
                    except Exception as fallback_error:
                        logger.warning(f"Fallback series {fallback_id} also failed: {str(fallback_error)}")
                        continue
            
            # If all attempts failed, re-raise the original exception
            raise e
    
    def _process_fred_data(self, fred_data: Dict[str, Any], config: Dict[str, str]) -> Dict[str, Any]:
        """Process FRED API response into our indicator format"""
        observations = fred_data["observations"]
        
        if not observations or len(observations) == 0:
            return self._create_fallback_indicator(config)
        
        # Get the latest observation (most recent)
        current_obs = observations[0]
        current_value = float(current_obs["value"]) if current_obs["value"] != "." else None
        
        # Get previous observation if available
        previous_value = None
        change_value = None
        change_percent = None
        
        if len(observations) > 1 and current_value is not None:
            prev_obs = observations[1]
            if prev_obs["value"] != ".":
                previous_value = float(prev_obs["value"])
                change_value = current_value - previous_value
                change_percent = (change_value / previous_value) * 100 if previous_value != 0 else 0
        
        # Handle special case for CPI inflation - calculate YoY change
        if config["fred_id"] == "CPIAUCSL":
            current_value, previous_value, change_value, change_percent = self._calculate_cpi_inflation(observations)
        
        # Determine trend
        trend = "stable"
        if change_value is not None:
            if change_value > 0:
                trend = "increasing"
            elif change_value < 0:
                trend = "decreasing"
        
        return {
            "name": config["name"],
            "symbol": config["symbol"],
            "current_value": current_value or 0.0,
            "previous_value": previous_value,
            "change_value": change_value,
            "change_percent": change_percent,
            "unit": config["unit"],
            "last_updated": current_obs["date"],
            "trend": trend,
            "data_source": "Federal Reserve Economic Data (FRED)"
        }
    
    def _calculate_cpi_inflation(self, observations: list) -> tuple:
        """Calculate year-over-year CPI inflation rate"""
        try:
            # For CPI, we want YoY inflation rate
            # We need at least 12 months of data to calculate YoY
            if len(observations) < 12:
                # Use available data for approximation
                current_cpi = float(observations[0]["value"]) if observations[0]["value"] != "." else None
                prev_cpi = float(observations[-1]["value"]) if observations[-1]["value"] != "." else None
                
                if current_cpi and prev_cpi:
                    # Calculate inflation rate
                    inflation_rate = ((current_cpi - prev_cpi) / prev_cpi) * 100
                    prev_inflation = 0  # Approximate
                    change = inflation_rate - prev_inflation
                    change_percent = (change / prev_inflation) * 100 if prev_inflation != 0 else 0
                    
                    return inflation_rate, prev_inflation, change, change_percent
            
            # If we have enough data, calculate proper YoY inflation
            current_cpi = float(observations[0]["value"]) if observations[0]["value"] != "." else None
            year_ago_cpi = float(observations[11]["value"]) if len(observations) > 11 and observations[11]["value"] != "." else None
            prev_year_ago_cpi = float(observations[12]["value"]) if len(observations) > 12 and observations[12]["value"] != "." else None
            
            if current_cpi and year_ago_cpi:
                current_inflation = ((current_cpi - year_ago_cpi) / year_ago_cpi) * 100
                
                if prev_year_ago_cpi:
                    prev_inflation = ((year_ago_cpi - prev_year_ago_cpi) / prev_year_ago_cpi) * 100
                    change = current_inflation - prev_inflation
                    change_percent = (change / prev_inflation) * 100 if prev_inflation != 0 else 0
                    return current_inflation, prev_inflation, change, change_percent
                
                return current_inflation, None, None, None
                
        except Exception as e:
            logger.error(f"Error calculating CPI inflation: {str(e)}")
        
        return 0.0, None, None, None
    
    def _create_fallback_indicator(self, config: Dict[str, str]) -> Dict[str, Any]:
        """Create a fallback indicator when data is not available"""
        return {
            "name": config["name"],
            "symbol": config["symbol"],
            "current_value": 0.0,
            "previous_value": None,
            "change_value": None,
            "change_percent": None,
            "unit": config["unit"],
            "last_updated": datetime.now().isoformat(),
            "trend": "stable",
            "data_source": "Data Unavailable"
        }
    
    async def _fetch_from_fred(self, series_id: str, limit: int = 2) -> Dict[str, Any]:
        """
        Fetch data from FRED API (Federal Reserve Economic Data)
        """
        url = f"{self.fred_base_url}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "limit": limit,
            "sort_order": "desc"
        }
        
        # For CPI data, we need more observations to calculate YoY inflation
        if series_id == "CPIAUCSL":
            params["limit"] = 15  # Get 15 months of data for YoY calculation
        
        try:
            logger.info(f"Fetching data for series {series_id} from FRED API")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if there's an error in the response
                if "error_code" in data:
                    logger.error(f"FRED API returned error for {series_id}: {data.get('error_message', 'Unknown error')}")
                    raise Exception(f"FRED API error: {data.get('error_message', 'Unknown error')}")
                
                # Check if we have observations
                if "observations" not in data or not data["observations"]:
                    logger.warning(f"No observations found for series {series_id}")
                    return {"observations": []}
                
                logger.info(f"Successfully fetched {len(data['observations'])} observations for {series_id}")
                return data
            else:
                logger.error(f"FRED API returned status code {response.status_code} for {series_id}")
                logger.error(f"Response: {response.text}")
                raise Exception(f"FRED API HTTP error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching from FRED API for {series_id}: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Error fetching from FRED API for {series_id}: {str(e)}")
            raise e
