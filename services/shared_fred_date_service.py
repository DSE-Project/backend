import httpx
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# FRED API Configuration
FRED_API_KEY = os.getenv("FRED_API_KEY")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FEDFUNDS_SERIES_ID = "FEDFUNDS"

@dataclass
class CachedFredDate:
    """Cached FRED date with timestamp"""
    date: str
    timestamp: datetime

class SharedFredDateService:
    """
    Shared service to fetch the latest FRED date once and cache it for a short period.
    This prevents multiple API calls when the dashboard loads all three predictions.
    """
    
    def __init__(self, cache_duration_minutes: int = 5):
        self._cached_date: Optional[CachedFredDate] = None
        self._cache_duration = timedelta(minutes=cache_duration_minutes)
        self._lock = asyncio.Lock()  # Prevent concurrent requests
    
    async def get_latest_fred_date(self) -> Optional[str]:
        """
        Get the latest FRED date, using cache if available and fresh.
        Returns None if unable to fetch from API.
        """
        async with self._lock:
            # Check if we have a valid cached date
            if self._is_cache_valid():
                logger.info(f"Using cached FRED date: {self._cached_date.date}")
                return self._cached_date.date
            
            # Cache is invalid or missing, fetch fresh data
            logger.info("Fetching fresh FRED date from API")
            fresh_date = await self._fetch_fresh_fred_date()
            
            if fresh_date:
                # Update cache
                self._cached_date = CachedFredDate(
                    date=fresh_date,
                    timestamp=datetime.now()
                )
                logger.info(f"Cached new FRED date: {fresh_date}")
                return fresh_date
            else:
                logger.error("Failed to fetch fresh FRED date")
                # Return cached date if available, even if expired
                if self._cached_date:
                    logger.warning(f"Using expired cached FRED date: {self._cached_date.date}")
                    return self._cached_date.date
                return None
    
    def _is_cache_valid(self) -> bool:
        """Check if the cached date is still valid"""
        if not self._cached_date:
            return False
        
        age = datetime.now() - self._cached_date.timestamp
        is_valid = age < self._cache_duration
        
        if not is_valid:
            logger.info(f"Cache expired (age: {age}, max: {self._cache_duration})")
        
        return is_valid
    
    async def _fetch_fresh_fred_date(self) -> Optional[str]:
        """Fetch the latest FRED date from API"""
        try:
            params = {
                "series_id": FEDFUNDS_SERIES_ID,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1
            }
            
            timeout_config = httpx.Timeout(30.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                logger.info(f"Fetching latest FRED date from FEDFUNDS series")
                response = await client.get(BASE_URL, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    logger.error("No data returned from FRED API")
                    return None
                    
                if "observations" not in data:
                    logger.error(f"No observations in FRED response: {data}")
                    return None
                    
                observations = data["observations"]
                if len(observations) == 0:
                    logger.error("Empty observations array from FRED API")
                    return None
                    
                latest_date = observations[0]["date"]
                logger.info(f"Successfully fetched latest FRED date: {latest_date}")
                return latest_date
                
        except httpx.TimeoutException as e:
            logger.error(f"Timeout fetching FRED date: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching FRED date: Status {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error fetching FRED date: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching FRED date: {type(e).__name__}: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear the cached date (useful for testing or manual refresh)"""
        logger.info("Clearing FRED date cache")
        self._cached_date = None
    
    def get_cache_info(self) -> dict:
        """Get information about the current cache state"""
        if not self._cached_date:
            return {
                "cached": False,
                "date": None,
                "timestamp": None,
                "age_seconds": None,
                "valid": False
            }
        
        age = datetime.now() - self._cached_date.timestamp
        return {
            "cached": True,
            "date": self._cached_date.date,
            "timestamp": self._cached_date.timestamp.isoformat(),
            "age_seconds": age.total_seconds(),
            "valid": self._is_cache_valid()
        }

# Global shared instance
shared_fred_date_service = SharedFredDateService(cache_duration_minutes=5)