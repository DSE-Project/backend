"""
FRED Data caching utility for macro indicators and economic charts
Caches FRED API responses to improve performance and reduce API calls
"""
import time
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import threading
import logging

logger = logging.getLogger(__name__)

class FredDataCache:
    """Thread-safe in-memory cache for FRED economic data"""
    
    def __init__(self, default_ttl_seconds: int = 1800):  # 30 minutes default (FRED data updates infrequently)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl_seconds
        self._lock = threading.RLock()
        
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry has expired"""
        return time.time() > cache_entry['expires_at']
    
    def _cleanup_expired(self):
        """Remove expired entries from cache"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items() 
            if current_time > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self.cache[key]
            
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired FRED cache entries")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if it exists and hasn't expired"""
        with self._lock:
            self._cleanup_expired()
            
            if key not in self.cache:
                logger.debug(f"FRED cache miss for key: {key}")
                return None
                
            entry = self.cache[key]
            if self._is_expired(entry):
                del self.cache[key]
                logger.debug(f"FRED cache expired for key: {key}")
                return None
                
            logger.debug(f"FRED cache hit for key: {key}")
            return entry['data']
    
    def set(self, key: str, data: Any, ttl_seconds: Optional[int] = None) -> None:
        """Cache a value with TTL"""
        ttl = ttl_seconds or self.default_ttl
        expires_at = time.time() + ttl
        
        with self._lock:
            self.cache[key] = {
                'data': data,
                'created_at': time.time(),
                'expires_at': expires_at,
                'ttl': ttl
            }
            
        # Convert expiry time to readable format for logging
        expiry_time = datetime.fromtimestamp(expires_at).strftime('%H:%M:%S')
        logger.info(f"Cached FRED data for key: {key}, expires at: {expiry_time}")
    
    def invalidate(self, key: str) -> bool:
        """Remove a specific cache entry"""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                logger.info(f"Invalidated FRED cache for key: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            cleared_count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared all FRED cache entries ({cleared_count} items)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            self._cleanup_expired()
            
            now = time.time()
            active_entries = len(self.cache)
            
            # Calculate remaining TTL for each entry
            entries_info = []
            for key, entry in self.cache.items():
                remaining_ttl = max(0, entry['expires_at'] - now)
                entries_info.append({
                    'key': key,
                    'created_at': datetime.fromtimestamp(entry['created_at']).isoformat(),
                    'expires_at': datetime.fromtimestamp(entry['expires_at']).isoformat(),
                    'remaining_seconds': int(remaining_ttl),
                    'remaining_minutes': round(remaining_ttl / 60, 1)
                })
            
            return {
                'active_entries': active_entries,
                'default_ttl_seconds': self.default_ttl,
                'default_ttl_minutes': self.default_ttl / 60,
                'entries': entries_info
            }
    
    def get_macro_indicators_cache_key(self) -> str:
        """Generate cache key for macro indicators"""
        # Cache for 30 minutes, so include 30-minute window in key
        current_window = datetime.now().strftime('%Y-%m-%d-%H-%M')[:16]  # YYYY-MM-DD-HH-M0 or M3
        window_30min = current_window[:-1] + ('0' if int(current_window[-1]) < 3 else '3')
        return f"macro_indicators_{window_30min}"
    
    def get_economic_charts_cache_key(self, period: str, indicators: Optional[List[str]] = None) -> str:
        """Generate cache key for economic charts data"""
        # Create hash of indicators for cache key
        indicators_str = "all" if not indicators else "_".join(sorted(indicators))
        indicators_hash = hashlib.md5(indicators_str.encode()).hexdigest()[:8]
        
        # Cache for 30 minutes
        current_window = datetime.now().strftime('%Y-%m-%d-%H-%M')[:16]
        window_30min = current_window[:-1] + ('0' if int(current_window[-1]) < 3 else '3')
        
        return f"economic_charts_{period}_{indicators_hash}_{window_30min}"

# Global cache instance
fred_data_cache = FredDataCache(default_ttl_seconds=1800)  # 30 minutes TTL

def cache_fred_data(cache_key_func, ttl_seconds: Optional[int] = None):
    """Decorator to cache FRED data responses"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key based on function arguments
            if hasattr(cache_key_func, '__call__'):
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = cache_key_func
            
            # Try to get cached response
            cached_response = fred_data_cache.get(cache_key)
            if cached_response is not None:
                logger.info(f"✅ Serving cached FRED data: {cache_key}")
                return cached_response
            
            # Cache miss - execute the function
            logger.info(f"⚠️ FRED cache miss: {cache_key} - fetching fresh data")
            try:
                result = await func(*args, **kwargs)
                
                # Cache the result
                fred_data_cache.set(cache_key, result, ttl_seconds)
                
                return result
                
            except Exception as e:
                logger.error(f"Error fetching FRED data for {cache_key}: {e}")
                raise
                
        # Preserve the original function's metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator