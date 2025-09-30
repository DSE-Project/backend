"""
Prediction caching utility for recession forecast endpoints
Caches prediction responses for 1 hour to improve performance and user experience
"""
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading
import logging

logger = logging.getLogger(__name__)

class PredictionCache:
    """Thread-safe in-memory cache for prediction responses"""
    
    def __init__(self, default_ttl_seconds: int = 3600):  # 1 hour default
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
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if it exists and hasn't expired"""
        with self._lock:
            self._cleanup_expired()
            
            if key not in self.cache:
                logger.debug(f"Cache miss for key: {key}")
                return None
                
            entry = self.cache[key]
            if self._is_expired(entry):
                del self.cache[key]
                logger.debug(f"Cache expired for key: {key}")
                return None
                
            logger.debug(f"Cache hit for key: {key}")
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
        logger.info(f"Cached prediction for key: {key}, expires at: {expiry_time}")
    
    def invalidate(self, key: str) -> bool:
        """Remove a specific cache entry"""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                logger.info(f"Invalidated cache for key: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            cleared_count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared all cache entries ({cleared_count} items)")
    
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
    
    def get_cache_key(self, timeframe: str, data_source: str = "auto") -> str:
        """Generate a cache key for prediction requests"""
        # Include hour in the key so cache naturally refreshes hourly
        current_hour = datetime.now().strftime('%Y-%m-%d-%H')
        return f"prediction_{timeframe}_{data_source}_{current_hour}"

# Global cache instance
prediction_cache = PredictionCache(default_ttl_seconds=3600)  # 1 hour TTL

def cache_prediction_response(timeframe: str):
    """Decorator to cache prediction responses"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = prediction_cache.get_cache_key(timeframe)
            
            # Try to get cached response
            cached_response = prediction_cache.get(cache_key)
            if cached_response is not None:
                logger.info(f"✅ Serving cached {timeframe} prediction")
                # Add cache metadata
                if hasattr(cached_response, 'feature_importance'):
                    cached_response.feature_importance = cached_response.feature_importance or {}
                    cached_response.feature_importance['cached'] = True
                    cached_response.feature_importance['cache_key'] = cache_key
                return cached_response
            
            # Cache miss - execute the function
            logger.info(f"⚠️ Cache miss for {timeframe} prediction - generating new prediction")
            try:
                result = await func(*args, **kwargs)
                
                # Cache the result
                prediction_cache.set(cache_key, result)
                
                # Add metadata to indicate this is a fresh prediction
                if hasattr(result, 'feature_importance'):
                    result.feature_importance = result.feature_importance or {}
                    result.feature_importance['cached'] = False
                    result.feature_importance['cache_key'] = cache_key
                
                return result
                
            except Exception as e:
                logger.error(f"Error generating {timeframe} prediction: {e}")
                raise
                
        # Preserve the original function's metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator