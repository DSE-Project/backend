import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class ExplainabilityCache:
    """Thread-safe in-memory cache for explainability responses"""
    
    def __init__(self, default_ttl_seconds: int = 7200):  # 2 hours default TTL
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
            logger.debug(f"Cleaned up {len(expired_keys)} expired explainability cache entries")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if it exists and hasn't expired"""
        with self._lock:
            self._cleanup_expired()
            
            if key not in self.cache:
                logger.debug(f"Explainability cache miss for key: {key}")
                return None
                
            entry = self.cache[key]
            if self._is_expired(entry):
                del self.cache[key]
                logger.debug(f"Explainability cache expired for key: {key}")
                return None
                
            logger.debug(f"Explainability cache hit for key: {key}")
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
        logger.info(f"Cached explainability for key: {key}, expires at: {expiry_time}")
    
    def invalidate(self, key: str) -> bool:
        """Remove a specific cache entry"""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                logger.info(f"Invalidated explainability cache for key: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            cleared_count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared all explainability cache entries ({cleared_count} items)")
    
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
    
    def get_explainability_cache_key(self, timeframe: str, explanation_type: str = "explain") -> str:
        """Generate a cache key for explainability requests"""
        # Cache refreshes every 2 hours (explanations are computationally expensive but stable)
        current_period = datetime.now().strftime('%Y-%m-%d-%H')
        # Round to 2-hour periods (00, 02, 04, 06, etc.)
        hour = int(current_period.split('-')[-1])
        period_hour = (hour // 2) * 2
        cache_period = current_period[:-2] + f"{period_hour:02d}"
        return f"explainability_{explanation_type}_{timeframe}_{cache_period}"

# Global cache instance
explainability_cache = ExplainabilityCache(default_ttl_seconds=7200)  # 2 hours TTL

def cache_explainability_response(timeframe: str, explanation_type: str = "explain"):
    """Decorator to cache explainability responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = explainability_cache.get_explainability_cache_key(timeframe, explanation_type)
            
            # Try to get cached response
            cached_response = explainability_cache.get(cache_key)
            if cached_response is not None:
                logger.info(f"✅ Serving cached {timeframe} {explanation_type}")
                # Add cache metadata
                cached_response['cached'] = True
                cached_response['cache_key'] = cache_key
                cached_response['cache_timestamp'] = datetime.now().isoformat()
                return cached_response
            
            # Cache miss - execute the function
            logger.info(f"⚠️ Cache miss for {timeframe} {explanation_type} - generating new explanation")
            try:
                result = await func(*args, **kwargs)
                
                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    result = {"data": result}
                
                # Cache the result
                explainability_cache.set(cache_key, result)
                
                # Add metadata to indicate this is a fresh explanation
                result['cached'] = False
                result['cache_key'] = cache_key
                result['cache_timestamp'] = datetime.now().isoformat()
                
                return result
                
            except Exception as e:
                logger.error(f"Error generating {timeframe} {explanation_type}: {e}")
                raise
                
        return wrapper
    return decorator