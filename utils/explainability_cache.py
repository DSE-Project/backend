import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ExplainabilityCache:
    """Thread-safe in-memory cache for SHAP explainers and background data"""
    
    def __init__(self, default_ttl_seconds: int = 7200):  # 2 hours default TTL
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self._lock = threading.RLock()
        self.default_ttl = default_ttl_seconds
        
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache if it exists and hasn't expired"""
        with self._lock:
            if key in self._cache:
                value, expiry_time = self._cache[key]
                if time.time() < expiry_time:
                    logger.debug(f"✅ Cache hit for explainer: {key}")
                    return value
                else:
                    # Expired - remove it
                    del self._cache[key]
                    logger.debug(f"⏰ Cache expired for explainer: {key}")
            
            logger.debug(f"❌ Cache miss for explainer: {key}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a value in cache with TTL"""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry_time = time.time() + ttl
        
        with self._lock:
            self._cache[key] = (value, expiry_time)
            logger.info(f"💾 Cached explainer: {key} (TTL: {ttl}s)")
    
    def delete(self, key: str) -> bool:
        """Delete a specific key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.info(f"🗑️ Deleted cached explainer: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached values"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"🧹 Cleared {count} cached explainers")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items"""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, (_, expiry_time) in self._cache.items():
                if current_time >= expiry_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
        
        if expired_keys:
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired explainer cache entries")
        
        return len(expired_keys)
    
    def get_cache_key(self, timeframe: str, seq_length: int = 12, num_samples: int = 100) -> str:
        """Generate a cache key for explainer instances"""
        # Include hour in the key so cache naturally refreshes every few hours
        current_4hour_block = datetime.now().strftime('%Y-%m-%d-%H')[:-1] + '0'  # Round to 4-hour blocks
        return f"explainer_{timeframe}_{seq_length}_{num_samples}_{current_4hour_block}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            current_time = time.time()
            active_count = 0
            expired_count = 0
            
            for _, expiry_time in self._cache.values():
                if current_time < expiry_time:
                    active_count += 1
                else:
                    expired_count += 1
            
            return {
                "total_entries": len(self._cache),
                "active_entries": active_count,
                "expired_entries": expired_count,
                "cache_keys": list(self._cache.keys())
            }

# Global cache instances for each timeframe
explainability_cache = ExplainabilityCache(default_ttl_seconds=7200)  # 2 hours TTL

class CachedExplainerData:
    """Container for cached explainer and its metadata"""
    def __init__(self, explainer: Any, background_data: np.ndarray, feature_names: list):
        self.explainer = explainer
        self.background_data = background_data
        self.feature_names = feature_names
        self.created_at = datetime.now()
    
    def is_valid(self) -> bool:
        """Check if the cached data is still valid"""
        # Could add additional validation logic here
        return (
            self.explainer is not None and 
            self.background_data is not None and 
            len(self.feature_names) > 0
        )