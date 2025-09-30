"""
Request Priority Management System
Provides priority-based request handling to ensure forecast prediction requests
are processed with higher priority than other API endpoints.
"""
import asyncio
import time
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import threading

logger = logging.getLogger(__name__)

class RequestPriority(IntEnum):
    """Priority levels for different request types"""
    CRITICAL = 1      # System health, cache management
    HIGH = 2          # Forecast predictions (dashboard)
    MEDIUM = 3        # Other forecast endpoints (test, status)
    LOW = 4           # Economic charts, macro indicators
    BACKGROUND = 5    # Scheduler, admin tasks

@dataclass
class PriorityRequest:
    """Container for prioritized requests"""
    request_id: str
    priority: RequestPriority
    created_at: float
    future: asyncio.Future
    endpoint: str
    method: str

class PriorityRequestManager:
    """Manages request prioritization and execution"""
    
    def __init__(self, max_concurrent_low_priority: int = 2, max_concurrent_high_priority: int = 10):
        self.max_concurrent_low_priority = max_concurrent_low_priority
        self.max_concurrent_high_priority = max_concurrent_high_priority
        
        # Separate queues for different priorities
        self.queues: Dict[RequestPriority, deque] = {
            priority: deque() for priority in RequestPriority
        }
        
        # Track active requests
        self.active_requests: Dict[RequestPriority, int] = {
            priority: 0 for priority in RequestPriority
        }
        
        # Semaphores for controlling concurrency
        self.high_priority_semaphore = asyncio.Semaphore(max_concurrent_high_priority)
        self.low_priority_semaphore = asyncio.Semaphore(max_concurrent_low_priority)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'queued_requests': 0,
            'average_wait_time': 0.0,
            'priority_stats': {priority.name: {'count': 0, 'avg_wait': 0.0} for priority in RequestPriority}
        }
        
        self._lock = threading.RLock()
        self._request_counter = 0
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        with self._lock:
            self._request_counter += 1
            return f"req_{self._request_counter}_{int(time.time() * 1000)}"
    
    def get_priority_for_endpoint(self, path: str, method: str) -> RequestPriority:
        """Determine priority based on endpoint path"""
        
        # CRITICAL priority (system health)
        if "/health" in path or "/cache/clear" in path:
            return RequestPriority.CRITICAL
            
        # HIGH priority (main dashboard predictions)
        if "/forecast/predict/" in path and method == "GET":
            if any(timeframe in path for timeframe in ["/1m", "/3m", "/6m"]):
                return RequestPriority.HIGH
                
        # MEDIUM priority (other forecast endpoints)
        if "/forecast/" in path:
            if "/status/" in path or "/test/" in path or "/cache/" in path:
                return RequestPriority.MEDIUM
                
        # LOW priority (charts and indicators)
        if any(endpoint in path for endpoint in ["/economic-charts", "/macro-indicators", "/yearly-risk"]):
            return RequestPriority.LOW
            
        # BACKGROUND priority (scheduler, admin)
        if "/scheduler/" in path or "/simulate/" in path:
            return RequestPriority.BACKGROUND
            
        # Default to MEDIUM for unknown endpoints
        return RequestPriority.MEDIUM
    
    async def execute_with_priority(self, request: Request, handler_func, *args, **kwargs):
        """Execute request with priority management"""
        priority = self.get_priority_for_endpoint(request.url.path, request.method)
        request_id = self._generate_request_id()
        
        # Update statistics
        with self._lock:
            self.stats['total_requests'] += 1
            self.stats['priority_stats'][priority.name]['count'] += 1
        
        # For high priority requests, execute immediately if possible
        if priority <= RequestPriority.MEDIUM:
            try:
                async with self.high_priority_semaphore:
                    start_time = time.time()
                    result = await handler_func(*args, **kwargs)
                    
                    # Update stats
                    wait_time = time.time() - start_time
                    self._update_completion_stats(priority, wait_time)
                    
                    logger.debug(f"✅ {priority.name} request {request_id} completed in {wait_time:.3f}s")
                    return result
                    
            except Exception as e:
                logger.error(f"❌ {priority.name} request {request_id} failed: {e}")
                raise
        
        # For low priority requests, use limited concurrency
        else:
            try:
                async with self.low_priority_semaphore:
                    # Check if there are high priority requests queued
                    high_priority_queued = sum(
                        len(self.queues[p]) for p in [RequestPriority.CRITICAL, RequestPriority.HIGH, RequestPriority.MEDIUM]
                    )
                    
                    if high_priority_queued > 0:
                        logger.info(f"⏳ Delaying {priority.name} request {request_id} - {high_priority_queued} high priority requests queued")
                        # Small delay to allow high priority requests to process
                        await asyncio.sleep(0.1)
                    
                    start_time = time.time()
                    result = await handler_func(*args, **kwargs)
                    
                    # Update stats
                    wait_time = time.time() - start_time
                    self._update_completion_stats(priority, wait_time)
                    
                    logger.debug(f"✅ {priority.name} request {request_id} completed in {wait_time:.3f}s")
                    return result
                    
            except Exception as e:
                logger.error(f"❌ {priority.name} request {request_id} failed: {e}")
                raise
    
    def _update_completion_stats(self, priority: RequestPriority, wait_time: float):
        """Update completion statistics"""
        with self._lock:
            self.stats['completed_requests'] += 1
            
            # Update priority-specific stats
            priority_stats = self.stats['priority_stats'][priority.name]
            old_avg = priority_stats['avg_wait']
            count = priority_stats['count']
            
            # Calculate new average
            new_avg = (old_avg * (count - 1) + wait_time) / count if count > 0 else wait_time
            priority_stats['avg_wait'] = new_avg
            
            # Update overall average
            total_completed = self.stats['completed_requests']
            old_overall_avg = self.stats['average_wait_time']
            self.stats['average_wait_time'] = (old_overall_avg * (total_completed - 1) + wait_time) / total_completed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current priority manager statistics"""
        with self._lock:
            active_by_priority = {
                priority.name: len(queue) for priority, queue in self.queues.items()
            }
            
            return {
                "total_requests": self.stats['total_requests'],
                "completed_requests": self.stats['completed_requests'],
                "success_rate": (self.stats['completed_requests'] / max(1, self.stats['total_requests'])) * 100,
                "average_wait_time_seconds": round(self.stats['average_wait_time'], 3),
                "concurrent_limits": {
                    "high_priority_max": self.max_concurrent_high_priority,
                    "low_priority_max": self.max_concurrent_low_priority,
                    "high_priority_available": self.high_priority_semaphore._value,
                    "low_priority_available": self.low_priority_semaphore._value
                },
                "priority_breakdown": self.stats['priority_stats'],
                "currently_queued": active_by_priority,
                "priority_mapping": {
                    "CRITICAL": "System health, cache management",
                    "HIGH": "Dashboard forecast predictions (/predict/1m, /predict/3m, /predict/6m)",
                    "MEDIUM": "Other forecast endpoints (status, test, cache)",
                    "LOW": "Economic charts, macro indicators, yearly risk",
                    "BACKGROUND": "Scheduler, simulation, admin tasks"
                }
            }

# Global priority manager instance
priority_manager = PriorityRequestManager(
    max_concurrent_low_priority=2,   # Limit low priority to 2 concurrent
    max_concurrent_high_priority=10  # Allow up to 10 high priority concurrent
)

async def get_priority_manager():
    """Dependency to get the priority manager"""
    return priority_manager

def with_priority_handling(handler_func):
    """Decorator to add priority handling to route handlers"""
    async def wrapper(request: Request, *args, **kwargs):
        return await priority_manager.execute_with_priority(request, handler_func, *args, **kwargs)
    
    wrapper.__name__ = handler_func.__name__
    wrapper.__doc__ = handler_func.__doc__
    return wrapper