"""
Priority Request Middleware
Automatically applies request prioritization to all incoming requests
"""
import asyncio
import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from utils.request_priority import priority_manager, RequestPriority

logger = logging.getLogger(__name__)

class PriorityMiddleware(BaseHTTPMiddleware):
    """Middleware to handle request prioritization"""
    
    def __init__(self, app, enable_logging: bool = True):
        super().__init__(app)
        self.enable_logging = enable_logging
    
    async def dispatch(self, request: Request, call_next):
        """Process request with priority handling"""
        start_time = time.time()
        
        # Get priority for this request
        priority = priority_manager.get_priority_for_endpoint(request.url.path, request.method)
        request_id = f"req_{int(time.time() * 1000)}"
        
        # Add priority info to request state
        request.state.priority = priority
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        if self.enable_logging:
            logger.info(f"🚀 {priority.name} request {request_id}: {request.method} {request.url.path}")
        
        try:
            # Execute request with priority management
            if priority <= RequestPriority.MEDIUM:
                # High priority - execute with high priority semaphore
                async with priority_manager.high_priority_semaphore:
                    response = await call_next(request)
            else:
                # Low priority - execute with low priority semaphore and check for high priority queue
                async with priority_manager.low_priority_semaphore:
                    # Brief delay if high priority requests are waiting
                    high_priority_active = (
                        priority_manager.max_concurrent_high_priority - 
                        priority_manager.high_priority_semaphore._value
                    )
                    
                    if high_priority_active > 5:  # If many high priority requests active
                        if self.enable_logging:
                            logger.info(f"⏳ Briefly delaying {priority.name} request {request_id} - high priority requests active")
                        await asyncio.sleep(0.05)  # 50ms delay
                    
                    response = await call_next(request)
            
            # Calculate and log response time
            duration = time.time() - start_time
            
            if self.enable_logging:
                logger.info(f"✅ {priority.name} request {request_id} completed in {duration:.3f}s")
            
            # Add priority headers to response
            response.headers["X-Request-Priority"] = priority.name
            response.headers["X-Request-Duration"] = f"{duration:.3f}"
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {priority.name} request {request_id} failed after {duration:.3f}s: {e}")
            
            # Return error response with priority info
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "request_id": request_id,
                    "priority": priority.name,
                    "error": str(e)
                },
                headers={
                    "X-Request-Priority": priority.name,
                    "X-Request-Duration": f"{duration:.3f}",
                    "X-Request-ID": request_id
                }
            )