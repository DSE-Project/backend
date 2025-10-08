from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging

from services.fred_data_scheduler import fred_scheduler

logger = logging.getLogger(__name__)

router = APIRouter()

# Response models
class SchedulerStatus(BaseModel):
    scheduler_running: bool
    last_update_times: Dict[str, Optional[str]]
    statistics: Dict[str, Any]
    next_scheduled_run: Optional[str]

class JobStatus(BaseModel):
    id: str
    name: str
    next_run_time: Optional[str]
    trigger: str

class UpdateResult(BaseModel):
    success: bool
    message: str
    updated: Optional[int] = None
    created: Optional[int] = None
    timeframe: Optional[str] = None

class ManualUpdateRequest(BaseModel):
    timeframe: Optional[str] = None  # '1m', '3m', '6m', or None for all

@router.get("/scheduler/status", response_model=SchedulerStatus)
async def get_scheduler_status():
    """Get current scheduler status and statistics"""
    try:
        stats = fred_scheduler.get_stats()
        return SchedulerStatus(**stats)
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scheduler/jobs", response_model=List[JobStatus])
async def get_scheduled_jobs():
    """Get information about all scheduled jobs"""
    try:
        jobs = fred_scheduler.get_job_status()
        return [JobStatus(**job) for job in jobs]
    except Exception as e:
        logger.error(f"Error getting scheduled jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scheduler/start", response_model=Dict[str, str])
async def start_scheduler():
    """Start the FRED data scheduler"""
    try:
        await fred_scheduler.start_scheduler()
        logger.info("Scheduler started via API")
        return {"status": "success", "message": "FRED data scheduler started successfully"}
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start scheduler: {str(e)}")

@router.post("/scheduler/stop", response_model=Dict[str, str])
async def stop_scheduler():
    """Stop the FRED data scheduler"""
    try:
        await fred_scheduler.stop_scheduler()
        logger.info("Scheduler stopped via API")
        return {"status": "success", "message": "FRED data scheduler stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop scheduler: {str(e)}")

@router.post("/scheduler/trigger", response_model=UpdateResult)
async def trigger_manual_update(
    background_tasks: BackgroundTasks,
    request: ManualUpdateRequest
):
    """Trigger a manual FRED data update"""
    try:
        # Validate timeframe if provided
        if request.timeframe and request.timeframe not in ['1m', '3m', '6m']:
            raise HTTPException(
                status_code=400, 
                detail="Invalid timeframe. Must be '1m', '3m', '6m', or None for all"
            )
        
        logger.info(f"Manual update triggered via API for {request.timeframe or 'all timeframes'}")
        
        # Run the update in background to avoid timeout
        background_tasks.add_task(
            run_manual_update_background, 
            request.timeframe
        )
        
        return UpdateResult(
            success=True,
            message=f"Manual update initiated for {request.timeframe or 'all timeframes'}",
            timeframe=request.timeframe
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering manual update: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger update: {str(e)}")

async def run_manual_update_background(timeframe: Optional[str]):
    """Background task for manual updates"""
    try:
        start_time = datetime.now()
        logger.info(f"Starting background manual update for {timeframe or 'all timeframes'}")
        
        result = await fred_scheduler.trigger_immediate_update(timeframe)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Background manual update completed in {duration:.1f}s: {result}")
        
    except Exception as e:
        logger.error(f"Background manual update failed: {e}")

@router.get("/scheduler/health")
async def scheduler_health_check():
    """Health check endpoint for the scheduler"""
    try:
        stats = fred_scheduler.get_stats()
        
        # Determine health status
        is_healthy = True
        issues = []
        
        if not stats['scheduler_running']:
            is_healthy = False
            issues.append("Scheduler is not running")
        
        # Check if there have been recent failures
        if stats['statistics']['failed_runs'] > stats['statistics']['successful_runs'] * 0.5:
            is_healthy = False
            issues.append("High failure rate detected")
        
        # Check if last run was too long ago (more than 8 days)
        if stats['statistics']['last_run']:
            try:
                last_run = datetime.fromisoformat(stats['statistics']['last_run'].replace('Z', '+00:00'))
                days_since_last_run = (datetime.now().replace(tzinfo=last_run.tzinfo) - last_run).days
                if days_since_last_run > 8:
                    is_healthy = False
                    issues.append(f"Last run was {days_since_last_run} days ago")
            except Exception as date_error:
                issues.append(f"Could not parse last run date: {date_error}")
        
        return {
            "healthy": is_healthy,
            "status": "healthy" if is_healthy else "unhealthy",
            "issues": issues,
            "scheduler_running": stats['scheduler_running'],
            "last_run": stats['statistics']['last_run'],
            "next_run": stats['next_scheduled_run'],
            "total_runs": stats['statistics']['total_runs'],
            "success_rate": (
                stats['statistics']['successful_runs'] / max(stats['statistics']['total_runs'], 1)
                if stats['statistics']['total_runs'] > 0 else 0
            )
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "healthy": False,
            "status": "unhealthy", 
            "issues": [f"Health check error: {str(e)}"],
            "scheduler_running": False
        }

@router.get("/scheduler/last-update/{timeframe}")
async def get_last_update_info(timeframe: str):
    """Get detailed information about the last update for a specific timeframe"""
    if timeframe not in ['1m', '3m', '6m']:
        raise HTTPException(status_code=400, detail="Invalid timeframe. Must be '1m', '3m', or '6m'")
    
    try:
        stats = fred_scheduler.get_stats()
        last_update = stats['last_update_times'].get(timeframe)
        
        return {
            "timeframe": timeframe,
            "last_update": last_update,
            "scheduler_running": stats['scheduler_running'],
            "statistics": stats['statistics']
        }
        
    except Exception as e:
        logger.error(f"Error getting last update info for {timeframe}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scheduler/config")
async def get_scheduler_config():
    """Get current scheduler configuration"""
    try:
        from config.scheduler_config import get_scheduler_summary, validate_config
        
        config_summary = get_scheduler_summary()
        config_issues = validate_config()
        
        return {
            "configuration": config_summary,
            "validation": {
                "valid": len(config_issues) == 0,
                "issues": config_issues
            }
        }
    except Exception as e:
        logger.error(f"Error getting scheduler config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scheduler/test-fred-connection")
async def test_fred_connection():
    """Test FRED API connectivity and configuration"""
    try:
        import os
        import socket
        import httpx
        
        # Check API key
        fred_api_key = os.getenv('FRED_API_KEY')
        if not fred_api_key:
            return {
                "status": "error",
                "issue": "FRED_API_KEY not set",
                "solution": "Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html"
            }
        
        # Test DNS resolution
        try:
            ip_address = socket.gethostbyname("api.stlouisfed.org")
        except socket.gaierror as e:
            return {
                "status": "error",
                "issue": f"DNS resolution failed: {e}",
                "solution": "Check internet connection, try different DNS (8.8.8.8), or check firewall"
            }
        
        # Test FRED API call
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "FEDFUNDS",
                "api_key": fred_api_key,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc"
            }
            
            timeout_config = httpx.Timeout(10.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if "observations" in data and len(data["observations"]) > 0:
                        obs = data["observations"][0]
                        return {
                            "status": "success",
                            "message": "FRED API connection successful",
                            "test_result": {
                                "series": "FEDFUNDS",
                                "latest_value": obs["value"],
                                "latest_date": obs["date"]
                            },
                            "dns_resolution": f"api.stlouisfed.org → {ip_address}"
                        }
                    else:
                        return {
                            "status": "error",
                            "issue": "FRED API returned unexpected format",
                            "response": data
                        }
                elif response.status_code == 400:
                    return {
                        "status": "error",
                        "issue": "Invalid FRED API key",
                        "solution": "Check your FRED_API_KEY - get a new one at https://fred.stlouisfed.org/docs/api/api_key.html"
                    }
                else:
                    return {
                        "status": "error",
                        "issue": f"FRED API returned status {response.status_code}",
                        "response": response.text[:500]
                    }
                    
        except httpx.TimeoutException:
            return {
                "status": "error",
                "issue": "FRED API request timed out",
                "solution": "Check internet connection or try again later"
            }
        except Exception as e:
            return {
                "status": "error",
                "issue": f"FRED API request failed: {e}",
                "solution": "Check network connectivity and firewall settings"
            }
        
    except Exception as e:
        logger.error(f"Error testing FRED connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scheduler/debug-series/{timeframe}")
async def debug_timeframe_series(timeframe: str):
    """Debug what series are being used for each timeframe"""
    if timeframe not in ['1m', '3m', '6m']:
        raise HTTPException(status_code=400, detail="Invalid timeframe. Must be '1m', '3m', or '6m'")
    
    try:
        config = fred_scheduler.series_mappings[timeframe]
        series_ids = config['series_ids']
        
        return {
            "timeframe": timeframe,
            "table_name": config['table'],
            "service_module": config['service_module'],
            "series_count": len(series_ids),
            "series_list": list(series_ids.keys()),
            "series_mapping": series_ids,
            "network_test_result": await fred_scheduler._test_network_connectivity()
        }
        
    except Exception as e:
        logger.error(f"Error debugging timeframe series: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scheduler/reset-stats")
async def reset_scheduler_stats():
    """Reset scheduler statistics (useful for testing/maintenance)"""
    try:
        fred_scheduler.update_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'last_run': None,
            'last_error': None,
            'records_updated': 0,
            'records_created': 0
        }
        
        logger.info("Scheduler statistics reset via API")
        return {"status": "success", "message": "Scheduler statistics reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting scheduler stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

