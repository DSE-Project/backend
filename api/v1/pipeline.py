from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from services.data_pipeline_service import pipeline_service
from services.scheduler_service import scheduler_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class PipelineRunRequest(BaseModel):
    force_retrain: bool = Field(default=False, description="Force model retraining regardless of new data amount")

class PipelineStatusResponse(BaseModel):
    pipeline_available: bool
    scheduler_running: bool
    last_run: Optional[str] = None
    next_scheduled_run: Optional[str] = None
    
class ScheduleUpdateRequest(BaseModel):
    daily_hour: Optional[int] = Field(None, ge=0, le=23, description="Hour for daily pipeline (0-23)")
    daily_minute: Optional[int] = Field(None, ge=0, le=59, description="Minute for daily pipeline (0-59)")
    weekly_day: Optional[str] = Field(None, description="Day of week for weekly pipeline (monday, tuesday, etc.)")
    weekly_hour: Optional[int] = Field(None, ge=0, le=23, description="Hour for weekly pipeline (0-23)")
    weekly_minute: Optional[int] = Field(None, ge=0, le=59, description="Minute for weekly pipeline (0-59)")
    timezone: Optional[str] = Field(default="US/Eastern", description="Timezone for scheduling")

@router.get("/status", response_model=PipelineStatusResponse, tags=["Pipeline Status"])
async def get_pipeline_status():
    """
    Get the current status of the data pipeline and scheduler
    """
    try:
        # Check scheduler status
        job_status = scheduler_service.get_job_status()
        
        # Find next scheduled run
        next_run = None
        if job_status['scheduler_running'] and job_status['jobs']:
            # Get the earliest next run time
            next_runs = [job.get('next_run_time') for job in job_status['jobs'] if job.get('next_run_time')]
            if next_runs:
                next_run = min(next_runs)
        
        return PipelineStatusResponse(
            pipeline_available=True,
            scheduler_running=job_status['scheduler_running'],
            next_scheduled_run=next_run
        )
        
    except Exception as e:
        logger.error(f"Error getting pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {str(e)}")

@router.get("/jobs", tags=["Pipeline Status"])
async def get_scheduled_jobs():
    """
    Get details of all scheduled pipeline jobs
    """
    try:
        return scheduler_service.get_job_status()
    except Exception as e:
        logger.error(f"Error getting scheduled jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get scheduled jobs: {str(e)}")

@router.post("/run", tags=["Pipeline Control"])
async def trigger_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """
    Manually trigger the data pipeline
    
    This will:
    1. Check for new data from FRED
    2. Fetch and validate new data if available
    3. Store data in the database
    4. Retrain models if sufficient new data is available (or if force_retrain=True)
    """
    try:
        logger.info(f"Manual pipeline trigger requested, force_retrain={request.force_retrain}")
        
        # Run pipeline in background to avoid timeout
        def run_pipeline_task():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(pipeline_service.run_pipeline(request.force_retrain))
                logger.info(f"Background pipeline completed: {result['status']}")
            except Exception as e:
                logger.error(f"Background pipeline failed: {str(e)}")
            finally:
                loop.close()
        
        background_tasks.add_task(run_pipeline_task)
        
        return {
            "status": "started",
            "message": "Pipeline execution started in background",
            "force_retrain": request.force_retrain,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger pipeline: {str(e)}")

@router.post("/run-sync", tags=["Pipeline Control"])
async def trigger_pipeline_sync(request: PipelineRunRequest):
    """
    Manually trigger the data pipeline and wait for completion
    
    WARNING: This endpoint may take several minutes to complete
    Use /run for background execution instead
    """
    try:
        logger.info(f"Synchronous pipeline trigger requested, force_retrain={request.force_retrain}")
        
        results = await pipeline_service.run_pipeline(request.force_retrain)
        
        return {
            "status": "completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in synchronous pipeline execution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@router.post("/trigger/daily", tags=["Pipeline Control"])
async def trigger_daily_pipeline():
    """
    Manually trigger the daily pipeline (normal data check and update)
    """
    try:
        result = await scheduler_service.trigger_daily_pipeline_now()
        
        if result['success']:
            return {
                "status": "success",
                "message": "Daily pipeline triggered successfully",
                "results": result
            }
        else:
            raise HTTPException(status_code=500, detail=f"Daily pipeline failed: {result.get('error', 'Unknown error')}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering daily pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger daily pipeline: {str(e)}")

@router.post("/trigger/weekly", tags=["Pipeline Control"])
async def trigger_weekly_pipeline():
    """
    Manually trigger the weekly comprehensive pipeline (forced retrain)
    """
    try:
        result = await scheduler_service.trigger_weekly_pipeline_now()
        
        if result['success']:
            return {
                "status": "success", 
                "message": "Weekly comprehensive pipeline triggered successfully",
                "results": result
            }
        else:
            raise HTTPException(status_code=500, detail=f"Weekly pipeline failed: {result.get('error', 'Unknown error')}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering weekly pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger weekly pipeline: {str(e)}")

@router.post("/retrain/{model_period}", tags=["Model Training"])
async def retrain_specific_model(model_period: str):
    """
    Retrain a specific model (1m, 3m, or 6m)
    
    Args:
        model_period: The model to retrain ('1m', '3m', or '6m')
    """
    if model_period not in ['1m', '3m', '6m']:
        raise HTTPException(
            status_code=400, 
            detail="Invalid model period. Must be one of: 1m, 3m, 6m"
        )
    
    try:
        from services.model_training_service import ModelTrainingService
        
        trainer = ModelTrainingService()
        result = await trainer.retrain_model(model_period)
        
        if result['success']:
            return {
                "status": "success",
                "message": f"{model_period} model retrained successfully",
                "results": result
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to retrain {model_period} model: {result.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retraining {model_period} model: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrain {model_period} model: {str(e)}"
        )

@router.post("/retrain/all", tags=["Model Training"])
async def retrain_all_models():
    """
    Retrain all models (1m, 3m, and 6m)
    
    WARNING: This may take a long time to complete
    """
    try:
        from services.model_training_service import ModelTrainingService
        
        trainer = ModelTrainingService()
        results = await trainer.retrain_all_models()
        
        return {
            "status": "completed",
            "message": f"Retrained {len(results['models_retrained'])} models successfully",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error retraining all models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrain models: {str(e)}")

@router.put("/schedule", tags=["Pipeline Configuration"])
async def update_schedule(request: ScheduleUpdateRequest):
    """
    Update the pipeline execution schedule
    """
    try:
        daily_config = {}
        weekly_config = {}
        
        # Build daily config
        if request.daily_hour is not None:
            daily_config['hour'] = request.daily_hour
        if request.daily_minute is not None:
            daily_config['minute'] = request.daily_minute
        if request.timezone:
            daily_config['timezone'] = request.timezone
        
        # Build weekly config  
        if request.weekly_day is not None:
            # Convert day name to number (APScheduler format)
            day_mapping = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            day_name = request.weekly_day.lower()
            if day_name in day_mapping:
                weekly_config['day_of_week'] = day_mapping[day_name]
            else:
                raise HTTPException(status_code=400, detail=f"Invalid day name: {request.weekly_day}")
        if request.weekly_hour is not None:
            weekly_config['hour'] = request.weekly_hour
        if request.weekly_minute is not None:
            weekly_config['minute'] = request.weekly_minute
        if request.timezone:
            weekly_config['timezone'] = request.timezone
        
        # Update the schedule
        success = scheduler_service.update_schedule(
            daily_config=daily_config if daily_config else None,
            weekly_config=weekly_config if weekly_config else None
        )
        
        if success:
            return {
                "status": "success",
                "message": "Schedule updated successfully",
                "daily_config": daily_config,
                "weekly_config": weekly_config
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update schedule")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update schedule: {str(e)}")

@router.post("/scheduler/start", tags=["Pipeline Configuration"])
async def start_scheduler():
    """
    Start the pipeline scheduler
    """
    try:
        if scheduler_service.is_running:
            return {
                "status": "already_running",
                "message": "Scheduler is already running"
            }
        
        scheduler_service.start()
        
        return {
            "status": "success",
            "message": "Scheduler started successfully"
        }
        
    except Exception as e:
        logger.error(f"Error starting scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start scheduler: {str(e)}")

@router.post("/scheduler/stop", tags=["Pipeline Configuration"])
async def stop_scheduler():
    """
    Stop the pipeline scheduler
    """
    try:
        if not scheduler_service.is_running:
            return {
                "status": "not_running",
                "message": "Scheduler is not currently running"
            }
        
        scheduler_service.stop()
        
        return {
            "status": "success",
            "message": "Scheduler stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Error stopping scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop scheduler: {str(e)}")

@router.get("/data/latest", tags=["Data Management"])
async def get_latest_data_info():
    """
    Get information about the latest data in the database
    """
    try:
        from services.database_service import db_service
        
        info = {}
        
        for table in ['historical_data_1m', 'historical_data_3m', 'historical_data_6m']:
            try:
                df = db_service.load_historical_data(table)
                if df is not None and not df.empty:
                    info[table] = {
                        'latest_date': df.index.max().strftime('%Y-%m-%d'),
                        'earliest_date': df.index.min().strftime('%Y-%m-%d'),
                        'total_records': len(df),
                        'columns': list(df.columns)
                    }
                else:
                    info[table] = {
                        'status': 'no_data'
                    }
            except Exception as e:
                info[table] = {
                    'error': str(e)
                }
        
        return {
            "status": "success",
            "data_info": info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting latest data info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get data info: {str(e)}")