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
    monthly_day: Optional[int] = Field(None, ge=1, le=31, description="Day of month for monthly pipeline (1-31)")
    monthly_hour: Optional[int] = Field(None, ge=0, le=23, description="Hour for monthly pipeline (0-23)")
    monthly_minute: Optional[int] = Field(None, ge=0, le=59, description="Minute for monthly pipeline (0-59)")
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

@router.post("/trigger/monthly", tags=["Pipeline Control"])
async def trigger_monthly_pipeline():
    """
    Manually trigger the monthly pipeline (comprehensive data check and forced retrain)
    """
    try:
        result = await scheduler_service.trigger_monthly_pipeline_now()
        
        if result['success']:
            return {
                "status": "success",
                "message": "Monthly pipeline triggered successfully",
                "results": result
            }
        else:
            raise HTTPException(status_code=500, detail=f"Monthly pipeline failed: {result.get('error', 'Unknown error')}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering monthly pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger monthly pipeline: {str(e)}")

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
    Update the monthly pipeline execution schedule
    """
    try:
        monthly_config = {}
        
        # Build monthly config
        if request.monthly_day is not None:
            monthly_config['day'] = request.monthly_day
        if request.monthly_hour is not None:
            monthly_config['hour'] = request.monthly_hour
        if request.monthly_minute is not None:
            monthly_config['minute'] = request.monthly_minute
        if request.timezone:
            monthly_config['timezone'] = request.timezone
        
        # Update the schedule
        success = scheduler_service.update_schedule(
            monthly_config=monthly_config if monthly_config else None
        )
        
        if success:
            return {
                "status": "success",
                "message": "Monthly schedule updated successfully",
                "monthly_config": monthly_config
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