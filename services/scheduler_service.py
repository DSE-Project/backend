import asyncio
import logging
from datetime import datetime, time
from typing import Optional, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from services.data_pipeline_service import pipeline_service

logger = logging.getLogger(__name__)

class SchedulerService:
    """
    Scheduler service for automated pipeline execution
    Handles monthly data pipeline runs (aligned with FRED data updates)
    """
    
    def __init__(self):
        """Initialize the scheduler service"""
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.pipeline = pipeline_service
        self.is_running = False
        
        # Configuration (can be moved to environment variables)
        self.monthly_schedule = {
            'day': 15,      # 15th of each month (mid-month after FRED updates)
            'hour': 6,      # 6 AM EST
            'minute': 30,   # 30 minutes past the hour
            'timezone': 'US/Eastern'
        }
        
        logger.info("SchedulerService initialized")
    
    def start(self):
        """Start the scheduler with configured jobs"""
        try:
            if self.is_running:
                logger.warning("⚠️ Scheduler is already running")
                return
            
            # Initialize the scheduler
            self.scheduler = AsyncIOScheduler()
            
            # Add monthly pipeline job
            self.scheduler.add_job(
                func=self._run_monthly_pipeline,
                trigger=CronTrigger(
                    day=self.monthly_schedule['day'],
                    hour=self.monthly_schedule['hour'],
                    minute=self.monthly_schedule['minute'],
                    timezone=self.monthly_schedule['timezone']
                ),
                id='monthly_pipeline',
                name='Monthly Data Pipeline',
                replace_existing=True,
                max_instances=1  # Prevent overlapping runs
            )
            
            # Add a health check job (every hour)
            self.scheduler.add_job(
                func=self._health_check,
                trigger=IntervalTrigger(hours=1),
                id='health_check',
                name='Pipeline Health Check',
                replace_existing=True,
                max_instances=1
            )
            
            # Start the scheduler
            self.scheduler.start()
            self.is_running = True
            
            logger.info("✅ Scheduler started successfully")
            logger.info(f"📅 Monthly pipeline scheduled: Day {self.monthly_schedule['day']} at {self.monthly_schedule['hour']:02d}:{self.monthly_schedule['minute']:02d} {self.monthly_schedule['timezone']}")
            logger.info("� Health checks every hour")
            
        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {str(e)}")
            self.is_running = False
            raise
    
    def stop(self):
        """Stop the scheduler"""
        try:
            if self.scheduler and self.is_running:
                self.scheduler.shutdown(wait=True)
                logger.info("📝 Scheduler stopped successfully")
            else:
                logger.warning("⚠️ Scheduler was not running")
            
            self.is_running = False
            
        except Exception as e:
            logger.error(f"❌ Error stopping scheduler: {str(e)}")
    
    async def _run_monthly_pipeline(self):
        """Execute the monthly pipeline job"""
        try:
            logger.info("📅 Monthly pipeline job started - checking for new FRED data")
            
            # Run the comprehensive pipeline with forced data check
            results = await self.pipeline.run_pipeline(force_retrain=True)
            
            # Log results
            if results['status'] == 'completed':
                logger.info("✅ Monthly pipeline completed successfully")
                if results.get('models_retrained', False):
                    updated_models = results.get('updated_models', [])
                    logger.info(f"🤖 Models retrained: {updated_models}")
                else:
                    logger.info("📊 No model updates needed - existing models performed better")
            else:
                logger.error(f"❌ Monthly pipeline failed: {results.get('errors', [])}")
            
        except Exception as e:
            logger.error(f"❌ Monthly pipeline job failed: {str(e)}", exc_info=True)
    
    async def _run_weekly_pipeline(self):
        """Execute the weekly comprehensive pipeline job"""
        try:
            logger.info("📅 Weekly comprehensive pipeline job started")
            
            # Run the pipeline with forced retraining for comprehensive check
            results = await self.pipeline.run_pipeline(force_retrain=True)
            
            # Log results
            if results['status'] == 'completed':
                logger.info("✅ Weekly comprehensive pipeline completed successfully")
                if results.get('models_retrained', False):
                    logger.info("🤖 Models were retrained during weekly run")
            else:
                logger.error(f"❌ Weekly pipeline failed: {results.get('errors', [])}")
            
        except Exception as e:
            logger.error(f"❌ Weekly pipeline job failed: {str(e)}", exc_info=True)
    
    async def _health_check(self):
        """Perform health check on pipeline components"""
        try:
            # Check database connectivity
            from services.database_service import db_service
            
            # Simple check - try to load a small amount of data
            test_data = db_service.load_historical_data('historical_data_1m')
            
            if test_data is not None:
                logger.debug("💚 Pipeline health check: Database connection OK")
            else:
                logger.warning("⚠️ Pipeline health check: Database connection issue")
            
        except Exception as e:
            logger.warning(f"⚠️ Pipeline health check failed: {str(e)}")
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get status of scheduled jobs"""
        if not self.scheduler or not self.is_running:
            return {
                'scheduler_running': False,
                'jobs': []
            }
        
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return {
            'scheduler_running': self.is_running,
            'jobs': jobs
        }
    
    async def trigger_monthly_pipeline_now(self) -> Dict[str, Any]:
        """Manually trigger the monthly pipeline"""
        try:
            logger.info("🔧 Manual trigger: Monthly pipeline")
            results = await self.pipeline.run_pipeline(force_retrain=True)
            return {
                'success': True,
                'trigger_time': datetime.now().isoformat(),
                'pipeline_results': results
            }
        except Exception as e:
            logger.error(f"❌ Manual monthly pipeline trigger failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'trigger_time': datetime.now().isoformat()
            }
    
    def update_schedule(self, monthly_config: Optional[Dict] = None):
        """
        Update the monthly schedule configuration
        
        Args:
            monthly_config: New monthly schedule config (day, hour, minute, timezone)
        """
        try:
            if not self.scheduler or not self.is_running:
                logger.error("❌ Cannot update schedule: Scheduler not running")
                return False
            
            if monthly_config:
                self.monthly_schedule.update(monthly_config)
                self.scheduler.modify_job(
                    'monthly_pipeline',
                    trigger=CronTrigger(
                        day=self.monthly_schedule['day'],
                        hour=self.monthly_schedule['hour'],
                        minute=self.monthly_schedule['minute'],
                        timezone=self.monthly_schedule['timezone']
                    )
                )
                logger.info(f"✅ Monthly schedule updated: {self.monthly_schedule}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating schedule: {str(e)}")
            return False

# Initialize the scheduler service
scheduler_service = SchedulerService()