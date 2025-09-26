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
    Handles daily and weekly data pipeline runs
    """
    
    def __init__(self):
        """Initialize the scheduler service"""
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.pipeline = pipeline_service
        self.is_running = False
        
        # Configuration (can be moved to environment variables)
        self.daily_schedule = {
            'hour': 6,      # 6 AM EST (after FRED typically updates)
            'minute': 30,   # 30 minutes past the hour
            'timezone': 'US/Eastern'
        }
        
        self.weekly_schedule = {
            'day_of_week': 0,        # Monday (0=Monday in APScheduler)
            'hour': 7,               # 7 AM EST
            'minute': 0,             # Top of the hour
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
            
            # Add daily pipeline job
            self.scheduler.add_job(
                func=self._run_daily_pipeline,
                trigger=CronTrigger(
                    hour=self.daily_schedule['hour'],
                    minute=self.daily_schedule['minute'],
                    timezone=self.daily_schedule['timezone']
                ),
                id='daily_pipeline',
                name='Daily Data Pipeline',
                replace_existing=True,
                max_instances=1  # Prevent overlapping runs
            )
            
            # Add weekly comprehensive pipeline job
            self.scheduler.add_job(
                func=self._run_weekly_pipeline,
                trigger=CronTrigger(
                    day_of_week=self.weekly_schedule['day_of_week'],
                    hour=self.weekly_schedule['hour'],
                    minute=self.weekly_schedule['minute'],
                    timezone=self.weekly_schedule['timezone']
                ),
                id='weekly_pipeline',
                name='Weekly Comprehensive Pipeline',
                replace_existing=True,
                max_instances=1
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
            logger.info(f"📅 Daily pipeline scheduled: {self.daily_schedule['hour']:02d}:{self.daily_schedule['minute']:02d} {self.daily_schedule['timezone']}")
            logger.info(f"📅 Weekly pipeline scheduled: {self.weekly_schedule['day_of_week']} {self.weekly_schedule['hour']:02d}:{self.weekly_schedule['minute']:02d} {self.weekly_schedule['timezone']}")
            
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
    
    async def _run_daily_pipeline(self):
        """Execute the daily pipeline job"""
        try:
            logger.info("🌅 Daily pipeline job started")
            
            # Run the pipeline with normal parameters
            results = await self.pipeline.run_pipeline(force_retrain=False)
            
            # Log results
            if results['status'] == 'completed':
                logger.info(f"✅ Daily pipeline completed successfully")
                if results.get('models_retrained', False):
                    logger.info("🤖 Models were retrained during daily run")
            else:
                logger.error(f"❌ Daily pipeline failed: {results.get('errors', [])}")
            
        except Exception as e:
            logger.error(f"❌ Daily pipeline job failed: {str(e)}", exc_info=True)
    
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
    
    async def trigger_daily_pipeline_now(self) -> Dict[str, Any]:
        """Manually trigger the daily pipeline"""
        try:
            logger.info("🔧 Manual trigger: Daily pipeline")
            results = await self.pipeline.run_pipeline(force_retrain=False)
            return {
                'success': True,
                'trigger_time': datetime.now().isoformat(),
                'pipeline_results': results
            }
        except Exception as e:
            logger.error(f"❌ Manual daily pipeline trigger failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'trigger_time': datetime.now().isoformat()
            }
    
    async def trigger_weekly_pipeline_now(self) -> Dict[str, Any]:
        """Manually trigger the weekly comprehensive pipeline"""
        try:
            logger.info("🔧 Manual trigger: Weekly comprehensive pipeline")
            results = await self.pipeline.run_pipeline(force_retrain=True)
            return {
                'success': True,
                'trigger_time': datetime.now().isoformat(),
                'pipeline_results': results
            }
        except Exception as e:
            logger.error(f"❌ Manual weekly pipeline trigger failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'trigger_time': datetime.now().isoformat()
            }
    
    def update_schedule(self, daily_config: Optional[Dict] = None, weekly_config: Optional[Dict] = None):
        """
        Update the schedule configuration
        
        Args:
            daily_config: New daily schedule config (hour, minute, timezone)
            weekly_config: New weekly schedule config (day_of_week, hour, minute, timezone)
        """
        try:
            if not self.scheduler or not self.is_running:
                logger.error("❌ Cannot update schedule: Scheduler not running")
                return False
            
            if daily_config:
                self.daily_schedule.update(daily_config)
                self.scheduler.modify_job(
                    'daily_pipeline',
                    trigger=CronTrigger(
                        hour=self.daily_schedule['hour'],
                        minute=self.daily_schedule['minute'],
                        timezone=self.daily_schedule['timezone']
                    )
                )
                logger.info(f"✅ Daily schedule updated: {self.daily_schedule}")
            
            if weekly_config:
                # Handle day_of_week conversion if it's a string
                if 'day_of_week' in weekly_config and isinstance(weekly_config['day_of_week'], str):
                    day_mapping = {
                        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                        'friday': 4, 'saturday': 5, 'sunday': 6
                    }
                    day_name = weekly_config['day_of_week'].lower()
                    if day_name in day_mapping:
                        weekly_config['day_of_week'] = day_mapping[day_name]
                    
                self.weekly_schedule.update(weekly_config)
                self.scheduler.modify_job(
                    'weekly_pipeline',
                    trigger=CronTrigger(
                        day_of_week=self.weekly_schedule['day_of_week'],
                        hour=self.weekly_schedule['hour'],
                        minute=self.weekly_schedule['minute'],
                        timezone=self.weekly_schedule['timezone']
                    )
                )
                logger.info(f"✅ Weekly schedule updated: {self.weekly_schedule}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating schedule: {str(e)}")
            return False

# Initialize the scheduler service
scheduler_service = SchedulerService()