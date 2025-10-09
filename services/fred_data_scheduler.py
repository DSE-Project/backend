import asyncio
import logging
import httpx
import os
import calendar
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.memory import MemoryJobStore

from services.database_service import db_service
from services.fred_data_service_1m import (
    SERIES_IDS,
    fetch_latest_observation,
    get_fred_latest_date,
    get_database_latest_date
)

logger = logging.getLogger(__name__)

class FREDDataScheduler:
    """
    Scheduler service for automated FRED data updates
    """
    
    def __init__(self):
        # Configure scheduler
        jobstores = {
            'default': MemoryJobStore()
        }
        executors = {
            'default': AsyncIOExecutor()
        }
        job_defaults = {
            'coalesce': False,
            'max_instances': 1
        }
        
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=timezone.utc
        )
        
        # Track last update times
        self.last_update_times = {
            '1m': None,
            '3m': None,
            '6m': None
        }
        
        # Track statistics
        self.update_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'last_run': None,
            'last_error': None,
            'records_updated': 0,
            'records_created': 0
        }
        
        # Configuration - all timeframes now use the same unified dataset
        self.series_mappings = {
            '1m': {
                'table': 'historical_data_1m',
                'series_ids': SERIES_IDS,
                'service_module': 'services.fred_data_service_1m'
            },
            '3m': {
                'table': 'historical_data_3m',
                'series_ids': SERIES_IDS,
                'service_module': 'services.fred_data_service_3m'
            },
            '6m': {
                'table': 'historical_data_6m',
                'series_ids': SERIES_IDS,
                'service_module': 'services.fred_data_service_6m'
            }
        }
        
        # Define quarterly series that need forward-filling (unified across all timeframes)
        self.quarterly_series = {
            '1m': ['GDP', 'REALGDP', 'PSTAX', 'COMREAL'],
            '3m': ['GDP', 'REALGDP', 'PSTAX', 'COMREAL'],
            '6m': ['GDP', 'REALGDP', 'PSTAX', 'COMREAL']
        }
        
        self._running = False
    

    
    async def start_scheduler(self):
        """Start the scheduler with weekly FRED data checks"""
        if self._running:
            logger.info("Scheduler is already running")
            return
        
        try:
            # Add weekly job for FRED data updates
            # Run every Tuesday at 10:00 AM UTC (most FRED data is updated on Tuesdays)
            self.scheduler.add_job(
                self.check_and_update_all_data,
                CronTrigger(day_of_week='tue', hour=10, minute=0),
                id='weekly_fred_update',
                name='Weekly FRED Data Update',
                replace_existing=True
            )
            
            # Add daily job for critical series monitoring
            # Run every day at 6:00 AM UTC for high-priority series
            self.scheduler.add_job(
                self.check_critical_series,
                CronTrigger(hour=6, minute=0),
                id='daily_critical_check',
                name='Daily Critical Series Check',
                replace_existing=True
            )
            
            # Start the scheduler
            self.scheduler.start()
            self._running = True
            
            logger.info("✅ FRED Data Scheduler started successfully")
            logger.info("📅 Weekly updates: Every Tuesday at 10:00 AM UTC")
            logger.info("📅 Daily critical checks: Every day at 6:00 AM UTC")
            
        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {e}")
            raise
    
    async def stop_scheduler(self):
        """Stop the scheduler"""
        if not self._running:
            logger.info("Scheduler is not running")
            return
        
        try:
            self.scheduler.shutdown()
            self._running = False
            logger.info("✅ FRED Data Scheduler stopped successfully")
        except Exception as e:
            logger.error(f"❌ Failed to stop scheduler: {e}")
            raise
    
    async def check_and_update_all_data(self):
        """
        Main scheduled job - checks and updates data for all timeframes
        """
        start_time = datetime.now(timezone.utc)
        self.update_stats['total_runs'] += 1
        self.update_stats['last_run'] = start_time.isoformat()
        
        logger.info("🔄 Starting weekly FRED data update check")
        
        try:
            total_updated = 0
            total_created = 0
            
            # Check each timeframe
            for timeframe in ['1m', '3m', '6m']:
                try:
                    logger.info(f"📊 Checking {timeframe} data...")
                    updated, created = await self.check_and_update_timeframe_data(timeframe)
                    total_updated += updated
                    total_created += created
                    
                    self.last_update_times[timeframe] = start_time.isoformat()
                    logger.info(f"✅ {timeframe} complete: {updated} updated, {created} created")
                    
                except Exception as e:
                    logger.error(f"❌ Error updating {timeframe} data: {e}")
                    continue
            
            # Update statistics
            self.update_stats['successful_runs'] += 1
            self.update_stats['records_updated'] += total_updated
            self.update_stats['records_created'] += total_created
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            logger.info(f"✅ Weekly update completed in {duration:.1f}s")
            logger.info(f"📈 Total: {total_updated} records updated, {total_created} records created")
            
        except Exception as e:
            self.update_stats['failed_runs'] += 1
            self.update_stats['last_error'] = str(e)
            logger.error(f"❌ Weekly update failed: {e}")
            raise
    
    async def check_critical_series(self):
        """
        Daily check for critical economic indicators that might update more frequently
        """
        logger.info("🔍 Performing daily critical series check")
        
        # Define critical series that might update more frequently
        critical_series = {
            'fedfunds': 'FEDFUNDS',
            'UNRATE': 'UNRATE',
            'recession': 'USREC'
        }
        
        try:
            for name, series_id in critical_series.items():
                try:
                    # Check if this series has newer data
                    has_update = await self.check_series_for_update(series_id, name, '1m')
                    if has_update:
                        logger.info(f"🚨 Critical update detected for {name}")
                        # Trigger full update for this timeframe
                        await self.check_and_update_timeframe_data('1m')
                        break
                except Exception as e:
                    logger.error(f"Error checking critical series {name}: {e}")
                    continue
            
            logger.info("✅ Daily critical series check completed")
            
        except Exception as e:
            logger.error(f"❌ Daily critical series check failed: {e}")
    
    async def check_and_update_timeframe_data(self, timeframe: str) -> Tuple[int, int]:
        """
        Correct method to check and update data for a specific timeframe
        - Database records always use YYYY-MM-01 format (first day of month)
        - Fetches last 12 months of data from database (last 12 records)
        - Fetches 12 months of FRED data for comparison
        - Updates existing records for back revisions
        - Creates maximum 1 new record for the most recent month
        
        Returns:
            Tuple of (records_updated, records_created)
        """
        if timeframe not in self.series_mappings:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        
        config = self.series_mappings[timeframe]
        table_name = config['table']
        series_ids = config['series_ids']
        
        logger.info(f"🔄 Correct update check for {timeframe} - {len(series_ids)} series")
        
        # Get last 12 months from database (last 12 records with YYYY-MM-01 format)
        db_records = await self.get_database_last_12_months(table_name)
        
        # Fetch 12 months of FRED data using proper date range
        fred_data = await self.fetch_12_months_fred_data_correct(series_ids)
        if not fred_data:
            logger.error(f"Failed to fetch FRED data for {timeframe}")
            return 0, 0
        
        # Process FRED data into monthly format (YYYY-MM-01)
        processed_fred_data = await self.process_fred_data_to_monthly(fred_data, timeframe)
        
        # Compare database records with processed FRED data
        updates_made = 0
        records_created = 0
        
        # Create lookup of existing database records by month
        db_by_month = {}
        if db_records:
            for record in db_records:
                month_key = pd.to_datetime(record['observation_date']).strftime('%Y-%m-01')
                db_by_month[month_key] = record
        
        # Get all months that should exist (last 12 months ending with previous month)
        current_date = datetime.now()
        
        # Start from previous month (economic data is always a month behind)
        if current_date.month == 1:
            start_month_date = datetime(current_date.year - 1, 12, 1)
        else:
            start_month_date = datetime(current_date.year, current_date.month - 1, 1)
        
        months_to_check = []
        for i in range(12):
            month_date = start_month_date - timedelta(days=32*i)
            month_date = month_date.replace(day=1)  # First day of month
            months_to_check.append(month_date.strftime('%Y-%m-01'))
        
        months_to_check.reverse()  # Oldest to newest
        
        # Process each month
        for month_key in months_to_check:
            if month_key in processed_fred_data:
                fred_month_data = processed_fred_data[month_key]
                
                if month_key in db_by_month:
                    # Update existing record if values changed
                    db_record = db_by_month[month_key]
                    updates = {}
                    
                    for series_name, fred_value in fred_month_data.items():
                        if series_name == 'observation_date':
                            continue
                        
                        db_value = db_record.get(series_name)
                        if db_value is None or abs(float(db_value) - float(fred_value)) > 0.001:
                            updates[series_name] = fred_value
                    
                    if updates:
                        updated = await self.update_existing_record_by_date(
                            updates, table_name, month_key
                        )
                        updates_made += updated
                        logger.info(f"✅ Updated {len(updates)} fields for {month_key}")
                
                else:
                    # Create new record (should be maximum 1 - the most recent month)
                    if month_key == max(processed_fred_data.keys()):  # Only create the most recent
                        # Check if we have sufficient data to create a meaningful record
                        non_null_count = len([v for k, v in fred_month_data.items() if k != 'observation_date' and v is not None])
                        total_series = len(config['series_ids'])
                        data_percentage = (non_null_count / total_series) * 100
                        
                        logger.info(f"📊 {timeframe} - Creating record for {month_key}: {non_null_count}/{total_series} series ({data_percentage:.1f}% complete)")
                        
                        # Create record if we have at least 25% of the expected data
                        if non_null_count >= max(1, total_series * 0.25):
                            created = await self.create_monthly_record(
                                fred_month_data, table_name, month_key
                            )
                            records_created += created
                            if created > 0:
                                logger.info(f"✅ {timeframe} - Successfully created new record for {month_key}")
                            else:
                                logger.warning(f"⚠️ {timeframe} - Failed to create record for {month_key}")
                        else:
                            logger.warning(f"⚠️ {timeframe} - Insufficient data for {month_key} ({data_percentage:.1f}% complete), skipping record creation")
        
        logger.info(f"✅ {timeframe} update complete: {updates_made} updates, {records_created} new records")
        return updates_made, records_created
    
    # analyze_data_updates function removed - no longer used by corrected implementation
    

    

    
    async def update_existing_record_by_date(
        self, 
        updates: Dict[str, Any], 
        table_name: str,
        target_date: str
    ) -> int:
        """
        Update an existing database record for a specific date
        """
        try:
            if not updates:
                return 0
            
            # Build update payload
            update_data = {}
            for series_name, value in updates.items():
                if series_name == "recession":
                    update_data[series_name] = int(value)
                else:
                    update_data[series_name] = float(value)
            
            # Update the record
            response = db_service.supabase.table(table_name)\
                .update(update_data)\
                .eq('observation_date', target_date)\
                .execute()
            
            if response.data:
                logger.info(f"✅ Updated {len(updates)} fields in {table_name} for {target_date}")
                return 1
            else:
                logger.error(f"❌ Failed to update record in {table_name} for {target_date}")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to update record for {target_date}: {e}")
            return 0
    

    
    def _is_weekly_series(self, series_name: str) -> bool:
        """Check if a series is weekly frequency - unified dataset uses only monthly data"""
        # The unified dataset no longer includes weekly series like ICSA
        return False
    
    def _same_month(self, date1: str, date2: str) -> bool:
        """Check if two dates are in the same month"""
        try:
            d1 = pd.to_datetime(date1)
            d2 = pd.to_datetime(date2)
            return d1.year == d2.year and d1.month == d2.month
        except:
            return False
    
    # update_existing_records function removed - replaced by update_existing_record_by_date
    
    # create_new_records function removed - replaced by create_monthly_record
    
    # fetch_all_timeframe_data function removed - replaced by fetch_12_months_fred_data_correct
    
    async def _test_network_connectivity(self) -> bool:
        """Test basic network connectivity to FRED API"""
        try:
            import socket
            import os
            
            # Quick DNS resolution test
            socket.gethostbyname("api.stlouisfed.org")
            
            # Test with an actual FRED API call instead of just the homepage
            fred_api_key = os.getenv('FRED_API_KEY')
            if not fred_api_key:
                logger.warning("No FRED API key available for connectivity test")
                return False
            
            # Test with a simple DNS and basic HTTP check first
            # If that fails, try a real API endpoint
            try:
                # Simple HTTP check to see if we can reach the domain
                timeout_config = httpx.Timeout(3.0)
                async with httpx.AsyncClient(timeout=timeout_config, follow_redirects=True) as client:
                    response = await client.get("https://api.stlouisfed.org/")
                    # Any response means we can connect
                    if response.status_code < 500:
                        return True
            except:
                pass
            
            # If simple check fails, try actual API endpoint
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "FEDFUNDS",
                "api_key": fred_api_key,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc"
            }
            
            timeout_config = httpx.Timeout(5.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.get(url, params=params)
                
                # Accept 200 OK or any successful response
                if response.status_code == 200:
                    data = response.json()
                    return "observations" in data
                elif response.status_code == 400:
                    # Bad request might mean API key issue, but network is working
                    logger.debug("FRED API returned 400 - possible API key issue, but network is working")
                    return True
                elif response.status_code < 500:
                    # Client errors are still considered "connected"
                    return True
                else:
                    # Server errors indicate actual connectivity issues
                    logger.warning(f"FRED API returned server error: {response.status_code}")
                    return False
                
        except Exception as e:
            logger.error(f"Network connectivity test failed: {e}")
            return False
    
    async def fetch_12_months_fred_data_correct(self, series_ids: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Correctly fetch the last 12 months of data from FRED API
        Uses proper date range with observation_start and observation_end
        
        Returns:
            Dict mapping series names to lists of observations
        """
        try:
            # Calculate proper date range - 12 months back from current month
            current_date = datetime.now()
            
            # End date is the last day of previous month (FRED data is always a month behind)
            if current_date.month == 1:
                end_month = 12
                end_year = current_date.year - 1
            else:
                end_month = current_date.month - 1
                end_year = current_date.year
            
            # Start date is 12 months before the end month
            start_year = end_year
            start_month = end_month - 11
            if start_month <= 0:
                start_month += 12
                start_year -= 1
            
            start_date_str = f"{start_year}-{start_month:02d}-01"
            end_date_str = f"{end_year}-{end_month:02d}-{calendar.monthrange(end_year, end_month)[1]}"
            
            logger.info(f"Fetching 12 months of FRED data from {start_date_str} to {end_date_str}")
            
            results = {}
            failed_series = []
            
            for series_name, series_id in series_ids.items():
                try:
                    await asyncio.sleep(0.1)  # Rate limiting
                    
                    params = {
                        'series_id': series_id,
                        'api_key': os.getenv('FRED_API_KEY'),
                        'file_type': 'json',
                        'observation_start': start_date_str,
                        'observation_end': end_date_str,
                        'sort_order': 'desc',  # Most recent first
                        'limit': 120  # Enough for weekly data aggregation
                    }
                    
                    timeout_config = httpx.Timeout(30.0)
                    async with httpx.AsyncClient(timeout=timeout_config) as client:
                        response = await client.get(
                            "https://api.stlouisfed.org/fred/series/observations",
                            params=params
                        )
                        response.raise_for_status()
                        data = response.json()
                        
                        if data and "observations" in data:
                            # Filter out missing values (".")
                            valid_observations = []
                            for obs in data["observations"]:
                                if obs["value"] != ".":
                                    try:
                                        valid_observations.append({
                                            "date": obs["date"],
                                            "value": float(obs["value"])
                                        })
                                    except (ValueError, TypeError):
                                        continue
                            
                            results[series_name] = valid_observations
                            logger.debug(f"Fetched {len(valid_observations)} observations for {series_name}")
                        else:
                            results[series_name] = []
                            logger.warning(f"No observations returned for {series_name}")
                            
                except Exception as e:
                    logger.error(f"Failed to fetch data for {series_name} ({series_id}): {e}")
                    results[series_name] = []
                    failed_series.append(series_name)
            
            if failed_series:
                logger.warning(f"Failed to fetch data for {len(failed_series)} series: {failed_series}")
            
            success_count = len([s for s in results.values() if s])
            logger.info(f"Successfully fetched data for {success_count}/{len(results)} series")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch 12 months FRED data: {e}")
            return {}
    
    async def process_fred_data_to_monthly(
        self, 
        fred_data: Dict[str, List[Dict[str, Any]]], 
        timeframe: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process FRED data into monthly format with YYYY-MM-01 dates
        Handle quarterly forward-filling and weekly averaging
        
        Returns:
            Dict mapping month strings (YYYY-MM-01) to series data
        """
        monthly_data = {}
        quarterly_series_list = self.quarterly_series.get(timeframe, [])
        
        # Get the last 12 months we need to process (ending with previous month)
        current_date = datetime.now()
        
        # Start from previous month (economic data is always a month behind)
        if current_date.month == 1:
            start_month_date = datetime(current_date.year - 1, 12, 1)
        else:
            start_month_date = datetime(current_date.year, current_date.month - 1, 1)
        
        months_needed = []
        for i in range(12):
            month_date = start_month_date - timedelta(days=32*i)
            month_date = month_date.replace(day=1)
            months_needed.append(month_date.strftime('%Y-%m-01'))
        
        months_needed.reverse()  # Oldest to newest
        
        for month_key in months_needed:
            month_data = {'observation_date': month_key}
            month_dt = pd.to_datetime(month_key)
            
            for series_name, observations in fred_data.items():
                if not observations:
                    continue
                
                if series_name in quarterly_series_list:
                    # Handle quarterly data with forward-filling
                    quarterly_value = self.get_quarterly_value_for_month(observations, month_dt)
                    if quarterly_value is not None:
                        month_data[series_name] = quarterly_value
                
                else:
                    # Handle monthly data - get the value for this specific month
                    monthly_value = self.get_monthly_value(observations, month_dt)
                    if monthly_value is not None:
                        month_data[series_name] = monthly_value
            
            # Only add month if we have at least some data
            if len(month_data) > 1:  # More than just observation_date
                monthly_data[month_key] = month_data
        
        logger.info(f"Processed FRED data into {len(monthly_data)} monthly records")
        return monthly_data
    
    def get_quarterly_value_for_month(self, observations: List[Dict[str, Any]], month_dt: datetime) -> Optional[float]:
        """Get quarterly value for a specific month with forward-filling logic"""
        quarter_starts = {
            1: [1, 2, 3],    # Q1: Jan, Feb, Mar
            2: [4, 5, 6],    # Q2: Apr, May, Jun  
            3: [7, 8, 9],    # Q3: Jul, Aug, Sep
            4: [10, 11, 12]  # Q4: Oct, Nov, Dec
        }
        
        # Determine which quarter this month belongs to
        month_num = month_dt.month
        target_quarter = None
        for quarter, months in quarter_starts.items():
            if month_num in months:
                target_quarter = quarter
                break
        
        # Look for data in the target quarter first
        for obs in observations:
            obs_dt = pd.to_datetime(obs['date'])
            obs_quarter = ((obs_dt.month - 1) // 3) + 1
            
            if obs_dt.year == month_dt.year and obs_quarter == target_quarter:
                logger.debug(f"Found quarterly data for {month_dt.strftime('%Y-%m')}: Q{target_quarter} = {obs['value']}")
                return obs['value']
        
        # If no data for target quarter, forward-fill from the most recent available quarter
        for obs in observations:
            obs_dt = pd.to_datetime(obs['date'])
            if obs_dt <= month_dt:  # Only use past data
                logger.debug(f"Forward-filling quarterly data for {month_dt.strftime('%Y-%m')} with {obs['value']} from {obs['date']}")
                return obs['value']
        
        return None
    

    
    def get_monthly_value(self, observations: List[Dict[str, Any]], month_dt: datetime) -> Optional[float]:
        """Get the monthly value for a specific month"""
        for obs in observations:
            obs_dt = pd.to_datetime(obs['date'])
            if obs_dt.year == month_dt.year and obs_dt.month == month_dt.month:
                return obs['value']
        
        return None
    
    async def create_monthly_record(
        self, 
        month_data: Dict[str, Any], 
        table_name: str, 
        month_key: str
    ) -> int:
        """
        Create a new monthly record with proper YYYY-MM-01 format
        """
        try:
            # Get all expected series for this table 
            config = None
            for tf, cfg in self.series_mappings.items():
                if cfg['table'] == table_name:
                    config = cfg
                    break
            
            if not config:
                logger.error(f"No configuration found for table {table_name}")
                return 0
            
            # Build complete record
            new_record = {"observation_date": month_key}
            
            for series_name in config['series_ids'].keys():
                if series_name in month_data:
                    if series_name == "recession":
                        new_record[series_name] = int(month_data[series_name])
                    else:
                        new_record[series_name] = float(month_data[series_name])
                else:
                    # Use NULL for missing values
                    new_record[series_name] = None
            
            # Insert the record
            response = db_service.supabase.table(table_name)\
                .insert(new_record)\
                .execute()
            
            if response.data:
                non_null_count = len([v for v in new_record.values() if v is not None]) - 1  # -1 for date
                total_series = len(config['series_ids'])
                logger.info(f"✅ Created monthly record for {month_key} with {non_null_count}/{total_series} non-null values")
                return 1
            else:
                logger.error(f"❌ Failed to create monthly record for {month_key}")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to create monthly record for {month_key}: {e}")
            return 0
    
    def is_quarterly_series(self, series_name: str, timeframe: str) -> bool:
        """Check if a series is quarterly and needs forward-filling"""
        return series_name in self.quarterly_series.get(timeframe, [])
    
    def get_current_quarter_months(self, current_date: datetime) -> List[str]:
        """Get the months for the current quarter"""
        month = current_date.month
        year = current_date.year
        
        if month <= 3:  # Q1: Jan, Feb, Mar
            quarter_months = [f"{year}-01-01", f"{year}-02-01", f"{year}-03-01"]
        elif month <= 6:  # Q2: Apr, May, Jun
            quarter_months = [f"{year}-04-01", f"{year}-05-01", f"{year}-06-01"]
        elif month <= 9:  # Q3: Jul, Aug, Sep
            quarter_months = [f"{year}-07-01", f"{year}-08-01", f"{year}-09-01"]
        else:  # Q4: Oct, Nov, Dec
            quarter_months = [f"{year}-10-01", f"{year}-11-01", f"{year}-12-01"]
        
        return quarter_months
    
    def get_last_quarterly_value(self, series_data: List[Dict[str, Any]], series_name: str) -> Optional[float]:
        """Get the most recent quarterly value for forward-filling"""
        if not series_data:
            return None
        
        # Quarterly data is released in specific months
        quarterly_months = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
        
        for obs in series_data:  # Already sorted by date desc
            obs_date = pd.to_datetime(obs["date"])
            if obs_date.month in quarterly_months:
                logger.info(f"Found quarterly value for {series_name}: {obs['value']} from {obs['date']}")
                return obs["value"]
        
        logger.warning(f"No quarterly value found for {series_name} in recent data")
        return None
    
    def should_create_record_with_minimal_data(self, current_data: Dict[str, Any], timeframe: str) -> bool:
        """
        Check if we should create a new record with minimal data (using NaN for missing values)
        Returns True if we have at least one non-quarterly feature value OR if it's a month 
        where we should have data based on last known data dates
        """
        quarterly_series_list = self.quarterly_series.get(timeframe, [])
        non_quarterly_count = 0
        
        for series_name, value in current_data.items():
            if series_name not in quarterly_series_list and value is not None:
                non_quarterly_count += 1
        
        # Create record if we have at least one non-quarterly feature
        return non_quarterly_count > 0
    
    def should_create_record_for_previous_month(self, timeframe: str) -> bool:
        """
        Check if we should create a record for the previous month based on FRED data availability.
        This checks if the month before the current scheduler run matches the month of the latest
        feature value, indicating FRED has released past month's data.
        """
        current_date = datetime.now()
        
        # Get the previous month's date
        if current_date.month == 1:
            prev_month_date = current_date.replace(year=current_date.year - 1, month=12, day=1)
        else:
            prev_month_date = current_date.replace(month=current_date.month - 1, day=1)
        
        prev_month_str = prev_month_date.strftime('%Y-%m-%d')
        
        logger.info(f"Checking if we should create record for previous month: {prev_month_str}")
        return True  # For now, always try to create previous month record if we have any data
    
    async def check_series_for_update(self, series_id: str, series_name: str, timeframe: str) -> bool:
        """
        Check if a specific series has updates available
        
        Returns:
            True if updates are available
        """
        try:
            # Fetch latest from FRED
            fred_data = await fetch_latest_observation(series_id)
            if not fred_data or "observations" not in fred_data:
                return False
            
            fred_obs = fred_data["observations"][0]
            fred_date = fred_obs["date"]
            
            # Get latest database date
            table_name = self.series_mappings[timeframe]['table']
            db_date = await self.get_database_latest_date(table_name)
            
            if not db_date:
                return True  # No database data means update needed
            
            # Compare dates
            return fred_date > db_date
            
        except Exception as e:
            logger.error(f"Error checking series {series_name}: {e}")
            return False
    
    async def get_database_latest_date(self, table_name: str) -> Optional[str]:
        """Get the latest observation date from a specific database table"""
        try:
            response = db_service.supabase.table(table_name)\
                .select("observation_date")\
                .order('observation_date', desc=True)\
                .limit(1)\
                .execute()
            
            if response.data and len(response.data) > 0:
                latest_date = response.data[0]["observation_date"]
                return pd.to_datetime(latest_date).strftime('%Y-%m-%d')
            return None
        except Exception as e:
            logger.error(f"Failed to fetch latest date from {table_name}: {e}")
            return None
    
    async def get_database_last_12_months(self, table_name: str) -> List[Dict[str, Any]]:
        """Get the last 12 months of data from a specific database table for back revision checking"""
        try:
            response = db_service.supabase.table(table_name)\
                .select("*")\
                .order('observation_date', desc=True)\
                .limit(12)\
                .execute()
            
            if response.data:
                logger.info(f"Retrieved {len(response.data)} records from {table_name} for back revision check")
                return response.data
            return []
        except Exception as e:
            logger.error(f"Failed to get last 12 months from {table_name}: {e}")
            return []
    
    async def get_latest_database_record(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get the complete latest record from a database table"""
        try:
            response = db_service.supabase.table(table_name)\
                .select("*")\
                .order('observation_date', desc=True)\
                .limit(1)\
                .execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to fetch latest record from {table_name}: {e}")
            return None
    
    # Manual trigger methods
    async def trigger_immediate_update(self, timeframe: Optional[str] = None):
        """
        Manually trigger an immediate data update
        
        Args:
            timeframe: Specific timeframe to update ('1m', '3m', '6m') or None for all
        """
        logger.info(f"🚀 Triggering immediate update for {timeframe or 'all timeframes'}")
        
        if timeframe:
            if timeframe not in self.series_mappings:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            updated, created = await self.check_and_update_timeframe_data(timeframe)
            logger.info(f"✅ Manual update complete: {updated} updated, {created} created")
            return {'updated': updated, 'created': created}
        else:
            # Update all timeframes
            await self.check_and_update_all_data()
            return self.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            'scheduler_running': self._running,
            'last_update_times': self.last_update_times,
            'statistics': self.update_stats,
            'next_scheduled_run': self.get_next_run_time()
        }
    
    def get_next_run_time(self) -> Optional[str]:
        """Get the next scheduled run time"""
        if not self._running:
            return None
        
        try:
            jobs = self.scheduler.get_jobs()
            if jobs:
                next_run = min(job.next_run_time for job in jobs if job.next_run_time)
                return next_run.isoformat()
        except:
            pass
        return None
    
    def get_job_status(self) -> List[Dict[str, Any]]:
        """Get status of all scheduled jobs"""
        if not self._running:
            return []
        
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        return jobs

# Global scheduler instance
fred_scheduler = FREDDataScheduler()