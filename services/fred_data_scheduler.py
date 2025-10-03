import asyncio
import logging
import httpx
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.memory import MemoryJobStore

from services.database_service import db_service
from services.fred_data_service_1m import (
    SERIES_IDS as SERIES_IDS_1M,
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
        
        # Configuration
        self.series_mappings = {
            '1m': {
                'table': 'historical_data_1m',
                'series_ids': SERIES_IDS_1M,
                'service_module': 'services.fred_data_service_1m'
            },
            '3m': {
                'table': 'historical_data_3m',
                'series_ids': self._get_3m_series_ids(),
                'service_module': 'services.fred_data_service_3m'
            },
            '6m': {
                'table': 'historical_data_6m',
                'series_ids': self._get_6m_series_ids(),
                'service_module': 'services.fred_data_service_6m'
            }
        }
        
        # Define quarterly series that need forward-filling
        self.quarterly_series = {
            '1m': ['GDP', 'REALGDP', 'PSTAX', 'COMREAL'],
            '3m': ['GDP'],
            '6m': ['GDP']
        }
        
        self._running = False
    
    def _get_3m_series_ids(self) -> Dict[str, str]:
        """Get series IDs for 3-month model - placeholder for now"""
        # Import from 3m service when available
        try:
            from services.fred_data_service_3m import SERIES_IDS_3M 
            return SERIES_IDS_3M
        except ImportError:
            # Fallback to 1m series for now
            return SERIES_IDS_1M
    
    def _get_6m_series_ids(self) -> Dict[str, str]:
        """Get series IDs for 6-month model"""
        # Import from 6m service when available
        try:
            from services.fred_data_service_6m import SERIES_IDS_6M
            logger.debug(f"Loaded 6m series IDs: {len(SERIES_IDS_6M)} series")
            return SERIES_IDS_6M
        except ImportError as e:
            logger.warning(f"Could not import 6m series IDs: {e}, falling back to 1m series")
            # Fallback to 1m series for now
            return SERIES_IDS_1M
    
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
        Enhanced method to check and update data for a specific timeframe
        - Fetches 12 months of data from both database and FRED
        - Handles back revisions properly
        - Supports quarterly data forward-filling
        - Creates records with NaN for missing values instead of waiting for 80% completion
        
        Returns:
            Tuple of (records_updated, records_created)
        """
        if timeframe not in self.series_mappings:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        
        config = self.series_mappings[timeframe]
        table_name = config['table']
        series_ids = config['series_ids']
        
        logger.info(f"🔄 Enhanced update check for {timeframe} - {len(series_ids)} series")
        logger.debug(f"Series for {timeframe}: {list(series_ids.keys())}")
        
        # Get last 12 months from database for back revision checking
        db_records = await self.get_database_last_12_months(table_name)
        if not db_records:
            logger.warning(f"No historical data found in {table_name}")
            # Still proceed to try creating initial data
        
        # Fetch 12 months of FRED data for comprehensive comparison
        fred_12_months = await self.fetch_12_months_fred_data(series_ids)
        if not fred_12_months:
            logger.error(f"Failed to fetch 12 months FRED data for {timeframe}")
            return 0, 0
        
        # Analyze what needs to be updated using 12-month comparison
        update_analysis = await self.analyze_12_month_data_updates(
            fred_12_months, db_records, table_name, timeframe
        )
        
        updates_made = 0
        records_created = 0
        
        # Handle back revisions (updates to existing records)
        if update_analysis['back_revisions']:
            logger.info(f"Processing {len(update_analysis['back_revisions'])} back revisions")
            for date, updates in update_analysis['back_revisions'].items():
                updated = await self.update_existing_record_by_date(
                    updates, table_name, date
                )
                updates_made += updated
        
        # Handle new record creation with enhanced logic
        if update_analysis['new_records']:
            logger.info(f"Creating {len(update_analysis['new_records'])} new records")
            for date, data in update_analysis['new_records'].items():
                # Apply quarterly forward-filling if needed
                enhanced_data = await self.apply_quarterly_forward_filling(
                    data, fred_12_months, timeframe, date
                )
                
                # Create record even with partial data (using NaN for missing values)
                if self.should_create_record_with_minimal_data(enhanced_data, timeframe):
                    created = await self.create_record_with_nan_handling(
                        enhanced_data, table_name, date
                    )
                    records_created += created
                    logger.info(f"✅ Created record for {date} with {len([v for v in enhanced_data.values() if v is not None])} non-null values")
                else:
                    logger.warning(f"Skipping record creation for {date} - insufficient data")
        
        logger.info(f"✅ {timeframe} update complete: {updates_made} updates, {records_created} new records")
        return updates_made, records_created
    
    async def analyze_data_updates(
        self, 
        fred_data: Dict[str, Any], 
        db_latest_date: str, 
        table_name: str
    ) -> Dict[str, Any]:
        """
        Analyze FRED data to determine what updates are needed
        
        Returns:
            Dict with 'value_updates' and 'new_data' keys
        """
        analysis = {
            'value_updates': {},  # Same date, different values
            'new_data': {},       # New dates requiring new records
            'no_changes': []      # Series with no changes
        }
        
        # Get the latest database record for comparison
        latest_db_record = await self.get_latest_database_record(table_name)
        if not latest_db_record:
            # If no database record, all FRED data is new
            analysis['new_data'] = fred_data
            return analysis
        
        db_date = pd.to_datetime(db_latest_date).strftime('%Y-%m-%d')
        
        for series_name, fred_info in fred_data.items():
            if not fred_info or fred_info['value'] is None:
                continue
            
            fred_date = fred_info['date']
            fred_value = fred_info['value']
            
            # Compare dates
            if fred_date == db_date:
                # Same date - check if value is different
                db_value = latest_db_record.get(series_name)
                if db_value is not None and abs(float(fred_value) - float(db_value)) > 1e-6:
                    analysis['value_updates'][series_name] = {
                        'new_value': fred_value,
                        'old_value': db_value,
                        'date': fred_date
                    }
                    logger.info(f"Value update needed for {series_name}: {db_value} -> {fred_value}")
                else:
                    analysis['no_changes'].append(series_name)
            
            elif fred_date > db_date:
                # New date - this series has new data
                analysis['new_data'][series_name] = fred_info
                logger.info(f"New data available for {series_name}: {fred_date}")
            
            elif self._is_weekly_series(series_name) and self._same_month(fred_date, db_date):
                # For weekly series, check if it's the same month but different value
                db_value = latest_db_record.get(series_name)
                if db_value is not None and abs(float(fred_value) - float(db_value)) > 1e-6:
                    analysis['value_updates'][series_name] = {
                        'new_value': fred_value,
                        'old_value': db_value,
                        'date': db_date  # Keep the database date for weekly updates
                    }
                    logger.info(f"Weekly series update for {series_name}: {db_value} -> {fred_value}")
        
        return analysis
    
    async def analyze_12_month_data_updates(
        self, 
        fred_12_months: Dict[str, List[Dict[str, Any]]],
        db_records: List[Dict[str, Any]],
        table_name: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Enhanced analysis comparing 12 months of FRED data with database records
        
        Returns:
            Dict with 'back_revisions' and 'new_records' keys
        """
        analysis = {
            'back_revisions': {},  # Date -> {series_name: new_value}
            'new_records': {},     # Date -> {series_name: value}
        }
        
        # Create a lookup of database records by date
        db_by_date = {}
        for record in db_records:
            date_key = pd.to_datetime(record['observation_date']).strftime('%Y-%m-%d')
            db_by_date[date_key] = record
        
        # Create a comprehensive date-series matrix from FRED data
        all_dates = set()
        for series_data in fred_12_months.values():
            for obs in series_data:
                all_dates.add(obs['date'])
        
        all_dates = sorted(all_dates, reverse=True)  # Most recent first
        logger.info(f"Analyzing {len(all_dates)} dates across 12 months for {timeframe}")
        
        for date in all_dates:
            date_updates = {}
            is_new_record = date not in db_by_date
            
            for series_name, series_data in fred_12_months.items():
                # Find FRED value for this date
                fred_value = None
                for obs in series_data:
                    if obs['date'] == date:
                        fred_value = obs['value']
                        break
                
                if fred_value is None:
                    continue  # No FRED data for this series on this date
                
                if is_new_record:
                    # This is a new date not in database
                    if date not in analysis['new_records']:
                        analysis['new_records'][date] = {}
                    analysis['new_records'][date][series_name] = fred_value
                else:
                    # Check for back revisions in existing records
                    db_record = db_by_date[date]
                    db_value = db_record.get(series_name)
                    
                    if db_value is None or abs(float(db_value) - fred_value) > 0.001:
                        # Significant difference or missing value in DB
                        if date not in analysis['back_revisions']:
                            analysis['back_revisions'][date] = {}
                        analysis['back_revisions'][date][series_name] = fred_value
                        
                        if db_value is None:
                            logger.info(f"Back-fill for {series_name} on {date}: None -> {fred_value}")
                        else:
                            logger.info(f"Back revision for {series_name} on {date}: {db_value} -> {fred_value}")
        
        logger.info(f"Analysis complete: {len(analysis['back_revisions'])} dates need revisions, {len(analysis['new_records'])} new records needed")
        return analysis
    
    async def apply_quarterly_forward_filling(
        self,
        current_data: Dict[str, Any],
        fred_12_months: Dict[str, List[Dict[str, Any]]],
        timeframe: str,
        target_date: str
    ) -> Dict[str, Any]:
        """
        Apply forward-filling for quarterly series (GDP, REALGDP, PSTAX, COMREAL)
        """
        enhanced_data = current_data.copy()
        quarterly_series_list = self.quarterly_series.get(timeframe, [])
        
        for series_name in quarterly_series_list:
            if series_name in enhanced_data and enhanced_data[series_name] is not None:
                continue  # We already have current data
            
            # Find the most recent quarterly value
            if series_name in fred_12_months:
                last_quarterly_value = self.get_last_quarterly_value(
                    fred_12_months[series_name], series_name
                )
                if last_quarterly_value is not None:
                    enhanced_data[series_name] = last_quarterly_value
                    logger.info(f"Forward-filled {series_name} for {target_date} with value {last_quarterly_value}")
        
        return enhanced_data
    
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
    
    async def create_record_with_nan_handling(
        self,
        data: Dict[str, Any],
        table_name: str,
        target_date: str
    ) -> int:
        """
        Create a new database record, using NaN for missing values instead of zeros
        """
        try:
            # Get all series for this table to ensure we have all columns
            config = None
            for tf, cfg in self.series_mappings.items():
                if cfg['table'] == table_name:
                    config = cfg
                    break
            
            if not config:
                logger.error(f"No configuration found for table {table_name}")
                return 0
            
            # Build complete record with NaN for missing values
            new_record = {"observation_date": target_date}
            
            for series_name in config['series_ids'].keys():
                if series_name in data and data[series_name] is not None:
                    if series_name == "recession":
                        new_record[series_name] = int(data[series_name])
                    else:
                        new_record[series_name] = float(data[series_name])
                else:
                    # Use NaN (None in database) instead of 0
                    if series_name == "recession":
                        new_record[series_name] = None  # Will be handled as NULL in database
                    else:
                        new_record[series_name] = None  # Will be handled as NULL in database
            
            # Insert the record
            response = db_service.supabase.table(table_name)\
                .insert(new_record)\
                .execute()
            
            if response.data:
                non_null_count = len([v for v in new_record.values() if v is not None]) - 1  # -1 for date
                logger.info(f"✅ Created record in {table_name} for {target_date} with {non_null_count} non-null values")
                return 1
            else:
                logger.error(f"❌ Failed to create record in {table_name} for {target_date}")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to create record for {target_date}: {e}")
            return 0
    
    def _is_weekly_series(self, series_name: str) -> bool:
        """Check if a series is weekly frequency (like ICSA)"""
        weekly_series = ['ICSA', 'CCSA', 'CONTINUED', 'INITIAL']  # Add more as needed
        return any(weekly in series_name.upper() for weekly in weekly_series)
    
    def _same_month(self, date1: str, date2: str) -> bool:
        """Check if two dates are in the same month"""
        try:
            d1 = pd.to_datetime(date1)
            d2 = pd.to_datetime(date2)
            return d1.year == d2.year and d1.month == d2.month
        except:
            return False
    
    async def update_existing_records(
        self, 
        value_updates: Dict[str, Any], 
        table_name: str, 
        target_date: str
    ) -> int:
        """
        Update existing database records with new values
        
        Returns:
            Number of records updated
        """
        if not value_updates:
            return 0
        
        try:
            # Prepare update data
            update_data = {}
            for series_name, update_info in value_updates.items():
                update_data[series_name] = update_info['new_value']
            
            # Update the record with the target date
            response = db_service.supabase.table(table_name)\
                .update(update_data)\
                .eq('observation_date', target_date)\
                .execute()
            
            if response.data:
                logger.info(f"✅ Updated {len(value_updates)} fields in {table_name} for {target_date}")
                return len(value_updates)
            else:
                logger.error(f"❌ Failed to update records in {table_name}")
                return 0
                
        except Exception as e:
            logger.error(f"❌ Error updating records in {table_name}: {e}")
            return 0
    
    async def create_new_records(self, new_data: Dict[str, Any], table_name: str) -> int:
        """
        Create new database records for new FRED data
        
        Returns:
            Number of records created
        """
        if not new_data:
            return 0
        
        # Group data by date (in case we have multiple dates)
        data_by_date = {}
        for series_name, fred_info in new_data.items():
            date = fred_info['date']
            if date not in data_by_date:
                data_by_date[date] = {}
            data_by_date[date][series_name] = fred_info['value']
        
        records_created = 0
        
        for date, series_data in data_by_date.items():
            try:
                # Check if we have enough data to create a complete record
                # We need most of the required fields to avoid too many NaN values
                if len(series_data) < len(self.series_mappings['1m']['series_ids']) * 0.8:
                    logger.warning(f"Insufficient data for {date}, skipping record creation")
                    continue
                
                # Prepare the new record
                new_record = {'observation_date': date}
                
                # Add all available series data
                for series_name, value in series_data.items():
                    if series_name == 'recession':
                        new_record[series_name] = int(float(value)) if value is not None else 0
                    else:
                        new_record[series_name] = float(value) if value is not None else 0.0
                
                # Fill missing series with default values
                all_series = self.series_mappings['1m']['series_ids'].keys()
                for series_name in all_series:
                    if series_name not in new_record:
                        if series_name == 'recession':
                            new_record[series_name] = 0
                        else:
                            new_record[series_name] = 0.0
                
                # Insert the new record
                response = db_service.supabase.table(table_name)\
                    .insert(new_record)\
                    .execute()
                
                if response.data:
                    logger.info(f"✅ Created new record in {table_name} for {date}")
                    records_created += 1
                else:
                    logger.error(f"❌ Failed to create record in {table_name} for {date}")
                    
            except Exception as e:
                logger.error(f"❌ Error creating record for {date}: {e}")
                continue
        
        return records_created
    
    async def fetch_all_timeframe_data(self, series_ids: Dict[str, str]) -> Dict[str, Any]:
        """
        Fetch latest data for all series in a timeframe with enhanced error handling
        """
        try:
            results = {}
            failed_series = []
            network_error_count = 0
            
            # Optional network connectivity pre-check (non-blocking)
            try:
                connectivity_ok = await self._test_network_connectivity()
                if connectivity_ok:
                    logger.debug("Network connectivity pre-check passed")
                else:
                    logger.warning("Network connectivity pre-check failed - proceeding anyway")
            except Exception as e:
                logger.warning(f"Network connectivity pre-check failed with error: {e} - proceeding anyway")
            
            for name, series_id in series_ids.items():
                try:
                    # Add small delay to avoid rate limiting
                    await asyncio.sleep(0.1)
                    
                    data = await fetch_latest_observation(series_id, timeout=30)
                    
                    if data and "observations" in data and len(data["observations"]) > 0:
                        obs = data["observations"][0]
                        results[name] = {
                            "date": obs["date"],
                            "value": float(obs["value"]) if obs["value"] != "." else None
                        }
                        logger.debug(f"✅ Successfully fetched {name}: {obs['value']}")
                    else:
                        failed_series.append(name)
                        results[name] = None
                        logger.warning(f"⚠️ No data returned for {name}")
                        
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to fetch {name} ({series_id}): {error_msg}")
                    failed_series.append(name)
                    results[name] = None
                    
                    # Track network-related errors
                    if "getaddrinfo failed" in error_msg or "ConnectError" in error_msg:
                        network_error_count += 1
                        
                        # If too many network errors, stop trying
                        if network_error_count >= 3:
                            logger.error(f"❌ Multiple network errors detected ({network_error_count}). Stopping fetch attempts.")
                            break
            
            if failed_series:
                logger.warning(f"Failed to fetch {len(failed_series)} series: {failed_series}")
                
                # If most series failed due to network issues, warn but don't abort completely
                if network_error_count >= len(series_ids) * 0.5:
                    logger.error(f"Network connectivity issues prevented fetching {network_error_count} series - continuing with available data")
            
            success_count = len(results) - len(failed_series)
            logger.info(f"Successfully fetched {success_count}/{len(results)} series")
            
            # Return results even if some failed, but log the issues
            return results
            
        except ConnectionError:
            # Re-raise connection errors
            raise
        except Exception as e:
            logger.error(f"Failed to fetch timeframe data: {e}")
            return {}
    
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
    
    async def fetch_12_months_fred_data(self, series_ids: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch the last 12 months of data from FRED API for back revision checking
        
        Returns:
            Dict mapping series names to lists of observations (most recent first)
        """
        try:
            # Calculate date range for last 12 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # Approximately 12 months
            
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
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
                        'sort_order': 'desc'  # Most recent first
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
                    logger.error(f"Failed to fetch 12 months data for {series_name} ({series_id}): {e}")
                    results[series_name] = []
                    failed_series.append(series_name)
            
            if failed_series:
                logger.warning(f"Failed to fetch 12 months data for {len(failed_series)} series: {failed_series}")
            
            success_count = len([s for s in results.values() if s])
            logger.info(f"Successfully fetched 12 months data for {success_count}/{len(results)} series")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch 12 months FRED data: {e}")
            return {}
    
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