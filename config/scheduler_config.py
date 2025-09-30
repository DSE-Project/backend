import os
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import time

@dataclass
class SchedulerConfig:
    """Configuration class for the FRED Data Scheduler"""
    
    # Scheduler timing settings
    WEEKLY_UPDATE_DAY: str = "tue"  # Tuesday (when most FRED data is released)
    WEEKLY_UPDATE_TIME: time = time(10, 0)  # 10:00 AM UTC
    DAILY_CHECK_TIME: time = time(6, 0)     # 6:00 AM UTC for critical series
    
    # Request settings
    REQUEST_TIMEOUT: int = 30  # seconds
    MAX_RETRIES: int = 3
    REQUEST_DELAY: float = 0.1  # seconds between requests to avoid rate limiting
    NETWORK_TEST_TIMEOUT: int = 5  # seconds for connectivity tests
    MAX_NETWORK_ERRORS: int = 3  # max network errors before stopping
    
    # Data validation settings
    MIN_SERIES_COMPLETION_RATE: float = 0.8  # 80% of series must be available for new record
    VALUE_COMPARISON_THRESHOLD: float = 1e-6  # Minimum difference to consider values different
    
    # Error handling
    MAX_FAILED_ATTEMPTS: int = 5  # Max consecutive failures before alerting
    HEALTH_CHECK_DAYS_THRESHOLD: int = 8  # Days without update before considering unhealthy
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Feature flags
    ENABLE_WEEKLY_UPDATES: bool = True
    ENABLE_DAILY_CRITICAL_CHECKS: bool = True
    ENABLE_VALUE_UPDATES: bool = True  # Allow updating existing values
    ENABLE_NEW_RECORD_CREATION: bool = True  # Allow creating new records
    ENABLE_NETWORK_TESTS: bool = True  # Enable pre-flight network connectivity tests
    FAIL_FAST_ON_NETWORK_ERRORS: bool = True  # Stop on multiple network errors

# Critical series that are checked daily for updates
CRITICAL_SERIES = {
    'fedfunds': 'FEDFUNDS',     # Federal Funds Rate
    'UNRATE': 'UNRATE',         # Unemployment Rate
    'USREC': 'USREC',           # US Recession Indicator
    'TB3MS': 'TB3MS',           # 3-Month Treasury
    'TB10YS': 'GS10'            # 10-Year Treasury
}

# Series that are known to be weekly frequency
WEEKLY_SERIES = [
    'ICSA',      # Initial Claims
    'CCSA',      # Continued Claims
    'CONTINUED', # Continued Claims (alternative)
    'INITIAL'    # Initial Claims (alternative)
]

# Series that typically update with a delay
DELAYED_SERIES = {
    'GDP': 30,        # GDP updates ~30 days after quarter end
    'REALGDP': 30,    # Real GDP updates ~30 days after quarter end
    'PCEPI': 15,      # PCE Price Index updates ~15 days after month end
    'PPIACO': 10      # Producer Price Index updates ~10 days after month end
}

# Database table mappings
TABLE_MAPPINGS = {
    '1m': 'historical_data_1m',
    '3m': 'historical_data_3m',
    '6m': 'historical_data_6m'
}

# Environment variable mappings
def get_config_from_env() -> Dict[str, Any]:
    """Get configuration values from environment variables"""
    return {
        'fred_api_key': os.getenv('FRED_API_KEY'),
        'supabase_url': os.getenv('SUPABASE_URL'),
        'supabase_key': os.getenv('SUPABASE_ANON_KEY'),
        'scheduler_enabled': os.getenv('SCHEDULER_ENABLED', 'true').lower() == 'true',
        'scheduler_log_level': os.getenv('SCHEDULER_LOG_LEVEL', 'INFO'),
        'scheduler_timezone': os.getenv('SCHEDULER_TIMEZONE', 'UTC')
    }

# Default scheduler configuration
DEFAULT_CONFIG = SchedulerConfig()

def validate_config() -> List[str]:
    """
    Validate configuration and return list of issues
    
    Returns:
        List of configuration issues (empty if all good)
    """
    issues = []
    
    # Check required environment variables
    env_config = get_config_from_env()
    
    if not env_config['fred_api_key']:
        issues.append("FRED_API_KEY environment variable is not set - Get one free at https://fred.stlouisfed.org/docs/api/api_key.html")
    elif len(env_config['fred_api_key']) < 20:
        issues.append("FRED_API_KEY appears to be too short - should be ~32 characters")
    
    if not env_config['supabase_url']:
        issues.append("SUPABASE_URL environment variable is not set")
    
    if not env_config['supabase_key']:
        issues.append("SUPABASE_ANON_KEY environment variable is not set")
    
    # Check scheduler settings
    if DEFAULT_CONFIG.REQUEST_TIMEOUT < 5:
        issues.append("REQUEST_TIMEOUT should be at least 5 seconds")
    
    if DEFAULT_CONFIG.MAX_RETRIES < 1:
        issues.append("MAX_RETRIES should be at least 1")
    
    if DEFAULT_CONFIG.MIN_SERIES_COMPLETION_RATE < 0.5 or DEFAULT_CONFIG.MIN_SERIES_COMPLETION_RATE > 1.0:
        issues.append("MIN_SERIES_COMPLETION_RATE should be between 0.5 and 1.0")
    
    return issues

def get_scheduler_summary() -> Dict[str, Any]:
    """Get a summary of the current scheduler configuration"""
    config = DEFAULT_CONFIG
    env_config = get_config_from_env()
    
    return {
        "timing": {
            "weekly_updates": f"Every {config.WEEKLY_UPDATE_DAY.title()} at {config.WEEKLY_UPDATE_TIME}",
            "daily_checks": f"Every day at {config.DAILY_CHECK_TIME}",
            "timezone": env_config['scheduler_timezone']
        },
        "settings": {
            "request_timeout": f"{config.REQUEST_TIMEOUT}s",
            "max_retries": config.MAX_RETRIES,
            "request_delay": f"{config.REQUEST_DELAY}s",
            "min_completion_rate": f"{config.MIN_SERIES_COMPLETION_RATE:.1%}"
        },
        "features": {
            "weekly_updates_enabled": config.ENABLE_WEEKLY_UPDATES,
            "daily_checks_enabled": config.ENABLE_DAILY_CRITICAL_CHECKS,
            "value_updates_enabled": config.ENABLE_VALUE_UPDATES,
            "new_records_enabled": config.ENABLE_NEW_RECORD_CREATION
        },
        "monitoring": {
            "critical_series_count": len(CRITICAL_SERIES),
            "weekly_series_count": len(WEEKLY_SERIES),
            "delayed_series_count": len(DELAYED_SERIES),
            "health_check_threshold": f"{config.HEALTH_CHECK_DAYS_THRESHOLD} days"
        }
    }