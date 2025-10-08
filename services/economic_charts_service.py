import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from fredapi import Fred
import os
from dotenv import load_dotenv
from utils.fred_data_cache import fred_data_cache, cache_fred_data

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class EconomicChartsService:
    """Service class for handling economic indicators historical data and charts"""
    
    def __init__(self):
        # Initialize FRED API - you'll need to set FRED_API_KEY environment variable
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            logger.error("FRED_API_KEY environment variable not set. FRED data is required.")
            raise ValueError("FRED_API_KEY environment variable is required for fetching economic data")
        else:
            self.fred = Fred(api_key=self.fred_api_key)
            logger.info("FRED API initialized successfully")
        
        # FRED series IDs for economic indicators
        self.fred_series = {
            'gdp': 'GDP',  # Gross Domestic Product
            'cpi': 'CPIAUCSL',  # Consumer Price Index for All Urban Consumers
            'unemployment_rate': 'UNRATE',  # Unemployment Rate
            'inflation': 'CPILFESL',  # Core CPI (for inflation calculation)
            'ppi': 'PPIACO',  # Producer Price Index
            'pce': 'PCE'  # Personal Consumption Expenditures
        }
        
        self.indicators_config = {
            'gdp': {
                'name': 'GDP',
                'full_name': 'Gross Domestic Product',
                'unit': 'Billions USD',
                'color': '#1f77b4',
                'series_id': 'GDP'
            },
            'cpi': {
                'name': 'CPI',
                'full_name': 'Consumer Price Index',
                'unit': 'Index',
                'color': '#ff7f0e',
                'series_id': 'CPIAUCSL'
            },
            'unemployment_rate': {
                'name': 'Unemployment Rate',
                'full_name': 'Unemployment Rate',
                'unit': 'Percentage',
                'color': '#d62728',
                'series_id': 'UNRATE'
            },
            'inflation': {
                'name': 'Inflation',
                'full_name': 'Inflation Rate (YoY)',
                'unit': 'Percentage',
                'color': '#2ca02c',
                'series_id': 'CPILFESL'  # We'll calculate YoY change from this
            },
            'ppi': {
                'name': 'PPI',
                'full_name': 'Producer Price Index',
                'unit': 'Index',
                'color': '#9467bd',
                'series_id': 'PPIACO'
            },
            'pce': {
                'name': 'PCE',
                'full_name': 'Personal Consumption Expenditures',
                'unit': 'Billions USD',
                'color': '#8c564b',
                'series_id': 'PCE'
            }
        }

    @cache_fred_data(lambda self, period="12m", indicators=None: fred_data_cache.get_economic_charts_cache_key(period, indicators))
    async def get_historical_data(self, period: str = "12m", indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get historical data for economic indicators from FRED API (cached for 30 minutes)
        
        Args:
            period: Time period (6m, 12m, 24m, all)
            indicators: List of specific indicators to include
            
        Returns:
            Dictionary containing historical data
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = self._get_start_date(period, end_date)
            
            # Filter indicators if specified
            if indicators:
                available_indicators = list(self.indicators_config.keys())
                filtered_indicators = [ind for ind in indicators if ind.lower() in available_indicators]
                if not filtered_indicators:
                    filtered_indicators = available_indicators
            else:
                filtered_indicators = list(self.indicators_config.keys())
            
            # Fetch data from FRED API
            if self.fred is not None:
                data = await self._fetch_fred_data(filtered_indicators, start_date, end_date)
            else:
                raise ValueError("FRED API not available. Cannot fetch economic data.")
            
            # Format data for frontend
            formatted_data = {
                'dates': data['dates'],
                'indicators': {},
                'metadata': {
                    'period': period,
                    'total_points': len(data['dates']),
                    'start_date': data['dates'][0] if data['dates'] else None,
                    'end_date': data['dates'][-1] if data['dates'] else None,
                    'source': 'Federal Reserve Economic Data (FRED)',
                    'cached': False  # Fresh data
                }
            }
            
            for indicator in filtered_indicators:
                if indicator in data and data[indicator] is not None:
                    config = self.indicators_config[indicator]
                    values = data[indicator] if isinstance(data[indicator], list) else data[indicator]
                    
                    formatted_data['indicators'][indicator] = {
                        'values': values,
                        'name': config['name'],
                        'full_name': config['full_name'],
                        'unit': config['unit'],
                        'color': config['color'],
                        'series_id': config.get('series_id', ''),
                        'current_value': values[-1] if values and len(values) > 0 else None,
                        'change_from_previous': self._calculate_change(values),
                        'trend': self._calculate_trend(values)
                    }
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error in get_historical_data: {str(e)}")
            raise e

    @cache_fred_data(lambda self: fred_data_cache.get_economic_charts_cache_key("5y_stats", ["all"]))
    async def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for economic indicators using FRED data (cached for 30 minutes)
        
        Returns:
            Dictionary containing correlation matrix and descriptive statistics
        """
        try:
            # Get data for the last 5 years for statistics
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1825)  # 5 years
            
            # Get all indicators
            indicators = list(self.indicators_config.keys())
            
            if self.fred is not None:
                data = await self._fetch_fred_data(indicators, start_date, end_date)
                
                # Create DataFrame
                df_data = {'dates': pd.to_datetime(data['dates'])}
                for indicator in indicators:
                    if indicator in data and data[indicator] is not None:
                        df_data[indicator] = data[indicator]
                
                df = pd.DataFrame(df_data)
                df.set_index('dates', inplace=True)
                
                # Remove any columns that are all NaN
                df = df.dropna(axis=1, how='all')
                
                if df.empty:
                    raise ValueError("No valid data available for statistics")
                
                # Calculate correlation matrix
                correlation_matrix = df.corr().round(3)
                
                # Calculate descriptive statistics
                descriptive_stats = df.describe().round(3)
                
                # Calculate volatility (standard deviation)
                volatility = df.std().round(3)
                
                return {
                    'correlation_matrix': correlation_matrix.to_dict(),
                    'descriptive_stats': descriptive_stats.to_dict(),
                    'volatility': volatility.to_dict(),
                    'metadata': {
                        'indicators_count': len(df.columns),
                        'data_points': len(df),
                        'period_covered': f"{data['dates'][0]} to {data['dates'][-1]}",
                        'source': 'Federal Reserve Economic Data (FRED)'
                    }
                }
            
        except Exception as e:
            logger.error(f"Error in get_summary_statistics: {str(e)}")
            raise e

    async def _fetch_fred_data(self, indicators: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Fetch data from FRED API for specified indicators
        
        Args:
            indicators: List of economic indicators to fetch
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary with dates and indicator values
        """
        try:
            all_data = {}
            common_dates = None
            
            for indicator in indicators:
                config = self.indicators_config[indicator]
                series_id = config['series_id']
                
                try:
                    # Fetch data from FRED
                    series_data = self.fred.get_series(
                        series_id, 
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d')
                    )
                    
                    if series_data.empty:
                        logger.warning(f"No data available for {indicator} ({series_id})")
                        continue
                    
                    # Handle special case for inflation (calculate YoY change)
                    if indicator == 'inflation':
                        # Calculate year-over-year percentage change
                        series_data = series_data.pct_change(periods=12) * 100
                        series_data = series_data.dropna()
                    
                    # Convert to list and store
                    dates = [d.strftime('%Y-%m-%d') for d in series_data.index]
                    values = series_data.values.tolist()
                    
                    all_data[indicator] = {
                        'dates': dates,
                        'values': values
                    }
                    
                    # Find common date range across all indicators
                    if common_dates is None:
                        common_dates = set(dates)
                    else:
                        common_dates = common_dates.intersection(set(dates))
                
                except Exception as e:
                    logger.error(f"Error fetching data for {indicator} ({series_id}): {str(e)}")
                    continue
            
            if not common_dates:
                logger.warning("No common dates found across indicators")
                return {'dates': [], **{ind: [] for ind in indicators}}
            
            # Sort common dates
            common_dates = sorted(list(common_dates))
            
            # Filter dates based on the requested date range
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Filter common dates to match the requested period
            filtered_dates = [
                date for date in common_dates 
                if start_date_str <= date <= end_date_str
            ]
            
            if not filtered_dates:
                logger.warning(f"No data found for the requested period {start_date_str} to {end_date_str}")
                return {'dates': [], **{ind: [] for ind in indicators}}
            
            # Align all data to filtered common dates
            aligned_data = {'dates': filtered_dates}
            
            for indicator in indicators:
                if indicator in all_data:
                    indicator_data = all_data[indicator]
                    date_to_value = dict(zip(indicator_data['dates'], indicator_data['values']))
                    
                    # Extract values for filtered common dates
                    aligned_values = []
                    for date in filtered_dates:
                        if date in date_to_value:
                            aligned_values.append(date_to_value[date])
                        else:
                            # Use forward fill for missing dates
                            aligned_values.append(aligned_values[-1] if aligned_values else None)
                    
                    aligned_data[indicator] = aligned_values
                else:
                    aligned_data[indicator] = [None] * len(filtered_dates)
            
            return aligned_data
            
        except Exception as e:
            logger.error(f"Error in _fetch_fred_data: {str(e)}")
            raise e

    def _get_start_date(self, period: str, end_date: datetime) -> datetime:
        """Get start date based on period"""
        if period == "6m":
            return end_date - timedelta(days=180)
        elif period == "12m":
            return end_date - timedelta(days=365)
        elif period == "24m":
            return end_date - timedelta(days=730)
        else:  # "all"
            return datetime(1947, 1, 1)  # Start of FRED data availability (earliest for most indicators)

    def _calculate_change(self, values: List[float]) -> Dict[str, float]:
        """Calculate change from previous period"""
        if len(values) < 2:
            return {'absolute': 0, 'percentage': 0}
        
        # Filter out None values
        valid_values = [v for v in values if v is not None]
        if len(valid_values) < 2:
            return {'absolute': 0, 'percentage': 0}
        
        current = valid_values[-1]
        previous = valid_values[-2]
        absolute_change = current - previous
        percentage_change = (absolute_change / previous * 100) if previous != 0 else 0
        
        return {
            'absolute': round(absolute_change, 3),
            'percentage': round(percentage_change, 2)
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate overall trend direction"""
        # Filter out None values
        valid_values = [v for v in values if v is not None]
        
        if len(valid_values) < 5:
            return 'insufficient_data'
        
        # Use linear regression to determine trend
        x = np.arange(len(valid_values))
        y = np.array(valid_values)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
