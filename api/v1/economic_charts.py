from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import logging
from services.economic_charts_service import EconomicChartsService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Initialize the economic charts service
try:
    economic_charts_service = EconomicChartsService()
    logger.info("✅ Economic charts service initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize economic charts service: {str(e)}")
    economic_charts_service = None

@router.get("/economic-charts/historical-data")
async def get_historical_data(
    period: str = Query(default="12m", description="Time period: 6m, 12m, 24m, all"),
    indicators: Optional[str] = Query(default=None, description="Comma-separated list of indicators")
):
    """
    Get historical economic indicators data
    
    Args:
        period: Time period for data (6m, 12m, 24m, all)
        indicators: Optional comma-separated list of specific indicators
        
    Returns:
        Historical economic data with metadata
    """
    try:
        if economic_charts_service is None:
            raise HTTPException(status_code=503, detail="Economic charts service not available")
            
        # Parse indicators if provided
        indicator_list = None
        if indicators:
            indicator_list = [indicator.strip().lower() for indicator in indicators.split(',')]
        
        # Fetch historical data
        data = await economic_charts_service.get_historical_data(period=period, indicators=indicator_list)
        
        return {
            "success": True,
            "data": data,
            "message": f"Historical data retrieved successfully for period: {period}"
        }
        
    except Exception as e:
        logger.error(f"Error in get_historical_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")

@router.get("/economic-charts/summary-stats")
async def get_summary_statistics():
    """
    Get summary statistics and correlations for economic indicators
    
    Returns:
        Statistical summary including correlations, descriptive stats, and volatility
    """
    try:
        if economic_charts_service is None:
            raise HTTPException(status_code=503, detail="Economic charts service not available")
            
        # Fetch summary statistics
        stats = await economic_charts_service.get_summary_statistics()
        
        return {
            "success": True,
            "data": stats,
            "message": "Summary statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in get_summary_statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch summary statistics: {str(e)}")

@router.get("/economic-charts/indicators")
async def get_available_indicators():
    """
    Get list of available economic indicators
    
    Returns:
        List of available indicators with their configurations
    """
    try:
        if economic_charts_service is None:
            raise HTTPException(status_code=503, detail="Economic charts service not available")
            
        indicators = economic_charts_service.indicators_config
        
        return {
            "success": True,
            "data": {
                "indicators": indicators,
                "count": len(indicators)
            },
            "message": "Available indicators retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in get_available_indicators: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch indicators: {str(e)}")

@router.get("/economic-charts/health")
async def health_check():
    """
    Health check endpoint for economic charts service
    
    Returns:
        Service health status
    """
    try:
        is_healthy = economic_charts_service is not None
        fred_status = "connected" if economic_charts_service and economic_charts_service.fred else "disconnected"
        
        return {
            "success": True,
            "data": {
                "service_status": "healthy" if is_healthy else "unhealthy",
                "fred_api_status": fred_status,
                "available_endpoints": [
                    "/economic-charts/historical-data",
                    "/economic-charts/summary-stats", 
                    "/economic-charts/indicators",
                    "/economic-charts/health"
                ]
            },
            "message": "Economic charts service health check completed"
        }
        
    except Exception as e:
        logger.error(f"Error in health_check: {str(e)}")
        return {
            "success": False,
            "data": {
                "service_status": "unhealthy",
                "error": str(e)
            },
            "message": "Economic charts service health check failed"
        }
