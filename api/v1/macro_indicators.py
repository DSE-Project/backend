from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from services.macro_indicators_service import MacroIndicatorsService
from utils.fred_data_cache import fred_data_cache

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/macro-indicators")
async def get_macro_indicators() -> Dict[str, Any]:
    """
    Get key macroeconomic indicators snapshot including:
    - Unemployment Rate
    - CPI Inflation
    - Yield Curve Spread (10y-2y)
    - Fed Funds Rate
    - PMI/ISM Manufacturing
    - Consumer Confidence Index
    
    Each indicator includes current value, change from last period, and last updated timestamp.
    """
    try:
        service = MacroIndicatorsService()
        indicators = await service.get_all_indicators()
        return {
            "success": True,
            "data": indicators,
            "message": "Macroeconomic indicators retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error fetching macro indicators: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch macroeconomic indicators: {str(e)}"
        )

@router.get("/macro-indicators/cache/stats")
async def get_fred_cache_stats():
    """Get FRED data cache statistics for macro indicators"""
    return {
        "success": True,
        "cache_stats": fred_data_cache.get_stats(),
        "message": "FRED cache statistics retrieved successfully"
    }

@router.post("/macro-indicators/cache/clear")
async def clear_fred_cache():
    """Clear all cached FRED data (admin function)"""
    try:
        fred_data_cache.clear()
        return {
            "success": True,
            "message": "FRED data cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing FRED cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear FRED cache: {str(e)}"
        )
