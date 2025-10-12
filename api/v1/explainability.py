from fastapi import APIRouter, HTTPException
from typing import Dict
import logging
from datetime import datetime

from services.fred_data_service_1m import get_latest_database_row_1m, convert_to_input_features_1m
from services.fred_data_service_3m import get_latest_database_row_3m, convert_to_input_features_3m
from services.fred_data_service_6m import get_latest_database_row_6m, convert_to_input_features_6m
from services.explainability_service_1m import get_explanation_1m, explainability_service_1m
from services.explainability_service_3m import get_explanation_3m, explainability_service_3m
from services.explainability_service_6m import get_explanation_6m, explainability_service_6m
from utils.explainability_cache import explainability_cache

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/explain/1m", response_model=Dict)
async def explain_1m_prediction():
    """Get SHAP and permutation importance explanations for 1-month prediction"""
    try:
        # Generate cache key for explanation results
        current_hour = datetime.now().strftime('%Y-%m-%d-%H')
        cache_key = f"explanation_result_1m_{current_hour}"
        
        # Try to get cached explanation result
        cached_result = explainability_cache.get(cache_key)
        if cached_result is not None:
            logger.info("✅ Serving cached 1M explanation result")
            # Add cache metadata
            cached_result['cached'] = True
            cached_result['cache_key'] = cache_key
            return cached_result
        
        # Cache miss - generate new explanation
        logger.info("🔍 Generating new 1M explanation")
        
        # Get latest data from database
        latest_row = get_latest_database_row_1m()
        if not latest_row:
            raise HTTPException(status_code=404, detail="No data available in database")
        
        # Convert to input features
        features = convert_to_input_features_1m(latest_row)
        
        # Generate explanation
        explanation = get_explanation_1m(features)
        
        # Cache the result for 1 hour
        explainability_cache.set(cache_key, explanation, ttl_seconds=3600)
        
        # Add metadata
        explanation['cached'] = False
        explanation['cache_key'] = cache_key
        explanation['timestamp'] = datetime.now().isoformat()
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating 1M explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@router.get("/explain/3m", response_model=Dict)
async def explain_3m_prediction():
    """Get SHAP and permutation importance explanations for 3-month prediction"""
    try:
        # Generate cache key for explanation results
        current_hour = datetime.now().strftime('%Y-%m-%d-%H')
        cache_key = f"explanation_result_3m_{current_hour}"
        
        # Try to get cached explanation result
        cached_result = explainability_cache.get(cache_key)
        if cached_result is not None:
            logger.info("✅ Serving cached 3M explanation result")
            # Add cache metadata
            cached_result['cached'] = True
            cached_result['cache_key'] = cache_key
            return cached_result
        
        # Cache miss - generate new explanation
        logger.info("🔍 Generating new 3M explanation")
        
        # Get latest data from database
        latest_row = get_latest_database_row_3m()
        if not latest_row:
            raise HTTPException(status_code=404, detail="No data available in database")
        
        # Convert to input features
        features = convert_to_input_features_3m(latest_row)
        
        # Generate explanation
        explanation = get_explanation_3m(features)
        
        # Cache the result for 1 hour
        explainability_cache.set(cache_key, explanation, ttl_seconds=3600)
        
        # Add metadata
        explanation['cached'] = False
        explanation['cache_key'] = cache_key
        explanation['timestamp'] = datetime.now().isoformat()
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating 3M explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@router.get("/explain/6m", response_model=Dict)
async def explain_6m_prediction():
    """Get SHAP and permutation importance explanations for 6-month prediction"""
    try:
        # Generate cache key for explanation results
        current_hour = datetime.now().strftime('%Y-%m-%d-%H')
        cache_key = f"explanation_result_6m_{current_hour}"
        
        # Try to get cached explanation result
        cached_result = explainability_cache.get(cache_key)
        if cached_result is not None:
            logger.info("✅ Serving cached 6M explanation result")
            # Add cache metadata
            cached_result['cached'] = True
            cached_result['cache_key'] = cache_key
            return cached_result
        
        # Cache miss - generate new explanation
        logger.info("🔍 Generating new 6M explanation")
        
        # Get latest data from database
        latest_row = get_latest_database_row_6m()
        if not latest_row:
            raise HTTPException(status_code=404, detail="No data available in database")
        
        # Convert to input features
        features = convert_to_input_features_6m(latest_row)
        
        # Generate explanation
        explanation = get_explanation_6m(features)
        
        # Cache the result for 1 hour
        explainability_cache.set(cache_key, explanation, ttl_seconds=3600)
        
        # Add metadata
        explanation['cached'] = False
        explanation['cache_key'] = cache_key
        explanation['timestamp'] = datetime.now().isoformat()
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating 6M explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@router.get("/explain/all", response_model=Dict)
async def explain_all_predictions():
    """Get SHAP and permutation importance explanations for all timeframes (1m, 3m, 6m)"""
    try:
        # Generate cache key for all explanations
        current_hour = datetime.now().strftime('%Y-%m-%d-%H')
        cache_key = f"explanation_result_all_{current_hour}"
        
        # Try to get cached explanation result
        cached_result = explainability_cache.get(cache_key)
        if cached_result is not None:
            logger.info("✅ Serving cached ALL explanation results")
            # Add cache metadata
            cached_result['cached'] = True
            cached_result['cache_key'] = cache_key
            return cached_result
        
        # Cache miss - generate new explanations
        logger.info("🔍 Generating new explanations for all timeframes")
        
        # Get all features
        latest_row_1m = get_latest_database_row_1m()
        latest_row_3m = get_latest_database_row_3m()
        latest_row_6m = get_latest_database_row_6m()
        
        if not latest_row_1m or not latest_row_3m or not latest_row_6m:
            raise HTTPException(status_code=404, detail="Missing data in database")
        
        features_1m = convert_to_input_features_1m(latest_row_1m)
        features_3m = convert_to_input_features_3m(latest_row_3m)
        features_6m = convert_to_input_features_6m(latest_row_6m)
        
        # Generate all explanations
        explanation_1m = get_explanation_1m(features_1m)
        explanation_3m = get_explanation_3m(features_3m)
        explanation_6m = get_explanation_6m(features_6m)
        
        result = {
            "1m": explanation_1m,
            "3m": explanation_3m,
            "6m": explanation_6m,
            "timestamp": features_1m.current_month_data.observation_date,
            "cached": False,
            "cache_key": cache_key,
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache the combined result for 1 hour
        explainability_cache.set(cache_key, result, ttl_seconds=3600)
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating all explanations: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@router.post("/debug/clear-cache", response_model=Dict)
async def clear_explainer_cache():
    """Debug endpoint to clear SHAP explainer cache and force reinitialization"""
    try:
        # Use the new global cache clearing
        from utils.explainability_cache import explainability_cache
        explainability_cache.clear()
        
        return {
            "message": "SHAP explainer cache cleared for all timeframes",
            "status": "success",
            "cache_stats": explainability_cache.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error clearing explainer cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@router.get("/cache/stats", response_model=Dict)
async def get_explainer_cache_stats():
    """Get explainability cache statistics"""
    try:
        from utils.explainability_cache import explainability_cache
        
        # Clean up expired entries first
        expired_count = explainability_cache.cleanup_expired()
        
        return {
            "status": "success",
            "cache_stats": explainability_cache.get_stats(),
            "expired_cleaned": expired_count,
            "message": "Explainability cache statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

@router.post("/cache/clear-results", response_model=Dict)
async def clear_explanation_results_cache():
    """Clear only cached explanation results (keep explainer instances cached)"""
    try:
        from utils.explainability_cache import explainability_cache
        
        # Get current cache stats
        stats_before = explainability_cache.get_stats()
        
        # Clear only explanation result cache entries
        cleared_count = 0
        cache_keys_to_delete = []
        
        with explainability_cache._lock:
            for key in explainability_cache._cache.keys():
                if key.startswith("explanation_result_"):
                    cache_keys_to_delete.append(key)
        
        for key in cache_keys_to_delete:
            if explainability_cache.delete(key):
                cleared_count += 1
        
        return {
            "status": "success",
            "cleared_results": cleared_count,
            "cache_stats_before": stats_before,
            "cache_stats_after": explainability_cache.get_stats(),
            "message": f"Cleared {cleared_count} cached explanation results (explainer instances kept)"
        }
        
    except Exception as e:
        logger.error(f"Error clearing explanation results cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear results cache: {str(e)}")
