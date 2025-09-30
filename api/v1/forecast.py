from fastapi import APIRouter, HTTPException,Body
from schemas.forecast_schema_1m import InputFeatures1M, ForecastResponse1M
from schemas.forecast_schema_3m import InputFeatures3M, ForecastResponse3M
from schemas.forecast_schema_6m import InputFeatures6M, ForecastResponse6M
from services.forecast_orchestrator import get_all_predictions, AllPredictionsResponse
from services.forecast_service_1m import predict_1m, initialize_1m_service
from services.forecast_service_3m import predict_3m, initialize_3m_service
from services.forecast_service_6m import predict_6m, initialize_6m_service
from pydantic import BaseModel
from utils.feature_preparation import prepare_features_1m, prepare_features_3m,prepare_features_6m
from utils.prediction_cache import cache_prediction_response, prediction_cache
import logging

router = APIRouter()

class AllInputFeatures(BaseModel):
    features_1m: InputFeatures1M
    features_3m: InputFeatures3M
    features_6m: InputFeatures6M

# Combined endpoint
@router.post("/predict/all", response_model=AllPredictionsResponse)
async def predict_all_timeframes(all_features: AllInputFeatures):
    """Get all recession probability forecasts (1m, 3m, 6m)"""
    try:
        return get_all_predictions(
            all_features.features_1m,
            all_features.features_3m,
            all_features.features_6m
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Individual endpoints - ALL CHANGED FROM POST to GET
@router.get("/predict/1m", response_model=ForecastResponse1M)
async def predict_1m_recession():
    """1-month recession probability forecast using latest FRED data (cached for 1 hour)"""
    logger = logging.getLogger(__name__)
    
    # Generate cache key
    cache_key = prediction_cache.get_cache_key("1m")
    
    # Try to get cached response
    cached_response = prediction_cache.get(cache_key)
    if cached_response is not None:
        logger.info("✅ Serving cached 1m prediction")
        # Add cache metadata
        if hasattr(cached_response, 'feature_importance'):
            cached_response.feature_importance = cached_response.feature_importance or {}
            cached_response.feature_importance['cached'] = True
            cached_response.feature_importance['cache_key'] = cache_key
        return cached_response
    
    # Cache miss - generate new prediction
    logger.info("⚠️ Cache miss for 1m prediction - generating new prediction")
    try:
        # Import the new service function
        from services.fred_data_service_1m import get_latest_prediction_1m
        
        # This function will handle all the logic:
        # 1. Check FRED for latest date
        # 2. Compare with database
        # 3. Fetch data accordingly
        # 4. Make prediction
        result = await get_latest_prediction_1m()
        
        # Cache the result
        prediction_cache.set(cache_key, result)
        
        # Add metadata to indicate this is a fresh prediction
        if hasattr(result, 'feature_importance'):
            result.feature_importance = result.feature_importance or {}
            result.feature_importance['cached'] = False
            result.feature_importance['cache_key'] = cache_key
        
        return result
        
    except RuntimeError as e:
        if "not loaded" in str(e):
            raise HTTPException(status_code=503, detail="1M model or scaler is not available. Please check service status.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/predict/3m", response_model=ForecastResponse3M)
async def predict_3m_recession():
    """3-month recession probability forecast using latest FRED data (cached for 1 hour)"""
    logger = logging.getLogger(__name__)
    
    # Generate cache key
    cache_key = prediction_cache.get_cache_key("3m")
    
    # Try to get cached response
    cached_response = prediction_cache.get(cache_key)
    if cached_response is not None:
        logger.info("✅ Serving cached 3m prediction")
        # Add cache metadata
        if hasattr(cached_response, 'feature_importance'):
            cached_response.feature_importance = cached_response.feature_importance or {}
            cached_response.feature_importance['cached'] = True
            cached_response.feature_importance['cache_key'] = cache_key
        return cached_response
    
    # Cache miss - generate new prediction
    logger.info("⚠️ Cache miss for 3m prediction - generating new prediction")
    try:
        # Import the new service function
        from services.fred_data_service_3m import get_latest_prediction_3m
        
        # This function will handle all the logic:
        # 1. Check FRED for latest date
        # 2. Compare with database
        # 3. Fetch data accordingly
        # 4. Make prediction
        result = await get_latest_prediction_3m()
        
        # Cache the result
        prediction_cache.set(cache_key, result)
        
        # Add metadata to indicate this is a fresh prediction
        if hasattr(result, 'feature_importance'):
            result.feature_importance = result.feature_importance or {}
            result.feature_importance['cached'] = False
            result.feature_importance['cache_key'] = cache_key
        
        return result
        
    except RuntimeError as e:
        if "not loaded" in str(e):
            raise HTTPException(status_code=503, detail="3M model or scaler is not available. Please check service status.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/predict/6m", response_model=ForecastResponse6M)
async def predict_6m_recession():
    """6-month recession probability forecast using latest FRED data (cached for 1 hour)"""
    logger = logging.getLogger(__name__)
    
    # Generate cache key
    cache_key = prediction_cache.get_cache_key("6m")
    
    # Try to get cached response
    cached_response = prediction_cache.get(cache_key)
    if cached_response is not None:
        logger.info("✅ Serving cached 6m prediction")
        # Add cache metadata
        if hasattr(cached_response, 'feature_importance'):
            cached_response.feature_importance = cached_response.feature_importance or {}
            cached_response.feature_importance['cached'] = True
            cached_response.feature_importance['cache_key'] = cache_key
        return cached_response
    
    # Cache miss - generate new prediction
    logger.info("⚠️ Cache miss for 6m prediction - generating new prediction")
    try:
        # Import the new service function
        from services.fred_data_service_6m import get_latest_prediction_6m
        
        # This function will handle all the logic:
        # 1. Check FRED for latest date
        # 2. Compare with database
        # 3. Fetch data accordingly (including weekly series averaging)
        # 4. Make prediction
        result = await get_latest_prediction_6m()
        
        # Cache the result
        prediction_cache.set(cache_key, result)
        
        # Add metadata to indicate this is a fresh prediction
        if hasattr(result, 'feature_importance'):
            result.feature_importance = result.feature_importance or {}
            result.feature_importance['cached'] = False
            result.feature_importance['cache_key'] = cache_key
        
        return result
        
    except RuntimeError as e:
        if "not loaded" in str(e):
            raise HTTPException(status_code=503, detail="6M model or scaler is not available. Please check service status.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Add status endpoints
@router.get("/status/1m")
async def get_1m_status():
    """Get 1-month model status"""
    from services.forecast_service_1m import get_model_info_1m
    return get_model_info_1m()

@router.get("/status/3m")
async def get_3m_status():
    """Get 3-month model status"""
    from services.forecast_service_3m import get_model_info_3m
    return get_model_info_3m()

@router.get("/status/6m")
async def get_6m_status():
    """Get 6-month model status"""
    from services.forecast_service_6m import get_model_info_6m
    return get_model_info_6m()

# Test endpoints
@router.get("/test/1m")
async def test_1m_service():
    """Test the 1-month forecasting service"""
    try:
        from services.forecast_service_1m import test_prediction_1m, get_model_info_1m
        
        # Get service status
        status = get_model_info_1m()
        
        # If service is not ready, try to initialize
        if not status.model_loaded or not status.scaler_loaded:
            if not initialize_1m_service():
                return {"error": "Service initialization failed", "status": status}
        
        # Run test prediction
        result = test_prediction_1m()
        
        if result:
            return {
                "status": "success",
                "test_result": {
                    "prob_1m": result.prob_1m,
                    "model_version": result.model_version,
                    "input_date": result.input_date
                },
                "service_status": get_model_info_1m()
            }
        else:
            return {"error": "Test prediction failed", "status": get_model_info_1m()}
            
    except Exception as e:
        return {"error": str(e)}

@router.get("/test/3m")
async def test_3m_service():
    """Test the 3-month forecasting service"""
    try:
        from services.forecast_service_3m import test_prediction_3m, get_model_info_3m
        
        # Get service status
        status = get_model_info_3m()
        
        # If service is not ready, try to initialize
        if not status.model_loaded or not status.scaler_loaded:
            if not initialize_3m_service():
                return {"error": "Service initialization failed", "status": status}
        
        # Run test prediction
        result = test_prediction_3m()
        
        if result:
            return {
                "status": "success",
                "test_result": {
                    "prob_3m": result.prob_3m,
                    "model_version": result.model_version,
                    "input_date": result.input_date
                },
                "service_status": get_model_info_3m()
            }
        else:
            return {"error": "Test prediction failed", "status": get_model_info_3m()}
            
    except Exception as e:
        return {"error": str(e)}

@router.get("/test/6m")
async def test_6m_service():
    """Test the 6-month forecasting service"""
    try:
        from services.forecast_service_6m import test_prediction_6m, get_model_info_6m
        
        # Get service status
        status = get_model_info_6m()
        
        # If service is not ready, try to initialize
        if not status.model_loaded or not status.scaler_loaded:
            if not initialize_6m_service():
                return {"error": "Service initialization failed", "status": status}
        
        # Run test prediction
        result = test_prediction_6m()
        
        if result:
            return {
                "status": "success",
                "test_result": {
                    "prob_6m": result.prob_6m,
                    "model_version": result.model_version,
                    "input_date": result.input_date
                },
                "service_status": get_model_info_6m()
            }
        else:
            return {"error": "Test prediction failed", "status": get_model_info_6m()}
            
    except Exception as e:
        return {"error": str(e)}

# Cache management endpoints
@router.get("/cache/stats")
async def get_cache_stats():
    """Get prediction cache statistics"""
    return {
        "status": "success",
        "cache_stats": prediction_cache.get_stats(),
        "message": "Cache statistics retrieved successfully"
    }

@router.post("/cache/clear")
async def clear_prediction_cache():
    """Clear all cached predictions (admin function)"""
    try:
        prediction_cache.clear()
        return {
            "status": "success",
            "message": "All prediction cache cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.delete("/cache/{timeframe}")
async def invalidate_timeframe_cache(timeframe: str):
    """Invalidate cache for a specific timeframe (1m, 3m, or 6m)"""
    if timeframe not in ["1m", "3m", "6m"]:
        raise HTTPException(status_code=400, detail="Invalid timeframe. Must be 1m, 3m, or 6m")
    
    try:
        cache_key = prediction_cache.get_cache_key(timeframe)
        invalidated = prediction_cache.invalidate(cache_key)
        
        return {
            "status": "success",
            "timeframe": timeframe,
            "cache_key": cache_key,
            "invalidated": invalidated,
            "message": f"Cache for {timeframe} predictions {'invalidated' if invalidated else 'was not cached'}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")

# Priority management endpoints
@router.get("/priority/stats")
async def get_priority_stats():
    """Get request priority statistics"""
    from utils.request_priority import priority_manager
    
    return {
        "status": "success",
        "priority_stats": priority_manager.get_stats(),
        "message": "Priority statistics retrieved successfully"
    }

