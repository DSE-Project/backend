from fastapi import APIRouter, HTTPException, Path, Query
from typing import Optional
from schemas.forecast_schema_1m import InputFeatures1M, ForecastResponse1M
from schemas.forecast_schema_3m import InputFeatures3M, ForecastResponse3M
from schemas.forecast_schema_6m import InputFeatures6M, ForecastResponse6M
from services.forecast_service_1m import predict_1m, initialize_1m_service
from services.forecast_service_3m import predict_3m
from services.forecast_service_6m import predict_6m

from schemas.simulate_schema import (
    ModelFeatureDefinitions, 
    AllFeatureDefinitions, 
    SimulateError,
    FeatureDefinition
)
from services.simulate_service import simulate_service

router = APIRouter()

@router.get("/features/{model_period}", response_model=ModelFeatureDefinitions)
async def get_model_features(
    model_period: str = Path(..., description="Model period: 1m, 3m, or 6m")
):
    """
    Get feature definitions for a specific model period
    
    - **model_period**: The model period (1m, 3m, or 6m)
    
    Returns feature definitions including name, description, min/max values, and defaults
    """
    if model_period not in ["1m", "3m", "6m"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model period: {model_period}. Must be one of: 1m, 3m, 6m"
        )
    
    try:
        features = simulate_service.get_feature_definitions(model_period)
        
        if not features:
            raise HTTPException(
                status_code=404, 
                detail=f"No feature definitions found for model period: {model_period}"
            )
        
        return features
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get feature definitions: {str(e)}"
        )

@router.get("/features", response_model=AllFeatureDefinitions)
async def get_all_features():
    """
    Get feature definitions for all supported model periods
    
    Returns a comprehensive list of all features for 1m, 3m, and 6m models
    """
    try:
        all_features = simulate_service.get_all_feature_definitions()
        
        if not all_features:
            raise HTTPException(
                status_code=404, 
                detail="No feature definitions found for any model"
            )
        
        return all_features
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get all feature definitions: {str(e)}"
        )

# Individual endpoints
@router.post("/predict/1m", response_model=ForecastResponse1M)
async def predict_1m_recession(features: InputFeatures1M):
    """1-month recession probability forecast"""
    try:
        # Try to initialize if not already done
        if not initialize_1m_service():
            raise HTTPException(status_code=503, detail="1M forecasting service could not be initialized")
        
        return predict_1m(features)
    except RuntimeError as e:
        # Check if it's a model loading issue
        if "not loaded" in str(e):
            raise HTTPException(status_code=503, detail="1M model or scaler is not available. Please check service status.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/predict/3m", response_model=ForecastResponse3M)
async def predict_3m_recession(features: InputFeatures3M):
    """3-month recession probability forecast"""
    try:
        return predict_3m(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/6m", response_model=ForecastResponse6M)
async def predict_6m_recession(features: InputFeatures6M):
    """6-month recession probability forecast"""
    try:
        return predict_6m(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features/{model_period}/important", response_model=list[FeatureDefinition])
async def get_important_features(
    model_period: str = Path(..., description="Model period: 1m, 3m, or 6m")
):
    """
    Get only important features for a specific model period
    
    - **model_period**: The model period (1m, 3m, or 6m)
    
    Returns only features marked as important (is_important = 1)
    """
    if model_period not in ["1m", "3m", "6m"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model period: {model_period}. Must be one of: 1m, 3m, 6m"
        )
    
    try:
        important_features = simulate_service.get_important_features(model_period)
        
        if important_features is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No feature definitions found for model period: {model_period}"
            )
        
        return important_features
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get important features: {str(e)}"
        )


@router.get("/summary")
async def get_feature_summary(
    model_period: Optional[str] = Query(None, description="Specific model period to summarize")
):
    """
    Get a summary of available features
    
    - **model_period**: Optional. If provided, returns summary for specific model only
    
    Returns feature count and other metadata for debugging/monitoring
    """
    try:
        if model_period and model_period not in ["1m", "3m", "6m"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model period: {model_period}. Must be one of: 1m, 3m, 6m"
            )
        
        summary = simulate_service.get_feature_summary(model_period)
        
        if "error" in summary:
            raise HTTPException(
                status_code=404, 
                detail=summary["error"]
            )
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get feature summary: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """
    Health check endpoint for simulate service
    
    Tests database connectivity and basic functionality
    """
    try:
        db_connected = simulate_service.test_database_connection()
        
        return {
            "status": "healthy" if db_connected else "unhealthy",
            "database_connected": db_connected,
            "supported_models": simulate_service.supported_models,
            "timestamp": simulate_service.get_feature_summary()["timestamp"] if db_connected else None
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "database_connected": False,
            "error": str(e),
            "supported_models": simulate_service.supported_models
        }