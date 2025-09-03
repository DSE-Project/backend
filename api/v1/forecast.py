from fastapi import APIRouter, HTTPException
from schemas.forecast_schema_1m import InputFeatures1M, ForecastResponse1M
from schemas.forecast_schema_3m import InputFeatures3M, ForecastResponse3M
from schemas.forecast_schema_6m import InputFeatures6M, ForecastResponse6M
from services.forecast_orchestrator import get_all_predictions, AllPredictionsResponse
from services.forecast_service_1m import predict_1m
from services.forecast_service_3m import predict_3m
from services.forecast_service_6m import predict_6m
from pydantic import BaseModel

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

# Individual endpoints
@router.post("/predict/1m", response_model=ForecastResponse1M)
async def predict_1m_recession(features: InputFeatures1M):
    """1-month recession probability forecast"""
    try:
        return predict_1m(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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