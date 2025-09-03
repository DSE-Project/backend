from fastapi import APIRouter, HTTPException
from schemas.forecast_schema import InputFeatures, ForecastResponse
from services import forecast_service

# Create a router to group related endpoints
router = APIRouter()

@router.post("/predict", response_model=ForecastResponse)
async def predict_recession(features: InputFeatures):
    """
    Receives economic feature values and returns the 1, 3, and 6-month 
    [cite_start]recession probability forecasts[cite: 134].
    """
    try:
        predictions = forecast_service.get_predictions(features)
        return predictions
    except RuntimeError as e:
        # This will catch the error if models failed to load
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # A general catch-all for other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")