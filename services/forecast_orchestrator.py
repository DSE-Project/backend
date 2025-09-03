from services.forecast_service_1m import predict_1m, InputFeatures1M
from services.forecast_service_3m import predict_3m, InputFeatures3M
from services.forecast_service_6m import predict_6m, InputFeatures6M
from pydantic import BaseModel

class AllPredictionsResponse(BaseModel):
    """Combined response from all three models"""
    prediction_1m: dict
    prediction_3m: dict
    prediction_6m: dict
    timestamp: str

def get_all_predictions(features_1m: InputFeatures1M, 
                       features_3m: InputFeatures3M, 
                       features_6m: InputFeatures6M) -> AllPredictionsResponse:
    """Get predictions from all three models"""
    # TODO: Implement orchestrated prediction logic
    pass

def get_model_statuses() -> dict:
    """Check status of all models"""
    # TODO: Return status of all three services
    pass

def initialize_all_services():
    """Initialize all forecasting services"""
    # TODO: Initialize all three services
    pass