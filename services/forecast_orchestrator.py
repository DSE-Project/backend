from datetime import datetime
from pydantic import BaseModel
from services.forecast_service_1m import predict_1m, InputFeatures1M
from services.forecast_service_3m import predict_3m, InputFeatures3M
from services.forecast_service_6m import predict_6m, InputFeatures6M

class AllPredictionsResponse(BaseModel):
    """Combined response from all three models"""
    prediction_1m: dict
    prediction_3m: dict
    prediction_6m: dict
    timestamp: str

def get_all_predictions(features_1m: InputFeatures1M, 
                        features_3m: InputFeatures3M, 
                        features_6m: InputFeatures6M,
                        history_1m=None,
                        history_3m=None,
                        history_6m=None) -> AllPredictionsResponse:
    """
    Get predictions from all three models.
    history_*: optional historical DataFrames for LSTM preprocessing
    """
    pred_1m_resp = predict_1m(features_1m, history_1m) if history_1m is not None else predict_1m(features_1m)
    pred_3m_resp = predict_3m(features_3m, history_3m) if history_3m is not None else predict_3m(features_3m)
    pred_6m_resp = predict_6m(features_6m, history_6m) if history_6m is not None else predict_6m(features_6m)

    response = AllPredictionsResponse(
        prediction_1m=pred_1m_resp.dict(),
        prediction_3m=pred_3m_resp.dict(),
        prediction_6m=pred_6m_resp.dict(),
        timestamp=datetime.utcnow().isoformat()
    )
    return response

def get_model_statuses() -> dict:
    """Check status of all models"""
    statuses = {
        "forecast_1m": "online",  # could be dynamic, e.g., forecast_service_1m.status()
        "forecast_3m": "online",
        "forecast_6m": "online"
    }
    return statuses

def initialize_all_services():
    """Initialize all forecasting services"""
    try:
        from services.forecast_service_1m import initialize_1m_service as init_1m
        from services.forecast_service_3m import initialize_3m_service as init_3m
        from services.forecast_service_6m import initialize_6m_service as init_6m

        init_1m()
        init_3m()
        init_6m()
        print("All forecasting services initialized successfully.")
    except Exception as e:
        print(f"Error initializing services: {e}")

if __name__ == "__main__":
    # Example initialization
    initialize_all_services()
    print(get_model_statuses())
