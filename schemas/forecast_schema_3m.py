from pydantic import BaseModel
from typing import Optional

class CurrentMonthData3M(BaseModel):
    """Current month economic indicators for 3-month recession prediction"""
    observation_date: str
    fedfunds: float
    TB3MS: float
    TB6MS: float
    TB1YR: float
    USTPU: float
    USGOOD: float
    SRVPRD: float
    USCONS: float
    MANEMP: float
    USWTRADE: float
    USTRADE: float
    USINFO: float
    UNRATE: float
    UNEMPLOY: float
    CPIFOOD: float
    CPIMEDICARE: float
    CPIRENT: float
    CPIAPP: float
    GDP: float
    REALGDP: float
    PCEPI: float
    PSAVERT: float
    PSTAX: float
    COMREAL: float
    COMLOAN: float
    SECURITYBANK: float
    PPIACO: float
    M1SL: float
    M2SL: float
    recession: int

class InputFeatures3M(BaseModel):
    """Features for 3-month recession probability model"""
    current_month_data: CurrentMonthData3M
    use_historical_data: bool = True  # Flag to indicate if historical data should be loaded
    historical_data_source: Optional[str] = "database"  # "csv" or "database"

    class Config:
        json_schema_extra = {
            "example": {
                "current_month_data": {
                    "observation_date": "1/2/2025",
                    "fedfunds": 4.40,
                    "TB3MS": 4.22,
                    "TB6MS": 4.14,
                    "TB1YR": 4.05,
                    "USTPU": 30000,
                    "USGOOD": 21670,
                    "SRVPRD": 13700,
                    "USCONS": 9000,
                    "MANEMP": 12800,
                    "USWTRADE": 7602,
                    "USTRADE": 15602,
                    "USINFO": 3200,
                    "UNRATE": 4.0,
                    "UNEMPLOY": 6600,
                    "CPIFOOD": 300,
                    "CPIMEDICARE": 600,
                    "CPIRENT": 1500,
                    "CPIAPP": 200,
                    "GDP": 25000,
                    "REALGDP": 21000,
                    "PCEPI": 140,
                    "PSAVERT": 5.0,
                    "PSTAX": 1100,
                    "COMREAL": 220000,
                    "COMLOAN": -0.3,
                    "SECURITYBANK": -2.0,
                    "PPIACO": 270,
                    "M1SL": 20000,
                    "M2SL": 150000,
                    "recession": 0
                },
                "use_historical_data": True,
                "historical_data_source": "database"
            }
        }
    

class ForecastResponse3M(BaseModel):
    prob_3m: float  # Recession probability (0.0 to 1.0)
    model_version: str  # Model version identifier
    timestamp: str  # Prediction timestamp
    input_date: str  # Date of the input data
    confidence_interval: Optional[dict] = None  # Optional confidence intervals
    feature_importance: Optional[dict] = None  # Optional feature importance scores
    

    class Config:
        json_schema_extra = {
            "example": {
                "prob_3m": 0.23,
                "model_version": "lstm_transformer_v1.0",
                "timestamp": "2024-08-01T10:30:00Z",
                "input_date": "1/2/2025",
                "confidence_interval": {
                    "threshold_used": 0.72,
                    "binary_prediction": 0,
                    "prediction_text": "No Recession Expected"
                },
                "feature_importance": {
                    "data_points_used": 120,
                    "feature_count": 30,
                    "sequence_length": 12,
                    "model_architecture": "LSTM+Transformer",
                    "lstm_units": 64,
                    "transformer_layers": 1,
                    "attention_heads": 2
                }
                
            }
        }

class ModelStatus3M(BaseModel):   
    """Status information for the 3-month model"""
    model_loaded: bool
    scaler_loaded: bool
    model_version: str
    last_updated: str
    historical_data_available: bool
    total_features: int
    model_architecture: Optional[str] = None
    sequence_length: Optional[int] = None
    threshold: Optional[float] = None