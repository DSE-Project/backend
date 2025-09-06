from pydantic import BaseModel
from typing import Optional

class CurrentMonthData3M(BaseModel):
    """Current month economic indicators for 1-month recession prediction"""
    observation_date: str
    ICSA: float
    CPIMEDICARE: float
    USWTRADE: float
    BBKMLEIX: float
    COMLOAN:float
    UMCSENT:float
    MANEMP:float
    fedfunds:float
    PSTAX:float
    USCONS:float
    USGOOD:float
    USINFO:float
    CPIAPP:float
    CSUSHPISA:float
    SECURITYBANK:float
    SRVPRD:float
    INDPRO:float
    TB6MS:float
    UNEMPLOY:float
    USTPU:float
    recession:int

class InputFeatures3M(BaseModel):
    """Features for 3-month recession probability model"""
    current_month_data: CurrentMonthData3M
    use_historical_data: bool = True  # Flag to indicate if historical data should be loaded
    historical_data_source: Optional[str] = "csv"  # "csv" or "database"

    class Config:
        json_schema_extra = {
            "example": {
                "current_month_data": {
                    "observation_date": "1/8/2024",
                    "ICSA": 237700,
                    "CPIMEDICARE":565.759,
                    "USWTRADE": 6147.9,
                    "BBKMLEIX" : 1.5062454,
                    "COMLOAN" :0.5,
                    "UMCSENT" :63.5,
                    "MANEMP" : 12845,
                    "fedfunds" :5.33,
                    "PSTAX" :3074.386,
                    "USCONS" :8221,
                    "USGOOD" :21683,
                    "USINFO" :2960,
                    "CPIAPP": 131.124,
                    "CSUSHPISA" : 322.425,
                    "SECURITYBANK" :-1.8,
                    "SRVPRD":136409,
                    "INDPRO" :102.8692,
                    "TB6MS" :4.97,
                    "UNEMPLOY" :7153,
                    "USTPU" :28911,
                    "recession": 0
                    
                },
                "use_historical_data": True,
                "historical_data_source": "csv"
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
                "model_version": "1m_v1.0",
                "timestamp": "2024-08-01T10:30:00Z",
                "input_date": "1/8/2024",
                "confidence_interval": {
                    "lower_bound": 0.18,
                    "upper_bound": 0.28
                },
                "feature_importance": {
                    "fedfunds": 0.15,
                    "UNRATE": 0.12,
                    "TB3MS": 0.10
                }
                
            }
        }

class ModelStatus3M(BaseModel):   
    """Status information for the 3-month model"""
    odel_loaded: bool
    scaler_loaded: bool
    model_version: str
    last_updated: str
    historical_data_available: bool
    total_features: int