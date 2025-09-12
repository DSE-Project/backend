from pydantic import BaseModel
from typing import Optional

class CurrentMonthData6M(BaseModel):
    """Current month economic indicators for 6-month recession prediction"""
    observation_date: str
    PSTAX:float
    USWTRADE: float
    MANEMP:float
    CPIAPP:float
    CSUSHPISA:float
    ICSA: float
    fedfunds:float
    BBKMLEIX: float
    TB3MS:float
    USINFO:float
    PPIACO : float
    CPIMEDICARE: float
    UNEMPLOY:float
    TB1YR: float
    USGOOD:float
    CPIFOOD: float
    UMCSENT: float
    SRVPRD: float
    GDP: float 
    INDPRO:float
    recession:int


class InputFeatures6M(BaseModel):
    """Features for 6-month recession probability model"""
    current_month_data: CurrentMonthData6M
    use_historical_data: bool = True  # Flag to indicate if historical data should be loaded
    historical_data_source: Optional[str] = "csv"  # "csv" or "database"

    class Config:
        json_schema_extra = {
            "example": {
                "current_month_data": {
                    "observation_date": "1/8/2024",
                    "PSTAX":3100.43,
                    "USWTRADE": 6155.9,
                    "MANEMP":12843,
                    "CPIAPP":131.327,
                    "CSUSHPISA":322.345,
                    "ICSA" : 237700,
                    "fedfunds":5.33,
                    "BBKMLEIX" : 1.49545,
                    "TB3MS":5.15,
                    "USINFO":2916,
                    "PPIACO" : 258.735,
                    "CPIMEDICARE": 565.857,
                    "UNEMPLOY": 7209,
                    "TB1YR": 4.52,
                    "USGOOD":21682,
                    "CPIFOOD": 305.999,
                    "UMCSENT" : 64.9,
                    "SRVPRD": 136419,
                    "GDP":29502.54 ,
                    "INDPRO" : 103.55,
                    "recession" :0
                    
                },
                "use_historical_data": True,
                "historical_data_source": "csv"
            }
        }
    

class ForecastResponse6M(BaseModel):
    prob_6m: float  # Recession probability (0.0 to 1.0)
    model_version: str  # Model version identifier
    timestamp: str  # Prediction timestamp
    input_date: str  # Date of the input data
    confidence_interval: Optional[dict] = None  # Optional confidence intervals
    feature_importance: Optional[dict] = None  # Optional feature importance scores
    

    class Config:
        json_schema_extra = {
            "example": {
                "prob_6m": 0.23,
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

class ModelStatus6M(BaseModel):   
    """Status information for the 6-month model"""
    model_loaded: bool
    scaler_loaded: bool
    model_version: str
    last_updated: str
    historical_data_available: bool
    total_features: int