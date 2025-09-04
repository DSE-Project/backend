from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class CurrentMonthData1M(BaseModel):
    """Current month economic indicators for 1-month recession prediction"""
    observation_date: str  # Format: "MM/DD/YYYY"
    fedfunds: float  # Federal funds rate
    TB3MS: float  # 3-Month Treasury Constant Maturity Rate
    TB6MS: float  # 6-Month Treasury Constant Maturity Rate
    TB1YR: float  # 1-Year Treasury Constant Maturity Rate
    USTPU: float  # US Total Private Units
    USGOOD: float  # US Goods
    SRVPRD: float  # Services Production
    USCONS: float  # US Construction
    MANEMP: float  # Manufacturing Employment
    USWTRADE: float  # US Wholesale Trade
    USTRADE: float  # US Trade
    USINFO: float  # US Information
    UNRATE: float  # Unemployment Rate
    UNEMPLOY: float  # Unemployment Level
    CPIFOOD: float  # CPI Food
    CPIMEDICARE: float  # CPI Medicare
    CPIRENT: float  # CPI Rent
    CPIAPP: float  # CPI Apparel
    GDP: float  # Gross Domestic Product
    REALGDP: float  # Real GDP
    PCEPI: float  # Personal Consumption Expenditures Price Index
    PSAVERT: float  # Personal Saving Rate
    PSTAX: float  # Personal Tax
    COMREAL: float  # Commercial Real Estate
    COMLOAN: float  # Commercial Loans
    SECURITYBANK: float  # Security Bank
    PPIACO: float  # Producer Price Index All Commodities
    M1SL: float  # M1 Money Stock
    M2SL: float  # M2 Money Stock
    recession: int  # Recession indicator (0 or 1)

class InputFeatures1M(BaseModel):
    """Input features for 1-month recession probability model"""
    current_month_data: CurrentMonthData1M
    use_historical_data: bool = True  # Flag to indicate if historical data should be loaded
    historical_data_source: Optional[str] = "csv"  # "csv" or "database"
    
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
                "historical_data_source": "csv"
            }
        }

class ForecastResponse1M(BaseModel):
    """Response for 1-month recession probability prediction"""
    prob_1m: float  # Recession probability (0.0 to 1.0)
    model_version: str  # Model version identifier
    timestamp: str  # Prediction timestamp
    input_date: str  # Date of the input data
    confidence_interval: Optional[dict] = None  # Optional confidence intervals
    feature_importance: Optional[dict] = None  # Optional feature importance scores
    
    class Config:
        json_schema_extra = {
            "example": {
                "prob_1m": 0.23,
                "model_version": "1m_v1.0",
                "timestamp": "2025-01-02T10:30:00Z",
                "input_date": "1/2/2025",
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

class ModelStatus1M(BaseModel):
    """Status information for the 1-month model"""
    model_loaded: bool
    scaler_loaded: bool
    model_version: str
    last_updated: str
    historical_data_available: bool
    total_features: int