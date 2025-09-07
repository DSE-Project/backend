from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class MonthlyRiskData(BaseModel):
    """Monthly recession risk data point"""
    observation_date: str  # Format: "MM/DD/YYYY"
    month: int  # Month number (1-12)
    year: int  # Year
    month_name: str  # Month name (e.g., "January")
    recession_probability: float  # Recession probability (0.0 to 1.0)
    risk_level: str  # "Low", "Medium", "High", "Very High"
    prediction_timestamp: str  # When this prediction was made

class YearlyRiskResponse(BaseModel):
    """Response for yearly recession risk analysis"""
    monthly_risks: List[MonthlyRiskData]
    analysis_period: dict  # Start and end dates
    summary_statistics: dict  # Average risk, highest risk month, etc.
    model_info: dict  # Model version and metadata
    total_months_analyzed: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "monthly_risks": [
                    {
                        "observation_date": "1/1/2024",
                        "month": 1,
                        "year": 2024,
                        "month_name": "January",
                        "recession_probability": 0.15,
                        "risk_level": "Low",
                        "prediction_timestamp": "2025-01-02T10:30:00Z"
                    },
                    {
                        "observation_date": "2/1/2024",
                        "month": 2,
                        "year": 2024,
                        "month_name": "February",
                        "recession_probability": 0.23,
                        "risk_level": "Medium",
                        "prediction_timestamp": "2025-01-02T10:30:00Z"
                    }
                ],
                "analysis_period": {
                    "start_date": "1/1/2024",
                    "end_date": "12/1/2024"
                },
                "summary_statistics": {
                    "average_risk": 0.18,
                    "highest_risk": 0.35,
                    "lowest_risk": 0.08,
                    "highest_risk_month": "September",
                    "trend": "increasing"
                },
                "model_info": {
                    "model_version": "1m_v1.0",
                    "prediction_model": "1-month LSTM"
                },
                "total_months_analyzed": 12
            }
        }

class YearlyRiskError(BaseModel):
    """Error response for yearly risk analysis"""
    error: str
    details: Optional[str] = None
    suggestions: Optional[List[str]] = None