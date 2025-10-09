from pydantic import BaseModel
from typing import Dict, List, Optional

class FeatureDefinition(BaseModel):
    """Individual feature definition"""
    feature_code: str
    name: str
    description: Optional[str] = None
    min_value: float
    max_value: float
    default_value: float
    is_important: int = 1
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature_code": "fedfunds",
                "name": "Federal Funds Rate",
                "description": "Interest rate set by Federal Reserve",
                "min_value": 0.0,
                "max_value": 10.0,
                "default_value": 4.40,
                "is_important": 1
            }
        }

class ModelFeatureDefinitions(BaseModel):
    """Feature definitions for a specific model (all models use the same unified dataset)"""
    model_period: str  # "1m", "3m", "6m"
    features: List[FeatureDefinition]
    total_features: int
    important_features_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_period": "1m",
                "features": [
                    {
                        "feature_code": "fedfunds",
                        "name": "Federal Funds Rate",
                        "description": "Interest rate set by Federal Reserve",
                        "min_value": 0.0,
                        "max_value": 10.0,
                        "default_value": 4.40,
                        "is_important": 1
                    }
                ],
                "total_features": 29,
                "important_features_count": 29
            }
        }

class AllFeatureDefinitions(BaseModel):
    """Feature definitions for all models (unified dataset shared across all models)"""
    models: Dict[str, ModelFeatureDefinitions]
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "models": {
                    "1m": {
                        "model_period": "1m",
                        "features": [],
                        "total_features": 29,
                        "important_features_count": 29
                    }
                },
                "timestamp": "2025-09-12T10:30:00Z"
            }
        }

class SimulateError(BaseModel):
    """Error response for simulate service"""
    error: str
    details: Optional[str] = None
    model_period: Optional[str] = None