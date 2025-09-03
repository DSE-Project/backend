from pydantic import BaseModel

class InputFeatures1M(BaseModel):
    """Features for 1-month recession probability model"""
    # TODO: Add actual features used by 1m model
    pass

class ForecastResponse1M(BaseModel):
    prob_1m: float
    model_version: str
    timestamp: str