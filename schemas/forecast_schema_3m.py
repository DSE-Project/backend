from pydantic import BaseModel

class InputFeatures3M(BaseModel):
    """Features for 3-month recession probability model"""
    # TODO: Add actual features used by 3m model
    pass

class ForecastResponse3M(BaseModel):
    prob_3m: float
    model_version: str
    timestamp: str