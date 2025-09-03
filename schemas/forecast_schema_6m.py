from pydantic import BaseModel

class InputFeatures6M(BaseModel):
    """Features for 6-month recession probability model"""
    # TODO: Add actual features used by 6m model
    pass

class ForecastResponse6M(BaseModel):
    prob_6m: float
    model_version: str
    timestamp: str