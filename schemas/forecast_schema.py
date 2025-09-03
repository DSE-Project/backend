from pydantic import BaseModel

class InputFeatures(BaseModel):
    unemployment_rate: float
    inflation_rate: float
    gdp_growth: float
    interest_rate: float
   

class ForecastResponse(BaseModel):
    prob_1m: float
    prob_3m: float
    prob_6m: float