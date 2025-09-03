from pydantic import BaseModel

class InputFeatures1M(BaseModel):
    """Features for 1-month recession probability model (df1 dataset)"""
    ICSA: float
    USWTRADE: float
    UMCSENT: float
    BBKMLEIX: float
    TB6MS: float
    SRVPRD: float
    COMLOAN: float
    CPIMEDICARE: float
    USINFO: float
    USGOOD: float
    PSAVERT: float
    UNEMPLOY: float
    PPIACO: float
    fedfunds: float
    CPIAPP: float
    TCU: float
    TB3MS: float
    SECURITYBANK: float
    CSUSHPISA: float
    MANEMP: float

class InputFeatures3M(BaseModel):
    """Features for 3-month recession probability model (df2 dataset)"""
    ICSA: float
    CPIMEDICARE: float
    USWTRADE: float
    BBKMLEIX: float
    COMLOAN: float
    UMCSENT: float
    MANEMP: float
    fedfunds: float
    PSTAX: float
    USCONS: float
    USGOOD: float
    USINFO: float
    CPIAPP: float
    CSUSHPISA: float
    SECURITYBANK: float
    SRVPRD: float
    INDPRO: float
    TB6MS: float
    UNEMPLOY: float
    USTPU: float

class InputFeatures6M(BaseModel):
    """Features for 6-month recession probability model (df2 dataset)"""
    PSTAX: float
    USWTRADE: float
    MANEMP: float
    CPIAPP: float
    CSUSHPISA: float
    ICSA: float
    fedfunds: float
    BBKMLEIX: float
    TB3MS: float
    USINFO: float
    PPIACO: float
    CPIMEDICARE: float
    UNEMPLOY: float
    TB1YR: float
    USGOOD: float
    CPIFOOD: float
    UMCSENT: float
    SRVPRD: float
    GDP: float
    INDPRO: float

class ForecastResponse(BaseModel):
    prob_1m: float
    prob_3m: float
    prob_6m: float

class AllInputFeatures(BaseModel):
    """Combined input features for all models"""
    features_1m: InputFeatures1M
    features_3m: InputFeatures3M
    features_6m: InputFeatures6M