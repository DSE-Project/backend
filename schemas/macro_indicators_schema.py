from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

class MacroIndicatorResponse(BaseModel):
    """Schema for individual macroeconomic indicator"""
    name: str = Field(..., description="Full name of the indicator")
    symbol: str = Field(..., description="Symbol/code of the indicator")
    current_value: float = Field(..., description="Current value of the indicator")
    previous_value: Optional[float] = Field(None, description="Previous period value")
    change_value: Optional[float] = Field(None, description="Change from previous period")
    change_percent: Optional[float] = Field(None, description="Percentage change from previous period")
    unit: str = Field(..., description="Unit of measurement (%, Index, etc.)")
    last_updated: str = Field(..., description="ISO timestamp of last update")
    trend: str = Field(..., description="Trend direction: increasing, decreasing, stable")
    data_source: str = Field(..., description="Source of the data")

class MacroIndicatorsSnapshot(BaseModel):
    """Schema for complete macroeconomic indicators snapshot"""
    indicators: Dict[str, MacroIndicatorResponse] = Field(..., description="Dictionary of all indicators")
    last_updated: str = Field(..., description="ISO timestamp of when snapshot was generated")
    data_sources: list[str] = Field(..., description="List of data sources used")

class MacroIndicatorsAPIResponse(BaseModel):
    """Main API response schema"""
    success: bool = Field(True, description="Whether the request was successful")
    data: MacroIndicatorsSnapshot = Field(..., description="The indicators data")
    message: str = Field(..., description="Response message")
