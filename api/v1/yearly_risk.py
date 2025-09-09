from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from schemas.yearly_risk_schema import YearlyRiskResponse, YearlyRiskError
from services.yearly_risk_service import analyze_yearly_recession_risk, get_monthly_risk_summary

router = APIRouter()

@router.get("/yearly-risk", response_model=YearlyRiskResponse)
async def get_yearly_recession_risk(
    months: Optional[int] = Query(
        default=12, 
        ge=1, 
        le=24, 
        description="Number of months to analyze (1-24)"
    )
):
    """
    Get recession risk analysis for the specified number of months
    
    This endpoint analyzes historical data to show recession probability 
    trends over the specified time period using the 1-month prediction model.
    
    - **months**: Number of months to analyze (default: 12, max: 24)
    """
    try:
        result = analyze_yearly_recession_risk(months_to_analyze=months)
        return result
        
    except RuntimeError as e:
        if "insufficient" in str(e).lower():
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data: {str(e)}"
            )
        elif "initialize" in str(e).lower():
            raise HTTPException(
                status_code=503, 
                detail="Forecasting service unavailable. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Analysis failed: {str(e)}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during yearly risk analysis: {str(e)}"
        )

@router.get("/yearly-risk/summary")
async def get_yearly_risk_summary(
    months: Optional[int] = Query(
        default=12, 
        ge=1, 
        le=24, 
        description="Number of months to analyze (1-24)"
    )
):
    """
    Get a quick summary of yearly recession risk
    
    Returns key statistics without detailed monthly breakdown.
    Faster than the full yearly-risk endpoint.
    """
    try:
        summary = get_monthly_risk_summary(months)
        
        if not summary.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=summary.get("error", "Unknown error occurred")
            )
        
        return {
            "months_analyzed": summary["months_analyzed"],
            "average_risk": summary["average_risk"],
            "highest_risk": summary["highest_risk"],
            "risk_trend": summary["trend"],
            "analysis_timestamp": "2025-01-02T10:30:00Z"  # You might want to make this dynamic
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate risk summary: {str(e)}"
        )

@router.get("/yearly-risk/test")
async def test_yearly_risk_endpoint():
    """
    Test endpoint to verify the yearly risk service is working
    
    Runs a quick test with 6 months of data to verify functionality.
    """
    try:
        from services.yearly_risk_service import test_yearly_risk_service
        
        result = test_yearly_risk_service()
        
        if result:
            return {
                "status": "success",
                "message": "Yearly risk service is working correctly",
                "test_results": {
                    "months_analyzed": result.total_months_analyzed,
                    "average_risk": result.summary_statistics["average_risk"],
                    "trend": result.summary_statistics["trend"]
                }
            }
        else:
            return {
                "status": "failed",
                "message": "Yearly risk service test failed"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Test failed: {str(e)}"
        )

# Health check for the yearly risk service
@router.get("/yearly-risk/health")
async def yearly_risk_health_check():
    """Health check for yearly risk service dependencies"""
    try:
        from services.forecast_service_1m import get_model_info_1m
        from services.yearly_risk_service import load_historical_data
        
        # Check 1M model status
        model_status = get_model_info_1m()
        
        # Check historical data availability
        historical_data = load_historical_data()
        
        return {
            "status": "healthy",
            "dependencies": {
                "1m_model_loaded": model_status.model_loaded,
                "1m_scaler_loaded": model_status.scaler_loaded,
                "historical_data_available": len(historical_data) > 0,
                "historical_records_count": len(historical_data)
            },
            "capabilities": {
                "max_months_analysis": min(24, len(historical_data) - 60) if len(historical_data) > 60 else 0
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }