from fastapi import APIRouter, HTTPException
from typing import Dict
import logging

from services.fred_data_service_1m import get_latest_database_row_1m, convert_to_input_features_1m
from services.fred_data_service_3m import get_latest_database_row_3m, convert_to_input_features_3m
from services.fred_data_service_6m import get_latest_database_row_6m, convert_to_input_features_6m
from services.explainability_service_1m import get_explanation_1m, explainability_service_1m
from services.explainability_service_3m import get_explanation_3m, explainability_service_3m
from services.explainability_service_6m import get_explanation_6m, explainability_service_6m

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/explain/1m", response_model=Dict)
async def explain_1m_prediction():
    """Get SHAP and permutation importance explanations for 1-month prediction"""
    try:
        logger.info("🔍 Generating explanations for 1M prediction")
        
        # Get latest data from database
        latest_row = get_latest_database_row_1m()
        if not latest_row:
            raise HTTPException(status_code=404, detail="No data available in database")
        
        # Convert to input features
        features = convert_to_input_features_1m(latest_row)
        
        # Generate explanation
        explanation = get_explanation_1m(features)
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating 1M explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@router.get("/explain/3m", response_model=Dict)
async def explain_3m_prediction():
    """Get SHAP and permutation importance explanations for 3-month prediction"""
    try:
        logger.info("🔍 Generating explanations for 3M prediction")
        
        # Get latest data from database
        latest_row = get_latest_database_row_3m()
        if not latest_row:
            raise HTTPException(status_code=404, detail="No data available in database")
        
        # Convert to input features
        features = convert_to_input_features_3m(latest_row)
        
        # Generate explanation
        explanation = get_explanation_3m(features)
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating 3M explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@router.get("/explain/6m", response_model=Dict)
async def explain_6m_prediction():
    """Get SHAP and permutation importance explanations for 6-month prediction"""
    try:
        logger.info("🔍 Generating explanations for 6M prediction")
        
        # Get latest data from database
        latest_row = get_latest_database_row_6m()
        if not latest_row:
            raise HTTPException(status_code=404, detail="No data available in database")
        
        # Convert to input features
        features = convert_to_input_features_6m(latest_row)
        
        # Generate explanation
        explanation = get_explanation_6m(features)
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating 6M explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@router.get("/explain/all", response_model=Dict)
async def explain_all_predictions():
    """Get SHAP and permutation importance explanations for all timeframes (1m, 3m, 6m)"""
    try:
        logger.info("🔍 Generating explanations for all predictions")
        
        # Get all features
        latest_row_1m = get_latest_database_row_1m()
        latest_row_3m = get_latest_database_row_3m()
        latest_row_6m = get_latest_database_row_6m()
        
        if not latest_row_1m or not latest_row_3m or not latest_row_6m:
            raise HTTPException(status_code=404, detail="Missing data in database")
        
        features_1m = convert_to_input_features_1m(latest_row_1m)
        features_3m = convert_to_input_features_3m(latest_row_3m)
        features_6m = convert_to_input_features_6m(latest_row_6m)
        
        # Generate all explanations
        explanation_1m = get_explanation_1m(features_1m)
        explanation_3m = get_explanation_3m(features_3m)
        explanation_6m = get_explanation_6m(features_6m)
        
        return {
            "1m": explanation_1m,
            "3m": explanation_3m,
            "6m": explanation_6m,
            "timestamp": features_1m.current_month_data.observation_date
        }
        
    except Exception as e:
        logger.error(f"Error generating all explanations: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@router.post("/debug/clear-cache", response_model=Dict)
async def clear_explainer_cache():
    """Debug endpoint to clear SHAP explainer cache and force reinitialization"""
    try:
        explainability_service_1m.clear_explainer_cache()
        explainability_service_3m.clear_explainer_cache()
        explainability_service_6m.clear_explainer_cache()
        
        return {
            "message": "SHAP explainer cache cleared for all timeframes",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error clearing explainer cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")
