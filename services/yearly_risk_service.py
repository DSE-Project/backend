import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from calendar import month_name

from schemas.yearly_risk_schema import YearlyRiskResponse, MonthlyRiskData, YearlyRiskError
from schemas.forecast_schema_1m import InputFeatures1M, CurrentMonthData1M
from services.forecast_service_1m import predict_1m, initialize_1m_service

# Historical data path
HISTORICAL_DATA_PATH = "data/historical_data_1m.csv"

def get_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability < 0.2:
        return "Low"
    elif probability < 0.4:
        return "Medium"
    elif probability < 0.7:
        return "High"
    else:
        return "Very High"

def format_date_for_prediction(date_obj) -> str:
    """Format date object to string compatible with prediction model"""
    try:
        # Use cross-platform date formatting
        return f"{date_obj.month}/{date_obj.day}/{date_obj.year}"
    except Exception as e:
        raise RuntimeError(f"Failed to format date {date_obj}: {e}")

def load_historical_data() -> pd.DataFrame:
    """Load and prepare historical data"""
    try:
        df = pd.read_csv(HISTORICAL_DATA_PATH, index_col='observation_date', parse_dates=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load historical data: {e}")

def prepare_prediction_data(row: pd.Series, observation_date: str) -> InputFeatures1M:
    """Convert a historical data row to InputFeatures1M format"""
    try:
        # Create CurrentMonthData1M from the row
        current_data = CurrentMonthData1M(
            observation_date=observation_date,
            fedfunds=float(row.get('fedfunds', 0)),
            TB3MS=float(row.get('TB3MS', 0)),
            TB6MS=float(row.get('TB6MS', 0)),
            TB1YR=float(row.get('TB1YR', 0)),
            USTPU=float(row.get('USTPU', 0)),
            USGOOD=float(row.get('USGOOD', 0)),
            SRVPRD=float(row.get('SRVPRD', 0)),
            USCONS=float(row.get('USCONS', 0)),
            MANEMP=float(row.get('MANEMP', 0)),
            USWTRADE=float(row.get('USWTRADE', 0)),
            USTRADE=float(row.get('USTRADE', 0)),
            USINFO=float(row.get('USINFO', 0)),
            UNRATE=float(row.get('UNRATE', 0)),
            UNEMPLOY=float(row.get('UNEMPLOY', 0)),
            CPIFOOD=float(row.get('CPIFOOD', 0)),
            CPIMEDICARE=float(row.get('CPIMEDICARE', 0)),
            CPIRENT=float(row.get('CPIRENT', 0)),
            CPIAPP=float(row.get('CPIAPP', 0)),
            GDP=float(row.get('GDP', 0)),
            REALGDP=float(row.get('REALGDP', 0)),
            PCEPI=float(row.get('PCEPI', 0)),
            PSAVERT=float(row.get('PSAVERT', 0)),
            PSTAX=float(row.get('PSTAX', 0)),
            COMREAL=float(row.get('COMREAL', 0)),
            COMLOAN=float(row.get('COMLOAN', 0)),
            SECURITYBANK=float(row.get('SECURITYBANK', 0)),
            PPIACO=float(row.get('PPIACO', 0)),
            M1SL=float(row.get('M1SL', 0)),
            M2SL=float(row.get('M2SL', 0)),
            recession=int(row.get('recession', 0))
        )
        
        return InputFeatures1M(current_month_data=current_data)
    
    except Exception as e:
        raise RuntimeError(f"Failed to prepare prediction data: {e}")

def analyze_yearly_recession_risk(months_to_analyze: int = 12) -> YearlyRiskResponse:
    """
    Analyze recession risk for the last N months using historical data
    """
    try:
        # Initialize 1M service if not already done
        if not initialize_1m_service():
            raise RuntimeError("Failed to initialize 1M forecasting service")
        
        # Load historical data
        historical_df = load_historical_data()
        
        if len(historical_df) < months_to_analyze + 60:  # Need extra data for LSTM sequence
            raise RuntimeError(f"Insufficient historical data. Need at least {months_to_analyze + 60} records")
        
        # Get the last N records for analysis
        analysis_records = historical_df.tail(months_to_analyze).copy()
        monthly_risks = []
        
        current_timestamp = datetime.now().isoformat()
        
        print(f"Analyzing recession risk for {len(analysis_records)} months...")
        
        for i, (date_index, row) in enumerate(analysis_records.iterrows()):
            try:
                # Convert date to string format using cross-platform method
                observation_date = format_date_for_prediction(date_index)
                
                # For prediction, we need to use historical data up to this point
                # Create a temporary historical dataset that excludes future data
                historical_subset = historical_df.loc[:date_index].iloc[:-1]  # Exclude current record
                
                if len(historical_subset) < 60:  # Need minimum data for LSTM
                    print(f"Skipping {observation_date}: insufficient historical data")
                    continue
                
                # Prepare input features
                input_features = prepare_prediction_data(row, observation_date)
                
                # Make prediction
                prediction_result = predict_1m(input_features)
                
                # Extract month and year
                month = date_index.month
                year = date_index.year
                month_name_str = month_name[month]
                
                # Create monthly risk data
                monthly_risk = MonthlyRiskData(
                    observation_date=observation_date,
                    month=month,
                    year=year,
                    month_name=month_name_str,
                    recession_probability=prediction_result.prob_1m,
                    risk_level=get_risk_level(prediction_result.prob_1m),
                    prediction_timestamp=current_timestamp
                )
                
                monthly_risks.append(monthly_risk)
                print(f"✅ Processed {observation_date}: {prediction_result.prob_1m:.3f} risk")
                
            except Exception as e:
                print(f"❌ Error processing {date_index}: {e}")
                continue
        
        if not monthly_risks:
            raise RuntimeError("No monthly risks could be calculated")
        
        # Calculate summary statistics
        probabilities = [risk.recession_probability for risk in monthly_risks]
        
        # Find highest risk month
        highest_risk_data = max(monthly_risks, key=lambda x: x.recession_probability)
        lowest_risk_data = min(monthly_risks, key=lambda x: x.recession_probability)
        
        # Simple trend analysis
        if len(probabilities) >= 3:
            recent_avg = np.mean(probabilities[-3:])
            earlier_avg = np.mean(probabilities[:3])
            if recent_avg > earlier_avg * 1.1:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        summary_stats = {
            "average_risk": float(np.mean(probabilities)),
            "highest_risk": float(np.max(probabilities)),
            "lowest_risk": float(np.min(probabilities)),
            "highest_risk_month": highest_risk_data.month_name,
            "lowest_risk_month": lowest_risk_data.month_name,
            "standard_deviation": float(np.std(probabilities)),
            "trend": trend
        }
        
        # Analysis period
        analysis_period = {
            "start_date": monthly_risks[0].observation_date,
            "end_date": monthly_risks[-1].observation_date,
            "total_months": len(monthly_risks)
        }
        
        # Model info
        model_info = {
            "model_version": "1m_v1.0",
            "prediction_model": "1-month LSTM",
            "analysis_timestamp": current_timestamp
        }
        
        return YearlyRiskResponse(
            monthly_risks=monthly_risks,
            analysis_period=analysis_period,
            summary_statistics=summary_stats,
            model_info=model_info,
            total_months_analyzed=len(monthly_risks)
        )
        
    except Exception as e:
        raise RuntimeError(f"Yearly risk analysis failed: {e}")

def get_monthly_risk_summary(months: int = 12) -> Dict:
    """Get a quick summary of monthly risks"""
    try:
        result = analyze_yearly_recession_risk(months)
        return {
            "success": True,
            "average_risk": result.summary_statistics["average_risk"],
            "highest_risk": result.summary_statistics["highest_risk"],
            "trend": result.summary_statistics["trend"],
            "months_analyzed": result.total_months_analyzed
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def test_yearly_risk_service():
    """Test the yearly risk service"""
    try:
        print("🧪 Testing yearly risk service...")
        
        # Test with last 6 months for faster testing
        result = analyze_yearly_recession_risk(months_to_analyze=6)
        
        print(f"✅ Analysis completed for {result.total_months_analyzed} months")
        print(f"📊 Average risk: {result.summary_statistics['average_risk']:.3f}")
        print(f"📈 Highest risk: {result.summary_statistics['highest_risk']:.3f} in {result.summary_statistics['highest_risk_month']}")
        print(f"📉 Trend: {result.summary_statistics['trend']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    # Test the service
    test_yearly_risk_service()