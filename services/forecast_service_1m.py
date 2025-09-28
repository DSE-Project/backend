import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime
from typing import Optional
from tensorflow.keras import backend as K
from statsmodels.tsa.seasonal import STL
from schemas.forecast_schema_1m import InputFeatures1M, ForecastResponse1M, CurrentMonthData1M, ModelStatus1M
from services.database_service import db_service

# Model and scaler paths
MODEL_1M_PATH = "ml_models/1m/model_1m.keras"
SCALER_1M_PATH = "ml_models/1m/scaler_1m.pkl"

# Global variables for model and scaler
model_1m = None
scaler_1m = None
historical_data_1m = None
_service_initialized = False

def focal_loss(gamma=2., alpha=0.25):
    """Focal loss function for the model"""
    def loss(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        bce_exp = K.exp(-bce)
        return K.mean(alpha * (1 - bce_exp) ** gamma * bce)
    return loss

def load_model_1m():
    """Load the 1-month recession prediction model"""
    global model_1m
    try:
        model_1m = tf.keras.models.load_model(
            MODEL_1M_PATH, 
            custom_objects={'loss': focal_loss(gamma=2., alpha=0.25)},
            compile=False
        )
        print("✅ 1M model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading 1M model: {e}")
        model_1m = None
        return False

def load_scaler_1m():
    """Load the scaler for 1-month model preprocessing"""
    global scaler_1m
    try:
        scaler_1m = joblib.load(SCALER_1M_PATH)
        print("✅ 1M scaler loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading 1M scaler: {e}")
        scaler_1m = None
        return False

def load_historical_data_1m():
    """Load historical data from Supabase database"""
    global historical_data_1m
    try:
        historical_data_1m = db_service.load_historical_data('historical_data_1m')
        if historical_data_1m is not None:
            print(f"✅ Historical data loaded from database: {len(historical_data_1m)} records")
            return True
        else:
            print("❌ Failed to load historical data from database")
            return False
    except Exception as e:
        print(f"❌ Error loading historical data from database: {e}")
        historical_data_1m = None
        return False

def create_features_vectorized(df):
    """Vectorized feature creation for better performance"""
    # Create lag features
    macro_cols = ['TB3MS', 'fedfunds', 'TB6MS', 'USGOOD']
    needed_lags = {'TB3MS': [6], 'fedfunds': [6], 'TB6MS': [6], 'USGOOD': [12]}
    
    for col in macro_cols:
        if col in df.columns:
            for lag in needed_lags.get(col, []):
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Create rolling features
    rolling_cols = ['fedfunds', 'TB3MS', 'SECURITYBANK', 'COMLOAN']
    for col in rolling_cols:
        if col in df.columns:
            df[f'{col}3'] = df[col].shift(1).rolling(3, min_periods=1).mean()
    
    return df

def preprocess_features_1m(features: InputFeatures1M, lookback_points=120) -> tuple:
    """
    Preprocess input features for 1-month model
    This will combine current month data with historical data for LSTM input
    """
    try:
        # Load historical data if not already loaded
        if historical_data_1m is None:
            if not load_historical_data_1m():
                raise RuntimeError("Failed to load historical data")
        
        # Keep only the last lookback_points for memory efficiency
        historical_data = historical_data_1m.tail(lookback_points+1).iloc[:-1].copy()
        
        # Convert all to float32 for memory efficiency
        historical_data = historical_data.astype(np.float32)
        
        print(f"Using only last {len(historical_data)} data points for processing")
        
        # Convert Pydantic model to dict
        current_data = features.current_month_data.dict()
        
        # Process current month data - match all historical columns
        historical_cols = historical_data.columns.tolist()
        filtered_current_data = {k: v for k, v in current_data.items() 
                               if k in historical_cols + ['observation_date']}
        
        current_month_df = pd.DataFrame([filtered_current_data], 
                                      index=[pd.to_datetime(filtered_current_data["observation_date"])])
        current_month_df = current_month_df.drop(columns=['observation_date'])
        
        # Combine and process
        df = pd.concat([historical_data, current_month_df], copy=False)
        df = create_features_vectorized(df)
        
        # STL decomposition for fedfunds
        if 'fedfunds' in df.columns and len(df) >= 24:
            try:
                stl = STL(df['fedfunds'], period=12, robust=True)
                res = stl.fit()
                df['fedfunds_trend'] = res.trend
                df['fedfunds_seasonal'] = res.seasonal
                df['fedfunds_resid'] = res.resid
            except:
                print("Warning: STL decomposition failed for fedfunds, using fallback")
                df['fedfunds_trend'] = df['fedfunds'].rolling(12, min_periods=1).mean()
                df['fedfunds_seasonal'] = 0
                df['fedfunds_resid'] = df['fedfunds'] - df['fedfunds_trend']
        
        # STL decomposition for UNRATE
        if 'UNRATE' in df.columns and len(df) >= 24:
            try:
                stl = STL(df['UNRATE'], period=12, robust=True)
                res = stl.fit()
                df['UNRATE_trend'] = res.trend
                df['UNRATE_seasonal'] = res.seasonal
                df['UNRATE_resid'] = res.resid
            except:
                print("Warning: STL decomposition failed for UNRATE, using fallback")
                df['UNRATE_trend'] = df['UNRATE'].rolling(12, min_periods=1).mean()
                df['UNRATE_seasonal'] = 0
                df['UNRATE_resid'] = df['UNRATE'] - df['UNRATE_trend']
        
        # Final processing
        df = df.dropna()
        cols_to_drop = ['recession'] if 'recession' in df.columns else []
        feature_cols = [c for c in df.columns if c not in cols_to_drop]
        
        X = df[feature_cols].values.astype(np.float32)
        X_scaled = scaler_1m.transform(X).astype(np.float32)
        
        return X_scaled, feature_cols, df
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise RuntimeError(f"Preprocessing failed: {e}")

def initialize_1m_service():
    """Initialize the 1-month forecasting service"""
    global _service_initialized
    
    # Return True if already initialized
    if _service_initialized:
        return True
    
    print("🚀 Initializing 1M forecasting service...")
    
    model_loaded = load_model_1m()
    scaler_loaded = load_scaler_1m()
    data_loaded = load_historical_data_1m()
    
    if model_loaded and scaler_loaded and data_loaded:
        print("✅ 1M service initialized successfully")
        _service_initialized = True
        return True
    else:
        print("❌ 1M service initialization failed")
        _service_initialized = False
        return False

def predict_1m(features: InputFeatures1M, seq_length=60, threshold=0.8) -> ForecastResponse1M:
    """Make 1-month recession probability prediction"""
    # Auto-initialize if not done
    if not _service_initialized:
        if not initialize_1m_service():
            raise RuntimeError("Failed to initialize 1M forecasting service")
    
    if model_1m is None or scaler_1m is None:
        raise RuntimeError("1M model or scaler is not loaded")
    
    try:
        # Preprocess features
        X_scaled, feature_cols, df = preprocess_features_1m(features)
        
        # Check if we have enough data for sequence
        if len(X_scaled) < seq_length:
            raise RuntimeError(f'Insufficient data. Need at least {seq_length} data points after preprocessing, got {len(X_scaled)}')
        
        # Create sequence for LSTM (only the last sequence we need)
        latest_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
        
        # Make prediction
        prediction = model_1m.predict(latest_sequence, verbose=0)
        prob_1m = float(prediction[0][0])
        
        # Binary prediction based on threshold
        binary_prediction = int(prob_1m > threshold)
        
        # Create response with additional metadata
        response = ForecastResponse1M(
            prob_1m=prob_1m,
            model_version="1m_v1.0",
            timestamp=datetime.now().isoformat(),
            input_date=features.current_month_data.observation_date,
            confidence_interval={
                "threshold_used": threshold,
                "binary_prediction": binary_prediction,
                "prediction_text": 'Recession Expected' if binary_prediction else 'No Recession Expected'
            },
            feature_importance={
                "data_points_used": len(df),
                "feature_count": len(feature_cols),
                "sequence_length": seq_length
            }
        )
        
        return response
    
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

def get_model_info_1m() -> ModelStatus1M:
    """Get information about the 1-month model"""
    return ModelStatus1M(
        model_loaded=model_1m is not None,
        scaler_loaded=scaler_1m is not None,
        model_version="1m_v1.0",
        last_updated="2025-01-01",  # Update this when you retrain
        historical_data_available=historical_data_1m is not None,
        total_features=len(historical_data_1m.columns) if historical_data_1m is not None else 0
    )

def test_prediction_1m():
    """Test function to verify the service works"""
    # Create test input using the same data from your example
    test_current_data = CurrentMonthData1M(
        observation_date="1/2/2025",
        fedfunds=4.40,
        TB3MS=4.22,
        TB6MS=4.14,
        TB1YR=4.05,
        USTPU=30000,
        USGOOD=21670,
        SRVPRD=13700,
        USCONS=9000,
        MANEMP=12800,
        USWTRADE=7602,
        USTRADE=15602,
        USINFO=3200,
        UNRATE=4.0,
        UNEMPLOY=6600,
        CPIFOOD=300,
        CPIMEDICARE=600,
        CPIRENT=1500,
        CPIAPP=200,
        GDP=25000,
        REALGDP=21000,
        PCEPI=140,
        PSAVERT=5.0,
        PSTAX=1100,
        COMREAL=220000,
        COMLOAN=-0.3,
        SECURITYBANK=-2.0,
        PPIACO=270,
        M1SL=20000,
        M2SL=150000,
        recession=0
    )
    
    test_features = InputFeatures1M(current_month_data=test_current_data)
    
    import time
    start_time = time.time()
    
    try:
        result = predict_1m(test_features)
        total_time = time.time() - start_time
        
        print("=== 1M RECESSION PREDICTION RESULTS ===")
        print(f"Recession Probability: {result.prob_1m:.4f}")
        print(f"Model Version: {result.model_version}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Input Date: {result.input_date}")
        print(f"Additional Info: {result.confidence_interval}")
        print(f"Feature Info: {result.feature_importance}")
        print(f"Total Time: {total_time:.3f} seconds")
        
        return result
    except Exception as e:
        print(f"❌ Test prediction failed: {e}")
        return None

if __name__ == "__main__":
    # Test the service
    print("Testing 1M forecasting service...")
    
    # Initialize service
    if initialize_1m_service():
        # Run test prediction
        test_prediction_1m()
    else:
        print("Service initialization failed - cannot run test")
    
    # Test model info
    info = get_model_info_1m()
    print(f"📊 Model info: {info}")