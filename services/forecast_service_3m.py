import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from datetime import datetime

from schemas.forecast_schema_3m import InputFeatures3M, ForecastResponse3M, CurrentMonthData3M, ModelStatus3M
import joblib
from tensorflow.keras import backend as K
from services.database_service import db_service

# Model and scaler paths
MODEL_3M_PATH = "ml_models/3m/model_3_months.keras"
SCALER_3M_PATH = "ml_models/3m/scaler_3.pkl"

# Global variables for model and scaler
model_3m = None
scaler_3m = None
historical_data_3m = None
_service_initialized_3m = False

def focal_loss(gamma=2., alpha=0.25):
    """Focal loss function for the model"""
    def loss(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        bce_exp = K.exp(-bce)
        return K.mean(alpha * (1 - bce_exp) ** gamma * bce)
    return loss

def load_model_3m():
    """Load the 3-month recession prediction model"""
    global model_3m
    try: 
        model_3m = tf.keras.models.load_model(
            MODEL_3M_PATH, 
            custom_objects={'loss': focal_loss(gamma=2., alpha=0.25)},
            compile=False
        )
        print("✅ 3M model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading 3M model: {e}")
        model_3m = None
        return False

def load_scaler_3m():
    """Load the scaler for 3-month model preprocessing"""
    global scaler_3m
    try:
        scaler_3m = joblib.load(SCALER_3M_PATH)
        print("✅ 3M scaler loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading 3M scaler: {e}")
        scaler_3m = None
        return False
    
def load_historical_data_3m():
    """Load historical data from Supabase database"""
    global historical_data_3m
    try:
        historical_data_3m = db_service.load_historical_data('historical_data_3m')
        if historical_data_3m is not None:
            print(f"✅ Historical data loaded from database: {len(historical_data_3m)} records")
            return True
        else:
            print("❌ Failed to load historical data from database")
            return False
    except Exception as e:
        print(f"❌ Error loading historical data from database: {e}")
        historical_data_3m = None
        return False    

def preprocess_features_3m(features: InputFeatures3M, lookback_points=120) -> tuple:
    """Preprocess input features for 3-month model"""
    try:
        # Load historical data if not already loaded
        if historical_data_3m is None:
            if not load_historical_data_3m():
                raise RuntimeError("Failed to load historical data")
        # Keep only the last lookback_points for memory efficiency

        historical_data = historical_data_3m.tail(lookback_points+1).iloc[:-1].copy()
        
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
        
        df = pd.concat([historical_data, current_month_df], copy=False)

        #print("Current month df columns:", current_month_df.columns)
        #print("Historical data columns:", historical_data.columns)
        #print("Final df shape after concat:", df.shape)
        #print("Tail of df:", df.tail(3))

        # Final processing
        df = df.dropna()
        cols_to_drop = ['recession'] if 'recession' in df.columns else []
        feature_cols = [c for c in df.columns if c not in cols_to_drop]
        
        X = df[feature_cols].values.astype(np.float32)
        X_scaled = scaler_3m.transform(X).astype(np.float32)

        return X_scaled, feature_cols, df
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise RuntimeError(f"Preprocessing failed: {e}")

def initialize_3m_service():
    """Initialize the 3 months forecasing service"""
    global _service_initialized_3m

    # Return True if already initialized
    if _service_initialized_3m:
        return True
    
    print("🚀 Initializing 3M forecasting service...")

    model_loaded = load_model_3m()
    scaler_loaded = load_scaler_3m()
    data_loaded = load_historical_data_3m()

    if model_loaded and scaler_loaded and data_loaded:
        print("✅ 3M service initialized successfully")
        _service_initialized_3m = True
        return True
    else:
        print("❌ 3M service initialization failed")
        _service_initialized_3m = False
        return False

def predict_3m(features: InputFeatures3M, seq_length=60, threshold=0.3) -> ForecastResponse3M:
    """Make 3-month recession probability prediction"""
    if not _service_initialized_3m:
        if not initialize_3m_service():
            raise RuntimeError("Failed to initialize 3M forecasting service")
    
    if model_3m is None or scaler_3m is None:
        raise RuntimeError("3M model or scaler is not loaded")
    
    try: 
        # Preprocess features
        X_scaled, feature_cols, df = preprocess_features_3m(features)
        
        # Check if we have enough data for sequence
        if len(X_scaled) < seq_length:
            raise RuntimeError(f'Insufficient data. Need at least {seq_length} data points after preprocessing, got {len(X_scaled)}')
        # Create sequence for LSTM (only the last sequence we need)
        latest_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
        
        # Make prediction
        prediction = model_3m.predict(latest_sequence, verbose=0)
        prob_3m = float(prediction[0][0])

        # Binary prediction based on threshold
        binary_prediction = int(prob_3m > threshold)

        
        # Create response with additional metadata
        response = ForecastResponse3M(
            prob_3m=prob_3m,
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

def get_model_info_3m() -> ModelStatus3M:
    """Get information about the 3-month model"""
    return ModelStatus3M(
        model_loaded=model_3m is not None,
        scaler_loaded=scaler_3m is not None,
        model_version="1m_v1.0",
        last_updated="2025-01-01",  # Update this when you retrain
        historical_data_available=historical_data_3m is not None,
        total_features=len(historical_data_3m.columns) if historical_data_3m is not None else 0
    )

def test_prediction_3m():
    """Test function to verify the service works"""
    test_current_data = CurrentMonthData3M(
        observation_date="1/8/2024",
        ICSA=237800,
        CPIMEDICARE=565.759,
        USWTRADE=6147.9,
        BBKMLEIX=1.5062454,
        COMLOAN=0.5,
        UMCSENT=63.5,
        MANEMP=12845,
        fedfunds=5.33,
        PSTAX=3074.386,
        USCONS=8221,
        USGOOD=21683,
        USINFO=2960,
        CPIAPP=131.124,
        CSUSHPISA=322.425,
        SECURITYBANK=-1.8,
        SRVPRD=136409,
        INDPRO=102.8692,
        TB6MS=4.97,
        UNEMPLOY=8153,
        USTPU=28911,
        recession=0,

    )
    test_features = InputFeatures3M(current_month_data=test_current_data)

    import time
    start_time = time.time()

    try:
        result = predict_3m(test_features)
        total_time = time.time() - start_time
        
        print("=== 3M RECESSION PREDICTION RESULTS ===")
        print(f"Recession Probability: {result.prob_3m:.4f}")
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
    if initialize_3m_service():
        # Run test prediction
        test_prediction_3m()
    else:
        print("Service initialization failed - cannot run test")
    
    # Test model info
    info = get_model_info_3m()
    print(f"📊 Model info: {info}")