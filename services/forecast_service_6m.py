import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from datetime import datetime

from schemas.forecast_schema_6m import InputFeatures6M, ForecastResponse6M, CurrentMonthData6M, ModelStatus6M
import joblib
from tensorflow.keras import backend as K
from services.database_service import db_service

# Model and scaler paths
MODEL_6M_PATH = "ml_models/6m/model_6_months.keras"
SCALER_6M_PATH = "ml_models/6m/scaler_6.pkl"
HISTORICAL_DATA_PATH_6M = "data/historical_data_6m.csv"

# Global variables for model and scaler
model_6m = None
scaler_6m = None
historical_data_6m = None
_service_initialized_6m = False

def focal_loss(gamma=2., alpha=0.25):
    """Focal loss function for the model"""
    def loss(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        bce_exp = K.exp(-bce)
        return K.mean(alpha * (1 - bce_exp) ** gamma * bce)
    return loss

def load_model_6m():
    """Load the 6-month recession prediction model"""
    global model_6m
    try: 
        model_6m = tf.keras.models.load_model(
            MODEL_6M_PATH, 
            custom_objects={'loss': focal_loss(gamma=2., alpha=0.25)},
            compile=False
        )
        print("✅ 6M model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading 6M model: {e}")
        model_6m = None
        return False

def load_scaler_6m():
    """Load the scaler for 6-month model preprocessing"""
    global scaler_6m
    try:
        scaler_6m = joblib.load(SCALER_6M_PATH)
        print("✅ 6M scaler loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading 6M scaler: {e}")
        scaler_6m = None
        return False
    
def load_historical_data_6m():
    """Load historical data from Supabase database"""
    global historical_data_6m
    try:
        historical_data_6m = db_service.load_historical_data('historical_data_6m')
        if historical_data_6m is not None:
            print(f"✅ Historical data loaded from database: {len(historical_data_6m)} records")
            return True
        else:
            print("❌ Failed to load historical data from database")
            return False
    except Exception as e:
        print(f"❌ Error loading historical data from database: {e}")
        historical_data_6m = None
        return False    

def preprocess_features_6m(features: InputFeatures6M, lookback_points=120) -> tuple:
    """Preprocess input features for 6-month model"""
    try:
        # Load historical data if not already loaded
        if historical_data_6m is None:
            if not load_historical_data_6m():
                raise RuntimeError("Failed to load historical data")
        # Keep only the last lookback_points for memory efficiency

        historical_data = historical_data_6m.tail(lookback_points).copy()
        
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
        X_scaled = scaler_6m.transform(X).astype(np.float32)

        return X_scaled, feature_cols, df
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise RuntimeError(f"Preprocessing failed: {e}")

def initialize_6m_service():
    """Initialize the 6 months forecasing service"""
    global _service_initialized_6m

    # Return True if already initialized
    if _service_initialized_6m:
        return True
    
    print("🚀 Initializing 6M forecasting service...")

    model_loaded = load_model_6m()
    scaler_loaded = load_scaler_6m()
    data_loaded = load_historical_data_6m()

    if model_loaded and scaler_loaded and data_loaded:
        print("✅ 6M service initialized successfully")
        _service_initialized_6m = True
        return True
    else:
        print("❌ 6M service initialization failed")
        _service_initialized_6m = False
        return False

def predict_6m(features: InputFeatures6M, seq_length=60, threshold=0.3) -> ForecastResponse6M:
    """Make 6-month recession probability prediction"""
    if not _service_initialized_6m:
        if not initialize_6m_service():
            raise RuntimeError("Failed to initialize 6M forecasting service")
    
    if model_6m is None or scaler_6m is None:
        raise RuntimeError("6M model or scaler is not loaded")
    
    try: 
        # Preprocess features
        X_scaled, feature_cols, df = preprocess_features_6m(features)
        
        # Check if we have enough data for sequence
        if len(X_scaled) < seq_length:
            raise RuntimeError(f'Insufficient data. Need at least {seq_length} data points after preprocessing, got {len(X_scaled)}')
        # Create sequence for LSTM (only the last sequence we need)
        latest_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
        
        # Make prediction
        prediction = model_6m.predict(latest_sequence, verbose=0)
        prob_6m = float(prediction[0][0])

        # Binary prediction based on threshold
        binary_prediction = int(prob_6m > threshold)

        
        # Create response with additional metadata
        response = ForecastResponse6M(
            prob_6m=prob_6m,
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

def get_model_info_6m() -> ModelStatus6M:
    """Get information about the 3-month model"""
    return ModelStatus6M(
        model_loaded=model_6m is not None,
        scaler_loaded=scaler_6m is not None,
        model_version="1m_v1.0",
        last_updated="2025-01-01",  # Update this when you retrain
        historical_data_available=historical_data_6m is not None,
        total_features=len(historical_data_6m.columns) if historical_data_6m is not None else 0
    )

def test_prediction_6m():
    """Test function to verify the service works"""
    test_current_data = CurrentMonthData6M(
        observation_date="1/8/2024",
        PSTAX= 3100.43,
        USWTRADE = 6155.9,
        MANEMP =12843,
        CPIAPP = 131.327,
        CSUSHPISA = 322.345,
        ICSA = 237700,
        fedfunds = 5.33,
        BBKMLEIX = 1.49545,
        TB3MS = 5.15,
        USINFO = 2916,
        PPIACO = 258.735,
        CPIMEDICARE=565.857,
        UNEMPLOY = 7209,
        TB1YR= 4.52,
        USGOOD= 21682,
        CPIFOOD= 305.999,
        UMCSENT = 64.9,
        SRVPRD = 136419,
        GDP = 29502.54,
        INDPRO = 103.55,
        recession = 0

    )

    test_features = InputFeatures6M(current_month_data=test_current_data)

    import time
    start_time = time.time()

    try:
        result = predict_6m(test_features)
        total_time = time.time() - start_time
        
        print("=== 3M RECESSION PREDICTION RESULTS ===")
        print(f"Recession Probability: {result.prob_6m:.4f}")
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
    if initialize_6m_service():
        # Run test prediction
        test_prediction_6m()
    else:
        print("Service initialization failed - cannot run test")
    
    # Test model info
    info = get_model_info_6m()
    print(f"📊 Model info: {info}")