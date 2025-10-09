import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from datetime import datetime
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

from schemas.forecast_schema_6m import InputFeatures6M, ForecastResponse6M, CurrentMonthData6M, ModelStatus6M
import joblib
from services.database_service import db_service

# Model and scaler paths
MODEL_6M_PATH = "ml_models/6m/lstm_transformer_model_6m.keras"
SCALER_6M_PATH = "ml_models/6m/lstm_transformer_scaler_6m.pkl"

# Global variables for model and scaler
model_6m = None
scaler_6m = None
historical_data_6m = None
_service_initialized_6m = False

# Model configuration (must match training config)
MODEL_CONFIG = {
    "seq_length": 12,
    "lstm_units": 64,
    "d_model": 32,
    "num_heads": 2,
    "ff_dim": 64,
    "num_layers": 1,
    "dropout": 0.18,
    "focal_gamma": 2.0,
    "focal_alpha": 0.9,
}

def focal_loss(gamma=2.0, alpha=0.9):
    """Focal loss function for the LSTM+Transformer model"""
    def loss(y_true, y_pred):
        # Clip to prevent log(0) issues
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        bce = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        mod_factor = (1 - p_t) ** gamma
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        return K.mean(alpha_factor * mod_factor * bce)
    return loss

class TransformerBlock(layers.Layer):
    """Transformer block for the hybrid model"""
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        
        # key_dim is dimension per head; ensure it's integer
        key_dim = max(1, d_model // num_heads)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate,
        })
        return config

class PositionalEncoding(layers.Layer):
    """Positional encoding for transformer"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x, training=None):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i//2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        # Cast positional encoding to same dtype as x
        pos_encoding = tf.cast(pos_encoding, x.dtype)
        return x + pos_encoding
    
    def get_config(self):
        return super().get_config()

def load_model_6m():
    """Load the LSTM+Transformer 6-month recession prediction model"""
    global model_6m
    try: 
        # Custom objects for loading the model
        custom_objects = {
            'focal_loss': focal_loss(gamma=MODEL_CONFIG['focal_gamma'], alpha=MODEL_CONFIG['focal_alpha']),
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding,
            'loss': focal_loss(gamma=MODEL_CONFIG['focal_gamma'], alpha=MODEL_CONFIG['focal_alpha'])
        }
        
        model_6m = tf.keras.models.load_model(
            MODEL_6M_PATH, 
            custom_objects=custom_objects,
            compile=False
        )
        print("✅ LSTM+Transformer 6M model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading LSTM+Transformer 6M model: {e}")
        model_6m = None
        return False

def load_scaler_6m():
    """Load the scaler for LSTM+Transformer 6-month model preprocessing"""
    global scaler_6m
    try:
        scaler_6m = joblib.load(SCALER_6M_PATH)
        print("✅ LSTM+Transformer 6M scaler loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading LSTM+Transformer 6M scaler: {e}")
        scaler_6m = None
        return False
    
def load_historical_data_6m():
    """Load historical data from Supabase database"""
    global historical_data_6m
    try:
        historical_data_6m = db_service.load_historical_data('historical_data')
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
    """
    Preprocess input features for LSTM+Transformer 6-month model
    
    Note: Since the trained model expects engineered features, we need to maintain 
    compatibility with the scaler that was trained on the feature-engineered dataset
    """
    try:
        # Load historical data if not already loaded
        if historical_data_6m is None:
            if not load_historical_data_6m():
                raise RuntimeError("Failed to load historical data")
        
        # Keep only the last lookback_points for memory efficiency
        historical_data = historical_data_6m.tail(lookback_points+1).iloc[:-1].copy()
        
        # Convert all to float32 for memory efficiency
        historical_data = historical_data.astype(np.float32)
        
        print(f"Using only last {len(historical_data)} data points for processing")
        print(f"Historical data columns: {list(historical_data.columns)}")
        
        # Convert Pydantic model to dict
        current_data = features.current_month_data.model_dump()

        # Process current month data - match all historical columns
        historical_cols = historical_data.columns.tolist()
        filtered_current_data = {k: v for k, v in current_data.items() 
                               if k in historical_cols + ['observation_date']}
        
        current_month_df = pd.DataFrame([filtered_current_data], 
                                      index=[pd.to_datetime(filtered_current_data["observation_date"])])
        current_month_df = current_month_df.drop(columns=['observation_date'])
        
        df = pd.concat([historical_data, current_month_df], copy=False)

        # Final processing - match training preprocessing
        df = df.dropna()
        
        # Drop columns same as training (adjust based on your training data columns)
        cols_to_drop = ['recession', 'recession_1m', 'recession_3m', 'recession_6m']
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        feature_cols = [c for c in df.columns if c not in cols_to_drop]
        
        print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        
        X = df[feature_cols].values.astype(np.float32)
        X_scaled = scaler_6m.transform(X).astype(np.float32)

        return X_scaled, feature_cols, df
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise RuntimeError(f"Preprocessing failed: {e}")

def create_sequences(X, seq_length):
    """
    Create sequences for LSTM+Transformer model
    Returns only the latest sequence needed for prediction
    """
    if len(X) < seq_length:
        raise RuntimeError(f'Insufficient data. Need at least {seq_length} data points, got {len(X)}')
    
    # Return only the last sequence we need for prediction
    latest_sequence = X[-seq_length:]
    return latest_sequence.reshape(1, seq_length, -1)

def initialize_6m_service():
    """Initialize the LSTM+Transformer 6 month forecasting service"""
    global _service_initialized_6m

    # Return True if already initialized
    if _service_initialized_6m:
        return True
    
    print("🚀 Initializing LSTM+Transformer 6M forecasting service...")

    model_loaded = load_model_6m()
    scaler_loaded = load_scaler_6m()
    data_loaded = load_historical_data_6m()

    if model_loaded and scaler_loaded and data_loaded:
        print("✅ LSTM+Transformer 6M service initialized successfully")
        _service_initialized_6m = True
        return True
    else:
        print("❌ LSTM+Transformer 6M service initialization failed")
        _service_initialized_6m = False
        return False

def predict_6m(features: InputFeatures6M, threshold=0.72) -> ForecastResponse6M:
    """
    Make 6-month recession probability prediction using LSTM+Transformer model
    
    Key changes from traditional models:
    - Uses sequence length of 12 (vs 60)
    - Uses threshold of 0.72 (configurable)
    - Different model architecture (LSTM+Transformer)
    - Targets 6-month ahead recession prediction
    """
    if not _service_initialized_6m:
        if not initialize_6m_service():
            raise RuntimeError("Failed to initialize LSTM+Transformer 6M forecasting service")
    
    if model_6m is None or scaler_6m is None:
        raise RuntimeError("LSTM+Transformer 6M model or scaler is not loaded")
    
    try: 
        # Preprocess features
        X_scaled, feature_cols, df = preprocess_features_6m(features)
        
        # Create sequence for LSTM+Transformer model (seq_length = 12)
        seq_length = MODEL_CONFIG['seq_length']
        latest_sequence = create_sequences(X_scaled, seq_length)
        
        # Make prediction
        prediction = model_6m.predict(latest_sequence, verbose=0)
        prob_6m = float(prediction[0][0])

        # Binary prediction based on threshold
        binary_prediction = int(prob_6m > threshold)

        # Create response with additional metadata
        response = ForecastResponse6M(
            prob_6m=prob_6m,
            model_version="lstm_transformer_6m_v1.0",
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
                "sequence_length": seq_length,
                "model_architecture": "LSTM+Transformer",
                "lstm_units": MODEL_CONFIG['lstm_units'],
                "transformer_layers": MODEL_CONFIG['num_layers'],
                "attention_heads": MODEL_CONFIG['num_heads'],
                "prediction_horizon": "6 Months Ahead"
            }
        )

        return response
    
    except Exception as e: 
        raise RuntimeError(f"LSTM+Transformer 6M prediction failed: {e}")   

def get_model_info_6m() -> ModelStatus6M:
    """Get information about the LSTM+Transformer 6-month model"""
    return ModelStatus6M(
        model_loaded=model_6m is not None,
        scaler_loaded=scaler_6m is not None,
        model_version="lstm_transformer_6m_v1.0",
        last_updated="2025-01-09",  # Update this when you retrain
        historical_data_available=historical_data_6m is not None,
        total_features=len(historical_data_6m.columns) if historical_data_6m is not None else 0
    )

def test_prediction_6m():
    """Test function to verify the LSTM+Transformer 6M service works"""
    # Create test input using the updated schema structure
    test_current_data = CurrentMonthData6M(
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

    test_features = InputFeatures6M(current_month_data=test_current_data)

    import time
    start_time = time.time()

    try:
        result = predict_6m(test_features)
        total_time = time.time() - start_time
        
        print("=== 6M RECESSION PREDICTION RESULTS ===")
        print(f"Recession Probability: {result.prob_6m:.4f}")
        print(f"Binary Prediction: {result.confidence_interval.get('binary_prediction', 'N/A')}")
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
    print("Testing LSTM+Transformer 6M forecasting service...")
    
    # Initialize service
    if initialize_6m_service():
        # Run test prediction
        test_prediction_6m()
    else:
        print("Service initialization failed - cannot run test")
    
    # Test model info
    info = get_model_info_6m()
    print(f"📊 Model info: {info}")