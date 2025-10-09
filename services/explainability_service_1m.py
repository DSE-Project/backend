import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Optional
from services.forecast_service_1m import (
    model_1m, scaler_1m, historical_data_1m, 
    preprocess_features_1m, initialize_1m_service
)
from schemas.forecast_schema_1m import InputFeatures1M

class ExplainabilityService1M:
    """Service for generating model explanations using SHAP and ELI5-style feature importance"""
    
    def __init__(self):
        self.explainer = None
        self.background_data = None
        
    def clear_explainer_cache(self):
        """Force reinitialization of the SHAP explainer"""
        self.explainer = None
        self.background_data = None
        print("🔄 SHAP explainer cache cleared - will reinitialize on next use")
        
    def initialize_explainer(self, seq_length=60, num_background_samples=100):
        """Initialize SHAP explainer with background data"""
        try:
            # Ensure model is loaded - MUST be called first
            if not initialize_1m_service():
                raise RuntimeError("Failed to initialize 1M forecasting service")
            
            # Re-import to get updated globals after initialization
            from services.forecast_service_1m import model_1m, scaler_1m, historical_data_1m
            
            if model_1m is None or scaler_1m is None or historical_data_1m is None:
                raise RuntimeError("Model, scaler, or historical data not loaded")
            
            # The 1m model needs feature engineering to go from 29 -> 43 features
            # Use enough data for lag features and STL decomposition
            recent_data = historical_data_1m.tail(num_background_samples + seq_length + 100)  # Extra padding for lags
            
            # Create a dummy current month to use the exact same preprocessing as forecast_service_1m
            from services.fred_data_service_1m import get_latest_database_row_1m, convert_to_input_features_1m
            latest_row = get_latest_database_row_1m()
            dummy_features = convert_to_input_features_1m(latest_row)
            
            # Use the EXACT same preprocessing function as forecast_service_1m
            X_scaled_full, feature_cols, df_processed = preprocess_features_1m(dummy_features, lookback_points=len(recent_data))
            
            # Take the processed data (now has 43 features)
            df = df_processed.copy()
            
            print(f"📊 Final background data: {len(df)} rows, {len(feature_cols)} features")
            
            # Check if we have enough data
            if len(df) == 0:
                raise RuntimeError("No data remaining after preprocessing. All rows were dropped due to NaN values.")
            
            if len(df) < seq_length:
                raise RuntimeError(f"Insufficient data after preprocessing: {len(df)} rows, need at least {seq_length} for sequence creation")
            
            X = df[feature_cols].values.astype(np.float32)
            X_scaled = scaler_1m.transform(X).astype(np.float32)
            
            # Create sequences
            sequences = []
            for i in range(len(X_scaled) - seq_length + 1):
                sequences.append(X_scaled[i:i+seq_length])
            
            if len(sequences) == 0:
                raise RuntimeError(f"No sequences created. Data length: {len(X_scaled)}, seq_length: {seq_length}")
            
            self.background_data = np.array(sequences[:num_background_samples])
            self.feature_names = feature_cols
            
            # Initialize SHAP explainer with GradientExplainer (most compatible with TensorFlow)
            try:
                self.explainer = shap.GradientExplainer(model_1m, self.background_data)
                print(f"✅ SHAP GradientExplainer initialized with {len(self.background_data)} background samples")
                return True
            except Exception as e:
                print(f"⚠️  GradientExplainer failed: {e}")
                print("🔄 Falling back to simpler feature attribution method...")
                
                # Simple feature attribution using input gradients
                self.explainer = "simple_gradients"
                print(f"✅ Simple gradient attribution initialized")
                return True
            
        except Exception as e:
            print(f"❌ Error initializing SHAP explainer: {e}")
            return False
    
    def get_shap_values(self, features: InputFeatures1M, seq_length=60) -> Dict:
        """Calculate SHAP values for the given prediction"""
        try:
            # Initialize explainer if not done
            if self.explainer is None:
                self.initialize_explainer(seq_length=seq_length)
            
            # Get fresh references to the loaded models
            from services.forecast_service_1m import model_1m
            
            if model_1m is None:
                raise RuntimeError("Model not loaded after initialization")
            
            # Use the exact same preprocessing as the forecast service
            X_scaled, feature_cols, df = preprocess_features_1m(features)
            
            if len(X_scaled) < seq_length:
                raise RuntimeError(f'Insufficient data for SHAP analysis')
            
            # Get the latest sequence
            latest_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
            
            # Calculate SHAP values using the appropriate method
            if self.explainer == "simple_gradients":
                # Fallback: Simple gradient-based importance calculation
                import tensorflow as tf
                with tf.GradientTape() as tape:
                    tape.watch(latest_sequence)
                    prediction = model_1m(latest_sequence)
                
                # Calculate gradients
                gradients = tape.gradient(prediction, latest_sequence)
                # Average across time steps and take absolute value for importance
                shap_values_mean = np.mean(np.abs(gradients[0].numpy()), axis=0)
                
            else:
                # Use GradientExplainer
                shap_values = self.explainer.shap_values(latest_sequence)
                
                # Process SHAP values for the sequence
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # Average across time dimension (keep sign for directional impact)
                shap_values_mean = np.mean(shap_values[0], axis=0)
            
            # Ensure it's a flat array
            if len(shap_values_mean.shape) > 1:
                shap_values_mean = shap_values_mean.flatten()
            
            # Get top 10 features by absolute importance but keep original values
            abs_values = np.abs(shap_values_mean)
            top_indices = np.argsort(abs_values)[-10:][::-1]
            top_indices = [int(idx) for idx in top_indices]  # Convert to int
            
            top_features = []
            for idx in top_indices:
                feature_name = feature_cols[idx] if idx < len(feature_cols) else f"unknown_{idx}"
                importance_val = float(shap_values_mean[idx])
                
                top_features.append({
                    "feature": feature_name,
                    "importance": importance_val,  # Keep sign for directional impact
                    "value": float(latest_sequence[0, -1, idx])  # Latest value
                })
            
            return {
                "shap_values": top_features,
                "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.5,
                "feature_count": len(feature_cols)
            }
            
        except Exception as e:
            print(f"❌ Error calculating SHAP values: {e}")
            raise RuntimeError(f"SHAP calculation failed: {e}")
    
    def get_eli5_feature_importance(self, features: InputFeatures1M, seq_length=60) -> Dict:
        """Get ELI5-style feature importance using permutation importance approach"""
        try:
            # Ensure model is loaded
            from services.forecast_service_1m import model_1m
            
            if model_1m is None:
                raise RuntimeError("Model not loaded")
            
            # Preprocess features
            X_scaled, feature_cols, df = preprocess_features_1m(features)
            
            if len(X_scaled) < seq_length:
                raise RuntimeError(f'Insufficient data for feature importance analysis')
            
            # Get the latest sequence
            latest_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
            
            # Get baseline prediction
            baseline_pred = model_1m.predict(latest_sequence, verbose=0)[0][0]
            
            # Calculate feature importance through perturbation
            feature_importance = []
            
            # Only perturb the last time step (most recent data)
            for idx, feature_name in enumerate(feature_cols):
                perturbed_sequence = latest_sequence.copy()
                
                # Perturb feature at last time step with mean value
                mean_value = np.mean(latest_sequence[0, :, idx])
                perturbed_sequence[0, -1, idx] = mean_value
                
                # Get perturbed prediction
                perturbed_pred = model_1m.predict(perturbed_sequence, verbose=0)[0][0]
                
                # Calculate importance as absolute difference
                importance = abs(baseline_pred - perturbed_pred)
                
                feature_importance.append({
                    "feature": feature_name,
                    "importance": float(importance),
                    "current_value": float(latest_sequence[0, -1, idx]),
                    "baseline_prediction": float(baseline_pred)
                })
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            # Return top 10
            return {
                "feature_importance": feature_importance[:10],
                "baseline_prediction": float(baseline_pred),
                "total_features": len(feature_cols)
            }
            
        except Exception as e:
            print(f"❌ Error calculating ELI5 feature importance: {e}")
            raise RuntimeError(f"Feature importance calculation failed: {e}")
    
    def get_combined_explanation(self, features: InputFeatures1M, seq_length=60) -> Dict:
        """Get both SHAP and ELI5 explanations in a combined format"""
        try:
            # Get SHAP values
            shap_result = self.get_shap_values(features, seq_length)
            
            # Get ELI5 feature importance
            eli5_result = self.get_eli5_feature_importance(features, seq_length)
            
            # Combine results
            return {
                "shap_explanation": shap_result,
                "eli5_explanation": eli5_result,
                "model_version": "1m_v1.0",
                "explanation_method": "SHAP + Permutation Importance"
            }
            
        except Exception as e:
            print(f"❌ Error generating combined explanation: {e}")
            raise RuntimeError(f"Explanation generation failed: {e}")

# Global instance
explainability_service_1m = ExplainabilityService1M()

def get_explanation_1m(features: InputFeatures1M) -> Dict:
    """Main function to get model explanation"""
    return explainability_service_1m.get_combined_explanation(features)
