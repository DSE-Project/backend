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

from typing import Dict, List, Tuple
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from utils.explainability_cache import explainability_cache, CachedExplainerData

class ExplainabilityService1M:
    """Service for generating model explanations using SHAP and permutation feature importance"""
    
    def __init__(self):
        # No longer store instance variables - use global cache instead
        pass
        
    def clear_explainer_cache(self):
        """Force reinitialization of the SHAP explainer"""
        # Clear from global cache instead of instance variables
        cache_key = explainability_cache.get_cache_key("1m")
        explainability_cache.delete(cache_key)
        print("🔄 SHAP explainer cache cleared (1M) - will reinitialize on next use")
        
    def get_or_initialize_explainer(self, seq_length=12, num_background_samples=100):
        """Get cached explainer or initialize new one"""
        # Generate cache key
        cache_key = explainability_cache.get_cache_key("1m", seq_length, num_background_samples)
        
        # Try to get from cache first
        cached_data = explainability_cache.get(cache_key)
        if cached_data and cached_data.is_valid():
            print(f"✅ Using cached SHAP explainer (1M)")
            return cached_data.explainer, cached_data.background_data, cached_data.feature_names
        
        # Cache miss - initialize new explainer
        print(f"🔄 Initializing new SHAP explainer (1M)")
        
        try:
            # Ensure model is loaded - MUST be called first
            if not initialize_1m_service():
                raise RuntimeError("Failed to initialize 1M forecasting service")
            
            # Re-import to get updated globals after initialization
            from services.forecast_service_1m import model_1m, scaler_1m, historical_data_1m
            
            if model_1m is None or scaler_1m is None or historical_data_1m is None:
                raise RuntimeError("Model, scaler, or historical data not loaded")
            
            # The new LSTM+Transformer 1m model uses the unified 29-feature dataset directly
            # Use enough data for sequence creation with the new shorter sequence length
            recent_data = historical_data_1m.tail(num_background_samples + seq_length + 50)  # Extra padding for sequences
            
            # Create a dummy current month to use the exact same preprocessing as forecast_service_1m
            from services.fred_data_service_1m import get_latest_database_row_1m, convert_to_input_features_1m
            latest_row = get_latest_database_row_1m()
            dummy_features = convert_to_input_features_1m(latest_row)
            
            # Use the EXACT same preprocessing function as forecast_service_1m
            X_scaled_full, feature_cols, df_processed = preprocess_features_1m(dummy_features, lookback_points=len(recent_data))
            
            # Take the processed data (unified 29-feature dataset)
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
            
            background_data = np.array(sequences[:num_background_samples])
            feature_names = feature_cols
            
            # Initialize SHAP explainer with GradientExplainer (most compatible with TensorFlow)
            explainer = None
            try:
                explainer = shap.GradientExplainer(model_1m, background_data)
                print(f"✅ SHAP GradientExplainer initialized with {len(background_data)} background samples")
            except Exception as e:
                print(f"⚠️  GradientExplainer failed: {e}")
                print("🔄 Falling back to simpler feature attribution method...")
                
                # Simple feature attribution using input gradients
                explainer = "simple_gradients"
                print(f"✅ Simple gradient attribution initialized")
            
            # Cache the initialized explainer
            cached_data = CachedExplainerData(explainer, background_data, feature_names)
            explainability_cache.set(cache_key, cached_data)
            
            return explainer, background_data, feature_names
            
        except Exception as e:
            print(f"❌ Error initializing SHAP explainer: {e}")
            raise RuntimeError(f"Failed to initialize explainer: {e}")
    
    def get_shap_values(self, features: InputFeatures1M, seq_length=12) -> Dict:
        """Calculate SHAP values for the given prediction"""
        try:
            # Get cached explainer or initialize new one
            explainer, background_data, feature_names = self.get_or_initialize_explainer(seq_length=seq_length)
            
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
            if explainer == "simple_gradients":
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
                shap_values = explainer.shap_values(latest_sequence)
                
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
                "base_value": float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.5,
                "feature_count": len(feature_cols)
            }
            
        except Exception as e:
            print(f"❌ Error calculating SHAP values: {e}")
            raise RuntimeError(f"SHAP calculation failed: {e}")
    
    def get_permutation_feature_importance(self, features: InputFeatures1M, seq_length: int = 12) -> Dict:
        """
        Get permutation feature importance focusing on the last time step of each sequence,
        using sklearn.inspection.permutation_importance, with n_jobs=1 to avoid pickling issues.
        Optimized for faster execution by limiting the number of windows and repeats.
        """
        try:
            # 1) ensure model is loaded
            from services.forecast_service_1m import model_1m
            if model_1m is None:
                raise RuntimeError("Model not loaded")

            # 2) preprocess -> X_scaled: shape (T, F)
            X_scaled, feature_cols, df = preprocess_features_1m(features)
            T, F = X_scaled.shape
            if T < seq_length:
                raise RuntimeError("Insufficient data for feature importance analysis")

            # 3) build rolling windows of length seq_length - OPTIMIZED: limit to recent windows
            max_windows = min(50, T - seq_length + 1)  # Limit to 50 most recent windows for performance
            start_idx = max(0, T - seq_length - max_windows + 1)
            N = min(max_windows, T - seq_length + 1)
            
            windows = np.stack([X_scaled[start_idx + i:start_idx + i + seq_length] for i in range(N)], axis=0)  # (N, L, F)
            last_steps = windows[:, -1, :]
            frozen_prefix = windows[:, :-1, :]

            # 4) sklearn-compatible estimator that only takes last-step features
            class _LastStepEstimator:
                def __init__(self, base_model, frozen_prefix):
                    self.base_model = base_model
                    self.frozen_prefix = frozen_prefix  # (N, L-1, F)

                def fit(self, X, y=None):
                    return self

                def predict(self, X):
                    N_local = X.shape[0]
                    seq = np.concatenate([self.frozen_prefix, X.reshape(N_local, 1, -1)], axis=1)
                    preds = self.base_model.predict(seq, verbose=0).reshape(-1)
                    return preds

            estimator = _LastStepEstimator(model_1m, frozen_prefix)

            # 5) baseline predictions (treated as "y" for stability scorer)
            y_hat = estimator.predict(last_steps)

            # 6) scorer: negative MAE vs baseline predictions (higher is better)
            def _neg_mae(y_true, y_pred):
                return -float(np.mean(np.abs(y_pred - y_true)))

            scorer = make_scorer(_neg_mae, greater_is_better=True)

            # 7) permutation importance (single-process to avoid pickling) - OPTIMIZED: fewer repeats
            result = permutation_importance(
                estimator=estimator,
                X=last_steps,
                y=y_hat,
                scoring=scorer,
                n_repeats=5,  # Reduced from 20 to 5 for faster execution
                random_state=0,
                n_jobs=1,  # avoid pickling Keras & local classes
            )

            importances_mean = np.abs(result.importances_mean)
            importances_std = result.importances_std

            # Latest window headline prediction (nice to report)
            latest_sequence = X_scaled[-seq_length:]
            latest_pred = float(model_1m.predict(latest_sequence.reshape(1, seq_length, F), verbose=0)[0][0])
            latest_last_step = latest_sequence[-1]

            feature_importance = []
            for idx, fname in enumerate(feature_cols):
                feature_importance.append({
                    "feature": fname,
                    "importance": float(importances_mean[idx]),
                    "importance_std": float(importances_std[idx]),
                    "current_value": float(latest_last_step[idx]),
                    "baseline_prediction": latest_pred
                })

            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

            return {
                "feature_importance": feature_importance[:10],
                "baseline_prediction": latest_pred,
                "total_features": len(feature_cols),
                "n_windows": int(N),
                "seq_length": int(seq_length),
                "note": "Optimized: Limited to 50 recent windows, 5 repeats for faster execution. Permutation permutes only last time-step features."
            }

        except Exception as e:
            print(f"❌ Error calculating permutation feature importance: {e}")
            raise RuntimeError(f"Feature importance calculation failed: {e}")
    
    def get_combined_explanation(self, features: InputFeatures1M, seq_length=12) -> Dict:
        """Get both SHAP and permutation importance explanations in a combined format"""
        try:
            # Get SHAP values
            shap_result = self.get_shap_values(features, seq_length)
            
            # Get permutation feature importance
            permutation_result = self.get_permutation_feature_importance(features, seq_length)
            
            # Combine results
            return {
                "shap_explanation": shap_result,
                "permutation_importance": permutation_result,
                "model_version": "lstm_transformer_1m_v1.0",
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
