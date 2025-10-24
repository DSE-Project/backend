import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import shap
from typing import Dict
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from services.forecast_service_6m import (
    model_6m, scaler_6m, historical_data_6m, 
    preprocess_features_6m, initialize_6m_service
)
from schemas.forecast_schema_6m import InputFeatures6M
from utils.explainability_cache import explainability_cache, CachedExplainerData

class ExplainabilityService6M:
    """Service for generating model explanations using SHAP and permutation feature importance"""
    
    def __init__(self):
        # No longer store instance variables - use global cache instead
        pass
        
    def clear_explainer_cache(self):
        """Force reinitialization of the SHAP explainer"""
        # Clear from global cache instead of instance variables
        cache_key = explainability_cache.get_cache_key("6m")
        explainability_cache.delete(cache_key)
        print("🔄 SHAP explainer cache cleared (6M) - will reinitialize on next use")
        
    def get_or_initialize_explainer(self, seq_length=12, num_background_samples=100):
        """Get cached explainer or initialize new one"""
        # Generate cache key
        cache_key = explainability_cache.get_cache_key("6m", seq_length, num_background_samples)
        
        # Try to get from cache first
        cached_data = explainability_cache.get(cache_key)
        if cached_data and cached_data.is_valid():
            print(f"✅ Using cached SHAP explainer (6M)")
            return cached_data.explainer, cached_data.background_data, cached_data.feature_names
        
        # Cache miss - initialize new explainer
        print(f"🔄 Initializing new SHAP explainer (6M)")
        
        try:
            # Ensure model is loaded - MUST be called first
            if not initialize_6m_service():
                raise RuntimeError("Failed to initialize 6M forecasting service")
            
            # Re-import to get updated globals after initialization
            from services.forecast_service_6m import model_6m, scaler_6m, historical_data_6m
            
            if model_6m is None or scaler_6m is None or historical_data_6m is None:
                raise RuntimeError("Model, scaler, or historical data not loaded")
            
            recent_data = historical_data_6m.tail(num_background_samples + seq_length)
            df = recent_data.copy()
            df = df.dropna()
            
            # Drop columns same as training (adjust based on your training data columns)
            cols_to_drop = ['recession', 'recession_1m', 'recession_3m', 'recession_6m']
            cols_to_drop = [c for c in cols_to_drop if c in df.columns]
            feature_cols = [c for c in df.columns if c not in cols_to_drop]
            
            X = df[feature_cols].values.astype(np.float32)
            X_scaled = scaler_6m.transform(X).astype(np.float32)
            
            sequences = []
            for i in range(len(X_scaled) - seq_length + 1):
                sequences.append(X_scaled[i:i+seq_length])
            
            background_data = np.array(sequences[:num_background_samples])
            feature_names = feature_cols
            
            print(f"Background data shape: {background_data.shape}")
            print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
            
            # Initialize SHAP explainer with GradientExplainer (most stable for TensorFlow models)
            explainer = None
            try:
                explainer = shap.GradientExplainer(model_6m, background_data)
                print(f"✅ SHAP GradientExplainer (6M) initialized with {len(background_data)} background samples")
            except Exception as grad_error:
                print(f"⚠️ GradientExplainer failed: {grad_error}")
                # Fallback to a simple manual feature importance calculation
                explainer = "simple_gradients"
                print("✅ Fallback to manual feature importance calculation (6M)")
            
            # Cache the initialized explainer
            cached_data = CachedExplainerData(explainer, background_data, feature_names)
            explainability_cache.set(cache_key, cached_data)
            
            return explainer, background_data, feature_names
            
        except Exception as e:
            print(f"❌ Error initializing SHAP explainer (6M): {e}")
            raise RuntimeError(f"Failed to initialize explainer: {e}")
    
    def get_shap_values(self, features: InputFeatures6M, seq_length=12) -> Dict:
        """Calculate SHAP values for the given prediction"""
        try:
            # Get cached explainer or initialize new one
            explainer, background_data, feature_names = self.get_or_initialize_explainer(seq_length=seq_length)
            
            # Get fresh references to the loaded models
            from services.forecast_service_6m import model_6m
            
            if model_6m is None:
                raise RuntimeError("Model not loaded after initialization")
            
            X_scaled, feature_cols, df = preprocess_features_6m(features)
            
            if len(X_scaled) < seq_length:
                raise RuntimeError(f'Insufficient data for SHAP analysis')
            
            latest_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
            
            # Calculate SHAP values using the appropriate method
            if explainer == "simple_gradients":
                # Fallback: Simple gradient-based importance calculation
                import tensorflow as tf
                with tf.GradientTape() as tape:
                    tape.watch(latest_sequence)
                    prediction = model_6m(latest_sequence)
                
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
            top_indices = np.argsort(np.abs(shap_values_mean))[-10:][::-1]
            top_indices = [int(idx) for idx in top_indices]  # Convert to int
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    "feature": feature_cols[idx],
                    "importance": float(shap_values_mean[idx]),  # Keep sign for directional impact
                    "value": float(latest_sequence[0, -1, idx])
                })
            
            return {
                "shap_values": top_features,
                "base_value": float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.5,
                "feature_count": len(feature_cols)
            }
            
        except Exception as e:
            print(f"❌ Error calculating SHAP values (6M): {e}")
            raise RuntimeError(f"SHAP calculation failed: {e}")
    
    def get_permutation_feature_importance(self, features: InputFeatures6M, seq_length: int = 12) -> Dict:
        """
        Get permutation feature importance focusing on the last time step of each sequence,
        using sklearn.inspection.permutation_importance, with n_jobs=1 to avoid pickling issues.
        Optimized for faster execution by limiting the number of windows and repeats.
        """
        try:
            # 1) ensure model is loaded
            from services.forecast_service_6m import model_6m
            if model_6m is None:
                raise RuntimeError("Model not loaded")

            # 2) preprocess -> X_scaled: shape (T, F)
            X_scaled, feature_cols, df = preprocess_features_6m(features)
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

            estimator = _LastStepEstimator(model_6m, frozen_prefix)

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
            latest_pred = float(model_6m.predict(latest_sequence.reshape(1, seq_length, F), verbose=0)[0][0])
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
            print(f"❌ Error calculating permutation feature importance (6M): {e}")
            raise RuntimeError(f"Feature importance calculation failed: {e}")
    
    def get_combined_explanation(self, features: InputFeatures6M, seq_length=12) -> Dict:
        """Get both SHAP and permutation importance explanations in a combined format"""
        try:
            shap_result = self.get_shap_values(features, seq_length)
            permutation_result = self.get_permutation_feature_importance(features, seq_length)
            
            return {
                "shap_explanation": shap_result,
                "permutation_importance": permutation_result,
                "model_version": "6m_v1.0",
                "explanation_method": "SHAP + Permutation Importance"
            }
            
        except Exception as e:
            print(f"❌ Error generating combined explanation (6M): {e}")
            raise RuntimeError(f"Explanation generation failed: {e}")

# Global instance
explainability_service_6m = ExplainabilityService6M()

def get_explanation_6m(features: InputFeatures6M) -> Dict:
    """Main function to get model explanation"""
    return explainability_service_6m.get_combined_explanation(features)
