import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import shap
from typing import Dict
from services.forecast_service_6m import (
    model_6m, scaler_6m, historical_data_6m, 
    preprocess_features_6m, initialize_6m_service
)
from schemas.forecast_schema_6m import InputFeatures6M

class ExplainabilityService6M:
    """Service for generating model explanations using SHAP and ELI5-style feature importance"""
    
    def __init__(self):
        self.explainer = None
        self.background_data = None
        
    def initialize_explainer(self, seq_length=60, num_background_samples=100):
        """Initialize SHAP explainer with background data"""
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
            
            cols_to_drop = ['recession'] if 'recession' in df.columns else []
            feature_cols = [c for c in df.columns if c not in cols_to_drop]
            
            X = df[feature_cols].values.astype(np.float32)
            X_scaled = scaler_6m.transform(X).astype(np.float32)
            
            sequences = []
            for i in range(len(X_scaled) - seq_length + 1):
                sequences.append(X_scaled[i:i+seq_length])
            
            self.background_data = np.array(sequences[:num_background_samples])
            self.feature_names = feature_cols
            
            self.explainer = shap.GradientExplainer(model_6m, self.background_data)
            
            print(f"✅ SHAP explainer (6M) initialized with {len(self.background_data)} background samples")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing SHAP explainer (6M): {e}")
            return False
    
    def get_shap_values(self, features: InputFeatures6M, seq_length=60) -> Dict:
        """Calculate SHAP values for the given prediction"""
        try:
            if self.explainer is None:
                self.initialize_explainer(seq_length=seq_length)
            
            # Get fresh references to the loaded models
            from services.forecast_service_6m import model_6m
            
            if model_6m is None:
                raise RuntimeError("Model not loaded after initialization")
            
            X_scaled, feature_cols, df = preprocess_features_6m(features)
            
            if len(X_scaled) < seq_length:
                raise RuntimeError(f'Insufficient data for SHAP analysis')
            
            latest_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
            shap_values = self.explainer.shap_values(latest_sequence)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            shap_values_mean = np.mean(np.abs(shap_values[0]), axis=0)
            top_indices = np.argsort(shap_values_mean)[-10:][::-1]
            top_indices = [int(idx) for idx in top_indices]  # Convert to int
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    "feature": feature_cols[idx],
                    "importance": float(shap_values_mean[idx]),
                    "value": float(latest_sequence[0, -1, idx])
                })
            
            return {
                "shap_values": top_features,
                "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.5,
                "feature_count": len(feature_cols)
            }
            
        except Exception as e:
            print(f"❌ Error calculating SHAP values (6M): {e}")
            raise RuntimeError(f"SHAP calculation failed: {e}")
    
    def get_eli5_feature_importance(self, features: InputFeatures6M, seq_length=60) -> Dict:
        """Get ELI5-style feature importance using permutation importance approach"""
        try:
            # Ensure model is loaded
            from services.forecast_service_6m import model_6m
            
            if model_6m is None:
                raise RuntimeError("Model not loaded")
            
            X_scaled, feature_cols, df = preprocess_features_6m(features)
            
            if len(X_scaled) < seq_length:
                raise RuntimeError(f'Insufficient data for feature importance analysis')
            
            latest_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
            baseline_pred = model_6m.predict(latest_sequence, verbose=0)[0][0]
            
            feature_importance = []
            
            for idx, feature_name in enumerate(feature_cols):
                perturbed_sequence = latest_sequence.copy()
                mean_value = np.mean(latest_sequence[0, :, idx])
                perturbed_sequence[0, -1, idx] = mean_value
                
                perturbed_pred = model_6m.predict(perturbed_sequence, verbose=0)[0][0]
                importance = abs(baseline_pred - perturbed_pred)
                
                feature_importance.append({
                    "feature": feature_name,
                    "importance": float(importance),
                    "current_value": float(latest_sequence[0, -1, idx]),
                    "baseline_prediction": float(baseline_pred)
                })
            
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                "feature_importance": feature_importance[:10],
                "baseline_prediction": float(baseline_pred),
                "total_features": len(feature_cols)
            }
            
        except Exception as e:
            print(f"❌ Error calculating ELI5 feature importance (6M): {e}")
            raise RuntimeError(f"Feature importance calculation failed: {e}")
    
    def get_combined_explanation(self, features: InputFeatures6M, seq_length=60) -> Dict:
        """Get both SHAP and ELI5 explanations in a combined format"""
        try:
            shap_result = self.get_shap_values(features, seq_length)
            eli5_result = self.get_eli5_feature_importance(features, seq_length)
            
            return {
                "shap_explanation": shap_result,
                "eli5_explanation": eli5_result,
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
