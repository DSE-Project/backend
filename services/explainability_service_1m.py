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
            
            # Prepare background data for SHAP
            # Use recent historical data as background
            recent_data = historical_data_1m.tail(num_background_samples + seq_length)
            
            # Create features for background data
            from services.forecast_service_1m import create_features_vectorized
            from statsmodels.tsa.seasonal import STL
            
            df = recent_data.copy()
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
                    df['UNRATE_trend'] = df['UNRATE'].rolling(12, min_periods=1).mean()
                    df['UNRATE_seasonal'] = 0
                    df['UNRATE_resid'] = df['UNRATE'] - df['UNRATE_trend']
            
            df = df.dropna()
            cols_to_drop = ['recession'] if 'recession' in df.columns else []
            feature_cols = [c for c in df.columns if c not in cols_to_drop]
            
            X = df[feature_cols].values.astype(np.float32)
            X_scaled = scaler_1m.transform(X).astype(np.float32)
            
            # Create sequences
            sequences = []
            for i in range(len(X_scaled) - seq_length + 1):
                sequences.append(X_scaled[i:i+seq_length])
            
            self.background_data = np.array(sequences[:num_background_samples])
            self.feature_names = feature_cols
            
            # Initialize SHAP explainer with GradientExplainer for deep learning models
            self.explainer = shap.GradientExplainer(model_1m, self.background_data)
            
            print(f"✅ SHAP explainer initialized with {len(self.background_data)} background samples")
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
            
            # Preprocess features to get the input sequence
            X_scaled, feature_cols, df = preprocess_features_1m(features)
            
            if len(X_scaled) < seq_length:
                raise RuntimeError(f'Insufficient data for SHAP analysis')
            
            # Get the latest sequence
            latest_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(latest_sequence)
            
            # Process SHAP values for the sequence
            # Average across time steps to get feature importance
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Average across time dimension
            shap_values_mean = np.mean(np.abs(shap_values[0]), axis=0)
            
            # Get top 10 features - ensure indices are integers
            top_indices = np.argsort(shap_values_mean)[-10:][::-1]
            top_indices = [int(idx) for idx in top_indices]  # Convert to int
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    "feature": feature_cols[idx],
                    "importance": float(shap_values_mean[idx]),
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
