import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
import logging

from schemas.simulate_schema import (
    FeatureDefinition, 
    ModelFeatureDefinitions, 
    AllFeatureDefinitions,
    SimulateError
)
from services.database_service import db_service

logger = logging.getLogger(__name__)

class SimulateService:
    """
    Service for managing feature definitions and simulation capabilities.
    
    Architecture Note: All models (1m, 3m, 6m) now use the same unified 29-feature dataset
    with LSTM+Transformer architecture. Previously each model had separate feature tables,
    but with the unified dataset, only one feature definitions table is needed.
    """
    def __init__(self):
        self.supported_models = ["1m", "3m", "6m"]
        # Since all models now use the same unified dataset, we only need one feature definitions table
        self.feature_definitions_table = "feature_definitions_1m"
    
    def get_feature_definitions(self, model_period: str) -> Optional[ModelFeatureDefinitions]:
        """Get feature definitions for a specific model period (all models use the same unified dataset)"""
        try:
            if model_period not in self.supported_models:
                logger.error(f"Unsupported model period: {model_period}")
                return None
            
            # All models use the same unified dataset, so use the same feature definitions table
            table_name = self.feature_definitions_table
            
            # Query feature definitions from database
            response = db_service.supabase.table(table_name).select("*").order('feature_code').execute()
            
            if not response.data:
                logger.warning(f"No feature definitions found for model {model_period}")
                return None
            
            # Convert to FeatureDefinition objects
            features = []
            important_count = 0
            
            for row in response.data:
                feature = FeatureDefinition(
                    feature_code=row['feature_code'],
                    name=row['name'],
                    description=row.get('description', ''),
                    min_value=float(row['min_value']),
                    max_value=float(row['max_value']),
                    default_value=float(row['default_value']),
                    is_important=int(row.get('is_important', 1))
                )
                features.append(feature)
                
                if feature.is_important == 1:
                    important_count += 1
            
            logger.info(f"✅ Loaded {len(features)} unified feature definitions for {model_period} model")
            
            return ModelFeatureDefinitions(
                model_period=model_period,
                features=features,
                total_features=len(features),
                important_features_count=important_count
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get feature definitions for {model_period}: {e}")
            return None
    
    def get_all_feature_definitions(self) -> Optional[AllFeatureDefinitions]:
        """Get feature definitions for all supported models"""
        try:
            models = {}
            
            for model_period in self.supported_models:
                model_features = self.get_feature_definitions(model_period)
                if model_features:
                    models[model_period] = model_features
                else:
                    logger.warning(f"Could not load features for {model_period} model")
            
            if not models:
                logger.error("No feature definitions could be loaded for any model")
                return None
            
            return AllFeatureDefinitions(
                models=models,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get all feature definitions: {e}")
            return None
    
    def get_important_features(self, model_period: str) -> Optional[List[FeatureDefinition]]:
        """Get only important features for a specific model"""
        try:
            model_features = self.get_feature_definitions(model_period)
            if not model_features:
                return None
            
            important_features = [f for f in model_features.features if f.is_important == 1]
            logger.info(f"✅ Found {len(important_features)} important features for {model_period} model")
            
            return important_features
            
        except Exception as e:
            logger.error(f"❌ Failed to get important features for {model_period}: {e}")
            return None
    
    def validate_feature_value(self, model_period: str, feature_code: str, value: float) -> tuple[bool, str]:
        """Validate if a feature value is within acceptable range"""
        try:
            model_features = self.get_feature_definitions(model_period)
            if not model_features:
                return False, f"Model {model_period} not found"
            
            # Find the specific feature
            feature = None
            for f in model_features.features:
                if f.feature_code == feature_code:
                    feature = f
                    break
            
            if not feature:
                return False, f"Feature {feature_code} not found for model {model_period}"
            
            # Check if value is within range
            if value < feature.min_value or value > feature.max_value:
                return False, f"Value {value} is outside acceptable range [{feature.min_value}, {feature.max_value}]"
            
            return True, "Valid"
            
        except Exception as e:
            logger.error(f"❌ Error validating feature value: {e}")
            return False, f"Validation error: {str(e)}"
    
    def get_feature_summary(self, model_period: str = None) -> Dict:
        """Get a summary of features for debugging/monitoring"""
        try:
            if model_period:
                # Single model summary
                model_features = self.get_feature_definitions(model_period)
                if not model_features:
                    return {"error": f"No features found for model {model_period}"}
                
                return {
                    "model_period": model_period,
                    "total_features": model_features.total_features,
                    "important_features": model_features.important_features_count,
                    "feature_codes": [f.feature_code for f in model_features.features],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # All models summary
                summary = {
                    "models": {},
                    "timestamp": datetime.now().isoformat()
                }
                
                for period in self.supported_models:
                    model_features = self.get_feature_definitions(period)
                    if model_features:
                        summary["models"][period] = {
                            "total_features": model_features.total_features,
                            "important_features": model_features.important_features_count
                        }
                
                return summary
                
        except Exception as e:
            logger.error(f"❌ Error getting feature summary: {e}")
            return {"error": str(e)}
    
    def test_database_connection(self) -> bool:
        """Test database connection for simulate service"""
        try:
            # Test connection to the unified feature definitions table
            response = db_service.supabase.table(self.feature_definitions_table).select("feature_code").limit(1).execute()
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False

# Global simulate service instance
simulate_service = SimulateService()

def test_simulate_service():
    """Test function for the simulate service"""
    try:
        print("🧪 Testing simulate service...")
        
        # Test database connection
        if not simulate_service.test_database_connection():
            print("❌ Database connection failed")
            return False
        
        # Test getting features for 1m model
        features_1m = simulate_service.get_feature_definitions("1m")
        if features_1m:
            print(f"✅ 1m model: {features_1m.total_features} features, {features_1m.important_features_count} important")
        else:
            print("❌ Failed to get 1m features")
        
        # Test getting all features
        all_features = simulate_service.get_all_feature_definitions()
        if all_features:
            print(f"✅ All models loaded: {list(all_features.models.keys())}")
        else:
            print("❌ Failed to get all features")
        
        # Test feature validation
        if features_1m and len(features_1m.features) > 0:
            first_feature = features_1m.features[0]
            valid, msg = simulate_service.validate_feature_value("1m", first_feature.feature_code, first_feature.default_value)
            print(f"✅ Validation test: {valid} - {msg}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_simulate_service()