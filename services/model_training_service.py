import os
import sys
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    precision_recall_curve,
    auc
)

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.database_service import db_service

logger = logging.getLogger(__name__)

class ModelTrainingService:
    """
    Service for retraining machine learning models with new economic data
    """
    
    def __init__(self):
        """Initialize the model training service"""
        self.model_configs = {
            '1m': {
                'model_path': 'ml_models/1m/model_1m.keras',
                'scaler_path': 'ml_models/1m/scaler_1m.pkl',
                'table_name': 'historical_data_1m',
                'sequence_length': 12,  # 12 months of historical data
                'prediction_horizon': 1   # Predict 1 month ahead
            },
            '3m': {
                'model_path': 'ml_models/3m/model_3_months.keras',
                'scaler_path': 'ml_models/3m/scaler_3.pkl',
                'table_name': 'historical_data_3m',
                'sequence_length': 24,  # 24 months of historical data
                'prediction_horizon': 3   # Predict 3 months ahead
            },
            '6m': {
                'model_path': 'ml_models/6m/model_6_months.keras',
                'scaler_path': 'ml_models/6m/scaler_6.pkl',
                'table_name': 'historical_data_6m',
                'sequence_length': 36,  # 36 months of historical data
                'prediction_horizon': 6   # Predict 6 months ahead
            }
        }
        
        # Feature columns for each model (based on your actual data structure)
        self.model_feature_columns = {
            '1m': [
                'fedfunds', 'TB3MS', 'TB6MS', 'TB1YR', 'USTPU', 'USGOOD', 'SRVPRD', 'USCONS', 
                'MANEMP', 'USWTRADE', 'USTRADE', 'USINFO', 'UNRATE', 'UNEMPLOY', 'CPIFOOD', 
                'CPIMEDICARE', 'CPIRENT', 'CPIAPP', 'GDP', 'REALGDP', 'PCEPI', 'PSAVERT', 
                'PSTAX', 'COMREAL', 'COMLOAN', 'SECURITYBANK', 'PPIACO', 'M1SL', 'M2SL'
            ],
            '3m': [
                'ICSA', 'CPIMEDICARE', 'USWTRADE', 'BBKMLEIX', 'COMLOAN', 'UMCSENT', 'MANEMP', 
                'fedfunds', 'PSTAX', 'USCONS', 'USGOOD', 'USINFO', 'CPIAPP', 'CSUSHPISA', 
                'SECURITYBANK', 'SRVPRD', 'INDPRO', 'TB6MS', 'UNEMPLOY', 'USTPU'
            ],
            '6m': [
                'PSTAX', 'USWTRADE', 'MANEMP', 'CPIAPP', 'CSUSHPISA', 'ICSA', 'fedfunds', 
                'BBKMLEIX', 'TB3MS', 'USINFO', 'PPIACO', 'CPIMEDICARE', 'UNEMPLOY', 'TB1YR', 
                'USGOOD', 'CPIFOOD', 'UMCSENT', 'SRVPRD', 'GDP', 'INDPRO'
            ]
        }
        
        # Performance comparison thresholds
        self.performance_thresholds = {
            'min_pr_auc_improvement': 0.01,  # Minimum 1% improvement in PR-AUC required
            'min_accuracy_threshold': 0.60,   # Minimum accuracy threshold
            'weight_pr_auc': 0.7,            # Weight for PR-AUC in combined score
            'weight_accuracy': 0.3           # Weight for accuracy in combined score
        }
        
        logger.info("ModelTrainingService initialized successfully")
    
    def focal_loss(self, gamma=2., alpha=0.25):
        """
        Focal loss function for handling imbalanced data
        This is the same loss function used in your existing models
        """
        def loss(y_true, y_pred):
            from tensorflow.keras import backend as K
            bce = K.binary_crossentropy(y_true, y_pred)
            bce_exp = K.exp(-bce)
            return K.mean(alpha * (1 - bce_exp) ** gamma * bce)
        return loss
    
    def get_model_required_features(self, period: str) -> List[str]:
        """
        Get the required features for a specific model period
        
        Args:
            period: Model period ('1m', '3m', '6m')
            
        Returns:
            List of required feature column names
        """
        if period not in self.model_feature_columns:
            raise ValueError(f"Unsupported model period: {period}")
        
        return self.model_feature_columns[period]
    
    def evaluate_existing_model(self, period: str, X_val: np.ndarray, y_val: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Evaluate the performance of the existing model
        
        Args:
            period: Model period ('1m', '3m', or '6m')
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dict with existing model performance metrics or None if model doesn't exist
        """
        try:
            config = self.model_configs[period]
            
            # Check if existing model files exist
            if not os.path.exists(config['model_path']) or not os.path.exists(config['scaler_path']):
                logger.info(f"📊 No existing {period} model found - will save new model")
                return None
            
            # Load existing model and scaler
            existing_model = tf.keras.models.load_model(
                config['model_path'],
                custom_objects={'loss': self.focal_loss(gamma=2., alpha=0.25)},
                compile=False
            )
            
            # Compile the model with the same settings as training
            existing_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=self.focal_loss(gamma=2., alpha=0.25),
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Evaluate existing model
            existing_predictions = existing_model.predict(X_val, verbose=0)
            existing_pred_binary = (existing_predictions > 0.5).astype(int).flatten()
            existing_pred_proba = existing_predictions.flatten()
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            existing_performance = {
                'accuracy': float(accuracy_score(y_val, existing_pred_binary)),
                'precision': float(precision_score(y_val, existing_pred_binary, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_val, existing_pred_binary, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_val, existing_pred_binary, average='weighted', zero_division=0)),
                'pr_auc': float(average_precision_score(y_val, existing_pred_proba)),
                'roc_auc': float(roc_auc_score(y_val, existing_pred_proba)) if len(np.unique(y_val)) > 1 else 0.5
            }
            
            # Calculate combined performance score
            existing_performance['combined_score'] = (
                existing_performance['pr_auc'] * self.performance_thresholds['weight_pr_auc'] +
                existing_performance['accuracy'] * self.performance_thresholds['weight_accuracy']
            )
            
            logger.info(f"📊 Existing {period} model performance:")
            logger.info(f"    PR-AUC: {existing_performance['pr_auc']:.4f}")
            logger.info(f"    Accuracy: {existing_performance['accuracy']:.4f}")
            logger.info(f"    Combined Score: {existing_performance['combined_score']:.4f}")
            
            return existing_performance
            
        except Exception as e:
            logger.warning(f"⚠️ Could not evaluate existing {period} model: {str(e)}")
            return None
    
    def should_save_new_model(self, existing_performance: Optional[Dict[str, float]], 
                             new_performance: Dict[str, float], period: str) -> Tuple[bool, str]:
        """
        Determine if the new model should be saved based on performance comparison
        
        Args:
            existing_performance: Performance metrics of existing model (None if no existing model)
            new_performance: Performance metrics of newly trained model
            period: Model period for logging
            
        Returns:
            Tuple of (should_save: bool, reason: str)
        """
        # If no existing model, save the new one
        if existing_performance is None:
            return True, "No existing model found"
        
        # Check minimum accuracy threshold
        if new_performance['accuracy'] < self.performance_thresholds['min_accuracy_threshold']:
            return False, f"New model accuracy ({new_performance['accuracy']:.4f}) below minimum threshold ({self.performance_thresholds['min_accuracy_threshold']})"
        
        # Check PR-AUC improvement
        pr_auc_improvement = new_performance['pr_auc'] - existing_performance['pr_auc']
        min_improvement = self.performance_thresholds['min_pr_auc_improvement']
        
        if pr_auc_improvement < min_improvement:
            return False, f"Insufficient PR-AUC improvement ({pr_auc_improvement:.4f} < {min_improvement})"
        
        # Check combined score improvement
        combined_improvement = new_performance['combined_score'] - existing_performance['combined_score']
        
        if combined_improvement <= 0:
            return False, f"No improvement in combined score ({combined_improvement:.4f})"
        
        # All checks passed - save the new model
        improvement_pct = (pr_auc_improvement / existing_performance['pr_auc']) * 100
        return True, f"PR-AUC improved by {improvement_pct:.2f}% ({existing_performance['pr_auc']:.4f} → {new_performance['pr_auc']:.4f})"
    
    async def retrain_model(self, period: str) -> Dict[str, Any]:
        """
        Retrain a specific model with new data
        
        Args:
            period: Model period ('1m', '3m', or '6m')
            
        Returns:
            Dict with retraining results
        """
        if period not in self.model_configs:
            return {
                'success': False,
                'error': f"Invalid model period: {period}. Must be one of {list(self.model_configs.keys())}"
            }
        
        config = self.model_configs[period]
        start_time = datetime.now()
        
        result = {
            'success': False,
            'period': period,
            'start_time': start_time.isoformat(),
            'data_shape': None,
            'training_samples': 0,
            'validation_samples': 0,
            'training_time_seconds': 0,
            'model_performance': {},
            'error': None
        }
        
        try:
            logger.info(f"🚀 Starting {period} model retraining...")
            
            # Step 1: Load and prepare data
            logger.info(f"📊 Loading data from {config['table_name']}...")
            raw_data = db_service.load_historical_data(config['table_name'])
            
            if raw_data is None or raw_data.empty:
                raise ValueError(f"No data available in {config['table_name']}")
            
            result['data_shape'] = raw_data.shape
            logger.info(f"✅ Loaded data: {raw_data.shape}")
            
            # Step 2: Prepare training data
            X, y, scaler = self.prepare_training_data(
                raw_data, 
                config['sequence_length'],
                config['prediction_horizon'],
                period
            )
            
            if X is None or y is None:
                raise ValueError("Failed to prepare training data")
            
            logger.info(f"📈 Prepared training data: X{X.shape}, y{y.shape}")
            
            # Step 3: Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            result['training_samples'] = len(X_train)
            result['validation_samples'] = len(X_val)
            
            logger.info(f"🔄 Data split - Train: {len(X_train)}, Val: {len(X_val)}")
            
            # Step 4: Build and train model
            model = self.build_lstm_model(X_train.shape[1], X_train.shape[2])
            
            logger.info("🤖 Starting model training...")
            training_start = datetime.now()
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=self.get_training_callbacks()
            )
            
            training_time = (datetime.now() - training_start).total_seconds()
            result['training_time_seconds'] = training_time
            
            logger.info(f"✅ Model training completed in {training_time:.2f} seconds")
            
            # Step 5: Evaluate existing model first
            logger.info("📊 Evaluating existing model performance...")
            existing_performance = self.evaluate_existing_model(period, X_val, y_val)
            
            # Step 6: Evaluate new model
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            val_predictions = model.predict(X_val, verbose=0)
            
            # Calculate comprehensive metrics for new model
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            val_pred_binary = (val_predictions > 0.5).astype(int).flatten()
            val_pred_proba = val_predictions.flatten()
            
            new_performance = {
                'validation_loss': float(val_loss),
                'accuracy': float(accuracy_score(y_val, val_pred_binary)),
                'precision': float(precision_score(y_val, val_pred_binary, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_val, val_pred_binary, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_val, val_pred_binary, average='weighted', zero_division=0)),
                'pr_auc': float(average_precision_score(y_val, val_pred_proba)),
                'roc_auc': float(roc_auc_score(y_val, val_pred_proba)) if len(np.unique(y_val)) > 1 else 0.5
            }
            
            # Calculate combined performance score
            new_performance['combined_score'] = (
                new_performance['pr_auc'] * self.performance_thresholds['weight_pr_auc'] +
                new_performance['accuracy'] * self.performance_thresholds['weight_accuracy']
            )
            
            result['model_performance'] = new_performance
            result['existing_performance'] = existing_performance
            
            logger.info(f"📊 New {period} model performance:")
            logger.info(f"    PR-AUC: {new_performance['pr_auc']:.4f}")
            logger.info(f"    Accuracy: {new_performance['accuracy']:.4f}")
            logger.info(f"    Combined Score: {new_performance['combined_score']:.4f}")
            
            # Step 7: Decide whether to save the new model
            should_save, reason = self.should_save_new_model(existing_performance, new_performance, period)
            
            result['model_saved'] = should_save
            result['save_decision_reason'] = reason
            
            if should_save:
                logger.info(f"✅ Saving new {period} model: {reason}")
                
                # Save model and scaler
                await self.save_model_artifacts(model, scaler, period, config)
                
                # Reload the corresponding service
                await self.reload_prediction_service(period)
                
                result['model_improvement'] = True
            else:
                logger.info(f"❌ Not saving new {period} model: {reason}")
                result['model_improvement'] = False
            
            result['success'] = True
            result['end_time'] = datetime.now().isoformat()
            
            logger.info(f"🎉 {period} model retraining completed successfully!")
            
        except Exception as e:
            error_msg = f"Error retraining {period} model: {str(e)}"
            result['error'] = error_msg
            result['end_time'] = datetime.now().isoformat()
            logger.error(f"❌ {error_msg}", exc_info=True)
        
        return result
    
    def prepare_training_data(self, data: pd.DataFrame, sequence_length: int, 
                            prediction_horizon: int, period: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[StandardScaler]]:
        """
        Prepare data for LSTM training
        
        Args:
            data: Historical economic data
            sequence_length: Number of time steps to use as input
            prediction_horizon: Number of months ahead to predict
            period: Model period ('1m', '3m', '6m') to determine feature columns
            
        Returns:
            Tuple of (X, y, scaler) or (None, None, None) if failed
        """
        try:
            logger.info("🔧 Preparing training data...")
            
            # Get the feature columns for this specific model
            required_features = self.model_feature_columns[period]
            
            # Check which features are available in the data
            available_features = [col for col in required_features if col in data.columns]
            missing_features = [col for col in required_features if col not in data.columns]
            
            if len(available_features) == 0:
                raise ValueError(f"No required feature columns found in data for {period} model")
            
            if missing_features:
                logger.warning(f"⚠️ Missing features for {period} model: {missing_features}")
                logger.info(f"📊 Using available features: {available_features}")
            
            # Use available features
            feature_data = data[available_features].copy()
            
            # Forward fill and interpolate missing values
            feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
            feature_data = feature_data.interpolate()
            
            # Remove any remaining NaN rows
            feature_data = feature_data.dropna()
            
            if len(feature_data) < sequence_length + prediction_horizon:
                raise ValueError(f"Insufficient data: need at least {sequence_length + prediction_horizon} rows, got {len(feature_data)}")
            
            # Scale the features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(feature_data)
            
            # Get recession labels - check if 'recession' column exists in the original data
            if 'recession' in data.columns:
                logger.info("📊 Using existing recession labels from data")
                recession_labels = data['recession'].values
                # Align with feature_data after cleaning
                recession_labels = recession_labels[data.index.isin(feature_data.index)]
            else:
                logger.info("📊 Creating synthetic recession labels based on economic indicators")
                recession_labels = self.create_recession_labels(feature_data)
            
            # Create sequences for LSTM
            X, y = self.create_sequences(scaled_data, recession_labels, sequence_length, prediction_horizon)
            
            logger.info(f"✅ Training data prepared: {len(X)} sequences, {len(available_features)} features")
            
            return X, y, scaler
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data: {str(e)}")
            return None, None, None
    
    def create_recession_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create recession labels based on economic indicators
        This is a simplified approach - in production, you'd use actual NBER recession dates
        
        Args:
            data: Economic data DataFrame
            
        Returns:
            Array of recession labels (0 = no recession, 1 = recession)
        """
        try:
            # Simple heuristic: recession if unemployment rate is rising and GDP is declining
            labels = np.zeros(len(data))
            
            if 'unemployment_rate' in data.columns:
                # Calculate rolling changes
                unemployment_change = data['unemployment_rate'].rolling(window=3).mean().diff()
                
                if 'gdp' in data.columns:
                    gdp_change = data['gdp'].rolling(window=2).mean().diff()
                    # Recession if unemployment rising and GDP falling
                    recession_mask = (unemployment_change > 0.1) & (gdp_change < -1.0)
                else:
                    # Just use unemployment if GDP not available
                    recession_mask = unemployment_change > 0.2
                
                labels[recession_mask] = 1
            
            # Smooth the labels to avoid single-month spikes
            labels = pd.Series(labels).rolling(window=3, center=True).mean().fillna(0).values
            labels = (labels > 0.3).astype(int)  # Convert back to binary
            
            recession_count = np.sum(labels)
            logger.info(f"📊 Created recession labels: {recession_count}/{len(labels)} recession periods")
            
            return labels
            
        except Exception as e:
            logger.error(f"❌ Error creating recession labels: {str(e)}")
            # Return all zeros as fallback
            return np.zeros(len(data))
    
    def create_sequences(self, scaled_data: np.ndarray, labels: np.ndarray, 
                        sequence_length: int, prediction_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            scaled_data: Scaled feature data
            labels: Recession labels
            sequence_length: Length of input sequences
            prediction_horizon: How many steps ahead to predict
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(len(scaled_data) - sequence_length - prediction_horizon + 1):
            # Input sequence
            sequence = scaled_data[i:i + sequence_length]
            
            # Target (recession label at future time point)
            target_idx = i + sequence_length + prediction_horizon - 1
            target = labels[target_idx]
            
            X.append(sequence)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, sequence_length: int, num_features: int) -> tf.keras.Model:
        """
        Build LSTM model architecture
        
        Args:
            sequence_length: Length of input sequences
            num_features: Number of features
            
        Returns:
            Compiled Keras model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self.focal_loss(gamma=2., alpha=0.25),
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def get_training_callbacks(self) -> list:
        """Get callbacks for model training"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0
            )
        ]
        return callbacks
    
    def backup_existing_model(self, period: str, config: Dict[str, Any]) -> bool:
        """
        Create a backup of existing model before replacing it
        
        Args:
            period: Model period ('1m', '3m', '6m')
            config: Model configuration
            
        Returns:
            bool: True if backup successful or no existing model, False on error
        """
        try:
            model_path = config['model_path']
            scaler_path = config['scaler_path']
            
            # Check if existing files exist
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger.info(f"📁 No existing {period} model to backup")
                return True
            
            # Create backup paths with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_model_path = f"{model_path}.backup_{timestamp}"
            backup_scaler_path = f"{scaler_path}.backup_{timestamp}"
            
            # Copy existing files to backup locations
            import shutil
            shutil.copy2(model_path, backup_model_path)
            shutil.copy2(scaler_path, backup_scaler_path)
            
            logger.info(f"💾 Backed up existing {period} model:")
            logger.info(f"    Model: {backup_model_path}")
            logger.info(f"    Scaler: {backup_scaler_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error backing up existing {period} model: {str(e)}")
            return False
    
    async def save_model_artifacts(self, model: tf.keras.Model, scaler: StandardScaler, 
                                 period: str, config: Dict[str, Any]):
        """
        Save trained model and scaler to disk (with backup of existing model)
        
        Args:
            model: Trained Keras model
            scaler: Fitted StandardScaler
            period: Model period ('1m', '3m', '6m')
            config: Model configuration
        """
        try:
            # Ensure directories exist
            model_dir = os.path.dirname(config['model_path'])
            scaler_dir = os.path.dirname(config['scaler_path'])
            
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(scaler_dir, exist_ok=True)
            
            # Backup existing model
            if not self.backup_existing_model(period, config):
                logger.warning(f"⚠️ Failed to backup existing {period} model, proceeding anyway")
            
            # Save new model
            model.save(config['model_path'])
            logger.info(f"✅ New model saved: {config['model_path']}")
            
            # Save new scaler
            joblib.dump(scaler, config['scaler_path'])
            logger.info(f"✅ New scaler saved: {config['scaler_path']}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model artifacts: {str(e)}")
            raise
    
    async def reload_prediction_service(self, period: str):
        """
        Reload the corresponding prediction service with the new model
        
        Args:
            period: Model period ('1m', '3m', '6m')
        """
        try:
            logger.info(f"🔄 Reloading {period} prediction service...")
            
            # Import and reinitialize the corresponding service
            if period == '1m':
                from services.forecast_service_1m import initialize_1m_service
                success = initialize_1m_service()
            elif period == '3m':
                from services.forecast_service_3m import initialize_3m_service  
                success = initialize_3m_service()
            elif period == '6m':
                from services.forecast_service_6m import initialize_6m_service
                success = initialize_6m_service()
            else:
                raise ValueError(f"Invalid period: {period}")
            
            if success:
                logger.info(f"✅ {period} prediction service reloaded successfully")
            else:
                logger.error(f"❌ Failed to reload {period} prediction service")
            
        except Exception as e:
            logger.error(f"❌ Error reloading {period} prediction service: {str(e)}")
    
    async def retrain_all_models(self) -> Dict[str, Any]:
        """
        Retrain all models (1m, 3m, 6m)
        
        Returns:
            Dict with results for all models
        """
        logger.info("🚀 Starting retraining for all models...")
        
        results = {
            'success': False,
            'models_retrained': [],
            'models_failed': [],
            'details': {}
        }
        
        for period in ['1m', '3m', '6m']:
            try:
                result = await self.retrain_model(period)
                results['details'][period] = result
                
                if result['success']:
                    results['models_retrained'].append(period)
                else:
                    results['models_failed'].append(period)
                    
            except Exception as e:
                logger.error(f"❌ Critical error retraining {period} model: {str(e)}")
                results['models_failed'].append(period)
                results['details'][period] = {
                    'success': False,
                    'error': str(e)
                }
        
        results['success'] = len(results['models_retrained']) > 0
        
        logger.info(f"🎉 All models retraining complete. Success: {len(results['models_retrained'])}, Failed: {len(results['models_failed'])}")
        
        return results