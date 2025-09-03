import numpy as np
import tensorflow as tf
from schemas.forecast_schema import AllInputFeatures, InputFeatures1M, InputFeatures3M, InputFeatures6M

# Paths to the trained models
MODEL_1M_PATH = "ml_models/model_1m.h5"
MODEL_3M_PATH = "ml_models/model_3m.h5"
MODEL_6M_PATH = "ml_models/model_6m.h5"

# Load the models once when the service is initialized
try:
    model_1m = tf.keras.models.load_model(MODEL_1M_PATH)
    model_3m = tf.keras.models.load_model(MODEL_3M_PATH)
    model_6m = tf.keras.models.load_model(MODEL_6M_PATH)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model_1m = model_3m = model_6m = None

def get_predictions(all_features: AllInputFeatures) -> dict:
    """
    Takes input features for all models and returns predictions from all three models.
    """
    if not all([model_1m, model_3m, model_6m]):
        raise RuntimeError("Models are not loaded. Cannot make predictions.")

    # Prepare input data for each model with features in the correct order
    
    # 1M model features (df1 dataset)
    input_1m = np.array([[
        all_features.features_1m.ICSA,
        all_features.features_1m.USWTRADE,
        all_features.features_1m.UMCSENT,
        all_features.features_1m.BBKMLEIX,
        all_features.features_1m.TB6MS,
        all_features.features_1m.SRVPRD,
        all_features.features_1m.COMLOAN,
        all_features.features_1m.CPIMEDICARE,
        all_features.features_1m.USINFO,
        all_features.features_1m.USGOOD,
        all_features.features_1m.PSAVERT,
        all_features.features_1m.UNEMPLOY,
        all_features.features_1m.PPIACO,
        all_features.features_1m.fedfunds,
        all_features.features_1m.CPIAPP,
        all_features.features_1m.TCU,
        all_features.features_1m.TB3MS,
        all_features.features_1m.SECURITYBANK,
        all_features.features_1m.CSUSHPISA,
        all_features.features_1m.MANEMP
    ]])

    # 3M model features (df2 dataset)
    input_3m = np.array([[
        all_features.features_3m.ICSA,
        all_features.features_3m.CPIMEDICARE,
        all_features.features_3m.USWTRADE,
        all_features.features_3m.BBKMLEIX,
        all_features.features_3m.COMLOAN,
        all_features.features_3m.UMCSENT,
        all_features.features_3m.MANEMP,
        all_features.features_3m.fedfunds,
        all_features.features_3m.PSTAX,
        all_features.features_3m.USCONS,
        all_features.features_3m.USGOOD,
        all_features.features_3m.USINFO,
        all_features.features_3m.CPIAPP,
        all_features.features_3m.CSUSHPISA,
        all_features.features_3m.SECURITYBANK,
        all_features.features_3m.SRVPRD,
        all_features.features_3m.INDPRO,
        all_features.features_3m.TB6MS,
        all_features.features_3m.UNEMPLOY,
        all_features.features_3m.USTPU
    ]])

    # 6M model features (df2 dataset)
    input_6m = np.array([[
        all_features.features_6m.PSTAX,
        all_features.features_6m.USWTRADE,
        all_features.features_6m.MANEMP,
        all_features.features_6m.CPIAPP,
        all_features.features_6m.CSUSHPISA,
        all_features.features_6m.ICSA,
        all_features.features_6m.fedfunds,
        all_features.features_6m.BBKMLEIX,
        all_features.features_6m.TB3MS,
        all_features.features_6m.USINFO,
        all_features.features_6m.PPIACO,
        all_features.features_6m.CPIMEDICARE,
        all_features.features_6m.UNEMPLOY,
        all_features.features_6m.TB1YR,
        all_features.features_6m.USGOOD,
        all_features.features_6m.CPIFOOD,
        all_features.features_6m.UMCSENT,
        all_features.features_6m.SRVPRD,
        all_features.features_6m.GDP,
        all_features.features_6m.INDPRO
    ]])

    # Make predictions using each model
    pred_1m = model_1m.predict(input_1m)
    pred_3m = model_3m.predict(input_3m)
    pred_6m = model_6m.predict(input_6m)

    # Format the predictions
    response = {
        "prob_1m": float(pred_1m[0][0]),
        "prob_3m": float(pred_3m[0][0]),
        "prob_6m": float(pred_6m[0][0]),
    }

    return response

# Individual prediction functions to call the models separately
def get_1m_prediction(features: InputFeatures1M) -> float:
    if not model_1m:
        raise RuntimeError("1M model is not loaded.")
    
    input_data = np.array([[
        features.ICSA, features.USWTRADE, features.UMCSENT, features.BBKMLEIX,
        features.TB6MS, features.SRVPRD, features.COMLOAN, features.CPIMEDICARE,
        features.USINFO, features.USGOOD, features.PSAVERT, features.UNEMPLOY,
        features.PPIACO, features.fedfunds, features.CPIAPP, features.TCU,
        features.TB3MS, features.SECURITYBANK, features.CSUSHPISA, features.MANEMP
    ]])
    
    pred = model_1m.predict(input_data)
    return float(pred[0][0])

def get_3m_prediction(features: InputFeatures3M) -> float:
    if not model_3m:
        raise RuntimeError("3M model is not loaded.")
    
    input_data = np.array([[
        features.ICSA, features.CPIMEDICARE, features.USWTRADE, features.BBKMLEIX,
        features.COMLOAN, features.UMCSENT, features.MANEMP, features.fedfunds,
        features.PSTAX, features.USCONS, features.USGOOD, features.USINFO,
        features.CPIAPP, features.CSUSHPISA, features.SECURITYBANK, features.SRVPRD,
        features.INDPRO, features.TB6MS, features.UNEMPLOY, features.USTPU
    ]])
    
    pred = model_3m.predict(input_data)
    return float(pred[0][0])

def get_6m_prediction(features: InputFeatures6M) -> float:
    if not model_6m:
        raise RuntimeError("6M model is not loaded.")
    
    input_data = np.array([[
        features.PSTAX, features.USWTRADE, features.MANEMP, features.CPIAPP,
        features.CSUSHPISA, features.ICSA, features.fedfunds, features.BBKMLEIX,
        features.TB3MS, features.USINFO, features.PPIACO, features.CPIMEDICARE,
        features.UNEMPLOY, features.TB1YR, features.USGOOD, features.CPIFOOD,
        features.UMCSENT, features.SRVPRD, features.GDP, features.INDPRO
    ]])
    
    pred = model_6m.predict(input_data)
    return float(pred[0][0])


