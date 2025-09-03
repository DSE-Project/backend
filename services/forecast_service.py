import numpy as np
import tensorflow as tf
from schemas.forecast_schema import InputFeatures

# Paths to the trained models
MODEL_1M_PATH = "ml_models/model_1m.h5"
MODEL_3M_PATH = "ml_models/model_3m.h5"
MODEL_6M_PATH = "ml_models/model_6m.h5"

# Load the models once when the service is initialized
# This is much more efficient than loading them on every request.
try:
    model_1m = tf.keras.models.load_model(MODEL_1M_PATH)
    model_3m = tf.keras.models.load_model(MODEL_3M_PATH)
    model_6m = tf.keras.models.load_model(MODEL_6M_PATH)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model_1m = model_3m = model_6m = None

def get_predictions(features: InputFeatures) -> dict:
    """
    Takes input features, prepares them, and returns predictions from all three models.
    """
    if not all([model_1m, model_3m, model_6m]):
        raise RuntimeError("Models are not loaded. Cannot make predictions.")

    # 1. Convert the Pydantic model to a list or NumPy array.
    # IMPORTANT: The order of features MUST match the order used during model training.
    input_data = np.array([[
        features.unemployment_rate,
        features.inflation_rate,
        features.gdp_growth,
        features.interest_rate,
        # ... ensure all features are here in the correct order
    ]])

    # 2. Make predictions using each model
    pred_1m = model_1m.predict(input_data)
    pred_3m = model_3m.predict(input_data)
    pred_6m = model_6m.predict(input_data)

    # 3. Format the predictions into the required response dictionary.
    # The output of .predict() might be a nested array, so we extract the scalar value.
    response = {
        "prob_1m": float(pred_1m[0][0]),
        "prob_3m": float(pred_3m[0][0]),
        "prob_6m": float(pred_6m[0][0]),
    }

    return response