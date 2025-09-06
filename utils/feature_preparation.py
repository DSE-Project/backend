import pandas as pd
from pathlib import Path
from schemas.forecast_schema_1m import InputFeatures1M
from schemas.forecast_schema_3m import InputFeatures3M
from schemas.forecast_schema_6m import InputFeatures6M,CurrentMonthData6M


DATA_FILE_1m = Path("data/historical_data_1m.csv")
DATA_FILE_3m = Path("data/historical_data_3m_new.csv")
DATA_FILE_6m = Path("data/historical_data_6m.csv")

def prepare_features_1m(user_features: dict) -> InputFeatures1M:
    df = pd.read_csv(DATA_FILE_1m, index_col=0)
    latest_row = df.iloc[-1].to_dict()

    # Merge user inputs with last row
    merged = {**latest_row, **user_features}

    # Ensure observation_date exists
    if "observation_date" not in merged:
        merged["observation_date"] = df.index[-1] if df.index.name == "observation_date" else "latest"

    # Wrap inside schema format
    return InputFeatures1M(
        current_month_data=merged,
        historical_data_source="csv",
        use_historical_data=True
    )


def prepare_features_3m(user_features: dict) -> InputFeatures3M:
    df = pd.read_csv(DATA_FILE_3m, index_col=0)
    latest_row = df.iloc[-1].to_dict()

    # Merge user input into the latest row
    merged = {**latest_row, **user_features}

    # Ensure observation_date exists
    if "observation_date" not in merged:
        merged["observation_date"] = df.index[-1] if df.index.name == "observation_date" else "latest"

    # Wrap inside schema format
    return InputFeatures3M(
        current_month_data=merged,
        historical_data_source="csv",
        use_historical_data=True
    )


def prepare_features_6m(user_features: dict) -> InputFeatures6M:
    """
    Merge user input with latest available values from CSV for 6-month prediction
    """
    # Load CSV
    df = pd.read_csv(DATA_FILE_6m, index_col=0)
    latest_row = df.iloc[-1].to_dict()  # last row in CSV

    # Merge user inputs with CSV defaults
    merged = {**latest_row, **user_features}

    # Ensure observation_date exists
    if "observation_date" not in merged:
        merged["observation_date"] = df.index[-1] if df.index.name == "observation_date" else "latest"

    # Wrap inside CurrentMonthData6M first if your schema expects it
    current_data = CurrentMonthData6M(**merged)

    # Wrap inside InputFeatures6M
    return InputFeatures6M(
        current_month_data=current_data,
        historical_data_source="csv",
        use_historical_data=True
    )
