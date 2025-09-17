import pandas as pd
from utils.supabase_client import supabase
from schemas.forecast_schema_1m import InputFeatures1M
from schemas.forecast_schema_3m import InputFeatures3M
from schemas.forecast_schema_6m import InputFeatures6M, CurrentMonthData6M


def fetch_latest_from_db(table_name: str) -> dict:
    """
    Fetch latest row from Supabase table and return as dict
    """
    response = supabase.table(table_name).select("*").order("observation_date", desc=True).limit(1).execute()

    if not response.data:
        raise ValueError(f"No data found in table {table_name}")

    return response.data[0]  # dict of latest row


def prepare_features_1m(user_features: dict) -> InputFeatures1M:
    latest_row = fetch_latest_from_db("historical_data_1m")
    
    # Check if this is a simple simulation (only 3 features) or advanced
    is_simple_simulation = all(key in ['UNRATE', 'CPIFOOD', 'TB3MS'] for key in user_features.keys())
    
    if is_simple_simulation:
        # For simple simulation, update only the user-provided features, keep others from database
        for key in ['UNRATE', 'CPIFOOD', 'TB3MS']:
            if key in user_features:
                latest_row[key] = user_features[key]
    else:
        # For advanced simulation, use all user-provided features
        latest_row.update(user_features)

    if "observation_date" not in latest_row:
        latest_row["observation_date"] = "latest"

    return InputFeatures1M(
        current_month_data=latest_row,
        historical_data_source="db",
        use_historical_data=True
    )


def prepare_features_3m(user_features: dict) -> InputFeatures3M:
    latest_row = fetch_latest_from_db("historical_data_3m")
    
    # Check if this is a simple simulation (only 3 features) or advanced
    is_simple_simulation = all(key in ['UNRATE', 'CPIFOOD', 'TB3MS'] for key in user_features.keys())
    
    if is_simple_simulation:
        # For simple simulation, update only the user-provided features, keep others from database
        for key in ['UNRATE', 'CPIFOOD', 'TB3MS']:
            if key in user_features:
                latest_row[key] = user_features[key]
    else:
        # For advanced simulation, use all user-provided features
        latest_row.update(user_features)

    if "observation_date" not in latest_row:
        latest_row["observation_date"] = "latest"

    return InputFeatures3M(
        current_month_data=latest_row,
        historical_data_source="db",
        use_historical_data=True
    )


def prepare_features_6m(user_features: dict) -> InputFeatures6M:
    latest_row = fetch_latest_from_db("historical_data_6m")
    
    # Check if this is a simple simulation (only 3 features) or advanced
    is_simple_simulation = all(key in ['UNRATE', 'CPIFOOD', 'TB3MS'] for key in user_features.keys())
    
    if is_simple_simulation:
        # For simple simulation, update only the user-provided features, keep others from database
        for key in ['UNRATE', 'CPIFOOD', 'TB3MS']:
            if key in user_features:
                latest_row[key] = user_features[key]
    else:
        # For advanced simulation, use all user-provided features
        latest_row.update(user_features)

    if "observation_date" not in latest_row:
        latest_row["observation_date"] = "latest"

    # Wrap inside schema
    current_data = CurrentMonthData6M(**latest_row)

    return InputFeatures6M(
        current_month_data=current_data,
        historical_data_source="db",
        use_historical_data=True
    )