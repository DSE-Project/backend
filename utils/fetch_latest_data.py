# utils/fetch_latest_data.py

import pandas as pd
from fredapi import Fred
import os

# -------------------------------
# CONFIG
# -------------------------------

API_KEY = 'ec45cf5fdb0f214fe7b404307f774615'  # Replace with your FRED API key
DATA_FOLDER = 'data'
FILENAME = 'last_60_months.csv'

# Map training names to actual FRED series IDs
FEATURE_MAP = {
    'ICSA': 'ICSA',
    'USWTRADE': 'USWTRADE',
    'UMCSENT': 'UMCSENT',
    'BBKMLEIX': 'BBKMLEIX',
    'TB6MS': 'TB6MS',
    'SRVPRD': 'SRVPRD',
    'COMLOAN': 'COMLOAN',
    'CPIMEDICARE': 'CPIMEDNS',  # FRED series ID
    'USINFO': 'USINFO',
    'USGOOD': 'USGOOD',
    'PSAVERT': 'PSAVERT',
    'UNEMPLOY': 'UNEMPLOY',
    'PPIACO': 'PPIACO',
    'fedfunds': 'FEDFUNDS',
    'CPIAPP': 'CPIAUCNS',  # FRED series ID
    'TCU': 'TCU',
    'TB3MS': 'TB3MS',
    'SECURITYBANK': 'BUSLOANS',  # FRED series ID
    'CSUSHPISA': 'CSUSHPISA',
    'MANEMP': 'MANEMP'
}

# -------------------------------
# FETCH DATA
# -------------------------------

def fetch_latest_us_data(api_key=API_KEY, feature_map=FEATURE_MAP, months=60):
    fred = Fred(api_key=api_key)
    data = pd.DataFrame()

    print("Fetching US economic data from FRED...")
    for name, fred_id in feature_map.items():
        try:
            series = fred.get_series(fred_id)
            # Resample to the 1st of each month
            series.index = pd.to_datetime(series.index)
            series = series.resample('MS').first()  # 'MS' = Month Start
            data[name] = series
            print(f"Fetched: {name} (FRED ID: {fred_id})")
        except Exception as e:
            print(f"Error fetching {name} ({fred_id}): {e}")

    # Keep only last `months` rows
    data = data.tail(months)

    # Ensure data folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Save to CSV
    filepath = os.path.join(DATA_FOLDER, FILENAME)
    data.to_csv(filepath, index=True)
    print(f"Data saved to {filepath}")

    return data

# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":
    df = fetch_latest_us_data()
    print(df.tail())
