import httpx
import json
import os

# Your FRED API key (set via environment variable or hardcode for testing)
FRED_API_KEY = "ef123a07a5f12077a0144db1f8cabf0d"

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Dictionary mapping indicator names to their FRED series IDs
SERIES_IDS = {
    "fedfunds": "FEDFUNDS",         # Federal funds rate
    "TB3MS": "TB3MS",               # 3-Month Treasury Constant Maturity Rate
    "TB6MS": "TB6MS",               # 6-Month Treasury Constant Maturity Rate
    "TB1YR": "TB1YR",               # 1-Year Treasury Constant Maturity Rate
    "USTPU": "USTPU",               # US Total Private Units
    "USGOOD": "USGOOD",             # US Goods
    "SRVPRD": "SRVPRD",             # Services Production
    "USCONS": "USCONS",             # US Construction
    "MANEMP": "MANEMP",             # Manufacturing Employment
    "USWTRADE": "USWTRADE",         # US Wholesale Trade
    "USTRADE": "USTRADE",           # US Trade
    "USINFO": "USINFO",             # US Information
    "UNRATE": "UNRATE",             # Unemployment Rate
    "UNEMPLOY": "UNEMPLOY",         # Unemployment Level
    "CPIFOOD": "CUSR0000SAF11",     # Consumer Price Index for All Urban Consumers: Food at Home in U.S. City Average
    "CPIMEDICARE": "CPIMEDSL",      # Consumer Price Index for All Urban Consumers: Medical Care in U.S. City Average
    "CPIRENT": "CUUR0000SEHA",      # Consumer Price Index for All Urban Consumers: Rent of Primary Residence in U.S. City Average
    "CPIAPP": "CPIAPPSL",           # Consumer Price Index for All Urban Consumers: Apparel in U.S. City Average
    "GDP": "GDP",                   # Gross Domestic Product
    "REALGDP": "GDPC1",             # Real GDP
    "PCEPI": "PCEPI",               # Personal Consumption Expenditures Price Index
    "PSAVERT": "PSAVERT",           # Personal Saving Rate
    "PSTAX": "W055RC1Q027SBEA",     # Personal Tax (proxy series)
    "COMREAL": "BOGZ1FL075035503Q",     # Commercial Real Estate Price Index
    "COMLOAN": "H8B1023NCBCMG",          # Commercial and Industrial Loans
    "SECURITYBANK": "H8B1002NCBCMG",       # Security Bank Obligations (approximate series)
    "PPIACO": "PPIACO",             # Producer Price Index All Commodities
    "M1SL": "M1SL",                 # M1 Money Stock
    "M2SL": "M2SL",                  # M2 Money Stock
    "recession": "USREC"          # Recession indicator (0 or 1)
}

async def fetch_latest_observation(series_id: str):
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(BASE_URL, params=params)
        r.raise_for_status()
        return r.json()

async def main():
    results = {}
    for name, series_id in SERIES_IDS.items():
        try:
            data = await fetch_latest_observation(series_id)
            results[name] = data
        except Exception as e:
            print(f"Failed to fetch {name} ({series_id}): {e}")
            results[name] = {"error": str(e)}

    # Save to JSON file
    with open("fred_latest_data.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
