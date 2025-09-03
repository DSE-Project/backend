# In main.py

from fastapi import FastAPI
from api import forecast

# Create the FastAPI app instance
app = FastAPI(
    title="RecessionScope API",
    # [cite_start]description="API for the US Recession Forecasting System[cite: 3, 62].",
    # version="1.0"
)

# Include the forecast router
# The prefix makes all endpoints in this router start with /api/v1
app.include_router(forecast.router, prefix="/api/v1", tags=["Forecasting"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the RecessionScope API!"}