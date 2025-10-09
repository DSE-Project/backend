import sys
import os

import asyncio

# Fix for Playwright on Windows (NotImplementedError)
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import pdfkit
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

import base64
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from io import BytesIO
from dotenv import load_dotenv

# Import routers
from api.v1.forecast import router as forecast_router
from api.v1.yearly_risk import router as yearly_risk_router
from api.v1.macro_indicators import router as macro_indicators_router
from api.v1.economic_charts import router as economic_charts_router
from api.v1 import economic

from io import BytesIO
from services.database_service import db_service
from middleware.priority_middleware import PriorityMiddleware
from services.pdf_utils import render_url_to_pdf_sync
# Configure pdfkit for cross-platform compatibility
# Try to find wkhtmltopdf automatically, or use None if not available
try:
    import shutil
    wkhtmltopdf_path = shutil.which('wkhtmltopdf')
    if wkhtmltopdf_path:
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
    else:
        config = None
        print("⚠️ wkhtmltopdf not found. PDF generation will not be available.")
except Exception as e:
    config = None
    print(f"⚠️ Could not configure pdfkit: {e}")

from api.v1.sentiment_component import router as sentiment_router
from api.v1.scheduler import router as scheduler_router
from api.v1.simulate import router as simulate_router
from api.v1.explainability import router as explainability_router

# Load environment variables
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

import logging

logging.basicConfig(
    level=logging.INFO,  # INFO and ERROR logs will be printed
    format="%(asctime)s [%(levelname)s] %(message)s"
)

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    print("⚠️ .env file not found. Make sure to create one with SUPABASE_URL and SUPABASE_ANON_KEY")

# Create the FastAPI app instance
app = FastAPI(
    title="RecessionScope API",
    description="API for the US Recession Forecasting System - Provides 1, 3, and 6-month recession probability predictions using machine learning models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["*"],  # Configure this properly for production
    allow_origins=[
        "http://localhost:5173",
        "https://localhost:5173",
        "http://127.0.0.1:5173",
        "https://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add priority request middleware (should be after CORS)
app.add_middleware(PriorityMiddleware, enable_logging=True)

# Include the routers
app.include_router(forecast_router, prefix="/api/v1/forecast", tags=["Forecasting"])

app.include_router(yearly_risk_router, prefix="/api/v1", tags=["yearly-risk"])
app.include_router(macro_indicators_router, prefix="/api/v1", tags=["macro-indicators"])
app.include_router(economic_charts_router, prefix="/api/v1", tags=["economic-charts"])
app.include_router(simulate_router, prefix="/api/v1/simulate", tags=["Simulation"])
app.include_router(economic.router, prefix="/api/v1/economic")
app.include_router(sentiment_router, prefix="/api/v1/sentiment", tags=["Sentiment Analysis"])
app.include_router(scheduler_router, prefix="/api/v1", tags=["FRED Data Scheduler"])
app.include_router(explainability_router, prefix="/api/v1/forecast", tags=["Model Explainability"])

# FRED Cache monitoring endpoint
from services.shared_fred_date_service import shared_fred_date_service

@app.get("/api/v1/fred-cache/status", tags=["FRED Cache"])
async def get_fred_cache_status():
    """Get the current status of the FRED date cache"""
    return {
        "cache_info": shared_fred_date_service.get_cache_info(),
        "description": "Shows whether FRED date is cached and cache validity"
    }

@app.post("/api/v1/fred-cache/clear", tags=["FRED Cache"])
async def clear_fred_cache():
    """Clear the FRED date cache (for testing purposes)"""
    shared_fred_date_service.clear_cache()
    return {"message": "FRED date cache cleared successfully"}

@app.get("/", tags=["Root"])
async def read_root():
    """Welcome endpoint with API information"""
    return {
        "message": "Welcome to the RecessionScope API!",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "all_predictions": "/api/v1/forecast/predict/all",
            "1m_prediction": "/api/v1/forecast/predict/1m",
            "3m_prediction": "/api/v1/forecast/predict/3m", 
            "6m_prediction": "/api/v1/forecast/predict/6m",
            "yearly_risk": "/api/v1/yearly-risk",
            "macro_indicators": "/api/v1/macro-indicators",
            "economic_charts": "/api/v1/economic-charts/historical-data",
            "chart_statistics": "/api/v1/economic-charts/summary-stats",
            "scheduler_status": "/api/v1/scheduler/status",
            "scheduler_health": "/api/v1/scheduler/health",
            "prediction_cache_stats": "/api/v1/forecast/cache/stats",
            "prediction_cache_clear": "/api/v1/forecast/cache/clear",
            "fred_cache_stats": "/api/v1/macro-indicators/cache/stats",
            "fred_cache_clear": "/api/v1/macro-indicators/cache/clear",
            "economic_charts_cache_stats": "/api/v1/economic-charts/cache/stats",
            "economic_charts_cache_clear": "/api/v1/economic-charts/cache/clear",
            "cache_stats": "/api/v1/forecast/cache/stats",
            "cache_clear": "/api/v1/forecast/cache/clear",
            "priority_stats": "/api/v1/forecast/priority/stats"

        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RecessionScope API"}

# Optional: Add startup event to initialize services
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # 1️⃣ Load historical data
        df = db_service.load_historical_data("historical_data_1m")
    
        
        # 2️⃣ Validate with Pandera
        if df is not None:
            validated_df = db_service.validate_dataframe(df, "historical_data_1m")
            if validated_df is None:
                print("⚠️ Data validation failed. Check logs for details.")
            else:
                print("✅ Data validation passed for historical_data_1m")
        else:
            print("⚠️ No data loaded from historical_data_1m")

        # Import here to avoid circular imports
        from services.forecast_service_1m import initialize_1m_service
        from services.forecast_service_3m import initialize_3m_service
        from services.forecast_service_6m import initialize_6m_service
        from services.fred_data_scheduler import fred_scheduler

        print("\n🚀 Initializing forecasting services...")
        
        # Initialize 1M service
        if initialize_1m_service():
            print("✅ 1M forecasting service initialized successfully")
        else:
            print("⚠️ Warning: 1M forecasting service failed to initialize")
        
        if initialize_3m_service():
            print("✅ 3M forecasting service initialized successfully")
        else:
            print("⚠️ 3M forecasting service failed to initialize")
        
        if initialize_6m_service():
            print("✅ 6M forecasting service initialized successfully")
        else:
            print("⚠️ 6M forecasting service failed to initialize")
        
        # Initialize FRED Data Scheduler
        print("\n📅 Initializing FRED Data Scheduler...")
        try:
            await fred_scheduler.start_scheduler()
            print("✅ FRED Data Scheduler initialized successfully")
        except Exception as scheduler_error:
            print(f"⚠️ Warning: FRED Data Scheduler failed to initialize: {scheduler_error}")

    except Exception as e:
        print(f"⚠️ Warning: Could not initialize some services: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on shutdown"""
    try:
        from services.fred_data_scheduler import fred_scheduler
        
        print("\n🛑 Shutting down services...")
        await fred_scheduler.stop_scheduler()
        print("✅ FRED Data Scheduler stopped successfully")
        
    except Exception as e:
        print(f"⚠️ Warning: Error during shutdown: {e}")

# Pydantic models
class ReportRequest(BaseModel):
    htmlContent: str

@app.get("/generate-report")
async def generate_report(url: str = Query(...), filename: str = Query("report.pdf")):
    pdf_bytes = await run_in_threadpool(render_url_to_pdf_sync, url)
    pdf_file = BytesIO(pdf_bytes)
    pdf_file.seek(0)
    return StreamingResponse(
        pdf_file,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# @app.get("/last-two/{table_name}")
# def get_last_two(table_name: str):
#     try:
#         df = db_service.load_last_n_rows(table_name, n=2)
#         if df is None or df.empty:
#             raise HTTPException(status_code=404, detail="No data found")
        
#         # Convert dataframe to JSON
#         return df.reset_index().to_dict(orient="records")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/last-two/{table_name}")
# def get_last_two(table_name: str):
#     try:
#         df = db_service.load_last_n_rows(table_name, n=2)
#         if df is None or df.empty:
#             raise HTTPException(status_code=404, detail="No data found")
        
#         # Convert dataframe to JSON
#         return df.reset_index().to_dict(orient="records")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/last-two/{table_name}")
def get_last_two(table_name: str):
    try:
        df = db_service.load_last_n_rows(table_name, n=10)  # load a few more rows just in case
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        

        df = df.dropna(how="all")
        df = df.tail(2)

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df.reset_index(drop=True).to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)