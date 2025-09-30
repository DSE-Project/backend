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


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.concurrency import run_in_threadpool
from services.pdf_utils import render_url_to_pdf_sync

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from api.v1.forecast import router as forecast_router
from api.v1.yearly_risk import router as yearly_risk_router
from api.v1.simulate import router as simulate_router
from api.v1.yearly_risk import router as yearly_risk_router
from api.v1.macro_indicators import router as macro_indicators_router
from api.v1.economic_charts import router as economic_charts_router
from fastapi.responses import StreamingResponse
from api.v1 import economic
from io import BytesIO

config = pdfkit.configuration(wkhtmltopdf=r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")

from api.v1.sentiment_component import router as sentiment_router
from api.v1.scheduler import router as scheduler_router

from dotenv import load_dotenv

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
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers
app.include_router(forecast_router, prefix="/api/v1/forecast", tags=["Forecasting"])
app.include_router(yearly_risk_router, prefix="/api/v1", tags=["yearly-risk"])
app.include_router(macro_indicators_router, prefix="/api/v1", tags=["macro-indicators"])
app.include_router(economic_charts_router, prefix="/api/v1", tags=["economic-charts"])
app.include_router(yearly_risk_router, prefix="/api/v1", tags=["yearly-risk"])
app.include_router(simulate_router, prefix="/api/v1/simulate", tags=["Simulation"])
app.include_router(economic.router, prefix="/api/v1/economic")
app.include_router(sentiment_router, prefix="/api/v1/sentiment", tags=["Sentiment Analysis"])
app.include_router(scheduler_router, prefix="/api/v1", tags=["FRED Data Scheduler"])

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
            "scheduler_health": "/api/v1/scheduler/health"
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

class ReportRequest(BaseModel):
    htmlContent: str

# @app.post("/generate-report")
# async def generate_report(request: ReportRequest):
#     pdf_bytes = pdfkit.from_string(request.htmlContent, False, configuration=config)
#     pdf_file = BytesIO(pdf_bytes)
#     pdf_file.seek(0)
#     return StreamingResponse(
#         pdf_file,
#         media_type="application/pdf",
#         headers={"Content-Disposition": "attachment; filename=report.pdf"}
#     )

@app.get("/generate-report")
async def generate_report(url: str = Query(...)):
    # Playwright can now access the public /reports-print route
    pdf_file = await run_in_threadpool(render_url_to_pdf_sync, url)
    return StreamingResponse(
        pdf_file,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename={filename}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)