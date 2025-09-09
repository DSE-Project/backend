import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1 import forecast

# Create the FastAPI app instance
app = FastAPI(
    title="RecessionScope API",
    description="API for the US Recession Forecasting System - Provides 1, 3, and 6-month recession probability predictions using machine learning models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # No trailing slash
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include the forecast router
app.include_router(forecast.router, prefix="/api/v1/forecast", tags=["Forecasting"])

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
            "simulate_1m": "/api/v1/forecast/simulate/1m"
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
        

    except Exception as e:
        print(f"⚠️ Warning: Could not initialize some services: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)