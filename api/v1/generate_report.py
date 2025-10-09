from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from io import BytesIO

from services.database_service import db_service
from services.pdf_utils import render_url_to_pdf_sync

router = APIRouter()

# Pydantic models
class ReportRequest(BaseModel):
    htmlContent: str

@router.get("/generate-report", tags=["Reports"])
async def generate_report(url: str = Query(...), filename: str = Query("report.pdf")):
    """Generate a PDF report from a given URL"""
    try:
        pdf_bytes = await run_in_threadpool(render_url_to_pdf_sync, url)
        pdf_file = BytesIO(pdf_bytes)
        pdf_file.seek(0)
        return StreamingResponse(
            pdf_file,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@router.get("/last-two/{table_name}", tags=["Database"])
async def get_last_two(table_name: str):
    """Get the last two records from a specified database table"""
    try:
        df = db_service.load_last_n_rows(table_name, n=2)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Convert dataframe to JSON
        return df.reset_index().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))