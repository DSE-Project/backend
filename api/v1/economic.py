from fastapi import APIRouter
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()

@router.get("/csv")
def get_csv():
    csv_path = Path.cwd() / "data" / "historical_data_1m.csv"
    if not csv_path.exists():
        return {"error": f"CSV file not found at {csv_path}"}
    return FileResponse(csv_path, media_type="text/csv", filename="historical_data_1m.csv")
