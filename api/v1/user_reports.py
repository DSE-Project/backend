"""
API routes for managing user PDF reports stored in Supabase storage
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from typing import List, Optional
import logging
from utils.supabase_client import supabase

router = APIRouter(prefix="/api/v1/user-reports", tags=["User Reports"])

logger = logging.getLogger(__name__)

@router.get("/list/{user_id}")
async def list_user_reports(user_id: str):
    """
    List all PDF reports for a specific user from Supabase storage
    """
    try:
        # List all files in the user's folder in the storage bucket
        result = supabase.storage.from_("user-reports").list(user_id)
        
        if not result:
            return {"success": True, "reports": []}
        
        # Format the report data
        reports = []
        for file in result:
            if file.get('name') and file['name'].endswith('.pdf'):
                # Get public URL for the file
                public_url = supabase.storage.from_("user-reports").get_public_url(f"{user_id}/{file['name']}")
                
                reports.append({
                    "name": file['name'],
                    "size": file.get('metadata', {}).get('size', 0),
                    "created_at": file.get('created_at'),
                    "updated_at": file.get('updated_at'),
                    "public_url": public_url['publicURL'] if 'publicURL' in public_url else None,
                    "download_path": f"{user_id}/{file['name']}"
                })
        
        # Sort by creation date (newest first)
        reports.sort(key=lambda x: x['created_at'] or '', reverse=True)
        
        return {
            "success": True, 
            "reports": reports,
            "total_count": len(reports)
        }
        
    except Exception as e:
        logger.error(f"Error listing user reports for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch user reports: {str(e)}"
        )

@router.get("/download/{user_id}/{file_name}")
async def download_report(user_id: str, file_name: str):
    """
    Get a signed URL to download a specific PDF report
    """
    try:
        # Generate a signed URL that expires in 1 hour (3600 seconds)
        signed_url_response = supabase.storage.from_("user-reports").create_signed_url(
            f"{user_id}/{file_name}", 
            expires_in=3600  # 1 hour
        )
        
        if 'error' in signed_url_response:
            raise HTTPException(
                status_code=404, 
                detail=f"Report not found: {signed_url_response['error']}"
            )
        
        # Redirect to the signed URL for direct download
        return RedirectResponse(url=signed_url_response['signedURL'])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report {file_name} for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate download URL: {str(e)}"
        )

@router.delete("/delete/{user_id}/{file_name}")
async def delete_report(user_id: str, file_name: str):
    """
    Delete a specific PDF report from storage
    """
    try:
        # Delete the file from storage
        result = supabase.storage.from_("user-reports").remove([f"{user_id}/{file_name}"])
        
        if not result:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "success": True, 
            "message": f"Report {file_name} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting report {file_name} for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete report: {str(e)}"
        )

@router.get("/info/{user_id}/{file_name}")
async def get_report_info(user_id: str, file_name: str):
    """
    Get metadata information about a specific PDF report
    """
    try:
        # Get file info from storage
        result = supabase.storage.from_("user-reports").list(user_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="User folder not found")
        
        # Find the specific file
        file_info = None
        for file in result:
            if file.get('name') == file_name:
                file_info = file
                break
        
        if not file_info:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Get public URL for the file
        public_url = supabase.storage.from_("user-reports").get_public_url(f"{user_id}/{file_name}")
        
        return {
            "success": True,
            "report": {
                "name": file_info['name'],
                "size": file_info.get('metadata', {}).get('size', 0),
                "created_at": file_info.get('created_at'),
                "updated_at": file_info.get('updated_at'),
                "public_url": public_url['publicURL'] if 'publicURL' in public_url else None,
                "download_path": f"{user_id}/{file_name}"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting info for report {file_name} for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get report info: {str(e)}"
        )