import os
import time
import io
import sys
import threading
import uuid
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
from core import workflow, agentstate
from dub import (
    create_dub_job,
    poll_job_until_complete,
    download_url_bytes,
    srt_to_plain_text,
    generate_notes_from_text
)

app = FastAPI(title="Askify API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for download progress
download_progress: Dict[str, Dict] = {}

class AskIn(BaseModel):
    query: str

@app.post("/api/ask")
async def api_ask(payload: AskIn):
    state: agentstate = {
        "query": payload.query,
        "lang": "english",
        "explanation": "",
        "audio_url": None,
        "summary": "",
    }

    result = await workflow.ainvoke(state)

    explanation = result.get("explanation", "") or ""
    summary = result.get("summary", "") or ""
    audio_b64: Optional[str] = None
    audio_obj = result.get("audio_url", None)
    if isinstance(audio_obj, io.BytesIO):
        import base64
        audio_b64 = base64.b64encode(audio_obj.read()).decode("utf-8")

    return {
        "explanation": explanation,
        "summary": summary,
        "audio_b64": audio_b64
    }

class DubIn(BaseModel):
    youtube_url: str
    target_locale: str

class DownloadIn(BaseModel):
    youtube_url: str

def download_video_async(download_id: str, youtube_url: str):
    """Download video in background thread"""
    try:
        download_progress[download_id] = {
            "status": "downloading",
            "progress": 0,
            "message": "Starting download...",
            "file_path": None,
            "error": None
        }
        
        from dub import download_youtube_highest_mp4
        
        # Update progress
        download_progress[download_id]["message"] = "Initializing download..."
        download_progress[download_id]["progress"] = 10
        
        # Download the video
        download_progress[download_id]["message"] = "Downloading video from YouTube..."
        download_progress[download_id]["progress"] = 30
        
        # Update progress during download
        download_progress[download_id]["message"] = "Downloading video from YouTube..."
        download_progress[download_id]["progress"] = 50
        
        try:
            mp4_path = download_youtube_highest_mp4(youtube_url)
            
            # Verify the file exists and has content
            if not os.path.exists(mp4_path):
                raise FileNotFoundError(f"Downloaded file not found: {mp4_path}")
            
            file_size = os.path.getsize(mp4_path)
            if file_size == 0:
                raise ValueError(f"Downloaded file is empty: {mp4_path}")
            
            print(f"Download completed successfully: {mp4_path} ({file_size} bytes)")
            
            # Update progress
            download_progress[download_id]["message"] = "Download completed!"
            download_progress[download_id]["progress"] = 100
            download_progress[download_id]["status"] = "completed"
            download_progress[download_id]["file_path"] = mp4_path
            print(f"Download progress updated for {download_id}: {mp4_path}")
            
        except Exception as download_error:
            # Handle specific download errors
            error_msg = str(download_error)
            if "Invalid data found when processing input" in error_msg:
                download_progress[download_id]["message"] = "Download failed: Video format not supported"
                download_progress[download_id]["error"] = "The video format is not supported or the video is corrupted. Please try a different YouTube video."
            elif "Video unavailable" in error_msg:
                download_progress[download_id]["message"] = "Download failed: Video unavailable"
                download_progress[download_id]["error"] = "This video is not available for download. It may be private, deleted, or region-restricted."
            elif "Sign in" in error_msg or "login" in error_msg.lower():
                download_progress[download_id]["message"] = "Download failed: Age-restricted video"
                download_progress[download_id]["error"] = "This video is age-restricted and requires authentication to download."
            else:
                download_progress[download_id]["message"] = f"Download failed: {error_msg}"
                download_progress[download_id]["error"] = error_msg
            
            download_progress[download_id]["status"] = "failed"
            download_progress[download_id]["progress"] = 0
        
    except Exception as e:
        download_progress[download_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"Download failed: {str(e)}",
            "file_path": None,
            "error": str(e)
        }

@app.post("/api/download")
def api_download_start(payload: DownloadIn):
    """Start YouTube download and return download ID for tracking"""
    try:
        print(f"Starting download for URL: {payload.youtube_url}")
        
        # Generate a download ID
        download_id = str(uuid.uuid4())
        
        # Initialize progress tracking
        download_progress[download_id] = {
            "status": "starting",
            "progress": 0,
            "message": "Initializing...",
            "file_path": None,
            "error": None
        }
        
        # Start download in background thread
        thread = threading.Thread(
            target=download_video_async,
            args=(download_id, payload.youtube_url)
        )
        thread.daemon = True
        thread.start()
        
        return {
            "download_id": download_id,
            "youtube_url": payload.youtube_url,
            "status": "started"
        }
    except Exception as e:
        import traceback
        print(f"Error in download start: {e}")
        print(traceback.format_exc())
        return {
            "error": "Download failed",
            "error_code": "DOWNLOAD_ERROR",
            "details": str(e)
        }

@app.get("/api/download_status")
def api_download_status(download_id: str):
    """Get download progress status"""
    if download_id not in download_progress:
        return {"error": "Download ID not found"}
    
    return download_progress[download_id]

@app.post("/api/dub")
def api_dub_start(payload: DubIn):
    try:
        print(f"Starting dubbing for URL: {payload.youtube_url}")
        from dub import create_dub_job
        
        # SIMPLE APPROACH: Just download the video directly in this endpoint
        # This eliminates the race condition completely
        print("Downloading video directly for dubbing...")
        from dub import download_youtube_highest_mp4
        
        try:
            mp4_path = download_youtube_highest_mp4(payload.youtube_url)
            print(f"Video downloaded successfully: {mp4_path}")
        except Exception as download_error:
            print(f"Download failed: {download_error}")
            return {
                "error": "Download failed",
                "error_code": "DOWNLOAD_ERROR",
                "details": str(download_error)
            }
        
        # Create the dubbing job
        print("Creating dubbing job...")
        try:
            job = create_dub_job(file_path=mp4_path, target_locale=payload.target_locale)
            print(f"Job created with ID: {job.id}")
            return {"job_id": job.id, "status": "success"}
        except Exception as job_error:
            print(f"Job creation failed: {job_error}")
            # Check if it's a credit-related error
            error_str = str(job_error).lower()
            if "credit" in error_str or "insufficient" in error_str:
                return {
                    "error": "Insufficient credits",
                    "error_code": "INSUFFICIENT_CREDITS",
                    "details": str(job_error)
                }
            else:
                return {
                    "error": "Job creation failed",
                    "error_code": "JOB_CREATION_ERROR",
                    "details": str(job_error)
                }
    except Exception as e:
        import traceback
        print(f"Error in dubbing: {e}")
        print(traceback.format_exc())
        return {
            "error": "Dubbing failed",
            "error_code": "GENERAL_ERROR",
            "details": str(e)
        }


@app.get("/api/debug")
def api_debug():
    """Debug endpoint to check system status"""
    return {
        "murf_api_key_set": bool(os.getenv("MURFDUB_API_KEY")),
        "downloads_dir_exists": os.path.exists("downloads"),
        "downloads_dir_files": os.listdir("downloads") if os.path.exists("downloads") else [],
        "current_dir": os.getcwd(),
        "python_version": sys.version,
        "server_status": "running",
        "download_progress": download_progress
    }

@app.get("/api/health")
def api_health():
    """Simple health check endpoint"""
    return {"status": "ok", "message": "Server is running"}

@app.get("/api/test_murf")
def api_test_murf():
    """Test MurfDub API connection"""
    try:
        from dub import murf_client
        # Try to get some basic info from MurfDub
        print(f"Testing MurfDub client: {murf_client}")
        return {"status": "success", "message": "MurfDub client initialized successfully"}
    except Exception as e:
        print(f"MurfDub test error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/dub_complete")
def api_dub_complete(job_id: str):
    """Get complete job results when job is finished"""
    try:
        # Use the polling function to wait for completion
        status = poll_job_until_complete(job_id=job_id)
        s = str(getattr(status, "status", "")).upper()

        if s not in ("COMPLETED", "FAILED", "ERROR"):
            return {"status": s.lower(), "message": "Job still in progress"}

        # Job is complete, get the results
        print(f"Job {job_id} status details: {status.__dict__}")
        
        # Extract video URL from download_details
        dubbed_video_url = None
        subtitles_url = None
        
        if hasattr(status, "download_details") and status.download_details:
            # Get the first download detail (usually the main dubbed video)
            download_detail = status.download_details[0]
            if hasattr(download_detail, "download_url"):
                dubbed_video_url = download_detail.download_url
            if hasattr(download_detail, "download_srt_url"):
                subtitles_url = download_detail.download_srt_url
        
        print(f"Extracted video URL: {dubbed_video_url}")
        print(f"Extracted subtitles URL: {subtitles_url}")

        notes = None
        if subtitles_url:
            try:
                srt_bytes = download_url_bytes(subtitles_url)
                transcript_text = srt_to_plain_text(srt_bytes)
                notes = generate_notes_from_text(transcript_text)
            except Exception as e:
                notes = f"- (Could not generate notes) {e}"

        return {
            "status": s.lower(),
            "dubbed_video_url": dubbed_video_url,
            "subtitles_url": subtitles_url,
            "notes": notes
        }
    except Exception as e:
        import traceback
        print(f"Error in dub complete: {e}")
        print(traceback.format_exc())
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dub_status")
def api_dub_status(job_id: str):
    try:
        # Just get the current status without waiting
        from dub import murf_client
        status = murf_client.dubbing.jobs.get_status(job_id=job_id)

        if hasattr(status, "to_dict"):
            status_dict = status.to_dict()
        elif hasattr(status, "__dict__"):
            status_dict = status.__dict__
        else:
            status_dict = dict(status)
        
        s = str(status_dict.get("status", "")).upper()
        print(f"Job {job_id} current status: {s}")
        
        # If job is still in progress, just return status
        if s not in ("COMPLETED", "FAILED", "ERROR"):
            return {"status": s.lower()}

        # Job failed - return detailed error information
        if s in ("FAILED", "ERROR"):
            failure_reason = status_dict.get("failure_reason", "Unknown error")
            failure_code = status_dict.get("failure_code", "UNKNOWN")
            credits_remaining = status_dict.get("credits_remaining", 0)
            
            return {
                "status": s.lower(),
                "error": failure_reason,
                "error_code": failure_code,
                "credits_remaining": credits_remaining
            }

        # Job is complete, get the results
        print(f"Job {job_id} status details: {status_dict}")
        
        # Extract video URL from download_details
        dubbed_video_url = None
        subtitles_url = None
        
        if hasattr(status, "download_details") and status.download_details:
            # Get the first download detail (usually the main dubbed video)
            download_detail = status.download_details[0]
            if hasattr(download_detail, "download_url"):
                dubbed_video_url = download_detail.download_url
            if hasattr(download_detail, "download_srt_url"):
                subtitles_url = download_detail.download_srt_url
        
        print(f"Extracted video URL: {dubbed_video_url}")
        print(f"Extracted subtitles URL: {subtitles_url}")

        notes = None
        if subtitles_url:
            try:
                srt_bytes = download_url_bytes(subtitles_url)
                transcript_text = srt_to_plain_text(srt_bytes)
                notes = generate_notes_from_text(transcript_text)
            except Exception as e:
                notes = f"- (Could not generate notes) {e}"

        return {
            "status": s.lower(),
            "dubbed_video_url": dubbed_video_url,
            "subtitles_url": subtitles_url,
            "notes": notes
        }
    except Exception as e:
        import traceback
        print(f"Error in dub status: {e}")
        print(traceback.format_exc())
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
