import os
import time
import io
import re
import requests
from typing import Optional, Tuple, List
from dotenv import load_dotenv
import yt_dlp  # ✅ switched from pytube
from murf import MurfDub
from langchain_core.prompts import PromptTemplate
from types import SimpleNamespace
from google import genai
import subprocess

load_dotenv()

MURFDUB_API_KEY = os.getenv("MURFDUB_API_KEY")
murf_client = MurfDub(api_key=MURFDUB_API_KEY)

client_gemini = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

def ask_gemini(prompt: str) -> str:
    try:
        response = client_gemini.models.generate_content(
            model = "gemini-2.0-flash",
            contents = prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return ""

# Target locales per Murf docs
TARGET_LOCALES = [
    "en_US","en_UK","en_IN","en_SCOTT","en_AU",
    "fr_FR","de_DE","es_ES","es_MX","it_IT","pt_BR","pl_PL",
    "hi_IN","ko_KR","ta_IN","bn_IN","ja_JP","zh_CN","nl_NL","fi_FI",
    "ru_RU","tr_TR","uk_UA","da_DK","id_ID","ro_RO","nb_NO",
]

# ---------- YT download ----------

def clean_youtube_url(url: str) -> str:
    """Return base YouTube video URL (remove &t=, &si=, etc)."""
    if "&" in url:
        url = url.split("&")[0]
    return url

def download_youtube_highest_mp4(url: str, out_dir: str = "downloads") -> str:
    """
    Downloads the highest quality MP4 (video+audio merged) using yt-dlp.
    Returns the file path to the saved video.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        url = clean_youtube_url(url)
        
        # Add a unique identifier to prevent file conflicts
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        output_template = os.path.join(out_dir, "%(title)s_" + unique_id + ".%(ext)s")
        
        # Store the expected final filename for easier tracking
        expected_filename = f"%(title)s_{unique_id}.%(ext)s"

        # Check if ffmpeg is available
        if not check_ffmpeg():
            print("Warning: ffmpeg not found. Download may fail for some videos.")
            # Use simpler format that doesn't require ffmpeg
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_template,
                'quiet': False,  # Show progress for debugging
                'no_warnings': False,
            }
        else:
            # Use ffmpeg for better quality
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'merge_output_format': 'mp4',
                'outtmpl': output_template,
                'quiet': False,  # Show progress for debugging
                'no_warnings': False,
            }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First extract info to check if video is available
            try:
                info = ydl.extract_info(url, download=False)
                print(f"Video info extracted: {info.get('title', 'Unknown')}")
            except Exception as info_error:
                print(f"Failed to extract video info: {info_error}")
                raise info_error
            
            # Now download
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)
            
            # Handle file extension properly
            if file_path.endswith('.webm'):
                file_path = file_path.replace('.webm', '.mp4')
            elif not file_path.endswith('.mp4'):
                # If no extension or different extension, ensure it's .mp4
                base_path = os.path.splitext(file_path)[0]
                file_path = base_path + '.mp4'
            
            # Verify file exists and has content
            if not os.path.exists(file_path):
                # Try to find the actual downloaded file
                for file in os.listdir(out_dir):
                    if file.endswith('.mp4'):
                        potential_file = os.path.join(out_dir, file)
                        if os.path.getsize(potential_file) > 0:  # Check file has content
                            file_path = potential_file
                            break
                else:
                    raise FileNotFoundError(f"Downloaded MP4 file not found in {out_dir}")
            
            # Verify file has content
            if os.path.getsize(file_path) == 0:
                raise ValueError(f"Downloaded file is empty: {file_path}")
            
            print(f"Successfully downloaded: {file_path} ({os.path.getsize(file_path)} bytes)")
            return file_path

    except Exception as e:
        print(f"YouTube download error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to download Youtube video with yt-dlp: {e}")

# ---------- Murf Dub API wrappers ----------

def create_dub_job(file_path: str, target_locale: str, priority: str = "LOW") -> dict:
    if target_locale not in TARGET_LOCALES:
        raise ValueError(f"target_locale '{target_locale}' not in supported TARGET_LOCALES.")
    
    # Verify file exists and is readable
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")
    
    try:
        print(f"Creating dubbing job for file: {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
        print(f"Target locale: {target_locale}")
        
        with open(file_path, "rb") as f:
            res = murf_client.dubbing.jobs.create(
                target_locales=[target_locale],
                file_name=os.path.basename(file_path),
                file=f,
                priority=priority
            )

        print(f"MurfDub response: {res}")
        
        if isinstance(res, dict):
            job_id = res.get("id") or res.get("job_id")
            print(f"Job created with ID: {job_id}")
            return SimpleNamespace(
                id=job_id,
                raw=res
            )
        else:
            try:
                job_id = getattr(res, "id", None) or getattr(res, "job_id", None)
                print(f"Job created with ID: {job_id}")
                return SimpleNamespace(
                    id=job_id,
                    raw=res.__dict__
                )
            except Exception as e:
                raise ValueError(f"Unexpected response format: {e}")
    except Exception as e:
        print(f"MurfDub API error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to create dubbing job: {e}")

def poll_job_until_complete(job_id: str, poll_interval: float = 3.0, timeout_sec: int = 1800) -> dict:
    start = time.time()
    attempt = 0

    while True:
        try:
            status = murf_client.dubbing.jobs.get_status(job_id=job_id)

            if hasattr(status, "to_dict"):
                status_dict = status.to_dict()
            elif hasattr(status, "__dict__"):
                status_dict = status.__dict__
            else:
                status_dict = dict(status)
            
            s = str(status_dict.get("status", "")).upper()
            print(f"Job {job_id} status: {s} (attempt {attempt})")
            
            if s in ("COMPLETED", "FAILED", "ERROR"):
                print(f"Job {job_id} final status: {s}")
                if s == "FAILED" or s == "ERROR":
                    print(f"Job {job_id} failed with details: {status_dict}")
                return SimpleNamespace(**status_dict)
            
            if time.time() - start > timeout_sec:
                raise TimeoutError("Polling timed out.")

            attempt+=1
            sleep_time = min(poll_interval * (1 + attempt // 10), 15)
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error polling job {job_id}: {e}")
            raise

def download_url_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

# ---------- Subtitles helpers ----------

def srt_to_plain_text(srt_bytes: bytes) -> str:
    text = srt_bytes.decode("utf-8", errors="ignore")
    text = re.sub(r"\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"(?m)^\d+\s*$", "", text)
    text = text.strip()
    return text

# ---------- Notes via LLM ----------

NOTES_PROMPT = PromptTemplate(
    template="""
You are a helpful teacher. Create compact class notes in bullet points from the following transcript text.

Rules:
- 4–7 concise bullets.
- Keep each bullet <= 2 lines.
- No new facts not present in text.
- Use plain language.

Transcript:
{text}

Notes:
""",
    input_variables=["text"]
)

def generate_notes_from_text(text: str) -> str:
    prompt_text = NOTES_PROMPT.format(text=text)
    try:
        resp = ask_gemini(prompt_text)
        return resp.strip()
    except Exception as e:
        return f"- (Error generating notes) {e}"

# ---------- Utility ----------

def save_bytes_to_tmpfile(b: bytes, suffix: str) -> str:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(b)
    return path
