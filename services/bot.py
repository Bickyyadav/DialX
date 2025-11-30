import datetime
import wave
import os
import httpx
import asyncio
from loguru import logger
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Request, BackgroundTasks, HTTPException
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url


cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_SECRET_KEY"),
    secure=True,
)


# async def upload_latest_recording():
#     recordings_dir = Path("./recordings")
#     if not recordings_dir.exists():
#         raise HTTPException(status_code=404, detail="No recordings directory found")

#     wav_files = sorted(
#         recordings_dir.glob("*.wav"),
#         key=lambda path: path.stat().st_mtime,
#         reverse=True,
#     )

#     if not wav_files:
#         raise HTTPException(status_code=404, detail="No recording files available")

#     latest_file = wav_files[0]
#     logger.info("Uploading latest recording: {}", latest_file)
#     return await upload_file_to_cloud(str(latest_file))


async def upload_file_to_cloud(file_path: str):
    """Upload a single audio file to Cloudinary and remove it locally."""
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Recording file not found")

    try:
        result = cloudinary.uploader.upload(
            file_path,
            folder="AudioFile",
            use_filename=True,
            resource_type="auto",
        )
        logger.info("Uploaded recording to Cloudinary: {}", result.get("secure_url"))
        return {
            "msg": "Audio uploaded successfully",
            "src": result.get("secure_url"),
        }
    except Exception as exc:
        logger.exception("Failed to upload recording %s", file_path)
        raise HTTPException(
            status_code=422,
            detail=f"Cloud upload failed: {exc}",
        ) from exc
    finally:
        try:
            os.remove(file_path)
        except OSError as cleanup_error:
            logger.warning("Could not delete local recording %s: %s", file_path, cleanup_error)
    


async def save_recording(buffer, audio, sample_rate, num_channels, call_id: str = None):
    """Save audio locally and upload complete file to webhook."""
    # Ensure recordings directory exists
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
        logger.info(f"Created recordings directory: {recordings_dir}")

    # Save audio file locally
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{recordings_dir}/conversation_{timestamp}.wav"

    # Create the WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
 
    await upload_file_to_cloud(filename)
    
