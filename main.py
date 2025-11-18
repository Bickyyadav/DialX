"""server.py

Webhook server to handle outbound call requests, initiate calls via Twilio API,
and handle subsequent WebSocket connections for Media Streams.
"""

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    BackgroundTasks,
    UploadFile,
    HTTPException,
    File,
)
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from server_utils import (
    DialoutResponse,
    dialout_request_from_request,
    generate_twiml,
    make_twilio_call,
    parse_twiml_request,
    read_excel_file,
    save_uploaded_file,
)

load_dotenv(override=True)


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.post("/dialout", response_model=DialoutResponse)
async def handle_dialout_request(request: Request) -> DialoutResponse:
    """Handle outbound call request and initiate call via Twilio.

    Args:
        request (Request): FastAPI request containing JSON with 'to_number' and 'from_number'.

    Returns:
        DialoutResponse: Response containing call_sid, status, and to_number.

    Raises:
        HTTPException: If request data is invalid or missing required fields.
    """
    logger.info("Received outbound call request")
    file_path = await save_uploaded_file(request)

    phone_number = read_excel_file(file_path)
    print(phone_number)

    for number in phone_number:
        await make_twilio_call(number)

    return DialoutResponse(
        call_sid="call_result.call_sid",
        status="call_initiated",
        to_number="call_result.to_number",
    )


@app.post("/twiml")
async def get_twiml(request: Request) -> HTMLResponse:
    """Return TwiML instructions for connecting call to WebSocket.

    This endpoint is called by Twilio when a call is initiated. It returns TwiML
    that instructs Twilio to connect the call to our WebSocket endpoint with
    stream parameters containing call metadata.

    Args:
        request (Request): FastAPI request containing Twilio form data with 'To' and 'From'.

    Returns:
        HTMLResponse: TwiML XML response with Stream connection instructions.
    """
    logger.info("Serving TwiML for outbound call")

    twiml_request = await parse_twiml_request(request)

    twiml_content = generate_twiml(twiml_request)
    print(twiml_content)
    return HTMLResponse(content=twiml_content, media_type="application/xml")


@app.post("/recording_callback")
async def recording_callback(request: Request, background_tasks: BackgroundTasks):
    """Handle Twilio RecordingStatusCallback and download the finished recording.

    Twilio will POST form data including RecordingSid and CallSid when a recording
    is available. We download the recording as a WAV and save it under RECORDINGS_DIR.
    """
    logger.info("Received recording status callback from Twilio")

    form = await request.form()
    recording_sid = form.get("RecordingSid")
    call_sid = form.get("CallSid")
    recording_status = form.get("RecordingStatus")

    if not recording_sid:
        logger.warning("Recording callback missing RecordingSid")
        return JSONResponse({"error": "missing RecordingSid"}, status_code=400)

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        logger.error("Twilio credentials missing for recording download")
        return JSONResponse({"error": "missing_twilio_credentials"}, status_code=500)

    recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Recordings/{recording_sid}.wav"

    recordings_dir = os.getenv("RECORDINGS_DIR", "./recordings")
    os.makedirs(recordings_dir, exist_ok=True)
    filename = os.path.join(recordings_dir, f"{call_sid or 'call'}_{recording_sid}.wav")

    def _download_recording(url, auth, path):
        # import requests inside the background function to avoid top-level import
        import requests

        try:
            logger.info(f"Downloading recording from {url} to {path}")
            resp = requests.get(url, auth=auth, stream=True, timeout=60)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Recording saved: {path}")
        except Exception:
            logger.exception("Failed to download recording")

    # pass the auth tuple and the destination path
    background_tasks.add_task(
        _download_recording, recording_url, (account_sid, auth_token), filename
    )

    return JSONResponse(
        {
            "status": "download_queued",
            "recording_sid": recording_sid,
            "call_sid": call_sid,
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection from Twilio Media Streams.

    This endpoint receives the WebSocket connection from Twilio's Media Streams
    and runs the bot to handle the voice conversation. Stream parameters passed
    from TwiML are available to the bot for customization.

    Args:
        websocket (WebSocket): FastAPI WebSocket connection from Twilio.
    """
    from bot import bot
    from pipecat.runner.types import WebSocketRunnerArguments

    await websocket.accept()
    logger.info("WebSocket connection accepted for outbound call")

    try:
        runner_args = WebSocketRunnerArguments(websocket=websocket)
        await bot(runner_args)
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {e}")
        await websocket.close()


if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", "7860"))
    logger.info(f"Starting Twilio outbound chatbot server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
