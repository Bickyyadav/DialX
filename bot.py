import asyncio
import json
import os
import re
import sys

import requests
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.frames.frames import (
    EndTaskFrame,
    LLMRunFrame,
    TTSSpeakFrame,
)
from services.bot import save_recording
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.transcriptions.language import Language
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.user_idle_processor import UserIdleProcessor
from deepgram import (
    LiveOptions,
)
from prompt_data import system_prompt, doctors, user

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def fetch_restaurant_recommendation(params: FunctionCallParams):
    await params.result_callback({"name": "The Everest hotel "})


async def fetch_doctor_details(params: FunctionCallParams):
    """Find doctors by specialty and geographic filters."""
    arguments = params.arguments or {}

    def _normalized(value: str | None) -> str:
        return "".join((value or "").strip().lower().split())

    raw_specialist = arguments.get("specialist")
    raw_state = arguments.get("state")
    raw_municipality = arguments.get("municipality")
    raw_neighborhood = arguments.get("neighborhood")
    print(
        "Doctor search inputs →",
        raw_specialist,
        raw_state,
        raw_municipality,
        raw_neighborhood,
    )

    missing = [
        label
        for label, value in (
            ("specialist", raw_specialist),
            ("state", raw_state),
            ("municipality", raw_municipality),
            ("neighborhood", raw_neighborhood),
        )
        if not value
    ]
    if missing:
        await params.result_callback(
            {
                "status": "error",
                "message": "कृपया सभी विवरण बताएं: विशेषज्ञता, राज्य, नगरपालिका, और पड़ोस।",
                "missing_fields": missing,
            }
        )
        return

    target_specialist = _normalized(raw_specialist)
    target_state = _normalized(raw_state)
    target_municipality = _normalized(raw_municipality)
    target_neighborhood = _normalized(raw_neighborhood)

    match = next(
        (
            doc
            for doc in doctors
            if _normalized(doc["specialist"]) == target_specialist
            and _normalized(doc["state"]) == target_state
            and _normalized(doc["municipality"]) == target_municipality
            and _normalized(doc["neighborhood"]) == target_neighborhood
        ),
        None,
    )
    print(match)

    if match:
        await params.result_callback(
            {
                "status": "found",
                "name": match["name"],
                "specialist": match["specialist"],
                "state": match["state"],
                "municipality": match["municipality"],
                "neighborhood": match["neighborhood"],
                "phone": match["phone"],
            }
        )
    else:
        await params.result_callback(
            {
                "status": "not_found",
                "message": "दिए गए क्षेत्र और विशेषज्ञता के लिए कोई डॉक्टर नहीं मिला।",
            }
        )


async def fetch_user_details(params: FunctionCallParams):
    """Lookup a user by name in the in-memory people list."""
    arguments = params.arguments or {}
    raw_name = arguments.get("name")
    print(f"Raw name (before cleaning) → {raw_name}")

    if not raw_name:
        await params.result_callback(
            {
                "status": "error",
                "message": "कृपया उस व्यक्ति का नाम बताएं जिसकी जानकारी चाहिए।",
            }
        )
        return

    target = raw_name.strip().lower().replace(" ", "")
    match = next((p for p in user if p["name"].lower() == target), None)
    print(match)

    if match:
        await params.result_callback(
            {
                "status": "found",
                "name": match["name"],
                "age": match["age"],
                "phone": match["phone"],
            }
        )
    else:
        await params.result_callback(
            {
                "status": "not_found",
                "message": f"{raw_name} हमारी सूची में नहीं मिला।",
            }
        )


async def google_web_search(params: FunctionCallParams):
    """Use Google Custom Search to answer user queries."""
    query = (params.arguments or {}).get("query")
    if not query:
        await params.result_callback("मुझे वेब पर खोज करने के लिए एक स्पष्ट सवाल चाहिए।")
        return

    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    search_cx = os.getenv("GOOGLE_SEARCH_CX")
    if not api_key or not search_cx:
        logger.warning("Google search credentials are missing")
        await params.result_callback(
            "वेब सर्च अभी उपलब्ध नहीं है क्योंकि सेवा कॉन्फ़िगर नहीं हुई है।"
        )
        return

    def _run_search():
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": api_key,
                "cx": search_cx,
                "q": query,
                "num": 3,
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    try:
        data = await asyncio.to_thread(_run_search)
    except Exception as exc:
        logger.error(f"Google search failed: {exc}")
        await params.result_callback(
            "मैं अभी वेब सर्च से जुड़ नहीं पा रही हूँ। कृपया थोड़ी देर बाद कोशिश करें।"
        )
        return

    items = data.get("items") or []
    if not items:
        await params.result_callback("मुझे इस सवाल के लिए कोई भरोसेमंद वेब परिणाम नहीं मिला।")
        return

    lines = []
    for item in items[:3]:
        title = item.get("title", "कोई शीर्षक नहीं")
        snippet = (item.get("snippet") or "").replace("\n", " ").strip()
        link = item.get("link", "")
        if len(snippet) > 180:
            snippet = snippet[:177].rstrip() + "..."
        lines.append(f"{title}: {snippet} {link}")

    await params.result_callback("यह जानकारी मुझे वेब पर मिली:\n" + "\n".join(lines))


def _extract_summary_payload(raw_text: str) -> dict[str, Any] | None:
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# async def summarize_conversation_with_llm(
#     transcript_history: list[dict[str, str]],
# ) -> dict[str, Any]:
#     default_summary = {
#         "summary_line": "Conversation summary unavailable.",
#         "interest_percentage": 0,
#     }
#     print("----------------------------------", transcript_history)

#     if not transcript_history:
#         return {
#             "summary_line": "Call ended before any conversation occurred.",
#             "interest_percentage": 0,
#         }

#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         logger.warning("Cannot summarize conversation without GOOGLE_API_KEY")
#         return default_summary

#     conversation_text = "\n".join(
#         f"{entry['role'].capitalize()}: {entry['content']}"
#         for entry in transcript_history
#     )

#     prompt = (
#         "You are a QA system that reads a phone conversation between a medical support agent "
#         "and a beneficiary. Provide a single concise sentence recap plus the likelihood that "
#         "the caller will book the requested specialist. Respond ONLY with JSON in this exact "
#         'schema: {"summary_line": "<short sentence>", "interest_percentage": <0-100 integer>}\n\n'
#         "Conversation:\n"
#         f"{conversation_text}"
#     )

#     def _invoke_llm():
#         response = requests.post(
#             "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent",
#             params={"key": api_key},
#             json={
#                 "contents": [
#                     {
#                         "role": "user",
#                         "parts": [
#                             {
#                                 "text": prompt,
#                             }
#                         ],
#                     }
#                 ]
#             },
#             timeout=25,
#         )
#         response.raise_for_status()
#         return response.json()

#     try:
#         data = await asyncio.to_thread(_invoke_llm)
#     except Exception as exc:
#         logger.error(f"Failed to summarize call with LLM: {exc}")
#         return default_summary

#     candidates = data.get("candidates") or []
#     if not candidates:
#         return default_summary

#     parts = candidates[0].get("content", {}).get("parts") or []
#     raw_text = parts[0].get("text", "") if parts else ""
#     # payload = _extract_summary_payload(raw_text)

#     if not payload:
#         return default_summary

#     summary_line = payload.get("summary_line") or payload.get("summary")
#     interest = payload.get("interest_percentage") or payload.get("interest")

#     try:
#         interest_value = max(0, min(100, int(round(float(interest)))))
#     except (TypeError, ValueError):
#         interest_value = 0

#     return {
#         "summary_line": summary_line or "Conversation summary unavailable.",
#         "interest_percentage": interest_value,
#     }


async def end_call_function(params: FunctionCallParams):
    """End the call when user requests it."""
    await params.llm.push_frame(TTSSpeakFrame("Goodbye! Ending the call now."))
    await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
    await params.result_callback({"status": "call_ended"})


async def run_bot(transport: BaseTransport, handle_sigint: bool, call_data: dict):
    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))
    llm.register_function(
        "get_restaurant_recommendation", fetch_restaurant_recommendation
    )
    llm.register_function("fetch_doctor_details", fetch_doctor_details)
    llm.register_function("end_call", end_call_function)
    # llm.register_function("web_search", google_web_search)
    llm.register_function("fetch_user_details", fetch_user_details)

    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        raise RuntimeError(
            "Missing DEEPGRAM_API_KEY environment variable for Deepgram STT"
        )

    stt = DeepgramSTTService(
        api_key=deepgram_api_key,
        live_options=LiveOptions(language=Language.EN),
    )

    tts = DeepgramTTSService(
        api_key=deepgram_api_key,
        voice_id="f91ab3e6-5071-4e15-b016-cde6f2bcd222",  # British Reading Lady
    )

    end_call_schema = FunctionSchema(
        name="end_call",
        description="End the conversation and hang up the call when the user requests to end it",
        properties={},
        required=[],
    )
    call_id = call_data["call_id"]

    restaurant_function = FunctionSchema(
        name="get_restaurant_recommendation",
        description="Get a restaurant recommendation",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        required=["location"],
    )

    doctor_details_function = FunctionSchema(
        name="fetch_doctor_details",
        description="Use state, municipality, neighborhood, and specialty to find an in-network doctor.",
        properties={
            "specialist": {
                "type": "string",
                "description": "जिस डॉक्टर की आवश्यकता है उसकी विशेषज्ञता (उदाहरण: Cardiologist)",
            },
            "state": {
                "type": "string",
                "description": "राज्य जैसा रिकॉर्ड में सूचीबद्ध है",
            },
            "municipality": {
                "type": "string",
                "description": "नगर पालिका/जिला जहाँ लाभार्थी स्थित है",
            },
            "neighborhood": {
                "type": "string",
                "description": "स्थानीय पड़ोस या क्षेत्र",
            },
        },
        required=["specialist", "state", "municipality", "neighborhood"],
    )

    user_details_function = FunctionSchema(
        name="fetch_user_details",
        description="This tool is used to verify the user by their given name",
        properties={
            "name": {
                "type": "string",
                "description": "पूरा नाम जैसा रिकॉर्ड में है (उदाहरण: Amit Sharma)",
            }
        },
        required=["name"],
    )

    web_search_function = FunctionSchema(
        name="web_search",
        description="Use Google search to gather the latest public information when unsure about an answer.",
        properties={
            "query": {
                "type": "string",
                "description": "Precise Hindi or English query to look up on the web",
            }
        },
        required=["query"],
    )

    tools = ToolsSchema(
        standard_tools=[
            # weather_function,
            restaurant_function,
            doctor_details_function,
            end_call_schema,
            # web_search_function,
            user_details_function,
        ]
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Say Hello to user and describe youself in short"},
    ]

    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

    from pipecat.frames.frames import LLMMessagesAppendFrame

    async def handle_idle(user_idle: UserIdleProcessor) -> None:
        await user_idle.push_frame(
            LLMMessagesAppendFrame(
                [
                    {
                        "role": "system",
                        "content": "Ask the user if they are still there and try to prompt for some input.",
                    }
                ],
                run_llm=True,
            )
        )

    user_idle = UserIdleProcessor(callback=handle_idle, timeout=10.0)
    transcript = TranscriptProcessor()

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    audiobuffer = AudioBufferProcessor(
        sample_rate=None,
        num_channels=2,
        buffer_size=0,
        enable_turn_audio=False,
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            user_idle,
            stt,  # Speech-To-Text
            transcript.user(),
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            transcript.assistant(),
            transport.output(),  # Websocket output to client
            audiobuffer,
            context_aggregator.assistant(),
        ]
    )

    transcript_history: list[dict[str, str]] = []

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    #! ----------------------------------------
    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio: bytes, sample_rate: int, num_channels: int):
        # Buffer audio data for later upload
        await save_recording(buffer, audio, sample_rate, num_channels, call_id)

    #! this is of uploading recording to the file from the twilio
    @transcript.event_handler("on_transcript_update")
    async def handle_update(processor, frame):
        for message in frame.messages:
            entry = {"role": message.role, "content": message.content}
            transcript_history.append(entry)
            logger.debug("Transcript update (%s): %s", message.role, message.content)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the outbound conversation, waiting for the user to speak first
        await audiobuffer.start_recording()
        await task.queue_frames([LLMRunFrame()])
        logger.info("Starting outbound call conversation")

    async def call_end_function():
        # summary = await summarize_conversation_with_llm(transcript_history)
        # logger.info("Call summary: %s", summary["summary_line"])
        # logger.info(
        #     "Interest to book specialist: %s%%",
        #     summary["interest_percentage"],
        # )
        # print(
        #     f"Call summary → {summary['summary_line']} | Interest: {summary['interest_percentage']}%"
        # )
        # return summary
        logger.info("User disconnected")
        await task.cancel()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Outbound call ended")
        summary = await call_end_function()

        transcript_file = os.path.join(os.getcwd(), "transcript.txt")
        try:
            with open(transcript_file, "w", encoding="utf-8") as fp:
                for entry in transcript_history:
                    fp.write(f"{entry['role']}: {entry['content']}\n")
                fp.write("\n--- Call Summary ---\n")
                fp.write(f"Summary: {summary['summary_line']}\n")
                fp.write(f"Interest Percentage: {summary['interest_percentage']}%\n")
            logger.info(f"Transcript saved to {transcript_file}")
            print(f"Transcript saved to {transcript_file}")
        except Exception as exc:
            logger.error(f"Failed to write transcript: {exc}")

        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


#! this code is of twilio
async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")
    logger.info(f"Auto-detected call_data: {call_data}")

    # Access custom stream parameters passed from TwiML
    # Use the body data to personalize the conversation
    # by loading customer data based on the to_number or from_number
    body_data = call_data.get("body", {})
    to_number = body_data.get("to_number")
    from_number = body_data.get("from_number")

    logger.info(f"Call metadata - To: {to_number}, From: {from_number}")

    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    handle_sigint = runner_args.handle_sigint

    await run_bot(transport, handle_sigint, call_data)
