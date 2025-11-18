import asyncio
import os
import sys

import requests

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

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


doctors = [
    {"name": "Abc", "age": 28, "phone": "9876543210"},
    {"name": "Priya Verma", "age": 25, "phone": "9123456780"},
    {"name": "Rohan Gupta", "age": 32, "phone": "9988776655"},
    {"name": "Neha Singh", "age": 27, "phone": "9090909090"},
    {"name": "Vicky Yadav", "age": 23, "phone": "9812345678"},
    {"name": "Sakshi Jain", "age": 30, "phone": "9001234567"},
    {"name": "Karan Mehta", "age": 35, "phone": "8899776655"},
    {"name": "Simran Kaur", "age": 26, "phone": "9797979797"},
    {"name": "Rahul Desai", "age": 29, "phone": "9345678901"},
    {"name": "Anjali Nair", "age": 24, "phone": "9456781234"},
]


book_appointment = [
    
]




async def fetch_restaurant_recommendation(params: FunctionCallParams):
    await params.result_callback({"name": "The Madharchod Sar hotel "})


async def fetch_doctor_details(params: FunctionCallParams):
    print("--------------------------")
    """Lookup a user by name in the in-memory people list."""
    arguments = params.arguments or {}
    raw_name = arguments.get("name")
    print("--------------------------raw_name")
    print(raw_name)

    if not raw_name:
        await params.result_callback(
            {
                "status": "error",
                "message": "कृपया उस व्यक्ति का नाम बताएं जिसकी जानकारी चाहिए।",
            }
        )
        return

    target = raw_name.strip().lower()
    # match = next((p for p in people if p["name"].lower() == target), None)
    match = next((p for p in doctors if p["name"].lower() == target), None)
    print("--------------------------")
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


async def end_call_function(params: FunctionCallParams):
    """End the call when user requests it."""
    await params.llm.push_frame(TTSSpeakFrame("Goodbye! Ending the call now."))
    await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
    await params.result_callback({"status": "call_ended"})


async def run_bot(transport: BaseTransport, handle_sigint: bool):
    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))
    llm.register_function(
        "get_restaurant_recommendation", fetch_restaurant_recommendation
    )
    llm.register_function("fetch_doctor_details", fetch_doctor_details)
    llm.register_function("end_call", end_call_function)
    llm.register_function("web_search", google_web_search)

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
        api_key=deepgram_api_key
        # voice_id="4b617ea7-b28a-4c3f-8535-8c33413e238f",  # British Reading Lady
    )

    end_call_schema = FunctionSchema(
        name="end_call",
        description="End the conversation and hang up the call when the user requests to end it",
        properties={},
        required=[],
    )
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
        description="Look up a doctor stored profile using their exact name.",
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
            web_search_function,
        ]
        # standard_tools=[weather_function, restaurant_function]
    )
    system_prompt = """ You are a helpful medical assistant designed to help users find doctors.

                        Your job:
                        1. First, politely ask the user for the doctor’s name.
                        2. When the user provides a doctor name, always call the tool `find_doctor_in_db` with the doctor’s name.
                        3. If the tool returns that the doctor is available, respond with:
                        - “Doctor found / available”
                        - Ask the user for the next required details (e.g., appointment date, time, patient name, or symptoms).
                        4. If the tool returns that the doctor is NOT found, politely say “Doctor not found” and ask the user to provide another doctor’s name.
                        5. Do not guess or hallucinate doctor names. Only rely on the database search result.
                        6. Keep your messages simple, friendly, and clear.
                        7. When calling the tool, output ONLY the JSON function call. No extra text.
                        8. Continue the conversation until all required information is collected from the user.

                        Your goal is to help the user easily find doctors and complete their appointment request.

                    """

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

    @transcript.event_handler("on_transcript_update")
    async def handle_update(processor, frame):
        for message in frame.messages:
            print(f"hi there {message.content}")
            entry = {"role": message.role, "content": message.content}
            transcript_history.append(entry)
            logger.debug("Transcript update (%s): %s", message.role, message.content)

        logger.trace("Transcript history size: %s", len(transcript_history))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the outbound conversation, waiting for the user to speak first
        await task.queue_frames([LLMRunFrame()])
        logger.info("Starting outbound call conversation")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Outbound call ended")

        transcript_file = os.path.join(os.getcwd(), "transcript.txt")
        try:
            with open(transcript_file, "w", encoding="utf-8") as fp:
                for entry in transcript_history:
                    fp.write(f"{entry['role']}: {entry['content']}\n")
            logger.info(f"Transcript saved to {transcript_file}")
        except Exception as exc:
            logger.error(f"Failed to write transcript: {exc}")

        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")

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

    await run_bot(transport, handle_sigint)
