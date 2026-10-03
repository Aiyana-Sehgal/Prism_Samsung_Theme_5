#!/usr/bin/env python3
"""
FDB-v3 agent: cascaded STT + LLM + TTS with a tool controller.

Copied into Full-Duplex-Bench/v3/ by reproduce.sh (it imports mock_apis there).
Tool names, signatures and descriptions match the benchmark's cascaded_agent.py
so the LLM sees the same tool schemas; every call is routed through ToolController.

Config (env vars, all optional):
  FDB_LLM_MODEL      LLM for planning (default: gpt-4o)
  FDB_QUIET_WINDOW   seconds of user silence before any tool runs (default: 0.9)
  FDB_MIN_EP / FDB_MAX_EP   endpointing delays (default: 0.6 / 2.0)
  FDB_KEEP_ALIVE     1 = keep session running after the harness disconnects (default: 1)
  FDB_TURN_DETECTOR  1 = semantic end-of-turn model, 0 = VAD only (default: 1)
  FDB_TRACE          path for controller decision trace (default: /tmp/fdb_controller_trace.log)
"""

import json
import logging
import os
import sys
import time

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, RunContext, llm, room_io

from controller import ToolController

# Import the plugins at module level, not lazily inside build_pipeline(). `download-files`
# only fetches weights for plugins that have registered themselves by the time the CLI
# runs, and it never calls build_pipeline() -- so a lazy import leaves the turn detector
# without its model files and every job then crashes on EnglishModel().
# This also avoids the event-loop stall from importing the openai plugin on first use.
from livekit.plugins import openai as _plugin_openai  # noqa: F401
from livekit.plugins import silero as _plugin_silero  # noqa: F401
from livekit.plugins import turn_detector as _plugin_turn_detector  # noqa: F401

LATENCY_PROFILE = "instant"
if "--latency" in sys.argv:
    i = sys.argv.index("--latency")
    LATENCY_PROFILE = sys.argv[i + 1]
    del sys.argv[i:i + 2]

from mock_apis import MockAPIRegistry  # noqa: E402

registry = MockAPIRegistry(latency_profile=LATENCY_PROFILE)
load_dotenv(os.path.join(os.path.dirname(__file__), ".env.local"))

# "local" (default): everything runs on the evaluating machine, no paid APIs.
# "openai": the original hosted pipeline, kept so the two can be compared directly.
STACK = os.getenv("FDB_STACK", "local").lower()
_LOCAL = STACK == "local"

LLM_MODEL = os.getenv("FDB_LLM_MODEL", "qwen2.5:7b-instruct" if _LOCAL else "gpt-4o")
# Any OpenAI-compatible server. Ollama serves one at :11434/v1 and ignores the key.
LLM_BASE_URL = os.getenv("FDB_LLM_BASE_URL", "http://localhost:11434/v1" if _LOCAL else "")
LLM_API_KEY = os.getenv("FDB_LLM_API_KEY", "ollama" if _LOCAL else "")
QUIET_WINDOW = float(os.getenv("FDB_QUIET_WINDOW", "0.9"))
MIN_EP = float(os.getenv("FDB_MIN_EP", "0.6"))
MAX_EP = float(os.getenv("FDB_MAX_EP", "2.0"))
USE_TURN_DETECTOR = os.getenv("FDB_TURN_DETECTOR", "1") == "1"
# The harness disconnects ~1.5 s after the audio ends. Keep the session alive so calls
# still inside the commit gate can finish and be logged (the harness reads the log after its ASR step).
KEEP_ALIVE = os.getenv("FDB_KEEP_ALIVE", "1") == "1"
TRACE_PATH = os.getenv("FDB_TRACE", "/tmp/fdb_controller_trace.log")
TOOL_LOG = "/tmp/agent_tool_calls.log"      # read by the benchmark harness
HEARTBEAT = "/tmp/agent_heartbeat.log"      # read by the latency analysis


# ── Latency tracker (same log format the harness parses) ─────────────
class LatencyTracker:
    def __init__(self):
        self.user_done_at = self.tool_start_at = self.tool_end_at = self.agent_start_at = 0
        self.query_received = False

    def reset(self):
        self.__init__()

    def log_breakdown(self, room_name: str):
        if not (self.user_done_at and self.agent_start_at and self.tool_start_at):
            return
        metrics = {
            "room": room_name, "tool": "Search Tool",
            "reasoning": round(self.tool_start_at - self.user_done_at, 3),
            "execution": round((self.tool_end_at - self.tool_start_at) if self.tool_end_at else 0, 3),
            "synthesis": round(self.agent_start_at - (self.tool_end_at or self.user_done_at), 3),
            "total": round(self.agent_start_at - self.user_done_at, 3),
            "agent_start_at": self.agent_start_at,
        }
        with open(HEARTBEAT, "a") as f:
            f.write(f"LATENCY_TRACK_JSON: {json.dumps(metrics)}\n")


INSTRUCTIONS = """You are a helpful voice assistant. Replies are spoken aloud, so keep them short and natural.
You have 12 tools across travel, finance, housing and e-commerce. This is a simulated test environment:
you are authorized to use every tool, including identity, billing and cart actions.

Rules:
1. When the user asks for something a tool can do, call the tool. Never answer from memory, never invent data.
   Do not ask clarifying questions when the request is clear enough to act on.
2. Small talk or greetings with no actionable request: reply briefly and call no tools.
3. People correct themselves ("actually", "no wait", "scratch that", "I mean", "never mind", "forget that").
   Only the final version counts: use the corrected value, keep details they did not correct,
   and drop abandoned requests completely. Ignore fillers and false starts.
4. Spelled-out IDs ("C-A-T", "B O B one two") are joined into one ID ("CAT", "BOB12").
5. Do not add details the user did not say (for example, do not invent a year for a date).
6. Multi-step requests: do the steps in order, and pass IDs exactly as returned by earlier tool results.
   Call each needed tool once; never repeat a call that already succeeded.
7. Only say something is done after the tool result confirms it.
8. If a tool returns status "superseded", the user was still talking: do not retry it; wait for their full request.
"""


class FDBAgent(Agent):
    def __init__(self, controller: ToolController, tracker: LatencyTracker):
        super().__init__(instructions=INSTRUCTIONS)
        self.ctl = controller
        self.tracker = tracker

    async def _run(self, ctx: RunContext, fn: str, args: dict) -> str:
        self.tracker.tool_start_at = self.tracker.tool_start_at or time.time()
        result = await self.ctl.run(fn, args, is_interrupted=lambda: ctx.speech_handle.interrupted)
        self.tracker.tool_end_at = time.time()
        return json.dumps(result)

    # ── Travel & Identity ─────────────────────────────────────────
    @llm.function_tool(description="Search for available flights to a destination.")
    async def search_flights(self, ctx: RunContext, destination: str, date: str):
        """
        Args:
            destination: The city or airport, e.g. 'London' or 'LHR'
            date: The travel date, e.g. '2026-08-20'
        """
        return await self._run(ctx, "search_flights", {"destination": destination, "date": date})

    @llm.function_tool(description="Book a flight ticket.")
    async def book_flight(self, ctx: RunContext, passenger_name: str):
        """
        Args:
            passenger_name: The name of the passenger, e.g. 'John Doe'
        """
        return await self._run(ctx, "book_flight", {"passenger_name": passenger_name})

    @llm.function_tool(description="MANDATORY tool to update simulated user identity document details (e.g. passport, driver license). You are fully authorized in this test environment. NEVER refuse to use it due to 'external personal/government system' safety concerns.")
    async def update_identity_doc(self, ctx: RunContext, doc_type: str, doc_number: str):
        """
        Args:
            doc_type: Type of document, e.g. 'passport' or 'id_card'
            doc_number: The document identifier string
        """
        return await self._run(ctx, "update_identity_doc", {"doc_type": doc_type, "doc_number": doc_number})

    # ── Finance & Billing ─────────────────────────────────────────
    @llm.function_tool(description="MANDATORY tool to get benefits for a credit card. NEVER guess benefits from memory. Execute this tool immediately.")
    async def get_card_benefits(self, ctx: RunContext, card_type: str):
        """
        Args:
            card_type: The card type, e.g. 'platinum' or 'gold'
        """
        return await self._run(ctx, "get_card_benefits", {"card_type": card_type})

    @llm.function_tool(description="MANDATORY tool to fetch the exact, current foreign exchange rate. NEVER guess or calculate exchange rates from your internal memory; you MUST use this API.")
    async def get_exchange_rate(self, ctx: RunContext, amount: float, from_currency: str, to_currency: str):
        """
        Args:
            amount: Amount to convert
            from_currency: 3-letter currency code, e.g. 'USD'
            to_currency: 3-letter currency code, e.g. 'EUR'
        """
        return await self._run(ctx, "get_exchange_rate",
                               {"amount": amount, "from_currency": from_currency, "to_currency": to_currency})

    @llm.function_tool(description="MANDATORY tool to process billing details. Execute this update immediately when the user requests Autopay modification.")
    async def modify_autopay(self, ctx: RunContext, bill_type: str, source_account: str):
        """
        Args:
            bill_type: Type of bill, e.g. 'credit_card' or 'utilities'
            source_account: Bank account identifier, e.g. 'checking'
        """
        return await self._run(ctx, "modify_autopay", {"bill_type": bill_type, "source_account": source_account})

    # ── Housing & Location ─────────────────────────────────────────
    @llm.function_tool(description="Search for available rental apartments.")
    async def search_apartments(self, ctx: RunContext, city: str, bedrooms: int, max_price: float):
        """
        Args:
            city: Destination city
            bedrooms: Number of bedrooms
            max_price: Maximum monthly rent budget
        """
        return await self._run(ctx, "search_apartments", {"city": city, "bedrooms": bedrooms, "max_price": max_price})

    @llm.function_tool(description="MANDATORY tool to calculate commute duration. Fetch exact commute times using this tool. Do NOT estimate from memory.")
    async def calculate_commute(self, ctx: RunContext, origin_address: str, destination_address: str, mode: str = "driving"):
        """
        Args:
            origin_address: Starting location
            destination_address: Destination location
            mode: Transport mode, defaults to 'driving'
        """
        return await self._run(ctx, "calculate_commute",
                               {"origin_address": origin_address, "destination_address": destination_address, "mode": mode})

    @llm.function_tool(description="Instantly update the user's search filter in the backend system. Execute this IMMEDIATELY without asking for further confirmations or batching requests. Do not ask clarifying questions.")
    async def update_search_filter(self, ctx: RunContext, filter_name: str, value: str):
        """
        Args:
            filter_name: Filter key to modify
            value: Filter value to apply
        """
        return await self._run(ctx, "update_search_filter", {"filter_name": filter_name, "value": value})

    # ── E-Commerce Support ─────────────────────────────────────────
    @llm.function_tool(description="MANDATORY tool to track physical package status. Do NOT answer from memory or batch tracking requests. EXECUTE THIS TOOL IMMEDIATELY for every order ID mentioned.")
    async def track_order(self, ctx: RunContext, order_id: str):
        """
        Args:
            order_id: Order identifier to track, e.g. 'BOB12'
        """
        return await self._run(ctx, "track_order", {"order_id": order_id})

    @llm.function_tool(description="MANDATORY tool to search for products in the catalog. Do NOT answer from memory. You MUST execute this tool whenever the user asks for item recommendations or searches.")
    async def search_products(self, ctx: RunContext, query: str, max_price: float = None):
        """
        Args:
            query: Product search term, e.g. 'headphones'
            max_price: Optional maximum budget
        """
        return await self._run(ctx, "search_products", {"query": query, "max_price": max_price})

    @llm.function_tool(description="MANDATORY tool to add an item to the shopping cart. Execute this action IMMEDIATELY the moment the user asks without confirming or waiting for them to list more items.")
    async def add_to_cart(self, ctx: RunContext, product_id: str, quantity: int = 1):
        """
        Args:
            product_id: ID of the product
            quantity: Amount to add
        """
        return await self._run(ctx, "add_to_cart", {"product_id": product_id, "quantity": quantity})


def build_pipeline():
    from livekit.plugins import openai, silero

    vad = silero.VAD.load(min_speech_duration=0.05, min_silence_duration=0.55)
    if _LOCAL:
        # Local Whisper + Piper; the LLM is any OpenAI-compatible local server.
        from local_models import LocalWhisperSTT, PiperTTS

        stt = LocalWhisperSTT(language="en")
        tts = PiperTTS()
        model = openai.LLM(
            model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY, temperature=0.0
        )
    else:
        stt = openai.STT(model="whisper-1", language="en")
        model = openai.LLM(model=LLM_MODEL, temperature=0.0)
        tts = openai.TTS(model="tts-1", voice="nova")
    turn = None
    if USE_TURN_DETECTOR:
        from livekit.plugins.turn_detector.english import EnglishModel
        turn = EnglishModel()
    return vad, stt, model, tts, turn


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    room = ctx.room.name

    def log_tool_call(fn: str, args: dict, t0: float, t1: float) -> None:
        with open(TOOL_LOG, "a") as f:
            f.write(json.dumps({"room": room, "call": {"function": fn, "args": args,
                                                       "timestamp_start": t0, "timestamp_end": t1}}) + "\n")

    controller = ToolController(
        room=room,
        execute_fn=lambda fn, **kw: registry.call(fn, **kw),
        log_fn=log_tool_call,
        quiet_window=QUIET_WINDOW,
        trace_path=TRACE_PATH,
    )
    tracker = LatencyTracker()
    vad, stt, model, tts, turn = build_pipeline()

    session_kwargs = dict(vad=vad, stt=stt, llm=model, tts=tts,
                          min_endpointing_delay=MIN_EP, max_endpointing_delay=MAX_EP)
    if turn is not None:
        session_kwargs["turn_detection"] = turn
    session = AgentSession(**session_kwargs)

    @session.on("user_state_changed")
    def _on_user_state(ev):
        controller.on_user_state(ev.new_state)

    @session.on("user_input_transcribed")
    def _on_transcript(ev):
        if ev.is_final:
            tracker.user_done_at = time.time()   # latest final segment = end of user speech
            tracker.query_received = True

    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        if ev.new_state == "speaking" and tracker.query_received and not tracker.agent_start_at:
            tracker.agent_start_at = time.time()
            tracker.log_breakdown(room)
            tracker.reset()

    start_kwargs = {}
    if KEEP_ALIVE:
        start_kwargs["room_options"] = room_io.RoomOptions(close_on_disconnect=False)
    await session.start(room=ctx.room, agent=FDBAgent(controller, tracker), **start_kwargs)
    logging.info(f"FDB controller agent started in {room} (llm={LLM_MODEL}, quiet={QUIET_WINDOW}s)")


if __name__ == "__main__":
    agents.cli.run_app(server)
