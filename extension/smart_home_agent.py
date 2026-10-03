#!/usr/bin/env python3
"""
EXTENSION USE CASE: hands-free smart-home assistant (beyond the FDB-v3 domains).

Same architecture as the benchmark agent: cascaded STT + LLM + TTS, with every
device action routed through ToolController's commit gate. What changes:

- Devices hold real state (the HOME dict), printed live in the terminal.
- Idempotency is state-based instead of session-based: if the device is already
  in the requested state, the tool reports that and does nothing. A user can still
  legitimately repeat an action later ("back to 22") after something changed it.

Run with your microphone:
    python smart_home_agent.py download-files     # first time only
    python smart_home_agent.py console

Try: "Set the AC to 22... no wait, the bedroom one, and make it 24."
     "Turn off the living room lights. Actually, dim them to 30 percent instead."
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))

from dotenv import load_dotenv  # noqa: E402
from livekit import agents  # noqa: E402
from livekit.agents import Agent, AgentServer, AgentSession, RunContext, llm  # noqa: E402

from controller import ToolController  # noqa: E402

# Registered at import time so `download-files` fetches their weights; a lazy import
# inside entrypoint() is too late and the turn detector then fails to initialise.
from livekit.plugins import openai as _plugin_openai  # noqa: F401,E402
from livekit.plugins import silero as _plugin_silero  # noqa: F401,E402
from livekit.plugins import turn_detector as _plugin_turn_detector  # noqa: F401,E402

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env.local"))

INITIAL_HOME = {
    "living room": {"ac": {"power": "on", "temperature": 25}, "lights": {"power": "on", "brightness": 100}},
    "bedroom":     {"ac": {"power": "off", "temperature": 26}, "lights": {"power": "off", "brightness": 0}},
    "kitchen":     {"lights": {"power": "on", "brightness": 80}},
    "front door":  {"lock": {"state": "unlocked"}},
}


# ── Home state + device tools (pure functions, testable offline) ─────
class Home:
    def __init__(self):
        self.state = copy.deepcopy(INITIAL_HOME)
        self.actions = []

    def _device(self, room: str, device: str):
        room, device = room.lower().strip(), device.lower().strip()
        if room not in self.state:
            return None, f"Unknown room '{room}'. Rooms: {', '.join(self.state)}."
        if device not in self.state[room]:
            return None, f"No {device} in the {room}. Devices there: {', '.join(self.state[room])}."
        return self.state[room][device], None

    def _changed(self, room, device, before, after):
        self.actions.append({"t": round(time.time(), 2), "room": room, "device": device,
                             "before": before, "after": after})
        self.render(f"{room} {device}: {before} -> {after}")

    def get_status(self, room: str) -> dict:
        room = room.lower().strip()
        if room == "all":
            return {"status": "success", "home": self.state}
        if room not in self.state:
            return {"status": "error", "message": f"Unknown room '{room}'."}
        return {"status": "success", "room": room, "devices": self.state[room]}

    def set_ac(self, room: str, temperature: int, power: str = "on") -> dict:
        dev, err = self._device(room, "ac")
        if err:
            return {"status": "error", "message": err}
        if not 16 <= int(temperature) <= 30:
            return {"status": "error", "message": "Temperature must be between 16 and 30."}
        target = {"power": power, "temperature": int(temperature)}
        if dev == target:
            return {"status": "no_change", "message": f"The {room} AC is already {power} at {temperature}."}
        before = dict(dev)
        dev.update(target)
        self._changed(room.lower(), "ac", before, dict(dev))
        return {"status": "success", "room": room, "ac": dict(dev)}

    def set_lights(self, room: str, power: str, brightness: int = 100) -> dict:
        dev, err = self._device(room, "lights")
        if err:
            return {"status": "error", "message": err}
        target = {"power": power, "brightness": 0 if power == "off" else max(1, min(100, int(brightness)))}
        if dev == target:
            return {"status": "no_change", "message": f"The {room} lights are already like that."}
        before = dict(dev)
        dev.update(target)
        self._changed(room.lower(), "lights", before, dict(dev))
        return {"status": "success", "room": room, "lights": dict(dev)}

    def set_lock(self, state: str) -> dict:
        dev, _ = self._device("front door", "lock")
        if dev["state"] == state:
            return {"status": "no_change", "message": f"The front door is already {state}."}
        before = dict(dev)
        dev["state"] = state
        self._changed("front door", "lock", before, dict(dev))
        return {"status": "success", "lock": dict(dev)}

    def render(self, event: str = ""):
        lines = ["", "=" * 56, f"  HOME STATE   {event}", "-" * 56]
        for room, devices in self.state.items():
            parts = []
            for name, d in devices.items():
                if name == "ac":
                    parts.append(f"AC {d['power']} {d['temperature']}C")
                elif name == "lights":
                    parts.append(f"lights {d['power']} {d['brightness']}%")
                else:
                    parts.append(f"lock {d['state']}")
            lines.append(f"  {room:<12} | " + " | ".join(parts))
        lines.append("=" * 56)
        print("\n".join(lines), flush=True)


INSTRUCTIONS = """You are a hands-free home assistant. Replies are spoken, so keep them to one short sentence.
Rooms: living room, bedroom, kitchen, front door. Use tools for every device action or status question.

People interrupt and correct themselves ("no wait", "actually", "the other one", "I mean").
Only the final version counts: apply the correction, keep details they did not change,
and drop anything they abandoned. Never act on a request they took back.
Only say something is done after the tool confirms it. If a tool returns "no_change", say it was already set.
If a tool returns "superseded", the user was still talking: do not retry; wait for their full request.
If the room is genuinely ambiguous, ask one short question instead of guessing.
"""


class HomeAgent(Agent):
    def __init__(self, ctl: ToolController):
        super().__init__(instructions=INSTRUCTIONS)
        self.ctl = ctl

    async def _run(self, ctx: RunContext, fn: str, args: dict) -> str:
        return json.dumps(await self.ctl.run(fn, args, is_interrupted=lambda: ctx.speech_handle.interrupted))

    @llm.function_tool(description="Read the current state of devices in a room, or 'all' for the whole home. Read-only.")
    async def get_status(self, ctx: RunContext, room: str):
        """
        Args:
            room: 'living room', 'bedroom', 'kitchen', 'front door', or 'all'
        """
        return await self._run(ctx, "get_status", {"room": room})

    @llm.function_tool(description="Set an air conditioner's temperature and power. Changes device state.")
    async def set_ac(self, ctx: RunContext, room: str, temperature: int, power: str = "on"):
        """
        Args:
            room: 'living room' or 'bedroom'
            temperature: Target temperature in Celsius, 16-30
            power: 'on' or 'off'
        """
        return await self._run(ctx, "set_ac", {"room": room, "temperature": temperature, "power": power})

    @llm.function_tool(description="Turn lights on or off, or set brightness. Changes device state.")
    async def set_lights(self, ctx: RunContext, room: str, power: str, brightness: int = 100):
        """
        Args:
            room: 'living room', 'bedroom' or 'kitchen'
            power: 'on' or 'off'
            brightness: 1-100 percent when on
        """
        return await self._run(ctx, "set_lights", {"room": room, "power": power, "brightness": brightness})

    @llm.function_tool(description="Lock or unlock the front door. Changes device state.")
    async def set_lock(self, ctx: RunContext, state: str):
        """
        Args:
            state: 'locked' or 'unlocked'
        """
        return await self._run(ctx, "set_lock", {"state": state})


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    from livekit.plugins import openai, silero
    from livekit.plugins.turn_detector.english import EnglishModel

    home = Home()
    home.render("session start")

    ctl = ToolController(
        room=ctx.room.name,
        execute_fn=lambda fn, **kw: getattr(home, fn)(**kw),
        log_fn=lambda fn, a, t0, t1: None,
        quiet_window=float(os.getenv("HOME_QUIET_WINDOW", "0.8")),
        trace_path=os.getenv("HOME_TRACE", "/tmp/home_controller_trace.log"),
        dedupe=False,  # state-based idempotency lives in the Home tools
    )

    # Same local, zero-cost stack as the benchmark agent.
    local = os.getenv("FDB_STACK", "local").lower() == "local"
    if local:
        from local_models import LocalWhisperSTT, PiperTTS

        stt_impl = LocalWhisperSTT(language="en")
        tts_impl = PiperTTS()
        llm_impl = openai.LLM(
            model=os.getenv("FDB_LLM_MODEL", "qwen2.5:7b-instruct"),
            base_url=os.getenv("FDB_LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("FDB_LLM_API_KEY", "ollama"),
            temperature=0.0,
        )
    else:
        stt_impl = openai.STT(model="whisper-1", language="en")
        tts_impl = openai.TTS(model="tts-1", voice="nova")
        llm_impl = openai.LLM(model=os.getenv("FDB_LLM_MODEL", "gpt-4o"), temperature=0.0)

    session = AgentSession(
        vad=silero.VAD.load(min_speech_duration=0.05, min_silence_duration=0.55),
        stt=stt_impl,
        llm=llm_impl,
        tts=tts_impl,
        turn_detection=EnglishModel(),
        min_endpointing_delay=0.6,
        max_endpointing_delay=2.5,
    )

    @session.on("user_state_changed")
    def _on_user_state(ev):
        ctl.on_user_state(ev.new_state)

    @session.on("user_input_transcribed")
    def _on_transcript(ev):
        if ev.is_final:
            print(f"  [user] {ev.transcript}", flush=True)

    await session.start(room=ctx.room, agent=HomeAgent(ctl))
    await session.say("Home assistant ready.", add_to_chat_ctx=False)


if __name__ == "__main__":
    agents.cli.run_app(server)
