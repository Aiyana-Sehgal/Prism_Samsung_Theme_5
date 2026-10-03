"""
ToolController: the coordination layer between the LLM and the tools.

The LLM only *proposes* tool calls. The controller decides whether a call
actually runs. Three rules:

1. Commit gate: a call runs only after the user has been silent for
   `quiet_window` seconds since their last speech. If they start speaking
   again while a call waits, the call is dropped as superseded and never
   logged. This stops actions on half-finished, self-corrected utterances.
2. Idempotency: an identical call (same tool, normalized args) runs at most
   once per session. Duplicates get the cached result and are not logged.
3. In-flight dedup: if the same call is proposed twice concurrently, the
   second waits for the first instead of running again.

Only calls that pass all three reach the mock API and the benchmark log.
Everything is session-scoped: one controller per LiveKit room.
"""

import asyncio
import json
import re
import time
from typing import Any, Callable, Optional

# Classified from the tool schemas (what each tool does), not from test items.
STATE_CHANGING = {
    "book_flight",
    "update_identity_doc",
    "modify_autopay",
    "update_search_filter",
    "add_to_cart",
}


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"[\s\-\._]", "", value).lower()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def call_key(fn: str, args: dict) -> str:
    norm = {k: _normalize(v) for k, v in sorted(args.items()) if v is not None}
    return f"{fn}|{json.dumps(norm, sort_keys=True, default=str)}"


class ToolController:
    def __init__(
        self,
        room: str,
        execute_fn: Callable[..., dict],
        log_fn: Callable[[str, dict, float, float], None],
        quiet_window: float = 0.9,
        max_wait: float = 8.0,
        trace_path: Optional[str] = None,
        dedupe: bool = True,
    ):
        self.room = room
        self._execute_fn = execute_fn  # sync function(fn_name, **args) -> dict
        self._log_fn = log_fn          # benchmark telemetry logger
        self.quiet_window = quiet_window
        self.max_wait = max_wait
        self.trace_path = trace_path
        # dedupe=True: session-level idempotency (benchmark). dedupe=False: the tools
        # themselves are state-idempotent (extension), so a user can legitimately repeat an action later.
        self.dedupe = dedupe

        self._user_speaking = False
        self._last_speech_end = time.monotonic()
        self._speech_generation = 0    # +1 each time the user starts speaking

        self._done: dict[str, dict] = {}
        self._inflight: dict[str, asyncio.Future] = {}

    # ── Signals from the voice pipeline ──────────────────────────────
    def on_user_state(self, new_state: str) -> None:
        if new_state == "speaking":
            self._user_speaking = True
            self._speech_generation += 1
        else:
            if self._user_speaking:
                self._last_speech_end = time.monotonic()
            self._user_speaking = False

    # ── Core ─────────────────────────────────────────────────────────
    async def _wait_for_quiet(self, generation: int) -> bool:
        """True if the user stayed quiet long enough; False if superseded."""
        deadline = time.monotonic() + self.max_wait
        while time.monotonic() < deadline:
            if self._speech_generation != generation:
                return False
            quiet_for = time.monotonic() - self._last_speech_end
            if not self._user_speaking and quiet_for >= self.quiet_window:
                return True
            await asyncio.sleep(0.03)
        return self._speech_generation == generation

    async def run(self, fn: str, args: dict, is_interrupted: Callable[[], bool] = lambda: False) -> dict:
        generation = self._speech_generation

        if not await self._wait_for_quiet(generation) or is_interrupted():
            self._trace("superseded", fn, args)
            return {
                "status": "superseded",
                "note": "The user kept talking, so this call was NOT executed. "
                        "Wait for their full request and use their latest wording.",
            }

        key = call_key(fn, args)
        if self.dedupe and key in self._done:
            self._trace("dedup", fn, args)
            return {**self._done[key], "note": "Already done earlier in this conversation; not repeated."}
        if key in self._inflight:
            self._trace("dedup_inflight", fn, args)
            return await self._inflight[key]

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            t0 = time.time()
            # The mock API uses time.sleep for latency injection; keep it off the event loop.
            result = await asyncio.to_thread(self._execute_fn, fn, **args)
            t1 = time.time()
            self._done[key] = result
            self._log_fn(fn, args, t0, t1)
            self._trace("executed", fn, args, state_changing=fn in STATE_CHANGING)
            future.set_result(result)
            return result
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            self._inflight.pop(key, None)

    def _trace(self, decision: str, fn: str, args: dict, **extra) -> None:
        if not self.trace_path:
            return
        with open(self.trace_path, "a") as f:
            f.write(json.dumps({"room": self.room, "t": time.time(), "decision": decision,
                                "function": fn, "args": args, **extra}, default=str) + "\n")
