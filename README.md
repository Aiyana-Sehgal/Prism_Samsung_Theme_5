# Interruptible Voice Agent with a Tool Commit Gate (Theme 05)

A LiveKit voice agent for Full-Duplex-Bench v3. The LLM only *proposes* tool calls; a small
coordination layer (`agent/controller.py`) decides whether each call actually runs. This stops the
agent from acting on half-finished or self-corrected sentences, and from repeating actions.

**Provider declaration:** every model runs **locally on the evaluating machine**. No paid API and
no model-provider account is used at evaluation time.

| Stage | Model | Runs on |
|---|---|---|
| VAD | Silero | CPU |
| Turn detection | LiveKit English turn detector | CPU |
| STT | faster-whisper (`small.en` default, `FDB_WHISPER_MODEL` to change) | CPU int8 |
| LLM | `qwen2.5:7b-instruct` via Ollama, an OpenAI-compatible local server | GPU if available, else CPU |
| TTS | Piper (`en_US-amy-medium`) | CPU |

The only credentials needed are **LiveKit Cloud** (free tier), because the benchmark harness drives
the agent over a LiveKit room. `reproduce.sh` downloads and starts the local LLM server itself — no
sudo, no system install, nothing outside the project directory.

**Why STT is on CPU by default.** faster-whisper's CTranslate2 backend links against CUDA 12, while
the pinned `torch` ships CUDA 13, so the two cannot share one GPU runtime out of the box. CPU int8
measured ~1.5× realtime on 16 cores here, which is fast enough, and it leaves the whole GPU to the
LLM — the component that actually benefits. To run STT on the GPU instead:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
# then in .env.local:  FDB_WHISPER_DEVICE=cuda
```

The adapter preloads those libraries itself and, if GPU transcription still fails mid-run, falls
back to CPU for the remainder of the session rather than aborting the run.

On a larger GPU set `FDB_WHISPER_MODEL=large-v3` for better transcription; `small.en` is the default
because it is accurate enough while staying quick on CPU.

> **Scoring note.** FDB-v3's official evaluation uses a hosted `gpt-4o` judge, which costs money.
> The headline numbers below are therefore the benchmark's **deterministic exact-match** scoring,
> which needs no judge and is bit-for-bit reproducible. `reproduce.sh` also runs the `--use-llm`
> judge against the *local* model and writes it to `*_llmjudge.json`. **Those LLM-judged numbers are
> not comparable to the paper's gpt-4o-judged baseline** and are reported only as a secondary signal.
> To make this possible, `reproduce.sh` applies a one-line patch to the benchmark's two evaluator
> scripts so the judge model is read from `FDB_JUDGE_MODEL` instead of being hardcoded to `gpt-4o`.
> No scoring logic is modified, and the report then records the model that actually judged.

## Architecture

```mermaid
flowchart TD
    A[User audio] --> B[Silero VAD + semantic turn detector]
    B --> C[Whisper STT]
    C --> D[LLM planner - proposes tool calls]
    D --> E{Tool controller}
    E -->|user resumed speaking| F[Superseded: not executed, not logged]
    E -->|duplicate call| G[Cached result: not repeated]
    E -->|quiet window passed| H[Mock API executes + call logged]
    H --> D
    D --> I[TTS reply]
```

The controller enforces three rules:

1. **Commit gate.** A call runs only once the user has been silent for a quiet window (default
   0.9 s, measured from the end of their speech, so LLM thinking time overlaps it). If they speak
   again, or LiveKit interrupts the reply, the call is dropped before execution.
2. **Idempotency.** Identical calls (same tool, normalized arguments) run once per session.
3. **In-flight dedup.** Concurrent identical proposals share one execution.

Why every tool is gated, including read-only ones: in FDB-v3's strict pass rate, any extra call fails
the scenario, so a stale "search Paris" before "actually, Berlin" is as costly as a stale booking.

Other decisions:
- The harness disconnects about 1.5 s after the audio ends; the session is kept alive
  (`close_on_disconnect=False`) so calls still inside the gate can finish and be logged.
- Mock tool latency uses `time.sleep`; execution runs in a thread so the event loop keeps
  handling speech events.
- Tool names, signatures and descriptions are identical to the benchmark's cascaded template.
  No benchmark items are hardcoded; prompt rules are general (corrections, spelled IDs, no invented details).

## Results (my best run)

| Metric | Cascaded template (baseline) | This agent |
|---|---|---|
| Strict pass rate | TODO | TODO |
| Tool selection F1 | TODO | TODO |
| Argument accuracy | TODO | TODO |
| First response latency | TODO | TODO |

Config: `FDB_QUIET_WINDOW=TODO`, `FDB_MAX_EP=TODO`, turn detector `TODO`, LLM `gpt-4o`, temperature 0.
Benchmark commit `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`. Full reports, tool logs and controller
traces are in `results/`. Hosted LLMs are not perfectly deterministic even at temperature 0;
TODO: note run-to-run variance.

## Reproduce (one command)

Requirements: Linux or macOS (on Windows, use WSL2 Ubuntu 22.04), Python 3.10, `ffmpeg`, `git`,
`unzip`, `curl`, `zstd`, and a free LiveKit Cloud project. **No model-provider API key is needed.**
A CUDA GPU is optional but makes the run much faster.

```bash
cp .env.example .env.local      # fill in LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
./reproduce.sh
```

The script creates a venv, installs pinned dependencies, clones the benchmark at the pinned commit,
downloads the official data, fetches and starts the local LLM server (Ollama, unpacked into `work/`,
no sudo) and the Piper TTS voice, runs all 100 examples, then evaluates with exact-match scoring and
again with the local LLM judge. Reports land in `results/`.

First run downloads roughly 10 GB (PyTorch + CUDA wheels, the LLM server, the 7B model, the
benchmark data and the TTS voice). Later runs reuse all of it.

**Important:** only one agent worker per LiveKit project may run during evaluation; LiveKit
dispatches rooms to any registered worker, and tool calls are logged on the machine that ran them.

Offline tests (no keys needed): `python tests/test_controller.py && python tests/test_extension.py`

## ⭐ Extension use case: interruptible smart-home assistant

`extension/smart_home_agent.py`: a hands-free home assistant (AC, lights, door lock) using the same
controller. It adds **state-based idempotency**: if a device is already in the requested state the
tool reports `no_change` and does nothing, while a user can still legitimately repeat an action later.
Session-level dedup (right for the benchmark) would wrongly block that in a real home.

```bash
python extension/smart_home_agent.py download-files   # first time
python extension/smart_home_agent.py console          # talk through your microphone
```

Live home state is printed in the terminal after every change. Example: *"Set the AC to 22... no
wait, the bedroom one, and make it 24"* changes only the bedroom AC.

## Limitations (honest)

- Latency trade-off: the commit gate adds delay when the LLM is faster than the quiet window.
- A pause longer than the quiet window, followed by a correction, can still let the first call run.
- **A local 7B model is weaker at function calling than a frontier hosted model.** The commit gate is
  model-agnostic, so the architecture's contribution holds, but absolute scores are lower than the
  same design would reach on `gpt-4o`. `FDB_STACK=openai` runs that comparison if a key is supplied.
- **Headline numbers use exact-match scoring, not the paper's `gpt-4o` judge**, so they are not
  directly comparable to the published baseline. See the scoring note above.
- `small.en` Whisper is a VRAM compromise on 8 GB; transcription errors on spelled-out IDs are the
  single biggest source of argument mismatches. A larger GPU should set `FDB_WHISPER_MODEL=large-v3`.
- Spoken replies are often cut off by the harness recording window, which affects response scoring for all systems.

## What I would do next

TODO: e.g. streaming STT for lower latency, a learned correction detector, a camera-grounded extension.
