# Run log

All runs use FDB-v3 at commit `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`, the local stack
(faster-whisper + `qwen2.5:7b-instruct` via Ollama + Piper), temperature 0.

Pass@1 is reported two ways: **exact** is the benchmark's deterministic exact-match scoring;
**judge** is its `--use-llm` judge pointed at the local 7B model. Official FDB-v3 uses a hosted
`gpt-4o` judge, which is not used here (no paid APIs), so neither column is directly comparable to
the published numbers. See README.

| # | Label | Hardware | Whisper | Scenarios | Pass@1 exact | Pass@1 judge | Tool sel | Arg acc | Latency | Interrupt | Turn-take |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `laptop-cpu-whisper` | RTX 4060 Laptop, 8 GB | `small.en` CPU int8 | 93/100 | 0.022 | — | 0.202 | 0.043 | 6.49 s | 65.4% | 0.28 |
| 2 | **`kaggle-t4-gpu-whisper`** | **T4, 16 GB** | **`medium.en` CUDA fp16** | **100/100** | **0.290** | **0.500** | **0.895** | **0.418** | **5.12 s** | **4.3%** | **0.94** |

Run 2 is the reported result. Its reports are the files in this directory.

## Run 1 — laptop (not reported, kept for the hardware finding)

Interrupted twice by host instability (a `DRIVER_POWER_STATE_FAILURE` bugcheck traced to the
laptop's Wi-Fi adapter power management, unrelated to the agent), then completed to 93/100 by
resuming without `--force`.

The low score is **not** an agent fault. On 8 GB of VRAM the LLM leaves no room for GPU Whisper, so
STT runs on CPU; mean response latency reaches 6.49 s against a harness that disconnects 1.5 s after
the audio ends. The agent therefore never responded in 67 of 93 scenarios (turn-take 0.28) and 65.4%
of the responses it did produce were interruptions. Controller decisions over the run were
`executed 35, superseded 4, dedup 1` — the gate behaved correctly throughout.

Config: `FDB_QUIET_WINDOW=0.9`, `FDB_MIN_EP=0.6`, `FDB_MAX_EP=2.0`, `FDB_TURN_DETECTOR=1`,
`FDB_KEEP_ALIVE=1`, `FDB_WHISPER_MODEL=small.en`, `FDB_WHISPER_DEVICE=cpu`.

## Run 2 — Kaggle T4 (reported)

Same commit, same code, same model, same prompt. Only the hardware and the Whisper placement
changed. `./reproduce.sh` was run unmodified on a clean Linux instance with no pre-existing venv,
model or data — the closest available analogue to a fresh judging machine.

Config: as run 1 except `FDB_WHISPER_MODEL=medium.en`, `FDB_WHISPER_DEVICE=cuda`,
`FDB_WHISPER_COMPUTE=float16`. Wall-clock: ~102 minutes including ~25 minutes of first-time
downloads.

Controller decisions: `executed 128 (85.3%)`, `superseded 18 (12.0%)`, `dedup 4 (2.7%)` across 96
rooms — 22 calls blocked before reaching the tool log, each of which would have failed its scenario
under the strict rule.

### Failure composition (71 failures)

| Category | Count |
|---|---|
| Wrong arguments only | 38 |
| Missing tool(s) | 31 |
| Unexpected (extra) tool(s) | 5 |

Argument failures by root cause: 41% genuine comprehension, 23% date reformatted or year invented,
18% verbosity (`savings_account` vs `savings`), 10% spelled-out ID punctuation (`D-L-5-5-5` vs
`DL555`), 8% expected an argument absent from the official tool schema.

## Tuning

No tuning runs were performed. The time budgeted for Phase 3 was spent on reproducibility defects
found while getting `reproduce.sh` to run on a clean machine (see below), and on the hardware
migration after the laptop proved unstable. The reported configuration is therefore the initial
default, not a tuned optimum — the quiet window of 0.9 s has not been swept.

`quick_eval.sh` exists for that sweep and runs a ~25-scenario subset of the hardest cases
(`./quick_eval.sh <label>`); candidates are `FDB_QUIET_WINDOW` 0.6/0.9/1.2, `FDB_MAX_EP`, and
`FDB_TURN_DETECTOR` 1 vs 0.

## Reproducibility defects found and fixed

Each of these would have broken a fresh run on the organizers' machine:

1. **Turn detector weights never downloaded.** The plugin was imported lazily inside
   `build_pipeline()`, so `download-files` — which only fetches for plugins registered by the time
   the CLI runs — never saw it. Every job then crashed on `EnglishModel()` with
   `Could not find file "languages.json"`, and a full 100-scenario run completed while scoring an
   agent that never started. Fixed by importing the plugins at module scope. (`36e8a19`)
2. **Silent virtualenv failure.** `python3 -m venv ... 2>/dev/null || true` hid a missing
   `python3-venv` behind a misleading `activate: No such file or directory` one line later. Now
   fails with the cause and the remedy. (`5c2d5af`)
3. **`TypeError` preloading CUDA 12 libraries.** `nvidia.cublas.lib` is a namespace package whose
   `__file__` is `None`, so `os.path.dirname(mod.__file__)` raised. Only triggered on a GPU host,
   so it never appeared on the CPU default. (`c9e72ce`)
4. **Model unloaded mid-run.** Ollama's default 5-minute keep-alive meant later scenarios paid a
   multi-second cold start, and late calls are scored as missing. Now pinned with
   `OLLAMA_KEEP_ALIVE=24h` plus an explicit warm-up of all three models. (`b12e9bc`)
