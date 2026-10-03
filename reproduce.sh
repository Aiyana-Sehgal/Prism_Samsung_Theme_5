#!/usr/bin/env bash
# One-command FDB-v3 reproduction. Fully local: every model runs on this machine.
#
# No paid API is used and no provider account is needed. The only credentials are
# LiveKit Cloud's (free tier), because the benchmark harness drives the agent over
# a LiveKit room.
#
# Needs: Linux or macOS, Python 3.10, ffmpeg, git, unzip, curl, zstd, and
#        .env.local (see .env.example) with LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
#
# Hardware: runs on CPU, but a CUDA GPU makes it far faster. On >=16 GB of VRAM set
#           FDB_WHISPER_MODEL=large-v3 in .env.local for better transcription.
set -euo pipefail

FDB_COMMIT="3e799c45a045256f47d5f1c9cda90157e2d2ec9e"   # pinned benchmark version
DATA_ID="1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz"             # official data from the v3 README
PROVIDER="controller"                                   # label for result/report files
OLLAMA_VERSION="v0.35.1"                                # pinned local LLM server
PIPER_VOICE="en_US-amy-medium"
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$HERE/work"

[[ -f "$HERE/.env.local" ]] || { echo "Missing .env.local (copy .env.example and fill it in)"; exit 1; }
for cmd in ffmpeg git unzip curl zstd; do
  command -v "$cmd" >/dev/null || { echo "$cmd is required"; exit 1; }
done

set -a; source "$HERE/.env.local"; set +a
LLM_MODEL="${FDB_LLM_MODEL:-qwen2.5:7b-instruct}"
JUDGE_MODEL="${FDB_JUDGE_MODEL:-$LLM_MODEL}"

echo "== 1/8 Python environment"
if [[ ! -x "$WORK/venv/bin/python" ]]; then
  mkdir -p "$WORK"
  # Fail loudly: silencing this hides the real cause (usually a missing python3-venv)
  # behind a confusing "activate: No such file or directory" on the next line.
  python3 -m venv "$WORK/venv" || {
    echo "Could not create a virtualenv at $WORK/venv."
    echo "Install the venv module first, e.g.:  sudo apt-get install -y python3-venv"
    exit 1
  }
fi
source "$WORK/venv/bin/activate"
python -c 'import sys; assert sys.version_info[:2] >= (3,10), sys.version' || {
  echo "Python 3.10+ required; got $(python -V)"; exit 1; }
pip install -q --upgrade pip
pip install -q -r "$HERE/requirements.txt"

echo "== 2/8 Benchmark at pinned commit"
if [[ ! -d "$WORK/Full-Duplex-Bench" ]]; then
  git clone -q https://github.com/DanielLin94144/Full-Duplex-Bench.git "$WORK/Full-Duplex-Bench"
fi
cd "$WORK/Full-Duplex-Bench" && git checkout -q "$FDB_COMMIT" && cd v3
cp "$HERE/agent/controller.py" "$HERE/agent/fdb_agent.py" "$HERE/agent/local_models.py" .
cp "$HERE/.env.local" .env.local

echo "== 3/8 Benchmark data"
if [[ ! -d fdb_v3_data_released ]]; then
  gdown -q "$DATA_ID" -O fdb_v3_data.zip
  unzip -q fdb_v3_data.zip && rm fdb_v3_data.zip
fi
[[ -d fdb_v3_data_released ]] || { echo "Data folder not found after unzip; check archive layout"; exit 1; }

echo "== 4/8 Local models (LLM server + TTS voice)"
# Ollama, unpacked into work/ (no sudo, no system install).
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="$WORK/ollama/models"
OLLAMA_BIN="$WORK/ollama/bin/ollama"
if [[ ! -x "$OLLAMA_BIN" ]]; then
  mkdir -p "$WORK/ollama"
  echo "   downloading ollama $OLLAMA_VERSION (~1.4 GB, once)"
  curl -fsSL -o "$WORK/ollama.tar.zst" \
    "https://github.com/ollama/ollama/releases/download/$OLLAMA_VERSION/ollama-linux-amd64.tar.zst"
  tar --zstd -xf "$WORK/ollama.tar.zst" -C "$WORK/ollama"
  rm -f "$WORK/ollama.tar.zst"
fi
[[ -x "$OLLAMA_BIN" ]] || { echo "ollama binary not found after extract"; exit 1; }

# Piper voice for TTS.
mkdir -p "$HERE/agent/piper"
for ext in onnx onnx.json; do
  [[ -f "$HERE/agent/piper/$PIPER_VOICE.$ext" ]] || \
    curl -fsSL -o "$HERE/agent/piper/$PIPER_VOICE.$ext" "$PIPER_BASE/$PIPER_VOICE.$ext"
done
export FDB_PIPER_VOICE="${FDB_PIPER_VOICE:-$HERE/agent/piper/$PIPER_VOICE.onnx}"

# Keep the model resident. The default 5 min unload would make any later scenario pay a
# multi-second cold start, and calls that land late count as missing.
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"

# Start the LLM server and pull the model.
"$OLLAMA_BIN" serve > "$HERE/ollama.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null && break
  sleep 1
done
curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null || { echo "ollama did not start; see ollama.log"; exit 1; }
echo "   pulling $LLM_MODEL (once)"
"$OLLAMA_BIN" pull "$LLM_MODEL"
[[ "$JUDGE_MODEL" == "$LLM_MODEL" ]] || "$OLLAMA_BIN" pull "$JUDGE_MODEL"

echo "== 5/8 Make the judge model configurable"
# The benchmark hardcodes model="gpt-4o" in its LLM-judge calls. This minimal, general
# patch reads FDB_JUDGE_MODEL instead so the judge can be the local model, and so the
# report records the model that actually judged. No scoring logic is changed. Idempotent.
sed -i 's/model="gpt-4o",/model=os.environ.get("FDB_JUDGE_MODEL", "gpt-4o"),/g' \
  evaluate_tool_calls.py evaluate_pass_rate.py

echo "== 6/8 Model files + warm up + clean logs"
python fdb_agent.py download-files
# Load the LLM into VRAM and the Whisper weights into RAM before the benchmark starts,
# so the first scenario is not penalised by a cold start.
curl -sf "http://$OLLAMA_HOST/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
  >/dev/null || echo "   (warm-up call failed; continuing)"
# Warming is an optimisation, never a reason to abort the run.
python -c "
import sys; sys.path.insert(0, '.')
from local_models import LocalWhisperSTT, PiperTTS
LocalWhisperSTT().prewarm(); PiperTTS()
print('   STT + TTS warm')
" || echo "   (warm-up skipped: $? - the agent will load models on first use)"
rm -f /tmp/agent_tool_calls.log /tmp/agent_heartbeat.log /tmp/fdb_controller_trace.log

echo "== 7/8 Start agent and run inference"
python fdb_agent.py start > "$HERE/agent_run.log" 2>&1 &
AGENT_PID=$!
trap 'kill $AGENT_PID $OLLAMA_PID 2>/dev/null || true' EXIT
sleep 20   # let the worker register with LiveKit Cloud
python run_tool_benchmark_all_released.py --provider "$PROVIDER" --root_dir fdb_v3_data_released --force
kill $AGENT_PID 2>/dev/null || true

echo "== 8/8 Evaluate"
# Primary numbers: deterministic exact-match scoring, no judge, reproducible by anyone.
python evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released \
  --provider "$PROVIDER" --output "${PROVIDER}_evaluation_report.json"
python evaluate_pass_rate.py --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released \
  --provider "$PROVIDER" --output "${PROVIDER}_pass_rate_report.json"

# Secondary: the LLM judge, run against the local model. NOT comparable to the paper's
# gpt-4o-judged baseline -- see README.
export OPENAI_BASE_URL="http://$OLLAMA_HOST/v1"
export OPENAI_API_KEY="${FDB_LLM_API_KEY:-ollama}"
export FDB_JUDGE_MODEL="$JUDGE_MODEL"
python evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released \
  --provider "$PROVIDER" --output "${PROVIDER}_evaluation_report_llmjudge.json" --use-llm || \
  echo "   (local LLM judge failed; exact-match reports above are the primary result)"
python evaluate_pass_rate.py --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released \
  --provider "$PROVIDER" --output "${PROVIDER}_pass_rate_report_llmjudge.json" --use-llm || \
  echo "   (local LLM judge failed; exact-match reports above are the primary result)"

python analyze_tool_latency.py --results-dir fdb_v3_data_released --provider "$PROVIDER"

mkdir -p "$HERE/results"
cp ${PROVIDER}_*report*.json "$HERE/results/" 2>/dev/null || true
cp /tmp/agent_tool_calls.log /tmp/fdb_controller_trace.log "$HERE/results/" 2>/dev/null || true
echo "Done. Reports and logs in $HERE/results/"
echo "  primary (exact match): ${PROVIDER}_pass_rate_report.json"
echo "  secondary (local judge): ${PROVIDER}_pass_rate_report_llmjudge.json"
