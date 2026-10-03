#!/usr/bin/env bash
# Fast iteration loop on a ~25-example subset (self-corrections, pauses, hard chains).
# Run ./reproduce.sh once first (it installs deps, the LLM server and the TTS voice).
# Usage:  ./quick_eval.sh [label]   e.g. ./quick_eval.sh qw09
# Tune via env vars in .env.local (FDB_QUIET_WINDOW, FDB_MAX_EP, FDB_TURN_DETECTOR, FDB_LLM_MODEL).
set -euo pipefail
LABEL="${1:-quick}"
HERE="$(cd "$(dirname "$0")" && pwd)"
V3="$HERE/work/Full-Duplex-Bench/v3"
WORK="$HERE/work"
source "$WORK/venv/bin/activate"

set -a; source "$HERE/.env.local"; set +a
LLM_MODEL="${FDB_LLM_MODEL:-qwen2.5:7b-instruct}"

cd "$V3"
cp "$HERE/agent/controller.py" "$HERE/agent/fdb_agent.py" "$HERE/agent/local_models.py" .
cp "$HERE/.env.local" .env.local

# Local LLM server (reuses what reproduce.sh unpacked; starts it only if not already up).
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="$WORK/ollama/models"
export FDB_PIPER_VOICE="${FDB_PIPER_VOICE:-$HERE/agent/piper/en_US-amy-medium.onnx}"
OLLAMA_PID=""
if ! curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null; then
  "$WORK/ollama/bin/ollama" serve > "$HERE/ollama.log" 2>&1 &
  OLLAMA_PID=$!
  for _ in $(seq 1 60); do
    curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null && break
    sleep 1
  done
fi

SUB="subset_$LABEL"; rm -rf "$SUB"; mkdir "$SUB"
python - "$SUB" <<'PY'
import json, os, sys
sub = sys.argv[1]
S = json.load(open("benchmark_data_v2.json"))["scenarios"]
pick = {s["id"] for s in S if s["state_rollback_test"]
        or {"SELF_CORRECTION", "PAUSE"} & set(s["disfluency_features"])
        or s["difficulty"] == "hard"}
pick = set(sorted(pick)[::2])          # every other one keeps runs short (~25 examples)
n = 0
for d in sorted(os.listdir("fdb_v3_data_released")):
    if any(d.startswith(i + "_") for i in pick):
        os.symlink(os.path.abspath(f"fdb_v3_data_released/{d}"), f"{sub}/{d}"); n += 1
print(f"subset: {n} examples")
PY

rm -f /tmp/agent_tool_calls.log /tmp/fdb_controller_trace.log
python fdb_agent.py start > "$HERE/agent_$LABEL.log" 2>&1 & PID=$!
trap 'kill $PID $OLLAMA_PID 2>/dev/null || true' EXIT
sleep 20
python run_tool_benchmark_all_released.py --provider "$LABEL" --root_dir "$SUB" --force
kill $PID 2>/dev/null || true

# Exact-match scoring: deterministic, and what tuning decisions are made on.
python evaluate_pass_rate.py --benchmark benchmark_data_v2.json --results-dir "$SUB" \
  --provider "$LABEL" --output "${LABEL}_pass_rate_report.json" | tail -25

mkdir -p "$HERE/results/$LABEL"
cp "${LABEL}_pass_rate_report.json" /tmp/fdb_controller_trace.log "$HERE/results/$LABEL/" 2>/dev/null || true
echo "Saved to results/$LABEL. Check the trace for 'superseded' and 'dedup' decisions on failed items."
