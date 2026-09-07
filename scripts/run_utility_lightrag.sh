#!/usr/bin/env bash
# Benign QA utility on LightRAG: start chat + embedding vLLM, run the experiment, stop both.
# Run from the project venv:
#   bash scripts/run_utility_lightrag.sh
#   bash scripts/run_utility_lightrag.sh --config configs/experiments/utility_isolation_lightrag_medical.yaml
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
EXPERIMENT="configs/experiments/utility_none_lightrag_medical.yaml"
LIGHTRAG_CONFIG="configs/lightrag/query_dual.yaml"
RESULTS_ROOT=""
KEEP=0
CHAT_PIDS=()
EMBED_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_utility_lightrag.sh [options]

  (default)          Start vLLM if needed, run utility QA, stop vLLM.
  --keep-vllm        Leave both vLLM processes running after the run.
  --config           Experiment YAML
                     (default: configs/experiments/utility_none_lightrag_medical.yaml)
  --lightrag-config  LightRAG / vLLM YAML (default: configs/lightrag/query_dual.yaml)
  --results-root     Override results directory
  -h, --help         Show this help.

Gold answers come from data/qa/<dataset>_questions.json (GraphRAG-Bench).
Run none and isolation as two experiments, then compare:

  python -m safe_rag.eval.report --utility \
    --compare results/utility_none_lightrag_medical \
              results/utility_isolation_lightrag_medical \
    --dest results/utility_lightrag_medical_compare.json

Logs (overwritten each run):
  logs/<attack>_<defense>_<system>_<dataset>/vllm-chat.log
  logs/<attack>_<defense>_<system>_<dataset>/vllm-embed.log
  logs/<attack>_<defense>_<system>_<dataset>/run.log
  logs/<attack>_<defense>_<system>_<dataset>/lightrag.log
  logs/<attack>_<defense>_<system>_<dataset>/turn_logs/

Results:
  results/<attack>_<defense>_<system>_<dataset>/
EOF
}

stage() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok() { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

yaml_get() {
  local file="$1"
  local key="$2"
  "$PYTHON" -c "
import yaml, sys
from pathlib import Path
raw = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
value = raw
for part in sys.argv[2].split('.'):
    if not isinstance(value, dict):
        value = ''
        break
    value = value.get(part, '')
print('' if value is None else value)
" "$file" "$key"
}

stop_pid() {
  local pid="${1:-}"
  local name="${2:-vLLM}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    ok "$name stopped"
  fi
}

start_role() {
  local role="$1"
  local log="$2"
  local replica="${3:-0}"
  local extra=()
  if [[ "$role" == "chat" ]]; then
    extra+=(--replica "$replica")
  fi
  if "$PYTHON" -m safe_rag.systems.lightrag.vllm_server ready --role "$role" --config "$LIGHTRAG_CONFIG" "${extra[@]}"; then
    ok "$role replica $replica already running"
    return 0
  fi
  "$PYTHON" -m safe_rag.systems.lightrag.vllm_server prepare-socket --role "$role" --config "$LIGHTRAG_CONFIG" "${extra[@]}"
  local devices
  devices="$("$PYTHON" -m safe_rag.systems.lightrag.vllm_server devices --role "$role" --config "$LIGHTRAG_CONFIG" "${extra[@]}")"
  local cmd=()
  mapfile -t cmd < <("$PYTHON" -m safe_rag.systems.lightrag.vllm_server argv --role "$role" --config "$LIGHTRAG_CONFIG" "${extra[@]}")
  ok "$role replica $replica GPUs: $devices"
  ok "$role logs → $log"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    CUDA_VISIBLE_DEVICES="$devices" \
    "${cmd[@]}" >"$log" 2>&1 &
  local pid=$!
  if [[ "$role" == "chat" ]]; then
    CHAT_PIDS+=("$pid")
  else
    EMBED_PID="$pid"
  fi
  if ! "$PYTHON" -m safe_rag.systems.lightrag.vllm_server wait --role "$role" --config "$LIGHTRAG_CONFIG" --pid "$pid" "${extra[@]}"; then
    err "$role vLLM replica $replica failed to start. Last log lines:"
    tail -n 40 "$log" >&2 || true
    return 1
  fi
  ok "$role replica $replica ready (pid $pid)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-vllm) KEEP=1 ;;
    --config)
      EXPERIMENT="$2"
      shift
      ;;
    --lightrag-config)
      LIGHTRAG_CONFIG="$2"
      shift
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

cleanup() {
  local status=$?
  if [[ "$KEEP" -eq 1 ]]; then
    local pid
    for pid in "${CHAT_PIDS[@]+"${CHAT_PIDS[@]}"}"; do
      ok "chat vLLM left running (pid $pid)"
    done
    [[ -n "$EMBED_PID" ]] && ok "embed vLLM left running (pid $EMBED_PID)"
  else
    stage "Stopping vLLM"
    stop_pid "$EMBED_PID" "embed vLLM"
    local pid
    for pid in "${CHAT_PIDS[@]+"${CHAT_PIDS[@]}"}"; do
      stop_pid "$pid" "chat vLLM"
    done
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

export PYTHONUNBUFFERED=1
export SAFE_RAG_LIGHTRAG_CONFIG="$LIGHTRAG_CONFIG"

if [[ ! -f "$EXPERIMENT" ]]; then
  err "experiment config not found: $EXPERIMENT"
  exit 1
fi
if [[ ! -f "$LIGHTRAG_CONFIG" ]]; then
  err "LightRAG config not found: $LIGHTRAG_CONFIG"
  exit 1
fi

SYSTEM="$(yaml_get "$EXPERIMENT" system)"
DATASET="$(yaml_get "$EXPERIMENT" dataset)"
ATTACK="$(yaml_get "$EXPERIMENT" attack)"
DEFENSE="$(yaml_get "$EXPERIMENT" defense)"
TURNS="$(yaml_get "$EXPERIMENT" turns)"
RUN_NAME="${ATTACK}_${DEFENSE}_${SYSTEM}_${DATASET}"
RESULT_DIR="${RESULTS_ROOT:-results}/${RUN_NAME}"
EXPERIMENT_LOG_DIR="logs/${RUN_NAME}"
QUESTIONS="$(yaml_get "$EXPERIMENT" utility.questions)"
QUESTIONS="${QUESTIONS:-data/qa/${DATASET}_questions.json}"

stage "Config"
ok "experiment: $EXPERIMENT"
ok "$ATTACK / $DEFENSE / $SYSTEM / $DATASET  questions=${TURNS:-200}"
ok "lightrag: $LIGHTRAG_CONFIG"
ok "gold QA: $QUESTIONS"

if [[ "$SYSTEM" != "lightrag" ]]; then
  err "this script starts LightRAG vLLM; experiment system is '$SYSTEM'"
  err "use: python -m safe_rag.eval.runner --config $EXPERIMENT"
  exit 1
fi
if [[ "$ATTACK" != "utility" ]]; then
  err "this script is for attack: utility; experiment attack is '$ATTACK'"
  err "use: bash scripts/run_agea_lightrag.sh --config $EXPERIMENT"
  exit 1
fi

WORKSPACE="data/lightrag/${DATASET}"
GRAPHML="$WORKSPACE/graph_chunk_entity_relation.graphml"
if [[ ! -f "$GRAPHML" ]]; then
  err "LightRAG workspace not found: $GRAPHML"
  err "build it first: bash scripts/build_lightrag.sh"
  exit 1
fi
ok "workspace: $WORKSPACE"

if [[ ! -f "$QUESTIONS" ]]; then
  err "gold QA file not found: $QUESTIONS"
  exit 1
fi

RUN_LOG="$EXPERIMENT_LOG_DIR/run.log"
mkdir -p "$EXPERIMENT_LOG_DIR"
export LOG_DIR="$EXPERIMENT_LOG_DIR"
ok "logs → $EXPERIMENT_LOG_DIR"

CHAT_NAME="$("$PYTHON" -m safe_rag.systems.lightrag.vllm_server name --role chat --config "$LIGHTRAG_CONFIG")"
CHAT_COUNT="$("$PYTHON" -m safe_rag.systems.lightrag.vllm_server replica-count --role chat --config "$LIGHTRAG_CONFIG")"
stage "vLLM chat ($CHAT_NAME x$CHAT_COUNT)"
for replica in $(seq 0 $((CHAT_COUNT - 1))); do
  if [[ "$CHAT_COUNT" -eq 1 ]]; then
    start_role chat "$EXPERIMENT_LOG_DIR/vllm-chat.log" "$replica"
  else
    start_role chat "$EXPERIMENT_LOG_DIR/vllm-chat-${replica}.log" "$replica"
  fi
done

EMBED_BACKEND="$("$PYTHON" -m safe_rag.systems.lightrag.vllm_server embedding-backend --config "$LIGHTRAG_CONFIG")"
if [[ "$EMBED_BACKEND" == "cpu" ]]; then
  stage "Embedding (CPU)"
  ok "skip embed vLLM; queries use $("$PYTHON" -m safe_rag.systems.lightrag.vllm_server name --role embed --config "$LIGHTRAG_CONFIG") on CPU"
else
  stage "vLLM embed ($("$PYTHON" -m safe_rag.systems.lightrag.vllm_server name --role embed --config "$LIGHTRAG_CONFIG"))"
  start_role embed "$EXPERIMENT_LOG_DIR/vllm-embed.log"
fi

stage "Utility QA"
ok "run log → $RUN_LOG"
{
  echo
  echo "===== $(date -Iseconds)  $EXPERIMENT ====="
} >"$RUN_LOG"
PY_ARGS=(-m safe_rag.eval.runner --config "$EXPERIMENT")
if [[ -n "$RESULTS_ROOT" ]]; then
  PY_ARGS+=(--results-root "$RESULTS_ROOT")
fi
"$PYTHON" "${PY_ARGS[@]}" | tee -a "$RUN_LOG"

stage "Done"
ok "run log → $RUN_LOG"
ok "turn logs → $EXPERIMENT_LOG_DIR/turn_logs"
ok "results → $RESULT_DIR"
ok "summary: $RESULT_DIR/run_summary.json"
ok "metrics: $RESULT_DIR/utility_metrics.json"
ok "history: $RESULT_DIR/query_history.json"
