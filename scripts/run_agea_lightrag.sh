#!/usr/bin/env bash
# AGEA on LightRAG: start chat + embedding vLLM, run the experiment, stop both.
# Run from WSL in the project venv:  bash scripts/run_agea_lightrag.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
EXPERIMENT="configs/experiments/agea_none_lightrag_medical.yaml"
LIGHTRAG_CONFIG="configs/lightrag/build.yaml"
RESULTS_ROOT=""
KEEP=0
CHAT_PID=""
EMBED_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_agea_lightrag.sh [options]

  (default)          Start vLLM if needed, run AGEA, stop vLLM.
  --keep-vllm        Leave both vLLM processes running after the attack.
  --config           Experiment YAML
                     (default: configs/experiments/agea_none_lightrag_medical.yaml)
  --lightrag-config  LightRAG / vLLM YAML (default: configs/lightrag/build.yaml)
  --results-root     Override results directory
  -h, --help         Show this help.

Attacker LLM: configs/agea/llm.yaml (or agea.llm_config in the experiment YAML).
Resume a partial run by setting agea.resume: true in the experiment YAML.

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
  local pid_var="$3"
  if "$PYTHON" -m safe_rag.systems.lightrag.vllm_server ready --role "$role" --config "$LIGHTRAG_CONFIG"; then
    ok "$role already running"
    return 0
  fi
  "$PYTHON" -m safe_rag.systems.lightrag.vllm_server prepare-socket --role "$role" --config "$LIGHTRAG_CONFIG"
  local devices
  devices="$("$PYTHON" -m safe_rag.systems.lightrag.vllm_server devices --role "$role" --config "$LIGHTRAG_CONFIG")"
  local cmd=()
  mapfile -t cmd < <("$PYTHON" -m safe_rag.systems.lightrag.vllm_server argv --role "$role" --config "$LIGHTRAG_CONFIG")
  ok "$role GPUs: $devices"
  ok "$role logs → $log"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    CUDA_VISIBLE_DEVICES="$devices" \
    "${cmd[@]}" >"$log" 2>&1 &
  local pid=$!
  printf -v "$pid_var" '%s' "$pid"
  if ! "$PYTHON" -m safe_rag.systems.lightrag.vllm_server wait --role "$role" --config "$LIGHTRAG_CONFIG" --pid "$pid"; then
    err "$role vLLM failed to start. Last log lines:"
    tail -n 40 "$log" >&2 || true
    return 1
  fi
  ok "$role ready (pid $pid)"
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
    [[ -n "$CHAT_PID" ]] && ok "chat vLLM left running (pid $CHAT_PID)"
    [[ -n "$EMBED_PID" ]] && ok "embed vLLM left running (pid $EMBED_PID)"
  else
    stage "Stopping vLLM"
    stop_pid "$EMBED_PID" "embed vLLM"
    stop_pid "$CHAT_PID" "chat vLLM"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

export PYTHONUNBUFFERED=1

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
LLM_CONFIG="$(yaml_get "$EXPERIMENT" agea.llm_config)"
LLM_CONFIG="${LLM_CONFIG:-configs/agea/llm.yaml}"
RUN_NAME="${ATTACK}_${DEFENSE}_${SYSTEM}_${DATASET}"
RESULT_DIR="${RESULTS_ROOT:-results}/${RUN_NAME}"
EXPERIMENT_LOG_DIR="logs/${RUN_NAME}"

stage "Config"
ok "experiment: $EXPERIMENT"
ok "$ATTACK / $DEFENSE / $SYSTEM / $DATASET  turns=${TURNS:-50}"
ok "lightrag: $LIGHTRAG_CONFIG"

if [[ "$SYSTEM" != "lightrag" ]]; then
  err "this script starts LightRAG vLLM; experiment system is '$SYSTEM'"
  err "use: python -m safe_rag.eval.runner --config $EXPERIMENT"
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

stage "Attacker LLM"
if [[ ! -f "$LLM_CONFIG" ]]; then
  err "AGEA LLM config not found: $LLM_CONFIG"
  err "copy configs/agea/llm.yaml.example to configs/agea/llm.yaml and fill in base_url + api_key"
  exit 1
fi
"$PYTHON" -c "
from safe_rag.attacks.agea.llm import load_agea_llm_settings
settings = load_agea_llm_settings('${LLM_CONFIG}')
print(settings.base_url)
print(settings.query_generator_model)
print(settings.graph_filter_model)
" | {
  IFS= read -r base_url
  IFS= read -r query_model
  IFS= read -r filter_model
  ok "config: $LLM_CONFIG"
  ok "base_url: $base_url"
  ok "query generator: $query_model"
  ok "graph filter: $filter_model"
}

CHAT_LOG="$EXPERIMENT_LOG_DIR/vllm-chat.log"
EMBED_LOG="$EXPERIMENT_LOG_DIR/vllm-embed.log"
ATTACK_LOG="$EXPERIMENT_LOG_DIR/run.log"
mkdir -p "$EXPERIMENT_LOG_DIR"
export LOG_DIR="$EXPERIMENT_LOG_DIR"
ok "logs → $EXPERIMENT_LOG_DIR"

stage "vLLM chat ($("$PYTHON" -m safe_rag.systems.lightrag.vllm_server name --role chat --config "$CONFIG"))"
start_role chat "$CHAT_LOG" CHAT_PID

stage "vLLM embed ($("$PYTHON" -m safe_rag.systems.lightrag.vllm_server name --role embed --config "$CONFIG"))"
start_role embed "$EMBED_LOG" EMBED_PID

stage "AGEA"
ok "attack log → $ATTACK_LOG"
{
  echo
  echo "===== $(date -Iseconds)  $EXPERIMENT ====="
} >"$ATTACK_LOG"
PY_ARGS=(-m safe_rag.eval.runner --config "$EXPERIMENT")
if [[ -n "$RESULTS_ROOT" ]]; then
  PY_ARGS+=(--results-root "$RESULTS_ROOT")
fi
# Keep stderr as a TTY so tqdm can redraw one bar in place.
# stdout (final metrics JSON, prints) still goes to the attack log.
"$PYTHON" "${PY_ARGS[@]}" | tee -a "$ATTACK_LOG"

stage "Done"
ok "attack log → $ATTACK_LOG"
ok "turn logs → $EXPERIMENT_LOG_DIR/turn_logs"
ok "results → $RESULT_DIR"
ok "summary: $RESULT_DIR/run_summary.json"
ok "history: $RESULT_DIR/query_history.json"
ok "graph:   $RESULT_DIR/extracted_graph.graphml"
