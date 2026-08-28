#!/usr/bin/env bash
# LightRAG build: start chat + embedding vLLM, insert corpus, stop both.
# Run from WSL in the project venv:  bash scripts/build_lightrag.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
CONFIG="configs/lightrag/build.yaml"
REBUILD=0
RESUME=0
KEEP=0
CHAT_PID=""
EMBED_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/build_lightrag.sh [options]

  (default)   Resume an existing workspace, or create one if missing.
  --rebuild   Delete data/lightrag/<dataset>/ and build from scratch.
  --resume    Keep the workspace; retry failed documents only.
  --keep-vllm Leave both vLLM processes running after the build.
  --config    Path to configs/lightrag/build.yaml
  -h, --help  Show this help.

Logs (see log_dir in the YAML, default logs/lightrag/; overwritten each run):
  logs/lightrag/vllm-chat.log
  logs/lightrag/vllm-embed.log
  logs/lightrag/build.log
EOF
}

stage() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok() { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

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
  if "$PYTHON" -m safe_rag.systems.lightrag.vllm_server ready --role "$role" --config "$CONFIG"; then
    ok "$role already running"
    return 0
  fi
  "$PYTHON" -m safe_rag.systems.lightrag.vllm_server prepare-socket --role "$role" --config "$CONFIG"
  local devices
  devices="$("$PYTHON" -m safe_rag.systems.lightrag.vllm_server devices --role "$role" --config "$CONFIG")"
  local cmd=()
  mapfile -t cmd < <("$PYTHON" -m safe_rag.systems.lightrag.vllm_server argv --role "$role" --config "$CONFIG")
  ok "$role GPUs: $devices"
  ok "$role logs → $log"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    CUDA_VISIBLE_DEVICES="$devices" \
    "${cmd[@]}" >"$log" 2>&1 &
  local pid=$!
  printf -v "$pid_var" '%s' "$pid"
  if ! "$PYTHON" -m safe_rag.systems.lightrag.vllm_server wait --role "$role" --config "$CONFIG" --pid "$pid"; then
    err "$role vLLM failed to start. Last log lines:"
    tail -n 40 "$log" >&2 || true
    return 1
  fi
  ok "$role ready (pid $pid)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild) REBUILD=1 ;;
    --resume) RESUME=1 ;;
    --keep-vllm) KEEP=1 ;;
    --config)
      CONFIG="$2"
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

if [[ "$REBUILD" -eq 1 && "$RESUME" -eq 1 ]]; then
  err "use only one of --rebuild or --resume"
  exit 1
fi

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

stage "Config"
ok "$CONFIG"

LOG_DIR="$("$PYTHON" -m safe_rag.systems.lightrag.vllm_server log-dir --config "$CONFIG")"
CHAT_LOG="$("$PYTHON" -m safe_rag.systems.lightrag.vllm_server log-file --role chat --config "$CONFIG")"
EMBED_LOG="$("$PYTHON" -m safe_rag.systems.lightrag.vllm_server log-file --role embed --config "$CONFIG")"
mkdir -p "$LOG_DIR"
export LOG_DIR="$LOG_DIR"
ok "logs → $LOG_DIR"

stage "vLLM chat ($("$PYTHON" -m safe_rag.systems.lightrag.vllm_server name --role chat --config "$CONFIG"))"
start_role chat "$CHAT_LOG" CHAT_PID

stage "vLLM embed ($("$PYTHON" -m safe_rag.systems.lightrag.vllm_server name --role embed --config "$CONFIG"))"
start_role embed "$EMBED_LOG" EMBED_PID

stage "Knowledge graph"
PY_ARGS=(-m safe_rag.systems.lightrag.build --config "$CONFIG")
if [[ "$REBUILD" -eq 1 ]]; then
  PY_ARGS+=(--rebuild)
elif [[ "$RESUME" -eq 1 ]]; then
  PY_ARGS+=(--resume)
fi
"$PYTHON" "${PY_ARGS[@]}"

stage "Done"
ok "workspace under data/lightrag/"
ok "graph: data/lightrag/*/graph_chunk_entity_relation.graphml"
