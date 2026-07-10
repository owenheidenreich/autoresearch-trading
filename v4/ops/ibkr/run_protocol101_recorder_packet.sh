#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
if [[ -z "$action" ]]; then
  echo "missing recorder packet action" >&2
  exit 64
fi

BUNDLE_ROOT="${BUNDLE_ROOT:?BUNDLE_ROOT is required}"
CAPTURE_ROOT="${CAPTURE_ROOT:-$HOME/.autoresearch-trading/live_runtime/ibkr_capture}"
MODEL_PYTHON="${MODEL_PYTHON:-$HOME/.autoresearch-trading/runtime-venv/bin/python}"
RECORDER_PYTHON="${RECORDER_PYTHON:-/usr/bin/python3}"
SESSION="${PROTOCOL101_PACKET_SESSION:-$(TZ=America/Los_Angeles date '+%Y-%m-%d')}"
DEVELOPMENT_SESSION="${PROTOCOL101_PACKET_DEVELOPMENT_SESSION:-${PROTOCOL101_PACKET_MONDAY_SESSION:-2026-07-06}}"
ALLOWED_SESSIONS="${PROTOCOL101_PACKET_ALLOWED_SESSIONS:-2026-07-06,2026-07-07,2026-07-08,2026-07-09,2026-07-10}"
GATE_MODE="${PROTOCOL101_PACKET_GATE_MODE:-development_session}"
CAPTURE_ID="protocol101-recorder-$SESSION"
CAPTURE_DIR="$CAPTURE_ROOT/$SESSION/$CAPTURE_ID"
GATE_PATH="$CAPTURE_ROOT/collection_gate_$DEVELOPMENT_SESSION.json"
LOG_DIR="${LOG_DIR:-$HOME/Library/Logs/autoresearch-trading}"
LABEL_PREFIX="${PROTOCOL101_RECORDER_LABEL_PREFIX:-com.autoresearch.protocol101.parityrecorder}"
REGISTRY="$BUNDLE_ROOT/runtime/PAPER_TRADING_DEFAULT.json"

mkdir -p "$CAPTURE_DIR" "$LOG_DIR"
export PYTHONPATH="$BUNDLE_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

session_allowed() {
  local item
  IFS=',' read -r -a allowed <<<"$ALLOWED_SESSIONS"
  for item in "${allowed[@]}"; do
    if [[ "$SESSION" == "$item" ]]; then
      return 0
    fi
  done
  return 1
}

if ! session_allowed; then
  echo "session $SESSION is outside the recorder parity packet: $ALLOWED_SESSIONS" >&2
  exit 0
fi

gate_allows_session() {
  if [[ "$GATE_MODE" == "none" || "$GATE_MODE" == "disabled" ]]; then
    return 0
  fi
  if [[ "$SESSION" == "$DEVELOPMENT_SESSION" ]]; then
    return 0
  fi
  "$RECORDER_PYTHON" - "$GATE_PATH" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
required = ("capture_integrity", "opening_context", "trace_extraction", "same_input_replay")
raise SystemExit(0 if payload.get("schema_version") == "Protocol101CollectionGateV1" and all(payload.get(key) == "pass" for key in required) else 1)
PY
}

local_hhmm_decimal() {
  local raw
  raw="$(TZ=America/Los_Angeles date '+%H%M')"
  printf '%d\n' "$((10#$raw))"
}

if ! gate_allows_session; then
  echo "collection gate did not pass; refusing to start $SESSION" >&2
  exit 0
fi

case "$action" in
  gateway)
    exec "$BUNDLE_ROOT/v4/ops/ibkr/retry_command.sh" "$BUNDLE_ROOT/v4/ops/ibkr/start_protocol101_recorder_gateway.sh"
    ;;
  preflight)
    export RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-60}"
    export RETRY_MAX_SLEEP_SECONDS="${RETRY_MAX_SLEEP_SECONDS:-10}"
    exec "$BUNDLE_ROOT/v4/ops/ibkr/retry_command.sh" \
      "$RECORDER_PYTHON" -m v4.ops.ibkr.run_protocol101_ibkr_recorder \
      --session "$SESSION" --client-id 156 --capture-root "$CAPTURE_ROOT" \
      --preflight-out "$CAPTURE_DIR/preflight.json"
    ;;
  recorder)
    export RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-100}"
    export RETRY_INITIAL_SLEEP_SECONDS="${RETRY_INITIAL_SLEEP_SECONDS:-2}"
    export RETRY_MAX_SLEEP_SECONDS="${RETRY_MAX_SLEEP_SECONDS:-30}"
    export RETRY_LOG_PATH="$LOG_DIR/protocol101-parityrecorder-retries.log"
    exec "$BUNDLE_ROOT/v4/ops/ibkr/retry_command.sh" \
      "$RECORDER_PYTHON" -m v4.ops.ibkr.run_protocol101_ibkr_recorder \
      --session "$SESSION" --capture-id "$CAPTURE_ID" --capture-root "$CAPTURE_ROOT" \
      --client-id 159 --strikes-around-atm 10 --heartbeat-seconds 5 \
      --ladder-refresh-seconds 30 --stop-time-et 16:05
    ;;
  health)
    health_args=(health --session "$SESSION" --capture-id "$CAPTURE_ID" --capture-root "$CAPTURE_ROOT" --max-heartbeat-age-seconds 10 --port 4002)
    current_hm="$(local_hhmm_decimal)"
    if (( current_hm >= 615 )); then
      health_args+=(--require-live --require-ladder)
    fi
    exec "$RECORDER_PYTHON" -m v4.ops.ibkr.protocol101_recorder_control "${health_args[@]}"
    ;;
  watchdog)
    while (( $(local_hhmm_decimal) < 1305 )); do
      health_args=(health --session "$SESSION" --capture-id "$CAPTURE_ID" --capture-root "$CAPTURE_ROOT" --max-heartbeat-age-seconds 20 --port 4002)
      current_hm="$(local_hhmm_decimal)"
      if (( current_hm >= 615 )); then
        health_args+=(--require-live --require-ladder)
      fi
      if ! "$RECORDER_PYTHON" -m v4.ops.ibkr.protocol101_recorder_control "${health_args[@]}"; then
        printf '%s recorder evidence unhealthy; restarting %s.recorder\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$LABEL_PREFIX" >>"$LOG_DIR/protocol101-parityrecorder-watchdog.log"
        launchctl kickstart -k "gui/$UID/$LABEL_PREFIX.recorder" 2>>"$LOG_DIR/protocol101-parityrecorder-watchdog.log" || true
        sleep 30
      else
        sleep 60
      fi
    done
    ;;
  shutdown)
    launchctl kill TERM "gui/$UID/$LABEL_PREFIX.watchdog" 2>/dev/null || true
    launchctl kill TERM "gui/$UID/$LABEL_PREFIX.recorder" 2>/dev/null || true
    launchctl kill TERM "gui/$UID/$LABEL_PREFIX.gateway" 2>/dev/null || true
    pkill -TERM -f "v4.ops.ibkr.run_protocol101_ibkr_recorder.*--session $SESSION" 2>/dev/null || true
    pkill -TERM -f "protocol101-parity-v1/.*/start_protocol101_recorder_gateway\\.sh" 2>/dev/null || true
    sleep 5
    pkill -KILL -f "v4.ops.ibkr.run_protocol101_ibkr_recorder.*--session $SESSION" 2>/dev/null || true
    pkill -KILL -f "protocol101-parity-v1/.*/start_protocol101_recorder_gateway\\.sh" 2>/dev/null || true
    osascript -e 'tell application "IB Gateway" to quit' >/dev/null 2>&1 || true
    pkill -TERM -f "java.*IBC|ibcstart\\.sh|IB Gateway|ibgateway|twslaunch|jts.*ibgateway" 2>/dev/null || true
    sleep 5
    pkill -KILL -f "java.*IBC|ibcstart\\.sh|IB Gateway|ibgateway|twslaunch|jts.*ibgateway" 2>/dev/null || true
    ;;
  finalize)
    exec "$RECORDER_PYTHON" -m v4.ops.ibkr.protocol101_recorder_control finalize \
      --session "$SESSION" --capture-id "$CAPTURE_ID" --capture-root "$CAPTURE_ROOT"
    ;;
  audit)
    "$RECORDER_PYTHON" -m v4.ops.ibkr.protocol101_recorder_control audit \
      --session "$SESSION" --capture-id "$CAPTURE_ID" --capture-root "$CAPTURE_ROOT" --require-complete-session
    "$MODEL_PYTHON" - "$REGISTRY" "$BUNDLE_ROOT" "$SESSION" "$CAPTURE_ROOT" "$GATE_PATH" "$DEVELOPMENT_SESSION" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
registry_path, bundle_root, session, capture_root, gate_path, development_session = sys.argv[1:]
registry = json.loads(Path(registry_path).read_text())
args = registry["models"]["protocol101"]["arguments"]
cmd = [
    sys.executable, "-m", "v4.scripts.run_protocol101_capture_replay",
    "--session", session,
    "--capture-root", capture_root,
    "--surface-manifest", args["surface_manifest"],
    "--protocol101-manifest", args["protocol101_manifest"],
    "--protocol101-summary", args["protocol101_summary"],
    "--lifecycle-manifest", args["lifecycle_manifest"],
]
if session == development_session:
    cmd.extend(["--gate-path", gate_path])
raise SystemExit(subprocess.call(cmd, cwd=bundle_root))
PY
    ;;
  *)
    echo "unknown recorder packet action: $action" >&2
    exit 64
    ;;
esac
