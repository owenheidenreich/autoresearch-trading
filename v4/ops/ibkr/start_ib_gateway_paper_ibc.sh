#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/gduby/Documents/autoresearch-trading}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
if [[ -x "$DEFAULT_PYTHON_BIN" && "$PYTHON_BIN" == "/usr/bin/python3" ]]; then
  PYTHON_BIN="$DEFAULT_PYTHON_BIN"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
INSTALL_IBC_SCRIPT="${INSTALL_IBC_SCRIPT:-$SCRIPT_DIR/install_ibc_macos.sh}"
WRITE_IBC_CONFIG_SCRIPT="${WRITE_IBC_CONFIG_SCRIPT:-$SCRIPT_DIR/write_ibc_runtime_config.py}"
PROBE_IBKR_API_SCRIPT="${PROBE_IBKR_API_SCRIPT:-$SCRIPT_DIR/probe_ibkr_api.py}"
WARM_LAUNCHD_PYTHON_DEPS_SCRIPT="${WARM_LAUNCHD_PYTHON_DEPS_SCRIPT:-$SCRIPT_DIR/warm_launchd_python_deps.py}"
IBC_VERSION="${IBC_VERSION:-3.23.0}"
IBC_PATH="${IBC_PATH:-$HOME/.autoresearch-trading/ibc/${IBC_VERSION}}"
IBC_INI="${IBC_INI:-$HOME/.autoresearch-trading/ibc/runtime/ibc-paper.ini}"
TWS_MAJOR_VRSN="${TWS_MAJOR_VRSN:-10.45}"
TWS_PATH="${TWS_PATH:-$HOME/Applications}"
IB_GATEWAY_API_PORTS="${IB_GATEWAY_API_PORTS:-4002,4000,7497,7496,4001}"
IB_GATEWAY_API_PORT="${IB_GATEWAY_API_PORT:-4002}"
IB_GATEWAY_WAIT_SECONDS="${IB_GATEWAY_WAIT_SECONDS:-240}"
IB_GATEWAY_STABLE_SECONDS="${IB_GATEWAY_STABLE_SECONDS:-10}"
IB_GATEWAY_KEEPALIVE_SECONDS="${IB_GATEWAY_KEEPALIVE_SECONDS:-0}"
IB_GATEWAY_PRELAUNCH_CLEANUP="${IB_GATEWAY_PRELAUNCH_CLEANUP:-YES}"
LOG_PATH="${LOG_PATH:-$HOME/Library/Logs/autoresearch-trading/ibc}"
PYTHON_IMPORT_LOCK="${PYTHON_IMPORT_LOCK:-$HOME/Library/Logs/autoresearch-trading/launchd-python-import.lock}"
AUTORESEARCH_TMP_DIR="${AUTORESEARCH_TMP_DIR:-$HOME/.autoresearch-trading/tmp}"

export PYTHON_BIN
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPYCACHEPREFIX="${AUTORESEARCH_PYCACHE_DIR:-$HOME/.autoresearch-trading/pycache}"
mkdir -p "$PYTHONPYCACHEPREFIX"
mkdir -p "$LOG_PATH" "$AUTORESEARCH_TMP_DIR"
export TMPDIR="$AUTORESEARCH_TMP_DIR/"
cd "$AUTORESEARCH_TMP_DIR"

terminate_existing_ibkr_stack() {
  if [[ "$IB_GATEWAY_PRELAUNCH_CLEANUP" != "YES" ]]; then
    return 0
  fi
  if pgrep -f "java.*IBC|ibcstart\\.sh|IB Gateway|ibgateway|twslaunch|jts.*ibgateway" >/dev/null 2>&1; then
    echo "Pre-launch cleanup: stopping existing IBKR Gateway/TWS/IBC processes before controlled paper startup." >&2
    pkill -TERM -f "java.*IBC|ibcstart\\.sh|IB Gateway|ibgateway|twslaunch|jts.*ibgateway" 2>/dev/null || true
    sleep 5
  fi
  if pgrep -f "java.*IBC|ibcstart\\.sh|IB Gateway|ibgateway|twslaunch|jts.*ibgateway" >/dev/null 2>&1; then
    echo "Pre-launch cleanup: force-stopping remaining IBKR Gateway/TWS/IBC processes." >&2
    pkill -KILL -f "java.*IBC|ibcstart\\.sh|IB Gateway|ibgateway|twslaunch|jts.*ibgateway" 2>/dev/null || true
    sleep 2
  fi
}

terminate_existing_ibkr_stack

if [[ ! -x "$IBC_PATH/scripts/ibcstart.sh" ]]; then
  if [[ ! -x "$INSTALL_IBC_SCRIPT" ]]; then
    INSTALL_IBC_SCRIPT="$REPO_ROOT/v4/ops/ibkr/install_ibc_macos.sh"
  fi
  "$INSTALL_IBC_SCRIPT" >/dev/null
fi

if [[ ! -f "$WRITE_IBC_CONFIG_SCRIPT" ]]; then
  WRITE_IBC_CONFIG_SCRIPT="$REPO_ROOT/v4/ops/ibkr/write_ibc_runtime_config.py"
fi

if ! "$PYTHON_BIN" "$WRITE_IBC_CONFIG_SCRIPT" \
  --out "$IBC_INI" \
  --api-port "$IB_GATEWAY_API_PORT" >/tmp/autoresearch-ibc-config.json; then
  echo "IBC runtime config was not created. Run v4/ops/ibkr/store_ibkr_paper_credentials.sh locally." >&2
  cat /tmp/autoresearch-ibc-config.json >&2 || true
  exit 2
fi

if [[ ! -f "$IBC_INI" ]]; then
  echo "IBC runtime config was not created. Run v4/ops/ibkr/store_ibkr_paper_credentials.sh locally." >&2
  cat /tmp/autoresearch-ibc-config.json >&2 || true
  exit 2
fi

if pgrep -f "java.*IBC.*${IBC_INI}" >/dev/null 2>&1; then
  echo "IBC already appears to be running for $IBC_INI"
else
  nohup "$IBC_PATH/scripts/ibcstart.sh" "$TWS_MAJOR_VRSN" \
    --gateway \
    --tws-path="$TWS_PATH" \
    --ibc-path="$IBC_PATH" \
    --ibc-ini="$IBC_INI" \
    --mode=paper \
    --on2fatimeout=exit \
    >"$LOG_PATH/ibc-gateway.out.log" \
    2>"$LOG_PATH/ibc-gateway.err.log" &
fi

if [[ ! -f "$PROBE_IBKR_API_SCRIPT" ]]; then
  PROBE_IBKR_API_SCRIPT="$REPO_ROOT/v4/ops/ibkr/probe_ibkr_api.py"
fi
if [[ ! -f "$WARM_LAUNCHD_PYTHON_DEPS_SCRIPT" ]]; then
  WARM_LAUNCHD_PYTHON_DEPS_SCRIPT="$REPO_ROOT/v4/ops/ibkr/warm_launchd_python_deps.py"
fi

if command -v lockf >/dev/null 2>&1; then
  if ! lockf -t "${LAUNCHD_PYTHON_IMPORT_LOCK_TIMEOUT_SECONDS:-120}" "$PYTHON_IMPORT_LOCK" \
    "$PYTHON_BIN" "$WARM_LAUNCHD_PYTHON_DEPS_SCRIPT" \
      --module numpy \
      --module ib_insync \
      --attempts "${LAUNCHD_PYTHON_IMPORT_ATTEMPTS:-5}" \
      --import-timeout-seconds "${LAUNCHD_PYTHON_IMPORT_TIMEOUT_SECONDS:-30}" \
      --sleep-seconds "${LAUNCHD_PYTHON_IMPORT_SLEEP_SECONDS:-2}" \
      >/tmp/autoresearch-launchd-python-deps.json; then
    echo "Launchd Python dependency warmup failed; continuing to IBKR API probe." >&2
    cat /tmp/autoresearch-launchd-python-deps.json >&2 || true
  fi
else
  if ! "$PYTHON_BIN" "$WARM_LAUNCHD_PYTHON_DEPS_SCRIPT" \
    --module numpy \
    --module ib_insync \
    --attempts "${LAUNCHD_PYTHON_IMPORT_ATTEMPTS:-5}" \
    --import-timeout-seconds "${LAUNCHD_PYTHON_IMPORT_TIMEOUT_SECONDS:-30}" \
    --sleep-seconds "${LAUNCHD_PYTHON_IMPORT_SLEEP_SECONDS:-2}" \
    >/tmp/autoresearch-launchd-python-deps.json; then
    echo "Launchd Python dependency warmup failed; continuing to IBKR API probe." >&2
    cat /tmp/autoresearch-launchd-python-deps.json >&2 || true
  fi
fi

if "$PYTHON_BIN" "$PROBE_IBKR_API_SCRIPT" \
  --ports "$IB_GATEWAY_API_PORTS" \
  --timeout-seconds "$IB_GATEWAY_WAIT_SECONDS" \
  --stable-seconds "$IB_GATEWAY_STABLE_SECONDS" \
  --hold-seconds "$IB_GATEWAY_KEEPALIVE_SECONDS" \
  --client-id 145 >/tmp/autoresearch-ibkr-api-probe.json; then
  cat /tmp/autoresearch-ibkr-api-probe.json
  exit 0
fi

echo "IBC started or attempted to start IB Gateway, but no stable IBKR API handshake was available after ${IB_GATEWAY_WAIT_SECONDS}s." >&2
echo "If IBKR Mobile/2FA is required, approve it; if credentials are missing, run store_ibkr_paper_credentials.sh." >&2
cat /tmp/autoresearch-ibkr-api-probe.json >&2 || true
exit 1
