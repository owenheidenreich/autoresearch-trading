#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/gduby/Documents/autoresearch-trading}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_PYTHON_BIN="${AUTORESEARCH_RUNTIME_PYTHON_BIN:-$HOME/.autoresearch-trading/runtime-venv/bin/python}"
DEFAULT_PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$RUNTIME_PYTHON_BIN" ]]; then
    PYTHON_BIN="$RUNTIME_PYTHON_BIN"
  else
    PYTHON_BIN="$DEFAULT_PYTHON_BIN"
  fi
fi
if [[ -x "$RUNTIME_PYTHON_BIN" && "$PYTHON_BIN" == "/usr/bin/python3" ]]; then
  PYTHON_BIN="$RUNTIME_PYTHON_BIN"
elif [[ -x "$DEFAULT_PYTHON_BIN" && "$PYTHON_BIN" == "/usr/bin/python3" ]]; then
  PYTHON_BIN="$DEFAULT_PYTHON_BIN"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
WAIT_FOR_IBKR_API_SCRIPT="${WAIT_FOR_IBKR_API_SCRIPT:-$SCRIPT_DIR/wait_for_ibkr_api.py}"
WARM_LAUNCHD_PYTHON_DEPS_SCRIPT="${WARM_LAUNCHD_PYTHON_DEPS_SCRIPT:-$SCRIPT_DIR/warm_launchd_python_deps.py}"
IB_GATEWAY_API_PORT="${IB_GATEWAY_API_PORT:-4002}"
IB_GATEWAY_API_PORTS="${IB_GATEWAY_API_PORTS:-4002,4000,7497,7496,4001}"
IB_GATEWAY_PREFLIGHT_TIMEOUT_SECONDS="${IB_GATEWAY_PREFLIGHT_TIMEOUT_SECONDS:-600}"
PYTHON_IMPORT_LOCK="${PYTHON_IMPORT_LOCK:-$HOME/Library/Logs/autoresearch-trading/launchd-python-import.lock}"
AUTORESEARCH_TMP_DIR="${AUTORESEARCH_TMP_DIR:-$HOME/.autoresearch-trading/tmp}"
AUTORESEARCH_PYCACHE_DIR="${AUTORESEARCH_PYCACHE_DIR:-$HOME/.autoresearch-trading/pycache}"

if [[ ! -f "$WAIT_FOR_IBKR_API_SCRIPT" ]]; then
  WAIT_FOR_IBKR_API_SCRIPT="$REPO_ROOT/v4/ops/ibkr/wait_for_ibkr_api.py"
fi
if [[ ! -f "$WARM_LAUNCHD_PYTHON_DEPS_SCRIPT" ]]; then
  WARM_LAUNCHD_PYTHON_DEPS_SCRIPT="$REPO_ROOT/v4/ops/ibkr/warm_launchd_python_deps.py"
fi

export PYTHON_BIN
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPYCACHEPREFIX="$AUTORESEARCH_PYCACHE_DIR"
mkdir -p "$AUTORESEARCH_TMP_DIR" "$AUTORESEARCH_PYCACHE_DIR"
export TMPDIR="$AUTORESEARCH_TMP_DIR/"
case ":${PYTHONPATH:-}:" in
  *":$REPO_ROOT:"*) ;;
  *) export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
cd "$REPO_ROOT"
if command -v lockf >/dev/null 2>&1; then
  if ! lockf -t "${LAUNCHD_PYTHON_IMPORT_LOCK_TIMEOUT_SECONDS:-120}" "$PYTHON_IMPORT_LOCK" \
    "$PYTHON_BIN" "$WARM_LAUNCHD_PYTHON_DEPS_SCRIPT" \
      --module numpy \
      --module pandas \
      --module ib_insync \
      --module v4.scripts.check_ibkr_live_data_entitlements \
      --attempts "${LAUNCHD_PYTHON_IMPORT_ATTEMPTS:-5}" \
      --import-timeout-seconds "${LAUNCHD_PYTHON_IMPORT_TIMEOUT_SECONDS:-30}" \
      --sleep-seconds "${LAUNCHD_PYTHON_IMPORT_SLEEP_SECONDS:-2}" \
      >/tmp/autoresearch-launchd-python-deps.json; then
    echo "Launchd Python dependency warmup failed; continuing to IBKR API preflight." >&2
    cat /tmp/autoresearch-launchd-python-deps.json >&2 || true
  fi
else
  if ! "$PYTHON_BIN" "$WARM_LAUNCHD_PYTHON_DEPS_SCRIPT" \
    --module numpy \
    --module pandas \
    --module ib_insync \
    --module v4.scripts.check_ibkr_live_data_entitlements \
    --attempts "${LAUNCHD_PYTHON_IMPORT_ATTEMPTS:-5}" \
    --import-timeout-seconds "${LAUNCHD_PYTHON_IMPORT_TIMEOUT_SECONDS:-30}" \
    --sleep-seconds "${LAUNCHD_PYTHON_IMPORT_SLEEP_SECONDS:-2}" \
    >/tmp/autoresearch-launchd-python-deps.json; then
    echo "Launchd Python dependency warmup failed; continuing to IBKR API preflight." >&2
    cat /tmp/autoresearch-launchd-python-deps.json >&2 || true
  fi
fi
exec "$PYTHON_BIN" "$WAIT_FOR_IBKR_API_SCRIPT" \
  --port "$IB_GATEWAY_API_PORT" \
  --auto-ports "$IB_GATEWAY_API_PORTS" \
  --timeout-seconds "$IB_GATEWAY_PREFLIGHT_TIMEOUT_SECONDS" \
  --run-entitlement-probe \
  --repo-root "$REPO_ROOT"
