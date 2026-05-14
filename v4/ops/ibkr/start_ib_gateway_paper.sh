#!/usr/bin/env bash
set -euo pipefail

APP_PATH="${IB_GATEWAY_APP:-/Users/gduby/Applications/IB Gateway 10.45/IB Gateway 10.45.app}"
PORTS="${IB_GATEWAY_API_PORTS:-${IB_GATEWAY_API_PORT:-4000,4002,7497,7496,4001}}"
WAIT_SECONDS="${IB_GATEWAY_WAIT_SECONDS:-180}"

if [[ ! -d "$APP_PATH" ]]; then
  echo "IB Gateway app not found: $APP_PATH" >&2
  exit 2
fi

open -na "$APP_PATH"

deadline=$((SECONDS + WAIT_SECONDS))
while [[ "$SECONDS" -lt "$deadline" ]]; do
  IFS=',' read -ra CANDIDATE_PORTS <<< "$PORTS"
  for port in "${CANDIDATE_PORTS[@]}"; do
    port="${port//[[:space:]]/}"
    if [[ -n "$port" ]] && lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "IB Gateway API listener is up on port $port"
      exit 0
    fi
  done
  sleep 5
done

echo "IB Gateway started, but no API port from [$PORTS] was listening after ${WAIT_SECONDS}s." >&2
echo "If Gateway is waiting for credentials/2FA, finish the login once and rerun the preflight." >&2
exit 1
