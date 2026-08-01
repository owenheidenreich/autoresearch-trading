#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/gduby/Documents/autoresearch-trading}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Backward-compatible wrapper. The launchd label is historical, but the feature
# is now the generic daily paper autopilot and resolves its model from
# v4/promotion/PAPER_TRADING_DEFAULT.json at run time.
if [[ -x "$SCRIPT_DIR/run_daily_paper_autopilot.sh" ]]; then
  exec /bin/bash "$SCRIPT_DIR/run_daily_paper_autopilot.sh" "$@"
fi
exec /bin/bash "$REPO_ROOT/v4/ops/ibkr/run_daily_paper_autopilot.sh" "$@"
