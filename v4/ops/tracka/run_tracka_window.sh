#!/bin/zsh
# Track A capture wrapper — fires from launchd, one invocation per window.
#
# Fail-closed by design. It refuses to run unless:
#   1. today is one of the DECLARED sessions (never substitute a session),
#   2. the declaration's authorization gate has been opened by the owner,
#   3. the recorder's frozen hash still matches the declaration.
#
# A weekday-based launchd trigger would also fire on 2026-08-12 and beyond; the
# session check below is what keeps the frozen declaration honest.

set -euo pipefail

REPO="/Users/gduby/Documents/autoresearch-trading"
WINDOW="${1:?usage: run_tracka_window.sh <open|midday>}"
DECL="$REPO/v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04/capture_declaration_v4.json"
LOGDIR="$REPO/v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04/run_logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y-%m-%dT%H%M%S)"
LOG="$LOGDIR/${STAMP}_${WINDOW}.log"

exec >> "$LOG" 2>&1
echo "=== Track A ${WINDOW} window @ ${STAMP} ==="

cd "$REPO"

# Preflight: declared session, authorization gate, frozen hashes. Any failure
# exits non-zero WITHOUT connecting, and leaves the reason in the log.
PYTHONPATH=. ./.venv/bin/python -m v4.research.pathd_phase0b_tracka_preflight \
    --declaration "$DECL" --window "$WINDOW"

echo "preflight passed; starting capture"
PYTHONPATH=. ./.venv/bin/python -m v4.scripts.capture_databento_live_opra_training_twin \
    --declaration "$DECL" --window "$WINDOW"

echo "=== window complete ==="
