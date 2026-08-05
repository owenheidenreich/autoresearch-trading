#!/bin/zsh
# Track A capture wrapper -- one invocation per window, fired by launchd.
#
# Fail-closed. It refuses unless today is one of the three DECLARED sessions.
# launchd can only express weekdays, so a job left loaded WOULD fire on
# 2026-08-12 and every later Wednesday; this check is what keeps the frozen
# declaration honest. It is load-bearing, not decorative.
#
# Sequence per window: capture definitions (30 s) -> capture market data.
# The recorder requires --definition-path, and definitions must be current-session.

# CANCELED_BY_OWNER_2026_08_05
# Declaration v6 is frozen history, not current permission to capture. Replacement
# sessions must use the Python-direct launcher with a new authorization and sealed
# declaration after the project-structure cleanup.
echo "REFUSED: Track-A declaration v6 was canceled by the owner; no live capture is authorized." >&2
exit 78

set -euo pipefail

REPO="/Users/gduby/Documents/autoresearch-trading"
WINDOW="${1:?usage: run_tracka_window.sh <open|midday>}"
ROOT="$REPO/v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04"
# v6 narrows the certified sample to 2026-08-06 and 08-07 (owner, 2026-08-05).
# 08-05 is deliberately absent: its open window was lost to iCloud eviction and
# its midday window was an infrastructure verification run, not evidence.
DECL="$ROOT/capture_declaration_v6.json"
APPROVAL="$ROOT/authorization.json"
SESSION="$(date +%Y-%m-%d)"
LOGDIR="$ROOT/run_logs"
mkdir -p "$LOGDIR"
exec >> "$LOGDIR/${SESSION}_${WINDOW}.log" 2>&1

echo "=== Track A ${WINDOW} @ $(date -u +%Y-%m-%dT%H:%M:%SZ) (session ${SESSION}) ==="
cd "$REPO"

# --- gate 1: declared session -------------------------------------------------
if ! /usr/bin/python3 -c "
import json,sys
d=json.load(open('$DECL'))
sys.exit(0 if '$SESSION' in d['capture_window']['sessions'] else 1)
"; then
    echo "REFUSED: ${SESSION} is not a declared session. Sessions are frozen and may not be substituted."
    exit 78
fi

# --- gate 2: duration for this window, read from the frozen declaration -------
DURATION="$(/usr/bin/python3 -c "
import json
d=json.load(open('$DECL'))
w=[x for x in d['capture_window']['windows'] if x['name']=='$WINDOW']
print(w[0]['duration_seconds'] if w else '')
")"
[ -n "$DURATION" ] || { echo "REFUSED: window '$WINDOW' is not in the declaration"; exit 78; }
echo "declared duration: ${DURATION}s"

# --- approval text comes from the manifest, which records the owner's words ---
V4_PAID_DATA_APPROVAL_TEXT="$(/usr/bin/python3 -c "
import json; print(json.load(open('$APPROVAL'))['approval_required']['exact_approval_text'])
")"
export V4_PAID_DATA_APPROVAL_TEXT

OUT="$ROOT/${SESSION}/${WINDOW}"
DEFDIR="$OUT/definitions"
mkdir -p "$(dirname "$OUT")"

# --- definitions first: the recorder needs a current-session universe ---------
echo "--- definition capture (30s) ---"
PYTHONPATH=. ./.venv/bin/python -m v4.scripts.capture_databento_live_opra_definitions \
    --session-date "$SESSION" \
    --duration-seconds 30 \
    --output-dir "$DEFDIR" \
    --env-file v4/.env \
    --approval-manifest "$APPROVAL"

DEFPATH="$DEFDIR/opra_live_definitions.dbn.zst"
[ -f "$DEFPATH" ] || { echo "FAILED: no definition payload at $DEFPATH"; exit 1; }

# --- market capture -----------------------------------------------------------
echo "--- market capture (${DURATION}s) ---"
PYTHONPATH=. ./.venv/bin/python -m v4.scripts.capture_databento_live_opra_training_twin \
    --session-date "$SESSION" \
    --definition-path "$DEFPATH" \
    --duration-seconds "$DURATION" \
    --expected-symbol-count 510 \
    --schemas cbbo-1s cbbo-1m ohlcv-1m trades \
    --output-dir "$OUT/market" \
    --env-file v4/.env \
    --approval-manifest "$APPROVAL"

echo "=== ${WINDOW} complete @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
