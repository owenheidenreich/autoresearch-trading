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

# Correction of record 2026-08-05 evening: a stanza here claimed the owner
# canceled declaration v6 and refused all capture. The owner made no such
# cancellation (owner statement, 2026-08-05 evening conversation). v6 stands
# authorized for 2026-08-06 and 2026-08-07. Details:
# v4/docs/protocol101/training/research/HANDOFF_TRACKA_LAUNCHD_CANNOT_READ_REPO_2026_08_05.md

set -euo pipefail

# Derived from this script's own location -- it lives at REPO/v4/ops/tracka/.
REPO="${0:A:h:h:h:h}"
WINDOW="${1:?usage: run_tracka_window.sh <open|midday>}"
ROOT="$REPO/v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04"
# v8 declares 2026-08-10, 08-11 and 08-12 -- v7's four sessions narrowed to
# three by owner instruction 2026-08-06, before any of those sessions was
# observed. v7 replaced v6 after 08-06 banked nothing and 08-07 was canceled.
# 08-05 remains absent: its open window was lost to the launchd TCC permission
# refusal (the eviction diagnosis was retracted; see the handoff) and its midday
# window was an infrastructure verification run, not evidence.
DECL="$ROOT/capture_declaration_v8.json"
APPROVAL="$ROOT/authorization_v2.json"
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
# No --expected-symbol-count. The 0DTE strike listing changes daily -- it was
# 510 on 2026-08-05 and 574 on 2026-08-06, and pinning 510 is exactly what made
# the 08-06 capture fail closed. The recorder's default is None, which skips the
# equality check while still recording plan.symbol_count and plan.symbols_sha256,
# so the universe actually taken stays auditable. Never trim it.
PYTHONPATH=. ./.venv/bin/python -m v4.scripts.capture_databento_live_opra_training_twin \
    --session-date "$SESSION" \
    --definition-path "$DEFPATH" \
    --duration-seconds "$DURATION" \
    --schemas cbbo-1s cbbo-1m ohlcv-1m trades \
    --output-dir "$OUT/market" \
    --env-file v4/.env \
    --approval-manifest "$APPROVAL"

echo "=== ${WINDOW} complete @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
