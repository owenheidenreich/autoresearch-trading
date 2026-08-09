#!/bin/zsh
# Track A -- read-only status check. Safe to run any time, as often as you like.
# It starts nothing, stops nothing, and writes nothing.
#
#   ./v4/ops/tracka/check_tracka.sh
#
# Answers the four questions that matter during a multi-day capture:
#   1. Is the runner still alive?
#   2. Is the Mac still on AC power (caffeinate -s is void on battery)?
#   3. What has actually banked so far?
#   4. What is armed next, and what did the log last say?

set -uo pipefail

# Derived from this script's own location -- it lives at REPO/v4/ops/tracka/.
REPO="${0:A:h:h:h:h}"
ROOT="$REPO/v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04"
DECL="$ROOT/capture_declaration_v8.json"
PY="$REPO/.venv/bin/python"

echo "=== Track A status @ $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo

# --- 1. runner ----------------------------------------------------------------
if pgrep -f "run_tracka_attended.sh" >/dev/null 2>&1; then
    echo "RUNNER    ALIVE"
    pgrep -fl "run_tracka_attended.sh" | sed 's/^/          /'
else
    echo "RUNNER    NOT RUNNING"
    echo "          If windows remain, start it with:"
    echo "            $REPO/v4/ops/tracka/start_tracka.sh"
    echo "          Restarting is safe: windows already past are skipped."
fi
echo

# --- 2. power -----------------------------------------------------------------
# Read pmset into a variable and match in the shell rather than piping into
# `grep -q`. Under `set -o pipefail` that pipeline reports FAILURE even when
# grep matches: grep -q exits at the first hit, pmset dies with SIGPIPE (141),
# and pipefail promotes 141 to the pipeline status. With the long `assertions`
# output that was deterministic -- measured 20 failures in 20 runs on
# 2026-08-09 -- so this check reported "no assertion held" while caffeinate was
# demonstrably holding PreventSystemSleep. The battery checks used the same
# idiom and happened to survive (0 in 200) only because their output is small
# enough that pmset finishes writing before grep exits.
BATT="$(pmset -g batt 2>/dev/null)"
ASSERTIONS="$(pmset -g assertions 2>/dev/null)"
if [[ "$BATT" == *"AC Power"* ]]; then
    echo "POWER     AC - good"
else
    echo "POWER     *** ON BATTERY *** caffeinate -s is void; the Mac will sleep"
    echo "          through its window. Plug it in now."
fi
echo "$BATT" | tail -1 | sed 's/^/          /'
if [[ "$ASSERTIONS" =~ 'PreventSystemSleep[[:space:]]+1' ]]; then
    echo "SLEEP     held off (PreventSystemSleep active)"
else
    echo "SLEEP     *** no PreventSystemSleep assertion held ***"
fi
echo

# --- 3. what has banked -------------------------------------------------------
echo "BANKED    (a window counts only when its market capture_summary.json exists)"
"$PY" - "$DECL" "$ROOT" <<'PYEOF'
import json, sys
from pathlib import Path

declaration_path, root = Path(sys.argv[1]), Path(sys.argv[2])
declaration = json.loads(declaration_path.read_text())
window = declaration["capture_window"]
banked = 0
for session in window["sessions"]:
    for spec in window["windows"]:
        name = spec["name"]
        summary = root / session / name / "market" / "capture_summary.json"
        if summary.is_file():
            try:
                payload = json.loads(summary.read_text())
                status = payload.get("status", "?")
                rows = payload.get("records_total", "?")
                symbols = payload.get("plan", {}).get("symbol_count", "?")
                print(f"          {session} {name:7s} {status} rows={rows} symbols={symbols}")
                banked += 1
            except (OSError, json.JSONDecodeError):
                print(f"          {session} {name:7s} SUMMARY UNREADABLE")
        else:
            print(f"          {session} {name:7s} -")
print(f"          {banked} of {len(window['sessions']) * len(window['windows'])} windows banked")
PYEOF
echo

# --- 4. log tail --------------------------------------------------------------
LATEST="$(ls -t "$ROOT"/run_logs/attended_console_*.log 2>/dev/null | head -1)"
if [ -n "$LATEST" ]; then
    echo "LOG       $LATEST"
    tail -6 "$LATEST" | sed 's/^/          /'
else
    echo "LOG       no attended console log yet"
fi
