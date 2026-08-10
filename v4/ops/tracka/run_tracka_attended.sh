#!/bin/zsh
# Track A capture -- ATTENDED runner. Start this in a Terminal (or the VS Code
# terminal) and leave the window open. It fires each declared window itself.
#
# WHY THIS EXISTS INSTEAD OF launchd
# ----------------------------------
# 2026-08-05: both launchd-fired captures failed and every shell-fired capture
# succeeded. The cause is macOS TCC, confirmed from the TCC database:
#
#   $ sqlite3 ~/Library/Application\ Support/com.apple.TCC/TCC.db \
#       'select service,client,auth_value from access
#          where service like "%DocumentsFolder%"'
#   ...|com.apple.Terminal|2          <- allowed
#   ...|com.microsoft.VSCode|2        <- allowed
#   ...|com.anthropic.claude-code|2   <- allowed
#   ...|/Users/gduby/.local/share/uv/python/cpython-3.12.13-.../python3.12|2
#
# That TCC.db output is quoted as observed on 2026-08-05 and is left unedited.
# Note as of 2026-08-06 the home path in the last line no longer exists, so that
# binary-path grant should be re-read before anyone relies on it. This attended
# runner does not depend on it -- it inherits the grant from Terminal or VS Code,
# which are granted as applications. Only the unattended launchd path needs it.
#
# ~/Documents is a TCC-protected location. The ONLY non-app binary holding a
# grant is that uv python3.12. /bin/zsh, /usr/bin/python3, head, dd, mkdir --
# every other binary the capture path uses -- hold no grant.
#
# A shell inherits the grant from the app that spawned it, so every interactive
# run works. A LaunchAgent has no such parent: TCC evaluates its own program,
# /bin/zsh, finds no entry, and refuses. zsh reports that refusal with the same
# words it uses for a missing file -- "can't open input file" -- which is why
# this first looked like iCloud eviction. It is not eviction. Proof that launchd
# itself is fine: com.autoresearch.protocol101.shadowasof.ledger runs a .py from
# inside ~/Documents every 60 s under launchd with exit code 0, because its
# program IS that granted uv python3.12.
#
# So: run the capture from a process tree that already holds the grant. That is
# this script. It needs no TCC change, no repo relocation, and no new launchd
# job -- three things that are all tier 1 and none of which can be validated
# before the 2026-08-06 open window.
#
# USAGE -- from the repository root, whatever its path:
#   ./v4/ops/tracka/run_tracka_attended.sh
#
# Leave the window open. Ctrl-C to disarm. Safe to start the evening before.
# The Mac must stay awake -- these windows do not survive system sleep.

# Correction of record 2026-08-05 evening: a stanza here claimed the owner
# canceled declaration v6 and refused to arm. The owner made no such
# cancellation (owner statement, 2026-08-05 evening conversation). Details:
# v4/docs/protocol101/training/research/HANDOFF_TRACKA_LAUNCHD_CANNOT_READ_REPO_2026_08_05.md
#
# Current declaration is v8: 2026-08-10, 08-11 and 08-12. v6's own sessions are
# spent -- 08-06 banked nothing and the owner canceled 08-07 for real on
# 2026-08-06, which is a genuine cancellation and not the retracted one above.

set -uo pipefail

# Derived from this script's own location -- it lives at REPO/v4/ops/tracka/.
REPO="${0:A:h:h:h:h}"
ROOT="$REPO/v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04"
DECL="$ROOT/capture_declaration_v9.json"
WRAPPER="$REPO/v4/ops/tracka/run_tracka_window.sh"
PY="$REPO/.venv/bin/python"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"; }

# How late a window may start and still be the window the declaration froze.
# The wait loop polls every second near the target, so an on-time fire is <2 s.
MAX_LATE_SECONDS=60

# --- gate 0: prove we hold the Documents grant BEFORE arming ------------------
# The whole point of this script is to run inside a granted process tree. If it
# was launched from something ungranted, it must say so now -- not at 06:28.
if ! /usr/bin/head -1 "$WRAPPER" >/dev/null 2>&1; then
    say "REFUSED: cannot read $WRAPPER"
    say "This shell does not hold Documents access. Start this from Terminal or"
    say "the VS Code terminal, not from a LaunchAgent, cron job, or ssh session."
    exit 77
fi
[ -x "$PY" ] || { say "REFUSED: no venv interpreter at $PY"; exit 1; }
say "Documents access OK (wrapper is readable from this shell)."

# --- read the frozen declaration; never hardcode the schedule -----------------
SCHEDULE="$("$PY" - "$DECL" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
cw = d["capture_window"]
for session in cw["sessions"]:
    for w in cw["windows"]:
        # start_local looks like "06:28:00 America/Los_Angeles"
        hhmmss, tz = w["start_local"].split(" ", 1)
        print(f"{session}\t{w['name']}\t{hhmmss}\t{tz}\t{int(w['duration_seconds'])}")
PYEOF
)"
[ -n "$SCHEDULE" ] || { say "REFUSED: could not read a schedule from $DECL"; exit 1; }

say "Declared windows (frozen in $(basename "$DECL")):"
echo "$SCHEDULE" | while IFS=$'\t' read -r s n t tz dur; do
    say "    $s  $n  $t $tz  (${dur}s)"
done

# --- arm ----------------------------------------------------------------------
echo "$SCHEDULE" | while IFS=$'\t' read -r session name hhmmss tz dur; do
    target_epoch="$("$PY" -c "
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
s, t, tz = sys.argv[1], sys.argv[2], sys.argv[3]
dt = datetime.strptime(s + ' ' + t, '%Y-%m-%d %H:%M:%S').replace(tzinfo=ZoneInfo(tz))
print(int(dt.timestamp()))
" "$session" "$hhmmss" "$tz")"

    now_epoch="$(date +%s)"
    wait_s=$(( target_epoch - now_epoch ))

    if [ "$wait_s" -lt -60 ]; then
        say "SKIP  $session $name -- start time already passed by $(( -wait_s ))s."
        continue
    fi

    say "ARMED $session $name -- firing in ${wait_s}s. Leave this window open."

    # Sleep in short chunks and re-check the wall clock. A single long sleep
    # silently overshoots if the machine suspends; this notices and still fires
    # (late, and it says so) rather than sleeping through the window.
    #
    # The loop also watches the power source. On 2026-08-06 the Mac was on AC at
    # launch and was moved to battery afterwards, where caffeinate -s is void, so
    # a start-time check alone would not have caught it. This cannot restore
    # power, but it timestamps the transition in the log so the cause is visible
    # immediately instead of being reconstructed from pmset days later.
    on_battery=0
    while :; do
        remaining=$(( target_epoch - $(date +%s) ))
        [ "$remaining" -le 0 ] && break
        # Matched in-shell rather than `pmset | grep -q`: under `set -o pipefail`
        # that pipeline can report failure via SIGPIPE even when grep matches,
        # which would log a false battery transition. See check_tracka.sh.
        if [[ "$(pmset -g batt 2>/dev/null)" == *"AC Power"* ]]; then
            if [ "$on_battery" -eq 1 ]; then
                say "POWER RESTORED -- back on AC."
                on_battery=0
            fi
        elif [ "$on_battery" -eq 0 ]; then
            say "POWER WARNING: now on BATTERY. caffeinate -s is void on battery and"
            say "               the Mac will sleep through this window. Plug it in."
            on_battery=1
        fi
        if [ "$remaining" -gt 300 ]; then
            sleep 60
        elif [ "$remaining" -gt 30 ]; then
            sleep 10
        else
            sleep 1
        fi
    done

    # A window that does not start when the frozen declaration says it starts is
    # NOT that window. On 2026-08-06 the open fired 1298 s late, which records
    # 09:49 ET while labelling the output "open" -- the 09:30 bell the envelope
    # law requires would have been absent from data presented as containing it.
    # Only the unrelated symbol-count guard stopped that from banking. Refuse.
    late=$(( $(date +%s) - target_epoch ))
    if [ "$late" -gt "$MAX_LATE_SECONDS" ]; then
        say "SKIP  $session $name -- ${late}s late (limit ${MAX_LATE_SECONDS}s)."
        say "      A late start is a different window than the one declared, so"
        say "      capturing it would bank mislabelled evidence. Nothing captured."
        continue
    fi
    if [ "$late" -gt 10 ]; then
        say "WARNING: firing ${late}s late -- the machine may have slept."
    fi

    say "FIRING $session $name"
    # The wrapper is the frozen, gated entry point. It re-checks the declared
    # session itself and refuses on any day that is not declared, so a stale
    # copy of this script cannot cause an out-of-declaration capture.
    # stdin comes from the schedule pipe; hand the wrapper /dev/null so nothing
    # downstream can swallow the remaining windows.
    /bin/zsh "$WRAPPER" "$name" < /dev/null
    rc=$?
    if [ "$rc" -eq 0 ]; then
        say "DONE  $session $name -- wrapper exited 0"
    else
        say "FAILED $session $name -- wrapper exited $rc"
        say "       see $ROOT/run_logs/${session}_${name}.log"
    fi
done

say "All declared windows have passed. Nothing further is armed."
