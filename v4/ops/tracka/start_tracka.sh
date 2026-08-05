#!/bin/zsh
# Track A -- one-command out-of-town start.
#
# Starts the attended runner in the background with:
#   caffeinate -is : holds a no-sleep assertion until the runner exits after
#                    the LAST declared window (Fri 2026-08-07 ~09:13 PDT),
#                    then releases so the Mac may sleep normally again
#   nohup + disown : keeps running if this terminal window is closed
#   console log    : run_logs/attended_console_<stamp>.log
#
# Start it from Terminal or the VS Code terminal -- a shell that holds the
# macOS Documents grant. It refuses anywhere else, like the runner itself.
#
# HARD REQUIREMENTS while it runs: Mac plugged into power, LID OPEN, no
# reboot, no log-out, no OS update. Locking the screen is fine. The display
# going dark is fine.

set -uo pipefail

REPO="/Users/gduby/Documents/autoresearch-trading"
ROOT="$REPO/v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04"
RUNNER="$REPO/v4/ops/tracka/run_tracka_attended.sh"
LOGDIR="$ROOT/run_logs"
LOG="$LOGDIR/attended_console_$(date +%Y%m%d_%H%M%S).log"

say() { echo "[start_tracka] $*"; }

# Same grant gate as the runner: fail here, loudly, not at 06:28.
if ! /usr/bin/head -1 "$RUNNER" >/dev/null 2>&1; then
    say "REFUSED: this shell cannot read the repo. Start from Terminal or the"
    say "VS Code terminal, not from a LaunchAgent, cron job, or ssh session."
    exit 77
fi

if pgrep -f "run_tracka_attended.sh" >/dev/null 2>&1; then
    say "REFUSED: an attended runner is already running:"
    pgrep -fl "run_tracka_attended.sh"
    say "Stop it first with: pkill -f run_tracka_attended.sh"
    exit 1
fi

if ! pmset -g batt | grep -q "AC Power"; then
    say "WARNING: not on AC power. Plug the Mac in NOW -- the schedule spans"
    say "two days and cannot complete on battery."
fi

mkdir -p "$LOGDIR"
nohup /usr/bin/caffeinate -is "$RUNNER" >> "$LOG" 2>&1 &
PID=$!
disown

sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
    say "FAILED: the runner exited immediately. Log follows:"
    cat "$LOG"
    exit 1
fi

say "RUNNING. Runner pid $PID; its caffeinate child holds the Mac awake until all windows finish."
say "Console log: $LOG"
say "Watch it:    tail -f $LOG"
say "Stop it:     pkill -f run_tracka_attended.sh"
say "It is now safe to close this terminal window and lock the screen."
say "Leave the Mac plugged in with the lid open."
echo
tail -n +1 "$LOG"
