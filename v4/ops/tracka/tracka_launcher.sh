#!/bin/zsh
# Track-A launchd entry point. Lives OUTSIDE ~/Documents on purpose.
#
# WHY THIS FILE EXISTS
# --------------------
# 2026-08-05 06:28 PDT the open window fired and died instantly with:
#     /bin/zsh: can't open input file: .../v4/ops/tracka/run_tracka_window.sh
# The file was present, executable, and parseable from an interactive shell.
#
# RETRACTED DIAGNOSIS (was: iCloud eviction). CORRECTED 2026-08-05.
# ----------------------------------------------------------------
# This file was written believing iCloud had evicted the wrapper to dataless.
# That was WRONG, and everything below it -- moving the entry point out of the
# synced tree, brctl download, dd, the retry loop -- treats a symptom that does
# not exist. The owner confirmed "Optimize Mac Storage" was already OFF, and the
# TCC database then settled it:
#
#   sqlite3 ~/Library/Application\ Support/com.apple.TCC/TCC.db \
#     'select client,auth_value from access
#        where service="kTCCServiceSystemPolicyDocumentsFolder"'
#
# ~/Documents is TCC-protected. Grants exist for com.apple.Terminal,
# com.microsoft.VSCode, com.anthropic.claude-code, com.openai.codex, and exactly
# one non-app binary: the uv python3.12 at
# ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12.
# There is NO grant for /bin/zsh, /usr/bin/python3, head, dd, or mkdir.
#
# An interactive shell inherits the grant from the app that spawned it, so every
# shell-fired run succeeds. A LaunchAgent has no granted parent: TCC evaluates
# /bin/zsh, finds nothing, and denies the open. zsh reports that denial with the
# same words it uses for ENOENT -- "can't open input file" -- which is what made
# eviction look plausible. The kernel reads the "#!" line during exec WITHOUT a
# TCC check, so the script starts and only then fails to read itself; that
# exec-succeeds-then-open-fails signature is diagnostic of TCC, not of eviction.
#
# launchd itself is fine. com.autoresearch.protocol101.shadowasof.ledger runs a
# .py from inside ~/Documents every 60 s under launchd, exit code 0 (verified
# live 2026-08-05: runs 5243 -> 5246 in 140 s) -- because its program IS that
# granted uv python3.12.
#
# CONSEQUENCE: the retry loop below CANNOT fix this. It retries a deterministic
# permission denial 60 times and still fails. Do not read its presence as a
# repair. Until a tier-1 fix lands (TCC grant, or moving the repo out of
# ~/Documents), use the attended runner instead:
#     v4/ops/tracka/run_tracka_attended.sh
# which fires the windows from a shell that already holds the grant.
#
# Everything below is retained rather than deleted: it is harmless under an
# attended run (the first read succeeds on attempt 1) and it is the record of
# what was tried.

set -uo pipefail

# Derived from this script's own location -- it lives at REPO/v4/ops/tracka/.
REPO="${0:A:h:h:h:h}"
WINDOW="${1:?usage: tracka_launcher.sh <open|midday>}"
LOGDIR="$HOME/.autoresearch-trading/tracka/logs"
mkdir -p "$LOGDIR"
exec >> "$LOGDIR/launcher_$(date +%Y-%m-%d)_${WINDOW}.log" 2>&1

echo "=== launcher ${WINDOW} @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- force-materialize everything the capture touches -------------------------
# brctl download is the supported way to pull an evicted item back. Reading the
# bytes is the belt-and-braces fallback: open() on a dataless file triggers the
# same fetch, and unlike brctl it fails loudly when the fetch does not happen.
materialize() {
    local target="$1"
    if [ ! -e "$target" ]; then
        echo "MISSING: $target"
        return 1
    fi
    /usr/bin/brctl download "$target" 2>/dev/null
    if [ -d "$target" ]; then
        /usr/bin/find "$target" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.json' \) 2>/dev/null \
            | while read -r f; do /bin/dd if="$f" of=/dev/null bs=1m count=1 2>/dev/null; done
    else
        /bin/dd if="$target" of=/dev/null bs=1m 2>/dev/null
    fi
    return 0
}

for p in \
    "$REPO/v4/ops/tracka/run_tracka_window.sh" \
    "$REPO/v4/scripts/capture_databento_live_opra_definitions.py" \
    "$REPO/v4/scripts/capture_databento_live_opra_training_twin.py" \
    "$REPO/v4/checks" \
    "$REPO/v4/research" \
    "$REPO/v4/.env" \
    "$REPO/v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04/capture_declaration_v9.json" \
    "$REPO/v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04/authorization.json"
do
    materialize "$p" || echo "WARN: could not materialize $p"
done

# The venv interpreter and its site-packages are also under the synced tree.
materialize "$REPO/.venv/bin/python" || echo "WARN: venv interpreter not materialized"

# Verify the thing we are about to exec is actually readable NOW. This is the
# precise check whose absence cost the 2026-08-05 open window: launchd produced
# a one-line error and an otherwise empty output tree, which looks identical to
# "never fired" unless you read the launchd stderr file specifically.
#
# WHY THIS RETRIES (2026-08-05 midday)
# -----------------------------------
# The single materialization pass above is NOT sufficient. brctl download is
# asynchronous, so the first read can still hit a dataless file. On 2026-08-05
# the launchd invocation at 16:10:00Z failed this check and exited 74; a second
# invocation 27 s later passed and captured normally.
#
# launchd did NOT perform that retry. KeepAlive is unset and StartCalendarInterval
# fires once, so the job's own record is runs=1, last exit code=74 -- a FAILURE.
# The retry came from an unrelated ad-hoc shell that happened to exist that day.
#
# CORRECTION 2026-08-05: the reasoning above is wrong. The 16:10:27Z success was
# not a late fetch completing -- it was a DIFFERENT INVOKER. It ran from a shell
# holding the Documents grant; the 16:10:00Z attempt ran under launchd, which
# does not. Same file, same minute, different permission. See the retracted
# diagnosis at the top. This loop will not save the 08-06 and 08-07 windows.
#
# Budget: 1 s spacing, 60 attempts. The observed recovery took 27 s. The window
# tolerates this -- the open capture is declared 09:28-09:33 ET to centre the
# 09:30 bell, so tens of seconds of drift still covers it.
readable=0
for attempt in $(seq 1 60); do
    if /usr/bin/head -1 "$REPO/v4/ops/tracka/run_tracka_window.sh" >/dev/null 2>&1; then
        readable=1
        echo "wrapper readable on attempt ${attempt}"
        break
    fi
    [ "$attempt" -eq 1 ] && echo "wrapper not yet readable -- retrying materialization"
    /usr/bin/brctl download "$REPO/v4/ops/tracka/run_tracka_window.sh" 2>/dev/null
    /bin/dd if="$REPO/v4/ops/tracka/run_tracka_window.sh" of=/dev/null bs=1m 2>/dev/null
    sleep 1
done

if [ "$readable" -ne 1 ]; then
    echo "FATAL: wrapper still unreadable after 60 attempts -- aborting"
    exit 74
fi
echo "materialization OK; handing off to the versioned wrapper"

exec /bin/zsh "$REPO/v4/ops/tracka/run_tracka_window.sh" "$WINDOW"
