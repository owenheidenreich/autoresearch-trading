#!/bin/zsh
# Track-A launchd entry point. Lives OUTSIDE ~/Documents on purpose.
#
# WHY THIS FILE EXISTS
# --------------------
# 2026-08-05 06:28 PDT the open window fired and died instantly with:
#     /bin/zsh: can't open input file: .../v4/ops/tracka/run_tracka_window.sh
# The file was present, executable, and parseable from an interactive shell.
#
# Cause: ~/Documents is the iCloud "Desktop & Documents" sync target
# (~/Library/Mobile Documents/com~apple~CloudDocs/Documents is a SYMLINK to it,
# and FXICloudDriveDesktop=1). iCloud had evicted the wrapper to dataless after
# ~19h untouched, and a launchd agent opening a dataless file fails. zsh reports
# a dataless/missing open with the same message it uses for ENOENT, which is why
# this first looked like a TCC denial -- it is not; no TCC denial was ever logged
# and the shadowasof agents exec a binary from outside the synced tree.
#
# So the entry point must live outside the synced tree, and it must force the
# repo paths it needs back onto local disk BEFORE invoking anything in them.

set -uo pipefail

REPO="/Users/gduby/Documents/autoresearch-trading"
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
    "$REPO/v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04/capture_declaration_v5.json" \
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
if ! /usr/bin/head -1 "$REPO/v4/ops/tracka/run_tracka_window.sh" >/dev/null 2>&1; then
    echo "FATAL: wrapper still unreadable after materialization -- aborting"
    exit 74
fi
echo "materialization OK; handing off to the versioned wrapper"

exec /bin/zsh "$REPO/v4/ops/tracka/run_tracka_window.sh" "$WINDOW"
