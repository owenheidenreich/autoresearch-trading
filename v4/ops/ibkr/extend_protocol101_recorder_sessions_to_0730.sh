#!/usr/bin/env bash
# Protocol101 parity recorder: extend the session allowlist through 2026-07-30.
#
# WHY: the deployed schedule (07-10, 07-13..17, 07-20..24) ends before the
# July FOMC (07-28/29). High-volatility sessions are the most valuable sealed
# confirmation evidence, and with 07-10 designated a validation day only 10
# sealed sessions remain. Extending adds 07-27..07-30 (~14 sealed days incl.
# FOMC). Sessions >= 2026-07-13 are sealed on arrival per
# v4/scripts/run_protocol101_sealed_day_assignment.py (rule fixed 2026-07-09).
#
# WHEN TO RUN: after the 2026-07-10 validation session finalizes and its
# manifests are reviewed green (owner decision 2026-07-09: do not perturb
# loaded launchd state before the pipeline's first live capture).
#
# WHAT IT DOES: backs up the eight parityrecorder plists, appends the four
# dates to PROTOCOL101_PACKET_ALLOWED_SESSIONS in each, then reloads each
# label. Idempotent: skips plists already containing 2026-07-30.

set -euo pipefail

OLD_LIST="2026-07-10,2026-07-13,2026-07-14,2026-07-15,2026-07-16,2026-07-17,2026-07-20,2026-07-21,2026-07-22,2026-07-23,2026-07-24"
NEW_LIST="${OLD_LIST},2026-07-27,2026-07-28,2026-07-29,2026-07-30"
AGENTS_DIR="$HOME/Library/LaunchAgents"
BACKUP_DIR="$AGENTS_DIR/backup_parityrecorder_$(date +%Y%m%d_%H%M%S)"
LABEL_PREFIX="com.autoresearch.protocol101.parityrecorder"

mkdir -p "$BACKUP_DIR"
changed=0
for plist in "$AGENTS_DIR/$LABEL_PREFIX".*.plist; do
  name="$(basename "$plist")"
  if grep -q "2026-07-30" "$plist"; then
    echo "skip (already extended): $name"
    continue
  fi
  if ! grep -q "$OLD_LIST" "$plist"; then
    echo "WARNING: $name does not contain the expected allowlist; NOT modified" >&2
    continue
  fi
  cp "$plist" "$BACKUP_DIR/$name"
  sed -i '' "s|$OLD_LIST|$NEW_LIST|" "$plist"
  label="${name%.plist}"
  launchctl unload "$plist" 2>/dev/null || true
  launchctl load "$plist"
  echo "extended + reloaded: $label"
  changed=$((changed + 1))
done

echo "---"
echo "plists changed: $changed (backups in $BACKUP_DIR)"
echo "verify loaded labels:"
launchctl list | grep "$LABEL_PREFIX" || true
echo "verify allowlist:"
grep -h "2026-07-30" "$AGENTS_DIR/$LABEL_PREFIX".*.plist | head -1 || echo "NO PLIST CONTAINS 2026-07-30 — extension failed"
