# Path-D Phase-1 SSD Setup Runbook

Date staged: 2026-08-03  
Status: commands reviewed but **not executed**; no external drive was present or touched.

This runbook is Phase B and may be used only after the owner physically attaches
the intended 2 TB SSD. It prepares an encrypted APFS workspace named
`AR_TRADING_DATA`, copies the already-owned corpus with checksums, and leaves the
internal source intact. It does not download data, train a model, open the
firewall, connect to a broker, or change the paper default.

Run all commands from:

```bash
cd /Users/gduby/Documents/autoresearch-trading
```

## 0. Hard stops

- Connect the Mac to power and connect the SSD directly with a known-good data
  cable.
- Close Disk Utility and any application using the prospective SSD.
- Do not proceed if more than one unfamiliar external physical disk is listed.
- Do not proceed if the capacity, manufacturer/model, or external status differs
  from the purchased SSD.
- Never use the startup disk, an internal disk, a partition identifier such as
  `disk7s1`, a wildcard, or an unverified shell variable as the erase target.
- The erase is irreversible. The owner must manually compare and type the exact
  **whole-disk** identifier before the erase command can run.

## 1. Identify the exact external whole disk — read only

List only external physical disks:

```bash
diskutil list external physical
```

From that output, inspect the intended whole disk. The following is a template,
not a runnable command: replace `diskN` with the exact observed identifier.

```text
diskutil info /dev/diskN
```

Manually verify all of the following in the output:

- `Device Location: External` or `Internal: No`;
- `Whole: Yes`;
- capacity is the purchased 2 TB device (approximately 2,000,000,000,000
  bytes, allowing normal manufacturer variation);
- model/media name matches the purchased SSD;
- the identifier is a whole disk such as `disk7`, not a slice such as
  `disk7s1`.

Stop if any item is uncertain.

## 2. DESTRUCTIVE — erase only the verified whole disk

**DESTRUCTIVE: every volume and file on the selected whole disk will be
destroyed.** The following guarded block requires the owner to type the exact
identifier, validates it again with `diskutil`, constrains capacity to a
plausible 2 TB range, and requires a disk-specific confirmation phrase. It does
not accept a slice or an internal disk.

```zsh
read "PHASE1_DISK_ID?Type the verified whole-disk identifier only (example: disk7): "
[[ "$PHASE1_DISK_ID" =~ ^disk[0-9]+$ ]] || { echo "STOP: whole-disk identifier required" >&2; return 1 2>/dev/null || exit 1; }

PHASE1_DISK_PLIST="$(mktemp -t pathd-phase1-disk.XXXXXX)" || { echo "STOP: mktemp failed" >&2; return 1 2>/dev/null || exit 1; }
trap 'rm -f "$PHASE1_DISK_PLIST"' EXIT
diskutil info -plist "/dev/$PHASE1_DISK_ID" > "$PHASE1_DISK_PLIST" || { echo "STOP: disk inspection failed" >&2; return 1 2>/dev/null || exit 1; }

[[ "$(plutil -extract DeviceIdentifier raw -o - "$PHASE1_DISK_PLIST")" == "$PHASE1_DISK_ID" ]] || { echo "STOP: identifier mismatch" >&2; return 1 2>/dev/null || exit 1; }
[[ "$(plutil -extract Whole raw -o - "$PHASE1_DISK_PLIST")" == "true" ]] || { echo "STOP: target is not a whole disk" >&2; return 1 2>/dev/null || exit 1; }
[[ "$(plutil -extract Internal raw -o - "$PHASE1_DISK_PLIST")" == "false" ]] || { echo "STOP: target is not external" >&2; return 1 2>/dev/null || exit 1; }

PHASE1_DISK_BYTES="$(plutil -extract TotalSize raw -o - "$PHASE1_DISK_PLIST")"
[[ "$PHASE1_DISK_BYTES" =~ ^[0-9]+$ ]] || { echo "STOP: unreadable capacity" >&2; return 1 2>/dev/null || exit 1; }
(( PHASE1_DISK_BYTES >= 1800000000000 && PHASE1_DISK_BYTES <= 2200000000000 )) || { echo "STOP: target is not a plausible 2 TB disk" >&2; return 1 2>/dev/null || exit 1; }

diskutil info "/dev/$PHASE1_DISK_ID"
echo "Review the external status, Whole=Yes, model, capacity, and identifier above."
read "PHASE1_ERASE_CONFIRM?Type exactly: ERASE $PHASE1_DISK_ID FOR AR_TRADING_DATA : "
[[ "$PHASE1_ERASE_CONFIRM" == "ERASE $PHASE1_DISK_ID FOR AR_TRADING_DATA" ]] || { echo "STOP: erase not confirmed" >&2; return 1 2>/dev/null || exit 1; }

diskutil eraseDisk APFS AR_TRADING_DATA GPT "/dev/$PHASE1_DISK_ID"
```

Do not copy any data yet. `eraseDisk APFS` creates the APFS volume first; the
next step encrypts it before use.

## 3. Encrypt the APFS volume before use

Resolve and validate the new APFS **volume** identifier, then enable encryption.
`diskutil` prompts interactively for the new disk passphrase, so it is not
placed in shell history.

```zsh
[[ -d /Volumes/AR_TRADING_DATA ]] || { echo "STOP: expected volume is not mounted" >&2; return 1 2>/dev/null || exit 1; }
PHASE1_VOLUME_ID="$(diskutil info -plist /Volumes/AR_TRADING_DATA | plutil -extract DeviceIdentifier raw -o - -)"
[[ "$PHASE1_VOLUME_ID" =~ ^disk[0-9]+s[0-9]+$ ]] || { echo "STOP: APFS volume identifier not resolved" >&2; return 1 2>/dev/null || exit 1; }
diskutil apfs encryptVolume "$PHASE1_VOLUME_ID" -user disk
```

Wait for encryption to finish. Re-run these checks until `Encrypted: Yes` and
the APFS encryption progress is complete:

```bash
diskutil info /Volumes/AR_TRADING_DATA
diskutil apfs list
```

Stop if the volume is not named exactly `AR_TRADING_DATA`, is not APFS, is not
encrypted, or is not external.

## 4. Create the frozen directory layout

These are the exact eight names frozen by `PHASE1_DIRECTORIES`:

```bash
mkdir -p \
  /Volumes/AR_TRADING_DATA/vendor \
  /Volumes/AR_TRADING_DATA/canonical \
  /Volumes/AR_TRADING_DATA/oof \
  /Volumes/AR_TRADING_DATA/exit_features \
  /Volumes/AR_TRADING_DATA/exit_labels \
  /Volumes/AR_TRADING_DATA/scratch \
  /Volumes/AR_TRADING_DATA/artifacts \
  /Volumes/AR_TRADING_DATA/reports
```

Verify exact names and ownership:

```bash
find /Volumes/AR_TRADING_DATA -mindepth 1 -maxdepth 1 -type d -print | sort
ls -ld /Volumes/AR_TRADING_DATA/{vendor,canonical,oof,exit_features,exit_labels,scratch,artifacts,reports}
```

## 5. Persist the three roots in zsh

Fail if a prior managed block exists, preserve the current `.zshrc`, append one
literal block, and load it:

```zsh
touch "$HOME/.zshrc"
if grep -Fq '# >>> PATHD_PHASE1_ROOTS >>>' "$HOME/.zshrc"; then
  echo "STOP: PATHD_PHASE1_ROOTS already exists; inspect it manually" >&2
  return 1 2>/dev/null || exit 1
fi
cp -p "$HOME/.zshrc" "$HOME/.zshrc.before-pathd-phase1-$(date +%Y%m%dT%H%M%S)"
{
  echo '# >>> PATHD_PHASE1_ROOTS >>>'
  echo 'export AR_TRADING_DATA_ROOT=/Volumes/AR_TRADING_DATA'
  echo 'export AR_TRADING_SCRATCH_ROOT=/Volumes/AR_TRADING_DATA'
  echo 'export AR_TRADING_ARTIFACT_ROOT=/Volumes/AR_TRADING_DATA/artifacts'
  echo '# <<< PATHD_PHASE1_ROOTS <<<'
} >> "$HOME/.zshrc"
source "$HOME/.zshrc"

[[ "$AR_TRADING_DATA_ROOT" == /Volumes/AR_TRADING_DATA ]] || { echo "STOP: data root mismatch" >&2; return 1 2>/dev/null || exit 1; }
[[ "$AR_TRADING_SCRATCH_ROOT" == /Volumes/AR_TRADING_DATA ]] || { echo "STOP: scratch root mismatch" >&2; return 1 2>/dev/null || exit 1; }
[[ "$AR_TRADING_ARTIFACT_ROOT" == /Volumes/AR_TRADING_DATA/artifacts ]] || { echo "STOP: artifact root mismatch" >&2; return 1 2>/dev/null || exit 1; }
```

The scratch root deliberately equals the volume root because trajectory code
writes the frozen `exit_features` and `exit_labels` directory names beneath it.

## 6. Run the authoritative storage preflight

This must report `PASS`, encrypted external APFS, the exact volume name, at
least 25% projected free space, and allocation no greater than 150 GB:

```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model \
  storage-preflight --create-roots
```

Also record the operating-system view:

```bash
diskutil info /Volumes/AR_TRADING_DATA
df -H /Volumes/AR_TRADING_DATA
```

Stop on any discrepancy. Do not bypass `phase1_storage` checks.

## 7. Checksum-copy the existing corpus; never move it

Verify the internal source exists:

```bash
test -d /Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31
du -sh /Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31
test ! -e /Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31
```

Keep the Mac on power. `caffeinate` prevents idle/display/system/disk sleep for
the lifetime of the copy command. The relocation command builds a SHA-256
manifest, copies with metadata, verifies every destination file, writes the
manifest only after verification, and never deletes the source:

```bash
caffeinate -dimsu env PYTHONPATH=. .venv/bin/python \
  -m v4.scripts.run_phase1_exit_model relocate-corpus \
  --source /Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31
```

The command must end with `COPIED_AND_VERIFIED` and
`"source_preserved": true`. If it stops or disconnects, do not delete the
`.relocation-incomplete` marker or overwrite the partial destination; inspect
and recover deliberately.

## 8. Independently verify identity, encryption, space, and checksums

Re-run the authoritative volume and allocation check:

```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model storage-preflight
diskutil info /Volumes/AR_TRADING_DATA
df -H /Volumes/AR_TRADING_DATA
```

Verify the persisted relocation manifest against both source and destination.
This hashes bytes only; it does not decode Parquet/DBN data or open the
firewall:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from pathlib import Path
import json

from v4.research.phase1_storage import verify_manifest

source = Path('/Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31')
destination = Path('/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31')
manifest_path = Path('/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31.manifest.json')
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
source_verified = verify_manifest(source, manifest)
destination_verified = verify_manifest(destination, manifest)
assert source_verified == destination_verified == manifest
print({
    'status': 'SOURCE_AND_DESTINATION_CHECKSUMS_MATCH',
    'file_count': manifest['file_count'],
    'total_bytes': manifest['total_bytes'],
    'manifest_sha256': manifest['manifest_sha256'],
})
PY
```

Keep the internal source unchanged until the entire Phase-1 campaign is safely
complete and a separate retention decision is made. Successful relocation is
not permission to remove it.

## 9. Prevent sleep and accidental eject during later long stages

For each Phase-C materialize/train/trajectory/replay command, prefix the exact
command with:

```text
caffeinate -dimsu env PYTHONPATH=. .venv/bin/python -m ...
```

Keep the Mac on external power, do not close the lid, do not disconnect the
cable, do not eject the volume in Finder, and do not allow a hub to power-cycle.
In a second terminal, this read-only command confirms the active power
assertion:

```bash
pmset -g assertions | grep -A12 -i caffeinate
```

Run `storage-preflight` before and after every long Phase-C stage. This is
required because the current CLI checks the 150 GB allocation at stage start,
not after every trajectory partition.

## 10. Safe unmount and eject

Do not unmount during a copy, hash verification, training, or replay. First
verify that no Phase-1 runner remains active:

```bash
pgrep -fl 'v4\.scripts\.run_phase1_exit_model' || true
```

If the output shows any active job, stop and let it finish or terminate it
deliberately before proceeding. Then flush writes and eject by the exact volume
mount point:

```bash
sync
diskutil eject /Volumes/AR_TRADING_DATA
```

Only disconnect the cable after `diskutil` reports a successful eject. If the
eject fails, use `lsof +D /Volumes/AR_TRADING_DATA` to identify open files;
close the owning application and retry. Do not use forced unmount or unplug the
drive.

## Completion gate for Phase B

Phase B is complete only when all of the following are recorded:

- the owner manually confirmed the exact external whole-disk identifier before
  the destructive command;
- the mounted volume is external, APFS, encrypted, and named exactly
  `AR_TRADING_DATA`;
- all eight frozen directories exist;
- all three roots resolve to that volume;
- `storage-preflight` reports `PASS`, at least 25% free, and allocation at or
  below 150 GB;
- relocation reports `COPIED_AND_VERIFIED` with the source preserved;
- independent source/destination manifest verification passes;
- the original internal corpus remains present.

Only then may the separately authorized Phase C begin.
