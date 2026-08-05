# RESOLVED — Track-A launchd jobs cannot read the repo. Cause confirmed: macOS file permissions.

**Opened 2026-08-05 morning. Closed 2026-08-05 10:00 PDT, before the next capture (06:28 PDT Thu 08-06).**

**One-line answer: macOS blocks background jobs from reading `~/Documents` unless the specific program
has been given permission. `/bin/zsh` has not been. That is the whole bug.**

**What to do tomorrow: do not rely on the scheduled job. Start
`./v4/ops/tracka/run_tracka_attended.sh` in a Terminal window tonight and leave it open.** It fires both
windows itself, from a process that already has the permission. Nothing else needs to change first.

---

## 1. What changed since this document was opened

It was opened saying the cause was unsettled between two guesses and that the shipped fix probably did not
work. Both halves are now answered, and **no LaunchAgent had to be installed to answer them** — the
discriminating test proposed in the original version (section 3, now removed) turned out to be unnecessary.

| Hypothesis | Verdict |
|---|---|
| **H1 — macOS permissions (TCC) on `~/Documents`** | **CONFIRMED** |
| H2 — iCloud evicting the file to "dataless" | **REFUTED** |
| H3 — machine asleep at the fire time, added during this investigation | **REFUTED** — `pmset -g log` shows no sleep or wake transition on 2026-08-04 or 2026-08-05 at all |

**The retry loop shipped in `685cbbc3` does not fix this and never could.** It retries a permission
refusal sixty times. Its comments have been corrected in place (`v4/ops/tracka/tracka_launcher.sh`) so the
next reader does not mistake it for a repair.

## 2. The evidence, in the order that settles it

### 2.1 The permission table names exactly who is allowed

macOS keeps folder permissions in a database. Reading it directly:

```
$ sqlite3 ~/Library/Application\ Support/com.apple.TCC/TCC.db \
    'select client, auth_value from access
       where service = "kTCCServiceSystemPolicyDocumentsFolder"'

com.apple.Terminal                                                     | 2   <- allowed
com.microsoft.VSCode                                                   | 2   <- allowed
com.anthropic.claude-code                                              | 2   <- allowed
com.openai.codex                                                       | 2   <- allowed
/Users/gduby/.local/share/uv/python/.../cpython-3.12.13/bin/python3.12 | 2   <- allowed
```

`auth_value = 2` means allowed. **That is the complete list of things permitted to read `~/Documents`.**
Four apps, and exactly one loose binary: a Python interpreter.

There is **no entry for `/bin/zsh`**, none for `/usr/bin/python3`, and none for `head`, `dd`, or `mkdir` —
every other program the capture path uses.

### 2.2 That single table explains every observation

| Time (PDT) | Invoker | Result | Why |
|---|---|---|---|
| 08-05 06:28 | launchd | FAILED | launchd's program is `/bin/zsh` → not in the table → refused |
| 08-05 08:03 | shell | SUCCEEDED | child of Terminal/VS Code → inherits their permission |
| 08-05 09:10:00 | launchd | FAILED | same as 06:28 |
| 08-05 09:10:27 | shell | SUCCEEDED | same as 08:03 |

The pair 27 seconds apart — the datum that looked so strange — is not a timing race at all. **It is two
different programs asking for the same file, one of which is on the list and one of which is not.**

A program run from Terminal inherits Terminal's permission. A scheduled job has no such parent, so macOS
checks the job's own program and finds nothing.

### 2.3 Why this masqueraded as a missing file

The 06:28 error was `/bin/zsh: can't open input file: <correct full path>` (verified byte-for-byte in
`run_logs/launchd.open.err`; the path has no typo). zsh prints the *same words* for "denied" as for "not
there," which is what sent the first diagnosis toward eviction.

More precisely, and this is the tell: the system read the script's `#!/bin/zsh` first line **without** a
permissions check — that read happens inside the kernel when starting a program, and is exempt. zsh then
started up and asked for the same file through the normal path, which **is** checked, and was refused. So
the script launched and then could not read itself. **That exact signature — starts fine, then can't open
itself — is a permissions denial and not eviction.**

### 2.4 Proof that scheduled jobs are not the problem, only ungranted ones

`com.autoresearch.protocol101.shadowasof.ledger` is an already-installed LaunchAgent that runs
`v4/scripts/run_protocol101_shadow_boundary_ledger.py` — **inside `~/Documents`** — every 60 seconds.

```
runs = 5243  →  5246   over 140 s, sampled live 2026-08-05 16:53Z–16:55Z
last exit code = 0
```

It works because its program is `~/.autoresearch-trading/runtime-venv/bin/python`, which is a symlink to
that one granted uv `python3.12`. **A scheduled job reads `~/Documents` perfectly well when its program is
on the list.** Track A's is not.

### 2.5 What was ruled out

- **Not eviction.** Owner confirmed "Optimize Mac Storage" was off; `find -flags dataless` over
  `v4/ops`, `v4/scripts`, `v4/checks` returns zero files; and eviction cannot produce a perfect split by
  invoker.
- **Not sleep/wake.** No sleep or wake transition on either day in `pmset -g log`.
- **Not a stale or wrong path.** The failing path in the launchd stderr is byte-identical to the real one.
- **Not launchd retrying and winning.** `KeepAlive` unset, `runs = 1`, `last exit code = 74`.

## 3. What has been done (no tier-1 action taken)

| Change | File | Tier |
|---|---|---|
| Attended runner — fires both windows from a permitted shell | `v4/ops/tracka/run_tracka_attended.sh` (new) | 3 |
| Retracted the eviction diagnosis in place; marked the retry loop as not-a-fix | `v4/ops/tracka/tracka_launcher.sh` | 3 |
| Fixed the launcher pre-loading declaration **v5** when the wrapper reads **v6** | `v4/ops/tracka/tracka_launcher.sh` | 3 |

**Deliberately not touched:** the two installed plists, the installed copy of `tracka_launcher.sh` at
`~/.autoresearch-trading/tracka/` (changing anything that runs unattended is tier 1), the frozen capture
scripts and their pinned hashes, `authorization.json`, the session gate, and `capture_declaration_v6.json`
— which is hash-sealed and still attributes the lost 08-05 open window to "iCloud file eviction". **That
attribution is now known to be wrong.** The declaration is not being edited, because it is sealed and
because the exclusion it records is still correct; this document is the correction of record.

The installed launcher and the repo copy now differ by comments only. Re-copying it is tier 1 and is not
required for the attended path.

## 4. Tomorrow, and the durable fix

### The attended runner is proven and needs no permission change

```
cd /Users/gduby/Documents/autoresearch-trading
./v4/ops/tracka/run_tracka_attended.sh
```

It reads the schedule out of the frozen declaration, refuses immediately and loudly if the shell it was
started in lacks Documents permission, waits, then calls the same frozen wrapper — which still enforces
its own declared-session gate, so a stray run cannot capture on an undeclared day. Verified this afternoon:
it read all four declared windows and armed correctly for 2026-08-06 06:28 (73,825 s out).

Two conditions: start it from **Terminal or the VS Code terminal**, and **keep the Mac awake** — the
windows do not survive system sleep. The machine has not slept in over two days, so this is a low bar.

### For the durable fix — this is the owner's call (tier 1)

| Option | What it does | Assessment |
|---|---|---|
| **A. Move the repo out of `~/Documents`** (e.g. `~/autoresearch-trading`) | Removes the permission barrier for every program at once | The real fix. But it rewrites absolute paths in the tracka plists, the shadowasof and parityrecorder plists, `.env`, and hardcoded `REPO=` lines. **Not something to attempt the night before a capture.** |
| **B. Grant Full Disk Access to `/bin/zsh`** | Puts zsh on the list | Broad and blunt — it grants every shell script on the machine full disk access. May not survive OS updates. |
| **C. Make the LaunchAgent's program the already-granted `python3.12`**, which then starts the wrapper | Cheapest and narrowest | Relies on child programs inheriting the parent's permission. That inheritance is how the Terminal case works, but it has **not** been tested for a LaunchAgent here, and testing it means installing a probe job (tier 1). |
| **D. Run attended for 08-06 and 08-07** | What section 4 describes | **Recommended for both remaining sessions.** Zero new risk, no permission change, proven three times today. |

**Recommendation: D now, then A afterward, unhurried, with the captures already banked.** Cost of D is
leaving a terminal window open overnight. Cost of getting A or C wrong is another lost bell window — and
the bell bursts on 08-06 and 08-07 are the only two samples left that set the availability clock, because
it is taken from the worst observed p99.

If you want C tested for the future, it is a two-minute probe job that contacts nothing, but it needs your
say-so to install and should be removed straight after.

## 5. What is still true from the original handoff

- **Do not widen or move the capture windows.** Declaration v6 freezes 2026-08-06 and 2026-08-07 at 06:28
  (300 s) and 09:10 (180 s) PDT. Narrowing is permitted; widening needs fresh authorization.
- **2026-08-05 is not evidence.** Open window lost; midday run is infrastructure verification only.
- **The capture code is healthy** — this was never a data or code problem. From the 08-05 midday run:
  `status: CAPTURE_RECEIPT_PRESENT, problems: []`, 510/510 symbols, cbbo-1m p99 527.6 ms, 51,232 receipt
  rows.
- Purpose: measure when OPRA data actually arrives, so features can be certified available at decision
  time. Unblocks 65 entry and 29 exit features (gate G3 in `STATUS.md`).

*Diagnosis corrected and closed by Claude Opus 5, 2026-08-05. The eviction diagnosis in the first version
of this document, and in `a6594ebb` / `685cbbc3`, is retracted.*
