# Handoff — Track-A launchd jobs cannot read the repo, and the 08-06/08-07 captures will fail

**Opened: 2026-08-05. Urgency: the next capture fires 06:28 PDT Thursday 2026-08-06.**
**One-line problem: every launchd-fired capture attempt has failed; every shell-fired attempt has
succeeded. The cause is not settled, and the fix currently in place probably does not work.**

---

## 1. Why this handoff exists

I diagnosed this as iCloud evicting files to "dataless" and shipped a retry loop
(`685cbbc3`). **The owner has since confirmed "Optimize Mac Storage" was already OFF when all of this
happened.** That substantially weakens my diagnosis, and if the real cause is a permissions denial, the
retry loop is useless — it will retry a deterministic refusal sixty times and still fail.

I am handing this over rather than shipping a second guess.

## 2. The evidence

### 2.1 The outcome pattern separates perfectly by *who invoked it*

| Time (PDT) | UTC | Invoker | Result |
|---|---|---|---|
| 2026-08-05 06:28 | 13:28Z | **launchd** | **FAILED** — `/bin/zsh: can't open input file: .../v4/ops/tracka/run_tracka_window.sh` |
| 2026-08-05 08:03 | 15:03Z | shell | SUCCEEDED — "materialization OK" |
| 2026-08-05 09:10:00 | 16:10:00Z | **launchd** | **FAILED** — "wrapper still unreadable after materialization" |
| 2026-08-05 09:10:27 | 16:10:27Z | shell | SUCCEEDED — full capture ran, 510/510 symbols |
| 2026-08-04 11:19, 11:27 | — | shell | reached the wrapper's session gate (so the file was readable) |

**Two launchd attempts, two failures. Every shell attempt succeeded.** The sharpest datum is the pair 27
seconds apart on the same file in the same minute: launchd failed, a shell succeeded.

### 2.2 What is NOT the explanation

- **Not a missing file.** `run_tracka_window.sh` exists, is executable, and reads fine interactively.
- **Not a stale launchd path.** Both plists were rewritten 08-05 08:03 and point at
  `/Users/gduby/.autoresearch-trading/tracka/tracka_launcher.sh`, outside the synced tree. That launcher
  *does* execute under launchd — it writes its own log — so launchd runs fine until it touches
  `~/Documents`.
- **Not launchd retrying and eventually winning.** `KeepAlive` is unset and `StartCalendarInterval` fires
  once. The midday job's own record is `runs = 1`, `last exit code = 74`. The 09:10:27 success came from an
  unrelated ad-hoc shell, **not** from launchd.
- **Not Optimize Mac Storage.** Owner confirms it was already off.

### 2.3 Current machine state

```
FXICloudDriveDesktop = 1          # Desktop & Documents sync is still ON (separate from Optimize)
run_tracka_window.sh              # NOT dataless right now; no special flags; readable
tccd log 06:25-06:32, 09:09-09:12 # no entries returned (may be retention, not absence of denial)
```

## 3. Two hypotheses, and how to tell them apart

### H1 — TCC permissions (I now think this is more likely)

`~/Documents` is a TCC-protected location on macOS. A LaunchAgent's process needs explicit
Documents-folder or Full Disk Access permission to read it. A shell inherits that permission from the
terminal/IDE that spawned it; a launchd agent does not.

- **Explains the perfect invoker-split**, which an intermittent storage condition does not.
- **zsh reports a permissions refusal with the same message it uses for a missing file** —
  `can't open input file` — which is exactly why this first looked like eviction.
- **If H1 is true, this is much worse than it looks.** The Python capture scripts also live in
  `~/Documents`, so launchd could never read them either. Relocating only the wrapper would not help, and
  **the retry loop cannot help at all.**
- Counter-evidence to weigh: no `tccd` denial was found in the log. That is weak — the query returned
  nothing at all for those windows, which is consistent with log retention rather than with a clean read.

### H2 — iCloud placeholder/eviction, despite Optimize being off

Desktop & Documents sync is still enabled, so files remain iCloud-managed and can in principle be
dematerialized or briefly unavailable during sync activity.

- Fits the 27-second recovery: the failing invocation *did* run `brctl download` + `dd` first, which could
  have triggered a fetch that completed moments later.
- Does not naturally explain why **only** launchd invocations failed.

### The discriminating test (run this first)

The file is materialized and readable **right now**. So:

1. Install a one-shot LaunchAgent scheduled 2–3 minutes out whose only job is
   `head -1 <repo>/v4/ops/tracka/run_tracka_window.sh` plus `id -u` and a `ls -lO` of the file, writing to
   `~/.autoresearch-trading/tracka/logs/tcc_probe.log`.
2. Immediately before it fires, read the same file from a shell to prove it is materialized.
3. **If the shell reads it and the launchd probe cannot → H1 (TCC), conclusively.** The retry loop is
   dead weight and the fix must be relocation or a permissions grant.
4. If both succeed, H2 is live and the retry loop is probably adequate — but then explain the two launchd
   failures before trusting Thursday.

Installing a LaunchAgent is **tier 1** — it needs owner authorization. It is cheap, contacts nothing, and
should be removed immediately after.

## 4. Options, if H1 is confirmed

| Option | What it does | Cost / risk |
|---|---|---|
| **A. Move the working copy out of `~/Documents`** (e.g. `~/autoresearch-trading`) | Removes TCC protection *and* iCloud sync in one move; fixes both hypotheses at once | Highest confidence. Touches paths in plists, the launcher, `.env`, and anything with an absolute path. Must not disturb the data corpus at `~/.autoresearch-trading/` or `/Volumes/AR_TRADING_DATA` |
| **B. Grant Full Disk Access to the agent's interpreter** | Lets the LaunchAgent read `~/Documents` | Granting FDA to `/bin/zsh` is broad and blunt; may not survive OS updates; hard to reason about |
| **C. Stage a self-contained copy outside `~/Documents`** — wrapper + capture scripts + venv | The capture runs entirely outside the protected tree | Creates a second copy that can drift from the repo; needs a sync step and a hash check each session |
| **D. Run the captures attended** for 08-06 and 08-07 | A shell-fired run works today, proven twice | Requires someone at the machine at 06:28 and 09:10 PDT both days; no code change; **the safest thing for Thursday specifically** |

**My recommendation for Thursday itself is D**, with A or C as the durable fix afterward — the captures are
free and repeatable, but a lost bell-burst window costs a session that cannot be recovered.

## 5. What must NOT be assumed

- **Do not assume the retry loop fixed this.** It was written for H2. If H1 holds it is useless, and its
  presence makes the system *look* repaired when it is not. That is the same failure mode as this morning:
  a launchd failure produces an empty output tree that is indistinguishable from "never fired" unless you
  read `launchd.*.err` specifically.
- **Do not widen or move the capture windows** to buy time. Declaration v6 freezes sessions 2026-08-06 and
  2026-08-07 with windows at 06:28 (300 s) and 09:10 (180 s) PDT. Narrowing is permitted; widening requires
  fresh owner authorization.
- **Do not count 2026-08-05 as evidence.** Its open window was lost and its midday run is an
  infrastructure verification run only, recorded as excluded in declaration v6.
- **Do not touch** the frozen capture implementations (three SHA-256s pinned in the declaration), the
  authorization manifest, or the session gate.

## 6. Verification that the capture itself is healthy

The capture path works when it is actually invoked. From 2026-08-05 midday:

```
status: CAPTURE_RECEIPT_PRESENT, problems: []
symbols: 510 (expected 510)
cbbo-1m p99: 527.6 ms     cbbo-1s p99: 259.4 ms     ohlcv-1m p99: 132.0 ms
local receipt rows: 51,232
```

**This is purely a scheduling/permissions problem, not a data or code problem.** Whoever picks this up
does not need to touch the capture logic.

## 7. Context

- Purpose of the capture: measure when OPRA data actually arrives, so features can be certified as
  available at decision time. It unblocks 65 entry and 29 exit features (gate G3 in `STATUS.md`).
- The current emission-lag placeholder is 2336 ms from a different vendor; measured OPRA CBBO-1m is
  ~528–585 ms in quiet windows. **The bell bursts on 08-06 and 08-07 are the samples that actually set the
  lag**, because it is taken from the worst observed p99. Losing them is the real cost of this bug.
- Read `STATUS.md` first, then
  `v4/docs/protocol101/training/contracts/PATHD_PROGRAMME_RESTART_RECORD_2026_08_05.md`.

## 8. Files involved

```
~/Library/LaunchAgents/com.autoresearch.tracka.capture.plist          # open,   06:28 PDT, Wed/Thu/Fri
~/Library/LaunchAgents/com.autoresearch.tracka.capture.midday.plist   # midday, 09:10 PDT, Wed/Thu/Fri
~/.autoresearch-trading/tracka/tracka_launcher.sh                     # entry point, outside ~/Documents
~/.autoresearch-trading/tracka/logs/                                  # launcher + launchd stderr
<repo>/v4/ops/tracka/run_tracka_window.sh                             # the file launchd cannot read
<repo>/v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04/
    capture_declaration_v6.json                                       # frozen: 08-06, 08-07
    authorization.json                                                # owner's authorized scope
    run_logs/                                                         # per-window wrapper logs
```

*Written by: Claude Opus 5 — 2026-08-05, after the owner refuted the eviction diagnosis.*
