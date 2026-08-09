# Track-A arrival capture — the required item that unblocks option features

**Meaning for the bot:** we cannot honestly train an option model until we have measured how late live
option quotes actually arrive, on several separate days, from the same data feed the model would read.
That measurement does not exist yet. One session has been recorded as an infrastructure test, and the two
sessions that would count as evidence have not happened. Until they do, the machinery correctly refuses to
build a training matrix, and that refusal is the intended behaviour rather than a fault to work around.

This is a dated requirement finding, not a status page. Current state and authorization come only from
[`v5/STATUS.md`](../../STATUS.md).

## 1. What the required item is

A **Track-A capture** is a short recording of the live option data feed. For three to five minutes we
subscribe to the exact set of SPX options expiring that day, and we write down our own clock reading the
instant each message reaches this machine. Comparing that reading to the market interval the message
describes gives one number per message: **how stale the quote already was when we first held it.**

Two recordings are taken per trading day, because staleness is not constant:

| Window | Local time | Length | Why this window |
|---|---|---:|---|
| Open | 06:28 | 300 s | Covers the 09:30 New York bell, the busiest and slowest moment of the day |
| Midday | 09:10 | 180 s | Quiet baseline, so we can see the range rather than one extreme |

Each recording writes the untouched raw feed, one line per message with its arrival stamp, and a
self-hashed summary containing the arrival percentiles. The implementation lives in
[`capture_databento_live_opra_definitions.py`](../../../v4/scripts/capture_databento_live_opra_definitions.py)
and
[`capture_databento_live_opra_training_twin.py`](../../../v4/scripts/capture_databento_live_opra_training_twin.py),
sequenced by [`run_tracka_window.sh`](../../../v4/ops/tracka/run_tracka_window.sh).

## 2. Why it must be done

Two numbers the project depends on are currently marked **uncertified** in
[`knobs.py`](../knobs.py) — meaning a value exists, but the evidence behind it does not support the use it
would be put to. Neither may be varied nor relied upon, and everything downstream is blocked.

| Uncertified number | Current value | Why it cannot be trusted |
|---|---:|---|
| `historical_arrival_lag_ms` | **0 ms** | The training corpus claims every quote was available the instant its minute ended, on all 47,707,186 rows. That is not a measurement; the vendor's historical files stamp a completed bar at the close of its own interval. |
| `emission_lag_ms` | **2,336 ms** | Derived from five samples of a *different* data source (an index feed, not the option feed). Wrong stream, and far too few samples to fit a trading system against. |

The consequence is concrete. A model fitted on the corpus as stored would learn to act on quotes at a
moment the live system cannot deliver them — it would appear skilful in testing by using information it
would not actually possess. That is not hypothetical: a prior entry model produced an apparently confirmed
result and was later withdrawn when exactly this defect was found in its clock, and the one-shot reserved
test set was spent proving the flaw rather than a strategy. Detail:
[historical arrival parity](HISTORICAL_ARRIVAL_PARITY_2026_08_05.md).

`knobs.py` names the remedy explicitly. `emission_lag_ms` unfreezes only on

> "A multi-session Track-A OPRA capture re-deriving the allowance from the same stream the features come
> from. It may not be lowered unless the lowering rule is pre-registered first."

That sentence is the requirement this document reports. Nothing else satisfies it.

## 3. What one capture actually produces

A capture was taken on **2026-08-05 midday** and is complete and well-formed: 180.3 seconds, 51,232
messages, the full 510-contract expiring set with no drift, and every safety flag false — no broker, no
orders, no model loaded, no reserved data opened.

| Stream | Messages | Median | 99th percentile |
|---|---:|---:|---:|
| One-minute option quotes — the model's feature stream | 1,361 | 369.6 ms | **527.6 ms** |
| One-second option quotes — the execution stream | 40,330 | 68.4 ms | **259.4 ms** |
| One-minute bars | 229 | 61.3 ms | 132.0 ms |

**The single most useful thing this capture told us is that the number moves.** The only earlier
measurement of the same one-minute option stream, on 2026-08-03, gave a 99th percentile of 319.5 ms. The
08-05 session gave 527.6 ms — **1.65× larger, from identical code.** Day-to-day variation, not
within-day noise, is the risk that has to be bounded, and it is exactly the thing a single session cannot
show. This is why the requirement says *multi-session* rather than *longer*.

Recording for more minutes on one day would sharpen a number that is already sharp and would say nothing
about the next day.

## 4. What we have, and what is still missing

The 08-05 midday recording **does not count as evidence.** The owner classified it as an infrastructure
verification run and removed 08-05 from the certified sample; see the frozen session list in
`capture_declaration_v6.json` and the note at the top of
[`run_tracka_window.sh`](../../../v4/ops/tracka/run_tracka_window.sh). That classification is correct and
is followed here: it proves the recording chain works end to end, and nothing about the market.

| Item | State |
|---|---|
| Recording chain proven to work | **Yes** — 08-05 midday, verified complete |
| Sessions accepted as evidence | **0 of 2** — 08-06 and 08-07 are declared and have not yet run |
| 08-05 open window | **Lost** — never recorded |
| Re-derived allowance from the option stream | **Not produced** |
| Signed feature ledger admitting option features | **Not produced** |

Two sessions is a deliberately modest target and is only defensible on one condition: **the allowance must
not be lowered.** The current 2,336 ms sits about 4.4× above the worst arrival yet observed, and that
margin is what absorbs the day-to-day spread two sessions cannot bound. Two sessions can support an
observed envelope; they cannot support a population worst case. If the goal ever becomes lowering the
allowance to buy decision speed, the sample requirement rises sharply and the lowering rule must be written
down before the data is inspected.

## 5. The blocker — scheduled jobs cannot read this repository

**Category: an environment defect that blocks the evidence branch, not a fault in the capture code.**

The capture is scheduled to run on its own at 06:28 and 09:10. On 2026-08-05 the scheduled job fired
exactly on time and died in under a second:

```
/bin/zsh: can't open input file:
  /Users/gduby/Documents/autoresearch-trading/v4/ops/tracka/run_tracka_window.sh
```

The file is present, executable, and reads normally from a terminal. The cause is that this repository
lives inside `~/Documents`, which macOS protects. Only a few programs hold permission to read it —
Terminal, the editors, and one Python interpreter. A program you start yourself in a terminal inherits
that permission; a scheduled job has no parent to inherit from, and `/bin/zsh` does not hold the grant on
its own.

**Two corrections belong on the record here, because both were mine and both could mislead a later
reader.**

1. I initially attributed the failure to iCloud evicting the file from local disk, and added a retry loop
   on that basis. That diagnosis was wrong. A permission denial does not resolve on retry, so the loop
   cannot help. The comment currently at the top of `run_tracka_window.sh` still says the 08-05 open
   window was "lost to iCloud eviction"; the accurate cause is the permission grant described above.
2. I also reported at one point that the capture had failed to fire at all. It did fire, on schedule. The
   distinction matters: a job that never runs and a job that runs and is denied need different fixes, and
   the evidence for the difference was in the scheduler's own error file rather than the capture output.

**This is not a Track-A problem.** It is a property of where the repository lives, so every unattended job
meets it. That includes the later live-shadow and guarded-paper rungs, neither of which can run on this
machine until it is resolved. The only complete fix is to move the repository out of `~/Documents`, which
should be scheduled deliberately after the 08-06 and 08-07 recordings are banked — a move touches many
absolute paths, including paths recorded inside signed evidence files, and a wrong path left behind fails
silently.

The present workaround is an attended runner started by hand from a terminal, which inherits the
permission and is armed for both remaining sessions.

## 6. Conflict with the status page that must be resolved

[`STATUS.md`](../../STATUS.md) says the conflict must be reported when another file disagrees with it. It
does disagree, in three places, because events moved after it was written:

| Status page says | Actually true on 2026-08-05 |
|---|---|
| Job 4 — "prior dates canceled; all jobs unloaded" | New dates 08-06 and 08-07 are declared and an attended runner is armed for both |
| §10 — "Nothing runs unattended" | Correct for scheduled jobs, but two long-running background agents remain loaded, and an attended capture runner is active |
| Job 4 waiting on "New owner authorization with exact dates" | Exact dates have been supplied |

The status page is the authority on what is authorized; this finding does not change it. Resolving the
rows above is a status-page edit, and it should be made before the 08-06 window so the register matches
what will actually happen.

> **Resolved 2026-08-09.** All three rows were reconciled in `STATUS.md`. Job 4 now records the armed
> runner and the v8 dates; §10 records that the two background agents are dead (exit 78) and that the
> Track-A launchd jobs were unloaded. The 08-06 window itself failed and banked nothing — see
> [STATUS §13](../../STATUS.md#13-track-a-capture-2026-08-06-failure-and-08-10-arming).

## 7. How to check the state yourself

> **Updated 2026-08-09.** The commands below originally named `/Users/gduby/...` and
> `capture_declaration_v6.json`. Neither resolves now: the home directory was renamed on 08-06, and v6's
> sessions (08-06/08-07) are spent and superseded by **v8** (08-10/11/12). The findings above are left as
> the record of what was measured on 08-05; only these instructions are corrected, because a runbook that
> cannot run is worth nothing. Paths are relative to the repository root.

```bash
# Which sessions are certified evidence, and which recordings exist
./.venv/bin/python -c "import json;print(json.load(open('v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04/capture_declaration_v8.json'))['capture_window']['sessions'])"
ls -d v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04/2026-08-*/*/

# Is the capture armed, on AC, and holding the no-sleep assertion?
./v4/ops/tracka/check_tracka.sh

# Verify a recording is complete rather than merely present
PYTHONPATH=. ./.venv/bin/python v4/ops/tracka/verify_tracka_window.py --session 2026-08-05 --window midday

# Rehearse the Phase-2 certification without issuing anything
./.venv/bin/python -m v5.ops.certify_tracka_arrival --dry-run-window 2026-08-05/midday
```

Read-only on owned local data. No model was fitted, no recording started, no vendor or broker contacted.
