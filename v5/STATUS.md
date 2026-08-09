# V5 Project Status — the one page

**Last updated: 2026-08-09.** This is the only current-state, job-register, and gate-chain document.
If another file disagrees, this page wins and the conflict must be reported.

> **The measurement review returned verdict B on 2026-08-05: only a large edge (~2.2–4.0 net
> points/session) is detectable on the owned year.** The owner accepted "large edge or stop," so the
> G1 direction screen is **released** under that reduced claim. Its 18-member family was **frozen
> 2026-08-06** before any outcome was inspected; the next step is the surrogate known-answer campaign,
> which must pass before real economics are read. See [`work/g1-direction/PLAN.md`](work/g1-direction/PLAN.md).
> G1 runs in parallel with the Track-A capture and does not depend on it.

## 0. Job register

A committed work packet must appear here. No row means no job.

| # | Job | Gate | State | Waiting on | Work packet |
|---|---|---|---|---|---|
| 1 | Build the clean v5 project boundary | infrastructure | **DONE 08-05** | — | this page and [v5 front door](README.md) |
| 2 | Independent review: is the project measurable? | G1/G4/G5/G8 | **DONE 08-05 — verdict B: only a large edge is detectable; limits verified from raw data and slightly conservative at lag 1** | — | [finding](research/findings/MEASUREMENT_REVIEW_2026_08_05.md), packet [`v5/work/measurement-review/`](work/measurement-review/) |
| 3 | ES direction screen: opening range and overnight gap | G1 | **RUNNING — family FROZEN 08-06 (18 members, hash `f43b92c2…`, reissued from `5ec8b5a4…` for a renamed home directory, one field of 213, no outcome inspected); next is the surrogate campaign** | Loader, gate, surrogates | [`v5/work/g1-direction/`](work/g1-direction/), [`family.py`](research/direction/family.py), [re-freeze](work/g1-direction/REFREEZE_2026_08_06.md) |
| 4 | Track-A option-feature arrival capture | G3 | **0 banked — runner ARMED and RUNNING since 08-09 08:45 PDT for 08-10/11/12; Phase-2 certification READY** | The six windows to fire; then one local certification command | [declaration v8](../v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04/capture_declaration_v8.json), [`certify_tracka_arrival.py`](ops/certify_tracka_arrival.py), [§13](#13-track-a-capture-2026-08-06-failure-and-08-10-arming) |
| 4a | Unattended jobs cannot read this repository | blocks G3/G7/G8 | **AUTHORIZED 08-05, NOT YET EXECUTED — a capture is now live, and the manifest forbids moving during one, so this waits until after 08-12** | Owner executing the [migration manifest](governance/REPO_MIGRATION_MANIFEST_2026_08_05.md); the TCC grant is also UNKNOWN since the rename | [requirement finding §5](research/findings/TRACK_A_ARRIVAL_CAPTURE_REQUIREMENT_2026_08_05.md#5-the-blocker--scheduled-jobs-cannot-read-this-repository), [§14](#14-home-directory-rename-2026-08-06) |
| 5 | Repair four defects in the validation gate | G5 | **DONE 08-05 — rebuilt natively** | — | [`validation/replay_gate.py`](research/validation/replay_gate.py), [§9](#9-g5-validation-is-defective) |
| 6 | Rebuild a confirmation firewall | G8 | **SIGNED 08-05 — every session from 2026-08-06 onward is confirmation-only** | — | [reservation declaration](governance/FORWARD_CONFIRMATION_RESERVATION_2026_08_06.md) |
| 7 | Programme restart after the 08-04 stop | all | **CLOSED 08-05** | — | [restart record](governance/PROGRAM_RESTART_RECORD_2026_08_05.md) |
| 8 | Quarantine stale v4 documentation | infrastructure | **DONE 08-05 — 117 files preserved** | — | [manifest](../_cleanup_quarantine/2026-08-05-docs/MANIFEST.md) |
| 9 | Inventory and promote training-readiness controls | infrastructure | **DONE 08-05** | — | [capability audit](research/findings/V5_WORKFLOW_CAPABILITY_AUDIT_2026_08_05.md) |
| 10 | Retire the documentation drawers and enforce it repo-wide | infrastructure | **DONE 08-05 — 75 files preserved** | — | [manifest](../_cleanup_quarantine/2026-08-05b-docs/MANIFEST.md), [`ops/check_project.py`](ops/check_project.py) |
| 11 | Build the two training preconditions | G3/G4 | **DONE 08-05 — built, not satisfied** | Rung 4 must supply usable latency and freshness receipts | [`research/knobs.py`](research/knobs.py), [arrival finding](research/findings/HISTORICAL_ARRIVAL_PARITY_2026_08_05.md) |
| 12 | Close six control-machinery gaps found by the training-readiness review: gate-pass receipts, computed power bound to the session index, trades→session aggregator, entry-freeze lock, knob-registry expansion, pipeline composition test | G1–G6 controls | **DONE 08-05 — 97 tests green** | — | [`research/gate_receipts.py`](research/gate_receipts.py), [`research/entry_stream.py`](research/entry_stream.py), [`research/validation/session_index.py`](research/validation/session_index.py) |
| 13 | Register every way training data can differ from what the live system sees | G4/G6/G7 | **BUILT 08-09 — 16 axes, 7 blocking; one owner decision raised** | Owner ruling on `product_session_existence`; the rest settle at their named gates | [`research/divergence.py`](research/divergence.py), [§15](#15-the-trainlive-divergence-register) |

## 1. What we are building

An automated day-trading bot that buys SPX 0DTE calls or puts when a model has defensible evidence about
the next 15–60 minutes of SPX direction. It must be causal, clear measured costs, survive validation, and
prove itself in guarded paper trading before any real-money decision.

## 2. The route to a trading bot

This is the only route document. Each rung names what it produces, which gate it clears, and what
authorizes it. A rung may not start until the row above it has passed — except rung 4, the one safe
parallel branch. Rung 1 completed 2026-08-05 with a favourable (verdict-B) answer; rung 2 is the
authorized frontier. Nothing below rung 2 is authorized today.

| # | Step | Produces | Gate | Authorized by | Where |
|---:|---|---|---|---|---|
| 1 | **Measurement review** | A verdict on whether 254 sessions can resolve any edge worth trading | precondition to G1 | **DONE 08-05 — verdict B** | [finding](research/findings/MEASUREMENT_REVIEW_2026_08_05.md) |
| 2 | **ES direction screen** | One raw, no-model replay of the frozen M1/M3 family on owned ES | G1 | **Released by rung 1 under "large edge or stop"** | [`work/g1-direction/`](work/g1-direction/) |
| 3 | **Option-dollar replay** | One locked 60-minute replay of the G1 policy in SPXW dollars | G2 | An exact 60-minute G1 pass | not yet registered |
| 4 | **Track-A capture and feature certification** | Multi-session OPRA arrival and freshness evidence; a re-derived emission lag; a signed feature ledger | G3 | Fresh owner authorization — vendor contact and unattended jobs are Tier 1 | [requirement finding](research/findings/TRACK_A_ARRIVAL_CAPTURE_REQUIREMENT_2026_08_05.md) |
| 5 | **Bounded entry search** | A frozen shallow entry ranker, chronological out-of-fold | G4 | A G1 pass **and** separate owner training authorization | not yet registered |
| 6 | **Exit search on a frozen entry stream** | An exit policy fitted only after entries are locked | G4 | Rung 5 complete and frozen | not yet registered |
| 7 | **Validation packet** | Authoritative `trades.csv` and manifest, SPX entry/exit chart, equity curve | G5 | Rung 6 complete; the corrected gate in [`validation/replay_gate.py`](research/validation/replay_gate.py) | not yet registered |
| 8 | **Runtime parity** | Same-input reproduction at `1e-12` absolute / `0` relative | G6 | A validated candidate | not yet registered |
| 9 | **No-order live shadow** | 20 consecutive sessions deciding live and submitting nothing | G7 | A parity-certified candidate | not yet registered |
| 10 | **Guarded paper** | Pre-registered forward sessions with real paper orders | G8 | Owner authorization; broker contact is Tier 1 | not yet registered |
| 11 | **Real money** | Out of research scope | G9 | Owner decision only | — |

Two preconditions cut across rungs 5–7 and are built but not satisfied:

- **Arrival parity.** The corpus records zero arrival latency on all 47,707,186 rows while live measured
  227 ms at the median. See [the arrival-parity finding](research/findings/HISTORICAL_ARRIVAL_PARITY_2026_08_05.md).
  Rung 4 must supply matching usable latency and sparse-minute/freshness receipts.
- **Frozen knobs.** [`research/knobs.py`](research/knobs.py) records which constants a search may vary and
  which are frozen. Any search touching a `FROZEN` or `UNCERTIFIED` knob is refused.

## 3. Where we actually are

**The plumbing works. We do not yet have a reason to trade.** Five research campaigns returned no edge.
The immediate uncertainty is more basic: the data may be unable to measure the size of edge the project
would plausibly find.

The only active research question is:

> Can we predict SPX/ES direction over 15–60 minutes well enough to clear 0.358 ES points per trade?

Until that is answered, option-model training cannot create useful information and is prohibited.

## 4. The measurement problem

With 254 owned ES sessions, the optimistic smallest detectable edge at 80% power is:

| Trades per session | Detectable points per trade | Relative to 0.358-point cost |
|---:|---:|---:|
| 1 | 2.104 | 5.9× |
| 6 | 0.859 | 2.4× |
| 26, the theoretical 15-minute maximum | 0.413 | 1.2× |

The sample can ask “is there a large edge?” but cannot reliably ask “is there any edge?” The protected
holdout is spent, the proposed replacement range was already used, and forward confirmation may require
roughly one year to multiple decades depending on effect size.

**Independently verified 2026-08-05 (job 2): every number above reproduces exactly from the raw owned
bars, and the independence assumptions are slightly conservative at lag 1 (measured session-to-session
autocorrelation is mildly negative). Verdict B — proceed under "large edge or stop."** Evidence:
[measurement review](research/findings/MEASUREMENT_REVIEW_2026_08_05.md) and the
[gate-chain audit](research/findings/GATE_CHAIN_AUDIT_2026_08_05.md).

## 5. Measured cost bars

| Item | Measured value |
|---|---:|
| ES futures round-trip friction | **0.358 points / $17.92** |
| SPX option round-trip friction | **$3.08** |
| Option friction as share of $565 average premium | **4.68%** |

A candidate that does not beat measured costs is not a strategy.

## 6. What is proven

- Guarded IBKR paper execution works on a paper account; this proves plumbing, not predictive edge.
- Safety guards refuse real-money accounts and missing acknowledgements.
- Live OPRA market-data capture and contract qualification work.
- The owned Path-D corpus contains 254 usable ES sessions and 251 option sessions. Its data stays outside
  v5 and is indexed rather than copied.
- In one paired session, all 914 observed live OPRA CBBO-1m rows matched Historical API rows after decode.
  This proves one-session value identity, not a multi-session latency bound or full feature parity.
- The training corpus contains **no arrival latency at all**: `receive_time` equals `event_time` on all
  47,707,186 rows across all 251 sessions, while the live CBBO-1m stream measured 226.932 ms at the median
  and 319.521 ms at p99. Training on these rows unrepaired would assume timing the live system cannot
  deliver. Evidence: [arrival-parity finding](research/findings/HISTORICAL_ARRIVAL_PARITY_2026_08_05.md).
- Current-session definitions and raw OSI contract identity are required; all 492 prior-session instrument
  IDs changed in the observed session.
- Native v5 clock, parity, feature-admission, and candidate-packet controls now exist. They are model-free,
  network-free building blocks and do not authorize training.
- The already-granted Python interpreter could run unattended and read this repository. **This is now
  UNKNOWN**: the macOS Documents grant recorded in the TCC database was keyed to that interpreter's
  absolute path under the old home directory, which the 2026-08-06 rename destroyed. It must be re-read
  before anything relies on it. No real job is loaded — the two Track-A jobs were unloaded on owner
  authorization 2026-08-06 and the two 2026-07-18 agents exit 78. See [§14](#14-home-directory-rename-2026-08-06).
- The prior-art refusal machinery blocks unchanged retries of closed ideas.
- Costs above are measured rather than assumed.

Evidence paths are catalogued in [the evidence index](evidence/INDEX.md).

### What is built for the future v5 workflow

- [`research/training_twin.py`](research/training_twin.py) applies one causal clock law to supplied
  historical and live rows, handles current-session definitions and warm-up, and emits content-addressed
  paired-parity receipts.
- [`research/feature_admission.py`](research/feature_admission.py) verifies ledger and receipt bytes and
  blocks unknown, barred, stale, or missing model inputs before a feature matrix is returned. The legacy
  ledger verifies as historical evidence but lacks validity windows and therefore cannot authorize a new
  fit.
- [`research/validation/candidate_packet.py`](research/validation/candidate_packet.py) exports a
  deterministic trade CSV, SPX entry/exit chart, equity curve, summary, and provenance manifest from an
  already-produced serial trade ledger. It rejects training rows, time leakage, overlapping positions,
  and cross-session 0DTE trades.
- [`research/knobs.py`](research/knobs.py) is the machine-readable registry of what a search may vary. A
  `FROZEN` constant carries its evidence path and the exact condition that would unfreeze it; an
  `UNCERTIFIED` one may be neither varied nor trusted; `SEARCHABLE` ranges open only when their gate does.
- [`research/validation/replay_gate.py`](research/validation/replay_gate.py) is the corrected economic
  gate. It requires a pre-run power receipt, absolute lower bounds at every fee and latency cell, a
  comparator frozen before evaluation, 4-of-5 folds on both absolute and paired results, and four negative
  controls that must each fail the same full gate.
- Vetted Path-D replay, experiment-control, capture, and paper-execution foundations remain frozen in v4
  and are classified in [TOOLBOX.md](TOOLBOX.md). They are not active v5 dependencies.

The complete dated inventory is the
[v5 workflow capability audit](research/findings/V5_WORKFLOW_CAPABILITY_AUDIT_2026_08_05.md).

## 7. What is closed

| Closed family | Finding |
|---|---|
| 0DTE long premium at minute cadence, any model or features | About −$13/trade before costs; negative in all five folds |
| SPX position within session range as a direct signal | −0.324 ES points/trade on raw data |
| Options-chain risk-reversal direction | Failed the five-fold gate |
| Prior 18-feature entry contract | No ranking power; deciles were flat and non-monotonic |
| Six-arm exit repair family | Every arm lost to its comparator |

The binding details and legitimate reopening conditions are in the
[do-not-retest ledger](research/history/DO_NOT_RETEST.md).

## 8. Gate chain

| Gate | Meaning | Current state |
|---|---|---|
| G1 Direction | Predict ES direction and clear 0.358 points/trade | **HOLD for measurement review; only a large edge is detectable** |
| G2 Option wrapper | Replay one locked 60-minute G1 policy in option dollars | Blocked by a 60-minute G1 pass |
| G3 Feature certification | Prove option features exist live at decision time | **8/73 scoped features currently admitted.** Phase-2 is ready to re-issue 51/73 only after v8 evidence banks; no new ledger exists yet |
| G4 Train | Fit a shallow model on admitted features | Blocked by G1 and separate owner authorization |
| G5 Validation | Beat comparator with positive confidence, 4/5 folds, and clean controls | Corrected gate rebuilt natively; the defective v4 replay now refuses to run |
| G6 Runtime parity | Reproduce offline decisions live | Existing tolerance frozen at 1e-12 absolute / 0 relative |
| G7 Live shadow | Run live and submit nothing | Not built; needs G1–G6 |
| G8 Guarded paper | Pre-registered paper orders on fresh sessions | No clean historical firewall; calendar-limited |
| G9 Real money | Separate safety and owner decision | Out of scope |

## 9. G5 validation is defective

Four confirmed defects in `v4/research/pathd_phase1_replay.py` threatened any future positive result:

1. The fee sensitivity cancels because policy and comparator receive the same fee.
2. The `time_shifted` control uses the next row through `shift(-1)`, importing future information.
3. “Positive skill” passes when only one of five folds is positive.
4. “Powered” checks row labels, not statistical power.

These defects do not weaken past negative results; they made a future pass untrustworthy.

**Repaired 2026-08-05.** The corrected gate is native v5 code at
[`research/validation/replay_gate.py`](research/validation/replay_gate.py); each defect has a regression
test proving the old behaviour is rejected. The v4 module was not edited into correctness — it stays as
the record of what past runs actually computed, and now refuses to execute so it cannot be used by
mistake. No past result changes; no candidate has been run through the new gate.

## 10. Current operating state

- **No scheduled job can run this repository's code.** `~/Documents` is protected by macOS and only a few
  granted programs may read it; a scheduled job has no parent whose permission it can inherit. This blocks
  the later live-shadow and guarded-paper rungs as well as Track-A capture. See
  [requirement finding §5](research/findings/TRACK_A_ARRIVAL_CAPTURE_REQUIREMENT_2026_08_05.md#5-the-blocker--scheduled-jobs-cannot-read-this-repository).
- **The home directory was renamed `gduby` → `och` on 2026-08-06, breaking every absolute path into the
  old home.** Real directories survived; absolute symlinks and hardcoded paths did not. Repaired the same
  day: both Python environments, the five Track-A shell scripts (now derived from their own location, so
  the pending repo migration cannot break them again), and the corpus paths in the G1 declaration and the
  corpus tests. See [§14](#14-home-directory-rename-2026-08-06).
- The two background agents loaded 2026-07-18 **no longer execute** — both exit 78, because their plists
  name the interpreter, script, `PYTHONPATH`, and working directory under the vanished home. The
  runtime venv they use was repaired on owner authorization 2026-08-06, which does **not** revive them:
  their plists still point at `/Users/gduby/...`. Rewriting those plists is an owner decision and has not
  been made, so they remain dead. They contacted nothing while working and contact nothing now.
- The two Track-A launchd capture jobs were **unloaded and their plists disabled on owner authorization
  2026-08-06** (`launchctl bootout`, plists renamed `*.disabled_20260806_stale_v6_dates`, fully
  recoverable). They were built for the spent v6 dates and fired Wednesday/Thursday/Friday, so 08-12
  would have collided with the attended runner. Track-A capture is attended-only.
- Track-A capture has **banked no evidence yet, and the runner is now armed and running.** It was started
  on owner authorization at **2026-08-09 08:45 PDT** on AC power, holds a verified `PreventSystemSleep`
  assertion, and is waiting to fire all six windows across **2026-08-10, 08-11 and 08-12**. The v6 dates
  08-06/08-07 are spent: 08-06 failed both windows and 08-07 was canceled by the owner. The replacement
  sessions are sealed in `capture_declaration_v8.json`, and analysis and all four runners are now pinned
  to that one file by test. The owner signed four sessions on 2026-08-06 (`authorization_v2.json`,
  declaration v7); v8 narrows that to three by owner instruction the same day, decided **before any of
  those sessions was observed**, so nothing was dropped after seeing an outcome. Narrowing needs no new
  signature; widening would. `authorization_v2.json` is deliberately left unedited as the record of what
  was signed. See [§13](#13-track-a-capture-2026-08-06-failure-and-08-10-arming).
  2026-08-05 remains excluded: its open window was lost to the permission defect and its midday window is
  an infrastructure test, not evidence.
- **Nothing automated arms this capture.** The runner is hand-started and monitors are advisory only. Any
  scheduled reminder that lives inside a desktop application depends on that application still running and
  is not an OS-level guarantee; it reports, it does not start. Treat "armed" as true only when
  `./v4/ops/tracka/check_tracka.sh` prints `RUNNER ALIVE` together with `SLEEP held off`.
- **A forward confirmation reservation is in force, signed 2026-08-05.** Every ES and SPXW session
  from **2026-08-06 onward is confirmation-only**: no research analysis, screen, model, chart, or
  summary may compute strategy economics on those sessions until a pre-registered protocol opens the
  reserve once. Development data ends **2026-08-05**. Infrastructure evidence that evaluates no
  policy — arrival latency, feed parity, capture completeness, including Track-A on 08-06/08-07 — is
  explicitly permitted. Full rules: [reservation declaration](governance/FORWARD_CONFIRMATION_RESERVATION_2026_08_06.md).
- No model is accepted for current research promotion.
- No training, data purchase, broker contact, paper submission, or runtime mutation is authorized by this
  status page.

## 11. Core paths

| Path | Purpose |
|---|---|
| [`README.md`](README.md) | Five-minute project front door |
| [`AGENTS.md`](AGENTS.md) | Binding working and safety rules |
| [`TOOLBOX.md`](TOOLBOX.md) | Vetted tools, legacy dependencies, and safety classes |
| [`work/`](work/) | Registered active work packets only |
| [`research/findings/V5_WORKFLOW_CAPABILITY_AUDIT_2026_08_05.md`](research/findings/V5_WORKFLOW_CAPABILITY_AUDIT_2026_08_05.md) | Dated map of built, proven, disproven, and missing workflow pieces |
| [`research/history/DO_NOT_RETEST.md`](research/history/DO_NOT_RETEST.md) | Closed experiments and reopening conditions |
| [`research/knobs.py`](research/knobs.py) | What a search may vary, what is frozen, and what is uncertified |
| [`evidence/INDEX.md`](evidence/INDEX.md) | Map to protected external evidence |
| [`ops/check_project.py`](ops/check_project.py) | Read-only structure check for both v5 and the repository |

## 12. Deferred cleanup

Not blocking, recorded so it is not rediscovered as new:

- `v2/` and `v3/` still contain stray planning and handoff files. They are closed eras that nothing writes
  to, so they were reported rather than quarantined. `check_project.py` names them in `CLOSED_ERA_TREES`.
- Two numbered sync duplicates sit inside protected evidence
  (`v4/audit/es_vwap/coverage 2.md` and one under `v4/audit/autoresearch/`). Evidence is never moved to
  satisfy a naming rule; removing them is an owner decision.
- The git object store holds 455 loose object filenames ending in `" 2"`. All referenced objects are
  present and none is missing, so this is disk clutter rather than corruption.

## 13. Track-A capture: 2026-08-06 failure and 08-10 arming

**Why this capture exists, stated wider than before.** It has been described as "get a latency number and
unblock 43 features." That is true and undersells it. Track-A is the **first systematic measurement of
how the training data differs from what the live system sees** — latency is simply the axis measured
first, and the same recording also settles sparse-minute coverage and confirms universe composition. Three
of the sixteen axes in [§15](#15-the-trainlive-divergence-register) are settled or repaired by this one
capture. Nothing about the recording changes; only the stated purpose is wider.

The first attended capture banked **no evidence**. Both causes are fixed in the v7 draft; neither was a
data or capture-code problem.

| What happened | Why |
|---|---|
| The open window fired 06:49:38 PDT instead of 06:28:00 — 1,298 s late | The Mac was moved to battery power. `caffeinate -s` is valid **only on AC power**; on battery the assertion is void and the machine slept. A 06:49 start records 09:49 ET, which is not the declared 09:30 bell. |
| That window then failed anyway | The definitions capture found **574** current-session SPXW symbols; the wrapper pinned **510**; the recorder failed closed with "do not trim the universe" and exited 1. The 0DTE strike listing changes daily — it was 510 on 08-05. |
| The midday window was skipped | The open wrapper did not return until 09:48:08 PDT because the machine slept mid-run, by which time the 09:10 start had passed by 2,288 s. |
| 2026-08-07 did not run | Canceled by the owner on 08-06; the laptop is away for the weekend. The armed runner was stopped at owner instruction. |

**The fail-closed universe check worked as designed** and prevented a silently trimmed capture. The defect
was pinning a daily-varying count in advance, not the check itself. Note what this means about the first
cause: had the universe happened to be 510 that morning, a capture recording 09:49 ET would have banked
labelled `open`, and only an unrelated guard stopped it. Both holes are now closed in code — the runner
refuses to start on battery, warns on any battery transition while waiting, and **skips** any window that
misses its declared start by more than 60 seconds rather than capturing a different window under the
declared name.

The replacement declaration therefore takes the complete current-session universe at whatever size it is
on the day — recording `symbol_count` and `symbols_sha256` rather than asserting a count in advance — and
records AC power and an open lid as operational preconditions. The owner signed four sessions
(08-10…08-13) on 08-06 and narrowed them to three (08-10/11/12) the same day, before any of those
sessions was observed.

Three sessions cannot support a population worst-case claim: the max of `n` days exceeds the daily
`q`-th percentile with probability `1 − q^n`, which is 14.3% at `q=0.95` and 3.0% at `q=0.99` for `n=3`
(18.5% and 3.9% at `n=4`). Certification wording stays limited to the observed three-session envelope. On
current evidence the practical difference is likely nil: the guard is `L = max(10,000 ms, 4 × worst p99)`,
and the measured CBBO-1m p99 of 527.622 ms on 08-05 leaves the 10-second floor binding unless a session's
p99 exceeds 2,500 ms.

**Phase-2 is ready as of 2026-08-08, but has issued nothing while the bank is empty.** The analysis and
capture runners are pinned by test to the same v8 declaration. A single local, model-free command now
recomputes every banked declared window, refuses partial or non-reproducing windows, signs separate latency
and sparse-minute/freshness receipts, applies the fixed guard above, and re-issues the ledger without
overwriting the legacy file:

```text
./.venv/bin/python -m v5.ops.certify_tracka_arrival --measured-on YYYY-MM-DD
```

The receipt expiry is pre-registered at **2026-11-10** and is not a command-line choice. Output goes to a
new `phase2_issuance_YYYY-MM-DD/` directory under the Track-A audit tree. The command adapts to the actual
number `N` of sessions represented by whatever windows bank: only `N=0` is fatal, and it prints
`1 − 0.95^N` and `1 − 0.99^N` alongside the required observed-`N` wording. Until a dated issuance exists,
G3 remains 8/73; the expected mechanical result after usable CBBO-1m evidence is 51/73 scoped features
(51/83 total), leaving 32 barred.

### The monitor was lying about the one thing that killed 08-06

Found and fixed 2026-08-09, while arming. `check_tracka.sh` reported
`*** no PreventSystemSleep assertion held ***` while `caffeinate` was demonstrably holding that exact
assertion — a **false negative on the precise failure that lost the 08-06 session.**

The cause is `pmset -g assertions | grep -q "..."` under `set -o pipefail`. `grep -q` exits at its first
match, `pmset` is then killed writing to a closed pipe (SIGPIPE, status 141), and `pipefail` promotes 141
to the pipeline's status, so a *successful* match reads as failure. Measured on 2026-08-09: **20 failures
in 20 runs.**

The same idiom guarded three other checks, including the battery **refusal** in `start_tracka.sh` — which
could have refused a valid start on AC power. Those survived only by luck: `pmset -g batt` output is small
enough that it finishes writing before `grep` exits, measured 0 failures in 200 runs. All four now read
`pmset` into a variable and match in-shell, so no pipeline exists to fail.

The lesson generalises past this script: **a safety check that fails open is worse than no check**, and a
monitor that cries wolf trains its reader to ignore the one alarm that matters.

## 14. Home-directory rename, 2026-08-06

The macOS account home was renamed `gduby` → `och`. Real directories and all data survived; every
absolute path pointing into the old home stopped resolving. 153 files in the repository still contain the
old prefix — most are receipts and findings, which are records of what was run and are **never edited**.

What it broke, and what was done, all on 2026-08-06:

| Broken | State |
|---|---|
| Both Python environments — `.venv` and the runtime venv — had dangling interpreter symlinks | **Repaired.** The runtime-venv repair was owner-authorized. `./.venv/bin/python` works; `check_project.py` and the 145 tests run again |
| `start_tracka.sh` and four sibling scripts hardcoded the repo path; the Sunday start would have exited 77 | **Repaired.** All five now derive the repo from their own location, so the pending repo migration cannot break them either |
| The G1 declaration's corpus root, which is inside the hashed freeze | **Re-frozen** on owner authorization: `5ec8b5a4…` → `f43b92c2…`, one field of 213, no outcome inspected. [Record](work/g1-direction/REFREEZE_2026_08_06.md) |
| Three corpus tests silently **skipped** while the suite still reported success | **Repaired.** Skip conditions are home-relative; the suite is 145 passed, 0 skipped |
| Two Track-A launchd jobs on stale Wednesday/Thursday/Friday dates | **Unloaded and disabled** on owner authorization; 08-12 would have collided with the attended runner |
| Two 2026-07-18 background agents | **Still dead** (exit 78). Their plists name the old home; rewriting them is an owner decision not yet made |
| The TCC Documents grant for the unattended interpreter | **UNKNOWN** — keyed to the old absolute path. Blocks job 4a until re-read |

The lesson worth keeping is the fifth row: an absolute path inside a test's skip condition turns a broken
environment into a green test run. The failure was invisible until the skip reasons were printed.

## 15. The train/live divergence register

**Every gate tests the model. None systematically tested whether the training data describes the same
game the live system plays.** G6 comes closest and is still not it: G6 is a *same-input* test — identical
inputs in, identical decisions out. Every divergence of this class lives upstream of it, in how the inputs
are built. If historical and live inputs are constructed differently, G6 passes perfectly and the model is
still learning a game nobody plays.

Two instances were already known — the corpus stores zero arrival lag on all 47,707,186 rows, and the
emission allowance comes from the wrong feed — but both were found by stumbling into them. "Look harder"
is not a control, so [`research/divergence.py`](research/divergence.py) makes it one: 16 axes, each with a
status, its evidence, the gate it binds at, and the exact condition that would settle it. An axis with no
exit condition is refused at import. `assert_no_unknown_on_path` refuses a fit that depends on an
unchecked axis, and refuses an axis name nobody declared — because the failure it exists to prevent is a
divergence nobody wrote down.

| Status | Count | Meaning |
|---|---:|---|
| `PROVEN_EQUAL` | 4 | Checked, with evidence |
| `MEASURED_DIFFERENT` | 5 | Known to differ; **blocks unless it names its repair** |
| `UNKNOWN` | 7 | Nobody has checked |

Seven axes currently block a fit. That is the honest number, and it is expected to be large on day one.

### What building it immediately found

- **Bar labelling is confirmed and now asserted.** A bar labelled 09:35 covers 09:35:00–09:36:00; all 254
  owned sessions open with a 09:30 bar. G1 was already correct, but by convention agreement rather than
  assertion — a flipped convention would shift every feature one minute and still look plausible. The
  loader now raises instead of assuming.
- **Nine sessions are short.** Seven close at 13:00 and two at 13:15. G1's horizons exit at 09:50, 10:05
  and 10:35, all clearing the earliest short close by more than two hours, so the frozen family is
  unaffected. A longer horizon would need a declared rule first.
- **Seven of G1's 254 sessions cannot be traded in the product.** ES trades a shortened session on US
  equity-market holidays when SPXW does not trade at all. Verified against the owned corpus: those seven
  sessions (2.8% of the M1 index, six of the 249 gap-eligible) have no SPXW option session whatsoever.

### The owner decision this raises

G1 measures ES direction **in order to justify buying SPXW options**. On those seven sessions the option
does not exist, so any edge measured there cannot be taken in the product. The question is whether the G1
index should exclude sessions on which the traded instrument does not trade.

No G1 outcome has been inspected, so deciding this now is still a **pre-outcome** narrowing rather than a
post-hoc one — the same standing that made the 08-06 re-freeze legitimate. It would change the frozen
family hash a second time. It is the owner's call, not the agent's, and G1's surrogate campaign is not
blocked while it is pending.

*Update this page when a job or gate changes. Do not create another status, roadmap, or gate file.*
