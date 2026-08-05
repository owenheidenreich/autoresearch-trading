# V5 Project Status — the one page

**Last updated: 2026-08-05.** This is the only current-state, job-register, and gate-chain document.
If another file disagrees, this page wins and the conflict must be reported.

> **The measurement review returned verdict B on 2026-08-05: only a large edge (~2.2–4.0 net
> points/session) is detectable on the owned year.** The owner accepted "large edge or stop," so the
> G1 direction screen is **released** under that reduced claim. The current job is implementing and
> running G1 per [`work/g1-direction/PLAN.md`](work/g1-direction/PLAN.md).

## 0. Job register

A committed work packet must appear here. No row means no job.

| # | Job | Gate | State | Waiting on | Work packet |
|---|---|---|---|---|---|
| 1 | Build the clean v5 project boundary | infrastructure | **DONE 08-05** | — | this page and [v5 front door](README.md) |
| 2 | Independent review: is the project measurable? | G1/G4/G5/G8 | **DONE 08-05 — verdict B: only a large edge is detectable; limits verified from raw data and slightly conservative at lag 1** | — | [finding](research/findings/MEASUREMENT_REVIEW_2026_08_05.md), packet [`v5/work/measurement-review/`](work/measurement-review/) |
| 3 | ES direction screen: opening range and overnight gap | G1 | **RELEASED 08-05 — current job; targets a large edge under "large edge or stop"** | Implementation | [`v5/work/g1-direction/`](work/g1-direction/) |
| 4 | Track-A option-feature arrival capture | G3 | **RUNNING ATTENDED — 0 of 2 evidence sessions banked** | The 08-06 and 08-07 recordings | [requirement finding](research/findings/TRACK_A_ARRIVAL_CAPTURE_REQUIREMENT_2026_08_05.md) |
| 4a | Unattended jobs cannot read this repository | blocks G3/G7/G8 | **OWNER DECIDED 08-05 — move approved; manifest drafted; execute after Track-A banks** | Owner executing the [migration manifest](governance/REPO_MIGRATION_MANIFEST_2026_08_05.md) | [requirement finding §5](research/findings/TRACK_A_ARRIVAL_CAPTURE_REQUIREMENT_2026_08_05.md#5-the-blocker--scheduled-jobs-cannot-read-this-repository) |
| 5 | Repair four defects in the validation gate | G5 | **DONE 08-05 — rebuilt natively** | — | [`validation/replay_gate.py`](research/validation/replay_gate.py), [§9](#9-g5-validation-is-defective) |
| 6 | Rebuild a confirmation firewall | G8 | **FORWARD DESIGN DECIDED 08-05 — reservation from 2026-08-06 drafted** | Owner signature on the [reservation declaration](governance/FORWARD_CONFIRMATION_RESERVATION_2026_08_06.md) | [§4](#4-the-measurement-problem) |
| 7 | Programme restart after the 08-04 stop | all | **CLOSED 08-05** | — | [restart record](governance/PROGRAM_RESTART_RECORD_2026_08_05.md) |
| 8 | Quarantine stale v4 documentation | infrastructure | **DONE 08-05 — 117 files preserved** | — | [manifest](../_cleanup_quarantine/2026-08-05-docs/MANIFEST.md) |
| 9 | Inventory and promote training-readiness controls | infrastructure | **DONE 08-05** | — | [capability audit](research/findings/V5_WORKFLOW_CAPABILITY_AUDIT_2026_08_05.md) |
| 10 | Retire the documentation drawers and enforce it repo-wide | infrastructure | **DONE 08-05 — 75 files preserved** | — | [manifest](../_cleanup_quarantine/2026-08-05b-docs/MANIFEST.md), [`ops/check_project.py`](ops/check_project.py) |
| 11 | Build the two training preconditions | G3/G4 | **DONE 08-05 — built, not satisfied** | Rung 4 must supply a latency receipt | [`research/knobs.py`](research/knobs.py), [arrival finding](research/findings/HISTORICAL_ARRIVAL_PARITY_2026_08_05.md) |
| 12 | Close six control-machinery gaps found by the training-readiness review: gate-pass receipts, computed power bound to the session index, trades→session aggregator, entry-freeze lock, knob-registry expansion, pipeline composition test | G1–G6 controls | **DONE 08-05 — 97 tests green** | — | [`research/gate_receipts.py`](research/gate_receipts.py), [`research/entry_stream.py`](research/entry_stream.py), [`research/validation/session_index.py`](research/validation/session_index.py) |

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
| 4 | **Track-A capture and feature certification** | Multi-session OPRA arrival evidence; a re-derived emission lag; a signed feature ledger | G3 | Fresh owner authorization — vendor contact and unattended jobs are Tier 1 | [requirement finding](research/findings/TRACK_A_ARRIVAL_CAPTURE_REQUIREMENT_2026_08_05.md) |
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
  Rung 4 must supply the missing latency receipt.
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
- The already-granted Python interpreter can run unattended and read this repository. All real jobs remain
  unloaded after the owner canceled the prior Track-A dates.
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
| G3 Feature certification | Prove option features exist live at decision time | Native enforcement exists; evidence branch on hold; only 8/73 scoped features admitted in the legacy ledger |
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
- Two long-running background agents remain loaded from 2026-07-18 and do execute on schedule, because
  their program is the one Python interpreter that holds the permission grant. They are self-gated and
  contact nothing. Removing them is an owner decision.
- Track-A capture dates are **2026-08-06 and 2026-08-07**, declared in `capture_declaration_v6.json` and
  driven by an attended runner started by hand. 2026-08-05 was deliberately excluded: its open window was
  lost to the defect above and its midday window is classified as an infrastructure test, not evidence.
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

*Update this page when a job or gate changes. Do not create another status, roadmap, or gate file.*
