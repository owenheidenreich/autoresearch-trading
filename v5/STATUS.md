# V5 Project Status — the one page

**Last updated: 2026-08-05.** This is the only current-state, job-register, and gate-chain document.
If another file disagrees, this page wins and the conflict must be reported.

> **The current job is an independent measurability review.** Do not implement or run the G1 direction
> screen until that review decides whether the owned sample can support a useful conclusion.

## 0. Job register

A committed work packet must appear here. No row means no job.

| # | Job | Gate | State | Waiting on | Work packet |
|---|---|---|---|---|---|
| 1 | Build the clean v5 project boundary | infrastructure | **DONE 08-05** | — | this page and [v5 front door](README.md) |
| 2 | Independent review: is the project measurable? | G1/G4/G5/G8 | **QUEUED — current job** | Independent reviewer | [`v5/work/measurement-review/`](work/measurement-review/) |
| 3 | ES direction screen: opening range and overnight gap | G1 | **HOLD — do not implement or run until job 2 answers** | Job 2 | [`v5/work/g1-direction/`](work/g1-direction/) |
| 4 | Track-A option-feature arrival capture | G3 | **HOLD — prior dates canceled; all jobs unloaded** | Project-structure checkpoint, then new owner authorization | legacy evidence indexed in [TOOLBOX](TOOLBOX.md) |
| 5 | Repair four defects in the validation gate | G5 | **OPEN, unassigned** | G1 before economic use | [§7](#7-g5-validation-is-defective) |
| 6 | Rebuild a confirmation firewall | G8 | **BLOCKED — no clean date range identified** | Job 2 | [§3](#3-the-measurement-problem) |
| 7 | Programme restart after the 08-04 stop | all | **CLOSED 08-05** | — | [restart record](governance/PROGRAM_RESTART_RECORD_2026_08_05.md) |
| 8 | Quarantine stale v4 documentation | infrastructure | **DONE 08-05 — 117 files preserved** | — | [manifest](../_cleanup_quarantine/2026-08-05-docs/MANIFEST.md) |

## 1. What we are building

An automated day-trading bot that buys SPX 0DTE calls or puts when a model has defensible evidence about
the next 15–60 minutes of SPX direction. It must be causal, clear measured costs, survive validation, and
prove itself in guarded paper trading before any real-money decision.

## 2. Where we actually are

**The plumbing works. We do not yet have a reason to trade.** Five research campaigns returned no edge.
The immediate uncertainty is more basic: the data may be unable to measure the size of edge the project
would plausibly find.

The only active research question is:

> Can we predict SPX/ES direction over 15–60 minutes well enough to clear 0.358 ES points per trade?

Until that is answered, option-model training cannot create useful information and is prohibited.

## 3. The measurement problem

With 254 owned ES sessions, the optimistic smallest detectable edge at 80% power is:

| Trades per session | Detectable points per trade | Relative to 0.358-point cost |
|---:|---:|---:|
| 1 | 2.104 | 5.9× |
| 6 | 0.859 | 2.4× |
| 26, the theoretical 15-minute maximum | 0.413 | 1.2× |

The calculation assumes trades within a session are independent, which is optimistic. The sample can ask
“is there a large edge?” but cannot reliably ask “is there any edge?” The protected holdout is spent, the
proposed replacement range was already used, and forward confirmation may require roughly one year to
multiple decades depending on effect size. The independent review must attack these claims before they
control the programme. Full evidence: [gate-chain audit](research/findings/GATE_CHAIN_AUDIT_2026_08_05.md).

## 4. Measured cost bars

| Item | Measured value |
|---|---:|
| ES futures round-trip friction | **0.358 points / $17.92** |
| SPX option round-trip friction | **$3.08** |
| Option friction as share of $565 average premium | **4.68%** |

A candidate that does not beat measured costs is not a strategy.

## 5. What is proven

- Guarded IBKR paper execution works on a paper account; this proves plumbing, not predictive edge.
- Safety guards refuse real-money accounts and missing acknowledgements.
- Live OPRA market-data capture and contract qualification work.
- The already-granted Python interpreter can run unattended and read this repository. All real jobs remain
  unloaded after the owner canceled the prior Track-A dates.
- The prior-art refusal machinery blocks unchanged retries of closed ideas.
- Costs above are measured rather than assumed.

Evidence paths are catalogued in [the evidence index](evidence/INDEX.md).

## 6. What is closed

| Closed family | Finding |
|---|---|
| 0DTE long premium at minute cadence, any model or features | About −$13/trade before costs; negative in all five folds |
| SPX position within session range as a direct signal | −0.324 ES points/trade on raw data |
| Options-chain risk-reversal direction | Failed the five-fold gate |
| Prior 18-feature entry contract | No ranking power; deciles were flat and non-monotonic |
| Six-arm exit repair family | Every arm lost to its comparator |

The binding details and legitimate reopening conditions are in the
[do-not-retest ledger](research/history/DO_NOT_RETEST.md).

## 7. Gate chain

| Gate | Meaning | Current state |
|---|---|---|
| G1 Direction | Predict ES direction and clear 0.358 points/trade | **HOLD for measurement review; only a large edge is detectable** |
| G2 Option wrapper | Replay one locked 60-minute G1 policy in option dollars | Blocked by a 60-minute G1 pass |
| G3 Feature certification | Prove option features exist live at decision time | Parallel branch on hold; not on G1’s path |
| G4 Train | Fit a shallow model on admitted features | Blocked by G1 and separate owner authorization |
| G5 Validation | Beat comparator with positive confidence, 4/5 folds, and clean controls | Defective; repair required |
| G6 Runtime parity | Reproduce offline decisions live | Existing tolerance frozen at 1e-12 absolute / 0 relative |
| G7 Live shadow | Run live and submit nothing | Not built; needs G1–G6 |
| G8 Guarded paper | Pre-registered paper orders on fresh sessions | No clean historical firewall; calendar-limited |
| G9 Real money | Separate safety and owner decision | Out of scope |

### 7. G5 validation is defective

Four confirmed defects threaten any future positive result:

1. The fee sensitivity cancels because policy and comparator receive the same fee.
2. The `time_shifted` control uses the next row through `shift(-1)`, importing future information.
3. “Positive skill” passes when only one of five folds is positive.
4. “Powered” checks row labels, not statistical power.

These defects do not weaken past negative results; they make a future pass untrustworthy until repaired.

## 8. Current operating state

- Nothing runs unattended.
- Track-A jobs are unloaded and the previous capture dates are canceled.
- No model is accepted for current research promotion.
- No training, data purchase, broker contact, paper submission, or runtime mutation is authorized by this
  status page.

## 9. Core paths

| Path | Purpose |
|---|---|
| [`README.md`](README.md) | Five-minute project front door |
| [`AGENTS.md`](AGENTS.md) | Binding working and safety rules |
| [`TOOLBOX.md`](TOOLBOX.md) | Vetted tools, legacy dependencies, and safety classes |
| [`work/`](work/) | Registered active work packets only |
| [`research/history/DO_NOT_RETEST.md`](research/history/DO_NOT_RETEST.md) | Closed experiments and reopening conditions |
| [`evidence/INDEX.md`](evidence/INDEX.md) | Map to protected external evidence |

*Update this page when a job or gate changes. Do not create another status, roadmap, or gate file.*
