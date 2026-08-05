# Project Status — the one page

**Last updated: 2026-08-05.** This is the only status document. If another document disagrees with this
one, this one wins and the other one is stale. Written in plain English on purpose.

> **2026-08-05 — the bottleneck moved.** A gate-chain audit found that the problem is no longer "we haven't
> found a signal." It is that **we cannot measure a signal of the size we would plausibly find, and we
> could not confirm one if we did.** See §3a. This affects G1, G4, G5, G8 and the project's viability
> premise, so it is queued for independent review before it drives any decision.

---

## 0. Open jobs — the register

Every active piece of work, its state, and where it lives. **A job does not exist unless it is on this
table.** If you create a document, add its row here in the same change.

| # | Job | Gate | State | Waiting on | Document |
|---|---|---|---|---|---|
| 1 | **Fable review — is this project measurable at all?** | G1/G4/G5/G8 | **QUEUED** | Owner to send it | [FABLE_REVIEW_MEASUREMENT_CAPACITY](v4/docs/protocol101/training/research/FABLE_REVIEW_MEASUREMENT_CAPACITY_2026_08_05.md) |
| 2 | G1 direction screen (M1 opening range, M3 overnight gap) | G1 | **HOLD** — do not start until job 1 answers, so the question is framed before results are seen | Job 1 | plan in job 3's §3 |
| 3 | Codex gate-chain audit | all | **DONE** 08-05; headline verified independently | — | [PATHD_GATE_CHAIN_RESEARCH](v4/docs/protocol101/training/research/PATHD_GATE_CHAIN_RESEARCH_2026_08_05.md) |
| 4 | Track-A arrival capture, sessions 08-06 + 08-07 | G3 | **RUNNING** — attended runner started by owner 08-05, armed for both sessions | Nothing; leave the terminal open and the Mac awake | declaration v6 (audit tree) |
| 5 | 🔴 **BUG (HIGH) — nothing scheduled can read this repo** | infra | **OPEN.** Cause confirmed 08-05 (macOS permissions, not iCloud); the durable fix is not done and is an owner decision | **Owner** — see §9a | [HANDOFF_TRACKA_LAUNCHD](v4/docs/protocol101/training/research/HANDOFF_TRACKA_LAUNCHD_CANNOT_READ_REPO_2026_08_05.md) |
| 6 | Repair the G5 validation gate (4 defects) | G5 | **OPEN**, unassigned. Must land before any future replay is believed | — | §7a below |
| 7 | Rebuild a confirmation firewall | G8 | **BLOCKED** — no clean date range identified; the one proposed was already used | Job 1 | §3a limit 2 |
| 8 | Programme reopened after the 08-04 stand-down | — | **CLOSED** 08-05 | — | [restart record](v4/docs/protocol101/training/contracts/PATHD_PROGRAMME_RESTART_RECORD_2026_08_05.md) |

## 1. What we are building

An automated day-trading bot that buys SPX 0DTE call and put options — a machine-learning model that
decides which way the market is about to go, buys the matching option, and sells it the same day.

## 2. Where we actually are

**The plumbing works. We do not yet have a reason to trade.**

Everything mechanical has been built and proven: we can pull live market data, turn it into features,
make a decision, qualify a real option contract, and place a guarded paper order with the broker. What we
have never demonstrated is a way to predict the market that makes money after costs.

Five separate research campaigns have each returned "no edge." That is not a broken machine — that is the
machine honestly reporting that the things we tried do not work.

## 3. The single most important finding

On 2026-08-04 the project proved something that changes the order of the work:

> **Buying SPX 0DTE options and holding them minute-to-minute loses about $13 per trade *before you pay
> any commission or spread*.** It lost money in all 5 test folds. This is the option losing value as time
> passes — it is a property of the instrument, not of the prediction. **No model, no neural network, no
> feature set, and no exit rule fixes it.**

This is recorded as row 181 of the do-not-retest ledger.

**Read carefully, because this does not kill the goal.** The same ledger row lists what would legitimately
re-open it, and one condition is **"demonstrated directional skill on the underlying"** — meaning: show
you can predict which way SPX itself moves. Row 180 confirms the option version is closed *because*
direction has not yet cleared that bar.

So the honest position is:

> **We cannot yet predict which way SPX is going. Until we can, wrapping that non-prediction in 0DTE
> options just adds decay and friction on top of a coin flip. Once we can, the 0DTE call/put bot becomes
> buildable again.**

That reduces the whole project to **one question**:

> ### Can we predict SPX direction over 15–60 minutes well enough to clear measured trading costs?

Everything else is downstream of that answer.

## 3a. The measurement problem (new, 2026-08-05)

Two independent limits, both about **resolution** rather than ideas. Neither is fixed by a better model,
more features, or a bigger network.

**Limit 1 — we cannot detect a cost-scale edge in the data we own.**

With 254 ES sessions, the smallest edge detectable at 80% power depends on how often the strategy trades:

| Trades per session | Smallest detectable edge, points/trade | vs the 0.358 cost bar |
|---:|---:|---:|
| 1 | 2.104 | **5.9×** |
| 6 (the frozen session cap) | 0.859 | **2.4×** |
| 26 (theoretical 15-min maximum) | 0.413 | **1.2×** |

Verified independently: the session standard deviations (13.486 / 18.099 / 24.807 points at 15/30/60 min)
reproduce exactly from the raw files, and the effect-size arithmetic is conservative, not inflated.

Read it this way: **even trading as often as the instrument physically allows, this year of data cannot
resolve an edge that merely covers costs** — and the table is optimistic, because it assumes trades within
a session are independent, which they are not. The honest reading is that the owned sample can answer
*"is there a large edge?"* and cannot answer *"is there an edge?"*

**Limit 2 — we could not confirm an edge if we found one.** The protected holdout is spent, no clean
replacement date range has been identified (the range previously proposed turned out to be already used),
and forward paper confirmation is estimated at roughly one year to multiple decades depending on effect
size.

**Consequence.** A negative result at G1 no longer means "no edge exists." It means "this data cannot find
one." Those are different conclusions and must never again be reported as the same thing.

## 4. The cost bar every idea must clear

Costs are measured, not assumed:

| Thing | Measured value |
|---|---|
| Round-trip friction, ES futures | **$17.92 = 0.358 ES points** |
| ES spread | 1.04 ticks (1.07 in the top volatility quartile) |
| Round-trip friction, SPX options | **$3.08** ($1.54/side) |
| Friction as a share of a $565 average option premium | **4.68%** |

A candidate that does not beat 0.358 points per trade gross is not a strategy, however good its
statistics look.

## 5. What is genuinely proven (do not redo these)

- **Live execution works.** Guarded paper round trip on account `DU***40`, 2026-08-04. The contract we
  qualify live (`SPXW  260804C07775000`) matches the training-data format exactly, so the
  data-to-broker bridge is closed.
- **Safety guards work.** They refuse a real-money account, refuse without the enable flag, refuse without
  each acknowledgement. Verified by direct exercise, not just unit tests.
- **Live market data works.** Databento OPRA definitions + market capture ran end to end, twice. Real
  arrival timing, CBBO-1m p99: **584.6 ms** (08-04 smoke) and **527.6 ms** (08-05 midday, 510/510 symbols).
  Both are quiet windows; the stressed open bursts are what actually set the number.
- **The research machinery works, with one exception.** Bootstrap confidence bounds, multiple-testing
  correction, and the prior-art check that blocks re-running closed experiments are all tested against
  known-bad inputs where they must fail — and they do. **The G5 replay gate is the exception: it has four
  confirmed defects (§7a) and must be repaired before any future verdict is trusted.**
- **Costs are measured** (§4), replacing three earlier wrong estimates.

## 6. What is closed and must not be retried

From the do-not-retest ledger — the most valuable document in this repo:

| Closed | Why |
|---|---|
| 0DTE long premium at minute cadence, **any** model or features | −$13.00/trade before costs, 5/5 folds |
| `omar` (SPX position within session range) as a tradable signal | −0.324 points/trade on raw data |
| Options-chain risk-reversal direction signal | Failed 5/5 fold gate (only 3/5 passed) |
| The 18-feature entry contract | Zero ranking power — deciles flat and non-monotonic |
| The exit-repair family (6 arms) | All lost to their comparator |

## 7. THE GATE CHAIN — everything between here and a working bot

*This is the gates file. There is no other one. The older `v4/docs/PROJECT_SECTIONS_AND_HILL_CLIMB_GATES.md`
is superseded.*

Each gate must pass before the next is attempted.

| # | Gate | Plain meaning | Status |
|---|---|---|---|
| **G1** | **Direction** | Predict SPX/ES direction over 15–60 min, beating 0.358 pts/trade gross, under one-position-at-a-time accounting | 🔴 **OPEN — question narrowed by §3a: it can only find a LARGE edge.** Chance of passing: 5–15% |
| **G2** | Option wrapper re-opens | Only if G1 passes. Row 181's stated re-entry condition | 🔴 blocked by G1. **Not structurally impossible** — measured 60-min directional break-even is 57.99% |
| **G3** | Feature certification | Prove each feature is actually available live at decision time, so we never train on data we would not have had | 🟡 capture 08-06/07, **but NOT on G1's critical path** — these are option features; G1 runs on ES bars. Two days give an observed envelope, not a worst case |
| **G4** | Train | Fit entry + exit models on certified features only | ⬜ needs G1. **A neural network is unjustified at 254 sessions** — training cannot create information G1 did not find |
| **G5** | Validation replay | Beat the comparator, positive confidence bound, 4/5 folds, negative controls clean | 🔴 **DEFECTIVE — must be repaired before any future replay.** See §7a |
| **G6** | Runtime parity | Live decisions match offline decisions **100%**. This is the gate that caught the `signed18` leak at 18.44% | ⬜ built and sound; tolerance already frozen at 1e-12 abs / 0 rel |
| **G7** | Live shadow | Model runs on live data, decides, submits nothing | ⬜ not built; ~3 engineering days + 20 sessions. **Also blocked by the §9a bug** — it must run unattended, and nothing scheduled can currently read this repo |
| **G8** | Guarded paper | Real paper orders over a pre-registered window. **This replaces the spent holdout** | ⬜ not built. **~1 year to decades** depending on effect size — see §3a limit 2. **Also blocked by §9a** for the same reason |
| **G9** | Real money | Separate owner decision and safety packet | ⬜ out of scope |

**G1 is what actually blocks everything. G3 is the only safe parallel branch — worth finishing because its
cost is already sunk, but it does not answer the direction question.**

## 7a. G5 is defective — repair before trusting any future verdict

Four defects, all confirmed in `v4/research/pathd_phase1_replay.py`:

| Line | Defect |
|---|---|
| — | The 8-cell fee/latency grid is really 4 latency tests run twice. `delta = learned − comparator` and both carry the same fee, so the fee cancels exactly. **Fee robustness is not measured at all.** |
| `:177` | The `time_shifted` negative control uses `.shift(-1)`, importing the **next row's** value. A control holding future information is not testing what it claims. |
| `:306` | "Positive skill" passes when **one** of five folds is positive. |
| `:311` | "Powered" means only ≥30 sessions and 5 fold labels present. **There is no power calculation in the gate** — which is exactly how the last campaign was surprised by `UNDERPOWERED` after the fact. |

**These do not threaten the five past negatives.** A gate that is too weak produces false *positives*; every
past campaign that ran through it still returned "no edge," which makes those results stronger, not weaker.
The defects threaten any *future* pass.

## 8. Why G6 and G8 exist (the expensive lesson)

A model called `signed18` posted a confirmed **+$540/session** edge. It was then found to have used an SPX
price bar stamped at time `t` whose closing value is not knowable until `t+60s`. Under the correct clock
only ~10% of its decisions reproduced. Proving it wrong consumed the protected holdout — the reserved
data set kept for final confirmation.

**Consequence: there is no clean historical confirmation left.** The only remaining out-of-sample test is
fresh live paper trading, which costs calendar time and cannot be bought with compute. That is why G8 is
not optional.

## 9. What is running right now

| What | When | How | Purpose |
|---|---|---|---|
| Track-A live OPRA capture | Thu 2026-08-06 and Fri 08-07, two windows each (06:28 and 09:10 PT) | **Attended only** — `./v4/ops/tracka/run_tracka_attended.sh`, started from Terminal and left running. The scheduled launchd job **cannot** do it | Measure when market data actually arrives, so features can be certified for G3 |
| `shadowasof.ledger` + `shadowasof.snapshot` | Every 60 s and every 900 s, continuously since 2026-07-18 | Two loaded LaunchAgents | **Leftovers from the stood-down programme.** Self-gated to weekday 06:29–13:06 PT, read-only, writing under `~/.autoresearch-trading/`. Owner intended to remove these — still loaded |
| Everything else | — | — | Nothing runs unattended |

**Correction 2026-08-05:** this table previously said nothing ran unattended. That was wrong — the two
`shadowasof` agents have been running since July. Verified live: `runs` went 5243 → 5246 over 140 s, exit
code 0. They are harmless, but they are also the proof that the scheduled-job bug below is a *permissions*
problem and not a launchd problem: those two work because their program is a Python binary that **does**
hold the grant.

**Why attended:** macOS blocks background jobs from reading `~/Documents` unless the specific program has
been granted permission, and `/bin/zsh` has not been. A shell started from Terminal inherits the grant; a
scheduled job does not. Confirmed from the permissions database — see job 5. The machine must also stay
awake.

2026-08-05's capture is **excluded from the sample**: its 06:28 open window was lost to that permission
denial, and its midday window ran only to prove the path works. Owner review: **Saturday 2026-08-08**.

## 9a. 🔴 OPEN BUG (HIGH) — nothing scheduled can read this repo

**Job 5. Diagnosed and worked around; not fixed. Needs an owner decision.**

**What is broken.** The repo lives in `~/Documents`, which macOS protects. A program may only read it if
that specific program has been granted permission. The complete list of what has been granted:

```
Terminal · VS Code · Claude Code · Codex · one Python binary (the uv python3.12)
```

`/bin/zsh` is **not** on it. Neither is `/usr/bin/python3`, nor `head`, `dd`, or `mkdir`.

A program you start in a terminal inherits that terminal's permission, so **everything works when a human
runs it**. A scheduled job has no such parent, so macOS checks the job's own program, finds nothing, and
refuses. That is why every launchd-fired capture failed and every hand-run one succeeded.

**Why it matters beyond Track A.** This is not a Track-A bug — it is a property of where the repo lives.
**Any** future unattended job hits it: G7 live shadow, G8 guarded paper, and any scheduled retraining or
monitoring. G7 and G8 both assume something can run without a person present. **Neither can be built on
this machine until this is fixed.**

**Why it was not obvious.** zsh reports "permission denied" using the same words it uses for "file not
found" (`can't open input file`), which sent the first diagnosis toward an iCloud sync fault. It is not
iCloud — that was checked and refuted, along with the machine being asleep.

**The three options, and the cost of each:**

| Option | What it does | Assessment |
|---|---|---|
| **A. Move the repo out of `~/Documents`** (e.g. to `~/autoresearch-trading`) | Removes the restriction for every program at once | **Recommended.** The only fix that is complete. Rewrites absolute paths in several launchd plists, `.env`, and hardcoded `REPO=` lines — a couple of hours with care, and it must not disturb the data at `~/.autoresearch-trading/` or `/Volumes/AR_TRADING_DATA` |
| B. Grant Full Disk Access to `/bin/zsh` | Puts zsh on the list | Blunt — grants *every* shell script on the machine full disk access, and may not survive OS updates |
| C. Make each scheduled job start via the already-granted Python | Narrowest change | Depends on child programs inheriting the parent's permission, which is **untested** here. Needs a throwaway probe job to verify |

**Recommendation: A, scheduled deliberately after the 08-06/08-07 captures are banked** — not before, and
not in a hurry. The attended runner covers the only work that needs to happen this week. Ask for help when
you want to do it; it is a mechanical change, but a wrong path left behind fails silently.

## 10. Where things live

| Path | What |
|---|---|
| `STATUS.md` | This page. The only status document. |
| `CLAUDE.md` / `AGENTS.md` | Ground rules for agents. Identical content. |
| `v4/docs/protocol101/training/history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md` | **The do-not-retest ledger.** Every closed idea and why. Read before proposing anything. |
| `v4/docs/protocol101/training/contracts/` | Signed decisions and build order |
| `v4/docs/protocol101/training/research/` | Individual studies — evidence, not plans |
| `v4/audit/autoresearch/` | Raw evidence: receipts, captures, results |

*Maintained by: Claude. Update this page whenever a gate moves — do not create a second status document.*
