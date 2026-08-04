# Fable Brief — What is the shortest honest path to a trained model? (2026-08-04)

**The owner's question, verbatim:** *"how can we get to actually training a model? im not sure where we're
at."*

We need a plan, not another diagnostic. Be blunt; `STOP` and "you are still building the wrong thing" are
respectable answers. Every number below is reproducible from committed receipts or artifacts on
`/Volumes/AR_TRADING_DATA`.

---

## 1. Today, in one page

Eleven commits today. **Zero models trained.** Sequence:

| Commit | What happened |
|---|---|
| `1f664c5e` | Ran the SPX directional-skill screen. IC **+0.522** → **artifact**; 78.5% reproduced by no-predictability surrogates. My session-shuffle control was blind to it. |
| `017e5692` | Owner **signed** the Build Order Amendment: 65-feature certification queue **FROZEN**; Databento capture **stood down** (verified: no launchd jobs, no plists). |
| `b29ed007`+`9ad1c34c` | Option 0 re-screen with a corrected null. **One survivor of sixty:** `omar` — price position within the session's realized range — excess IC +0.052/+0.063/+0.078 @15/30/60m, maxT p ≤ 0.0002, 5/5 folds, **both independent nulls agreeing**. `spx_vwap_gap_*`, `session_range_bps`, all momentum, and **`minute_of_session`** all died. |
| `78df5eb3` | **I retracted my own economic sizing the same day.** Within-session deciles are look-ahead (the 10th-pctile cut ranges −0.941→+0.852 across 40 sessions). Redone causally, three defensible estimators give **+0.82 / +4.54 / +3.57** points/trade. **`NOT_IDENTIFIED`.** Worse: at fixed causal omar the **real** forward move is flat across every bin; the whole "corrected" signal is my surrogate's slope, caused by the wild bootstrap imposing session drift at *every* minute. |
| `1f532f93`+`32bb5142` | Owner overrode my advice and ran Option A. **ES spread measured: 1.0397 ticks unconditional, 1.0734 in the top RV quartile.** The 1-tick assumption **held**; the 2-tick scenario that would have closed the ES branch does not occur. Friction **$17.9176**. Hurdles **52.98/52.12/51.50%** vs passive 0DTE 53.9–56.6%. $1.478 spent. |

## 2. State of the assets

**Proven and durable**
- **Execution plane.** IBKR paper guard + executor, guarded round trip on `DU***40`; contract identity
  `SPXW  260804C07775000` matches the training corpus OSI format exactly. Strategy-agnostic, zero cost to hold.
- **Research machinery.** `v4/research/pathd_model_gate.py` (4 rejection tests + 4 charter diagnostics),
  `pathd_research_loop.py` (bounded waves, maxT family, semantic dedup, prior-art blocking),
  `pathd_phase1_replay.py` (four-box, negative controls, bootstrap LCB). All validated against known-bad
  artifacts where they must fail.
- **A measured friction bar.** **≥ 0.358 ES points per trade** ($17.9176 ÷ $50), horizon-independent in
  points. `v4/docs/protocol101/training/research/PATHD_ES_SPREAD_MEASUREMENT_RESULTS_2026_08_04.md`

**Owned data (no purchase needed)**
| Corpus | Coverage |
|---|---|
| **GLBX ES OHLCV-1m** | **261 sessions, 2025-08-01 → 2026-07-31 — a full year** |
| SPX 1m (ThetaData, official) | 505 files |
| VIX 1m | 338 files |
| Databento OPRA SPXW CBBO-1m/1s | 251 sessions |
| GLBX ES bbo-1s | 20 sessions (today's measurement) |

**Closed / blocked**
- **0DTE long-premium class: CLOSED, structural.** −$13.00/trade gross with all friction removed, negative
  5/5 folds. Recorded in the do-not-retest ledger §4.
- **Feature certification queue: FROZEN** by signed amendment (8 ADMITTED / 75 BARRED). Enforcement refuses
  any non-admitted feature — so **no SPXW-option model can be trained today**.
- **Protected 36-session holdout: SPENT.** `holdout_open_count` must stay 0. The only remaining
  confirmation is fresh live paper.
- **Protocol 028 rejected stitched ES futures VWAP as a feature** (March −$80 PF .928; Q2 −$2,130 PF .552;
  Q4 −$2,790 PF .890). There is no positive evidence in the record that ES helps *as a feature*. Note the
  distinction: ES as a *tradable instrument* is a different claim and is the one now supported.

## 3. My read — attack it

**Nothing is blocking a training run except the absence of something worth training.** The certification
apparatus, the frozen queue, the spent holdout — none of those stop an **ES** model. We own a full year of
ES 1-minute bars, and ES is not the 0DTE class, not covered by the frozen queue, and not in the
do-not-retest closure.

So the shortest path I can see is:

1. **Identify the numerator.** `omar`'s within-session association is real and survives two nulls, but the
   *tradable* size is unidentified. Test whether a **causal** normalization captures it — trailing-RV
   scaling, or a causal running forecast of the session's eventual range — instead of a look-ahead
   within-session rank. Fix the surrogate defect (resample session drift; do not impose it every minute).
   **Owned data, zero cost, pass/fail at 0.358 points.**
2. **If it clears: train the first ES model.** A shallow ranker on ES 1-minute bars, causal clock, the
   existing four-box replay and gate, 261 sessions, session-blocked maxT.
3. **If it does not clear: stop the directional programme** and keep the execution plane warm.

**What I am least sure about, and want challenged:**

- **Am I about to repeat the whole 0DTE story on ES?** The pattern was: a feature association → a model →
  friction eats it. ES friction is 3.00% of the mean 60-minute move vs 4.68% of premium for 0DTE — better,
  but not free. Is a single mean-reversion feature a strategy, or am I one step from a fifth negative?
- **Should step 1 even come first?** An alternative is to skip the omar identification and go straight to a
  proper ES feature study with a preregistered wave, on the grounds that one feature was never going to be
  a trader. That risks the exact "search for a signal with a spent holdout" failure this project has paid
  for twice.
- **Is `omar` on SPX even the right thing to trade on ES?** The signal is measured on the index; ES has
  basis and a nearly 24-hour session, so "session range" is not the same object. Does the omar
  construction need to be rebuilt natively on ES, and does that invalidate the SPX evidence?
- **The holdout is SPENT.** Any ES result can only ever be "worth a forward live-paper test". Does that
  make a training campaign premature until we decide we would actually paper-trade the output?

## 4. What we want from you

1. **A concrete, ordered plan to a trained and validated model** — with the gate at each step and an
   explicit stop rule. Name the first command.
2. **Is the ES pivot sound, or is it the 0DTE mistake wearing a new instrument?**
3. **What should we stop doing?** Today produced eleven commits and zero models; nine of the last twelve
   were me correcting my own errors. Is that healthy verification or a research programme that cannot
   finish anything?
4. **Anything obviously missed.**

**Hard constraints:** holdout SPENT (`holdout_open_count` = 0); causal clock t−60s; no paid data without
fresh owner authorization; `FILL_LAW`, the clock, the label law and the OOF firewall are frozen; Option B
(longer-tenor options data) is a hard stop; the 0DTE long-premium class stays closed.

*Prepared by Claude Opus 5 — 2026-08-04.*
