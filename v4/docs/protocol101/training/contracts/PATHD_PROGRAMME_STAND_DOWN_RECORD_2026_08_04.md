# Path-D — Programme Stand-Down Record (Option C)

**Status: OWNER-DECIDED — the directional research programme is STOOD DOWN as of 2026-08-04.**

Owner instruction: *"proceed with option C"*, 2026-08-04, following the closure of the last live
hypothesis. Decision key and options:
[`PATHD_NEXT_CLASS_OWNER_DECISION_2026_08_04.md`](PATHD_NEXT_CLASS_OWNER_DECISION_2026_08_04.md).

This is the terminal record. It exists so that a future restart begins from what is true rather than from
what was hoped.

---

## 1. Why

Five independent negatives, none of which repairs another:

| # | Finding | Number |
|---|---|---|
| 1 | 0DTE long premium is negative-EV **before any cost** | −$13.00/trade gross, negative 5/5 folds |
| 2 | The 18-feature entry contract has **zero ranking power** | deciles top −$15.37, **middle −$10.79**, bottom −$16.32 |
| 3 | The April-validated exit repair **failed** | 6/6 arms lost, maxT p 0.967–0.994 |
| 4 | Outcome shape is **incompatible with the signed Charter** | 23–69% big-loss share vs a 2% limit, at every horizon 5–240 min |
| 5 | **`omar`, the last survivor of sixty, closes on raw economics** | causal fade rule **−0.324 pts/trade** on raw real data; −$34 net of measured friction |

Finding 5 is the terminal one, and it needed no surrogate machinery: the causal fade rule at fixed omar
thresholds is negative on the raw data, and the momentum flip (+0.324) is still under the measured
0.358-point bar. Independently reproduced from a separate artifact at **−0.3578 pts/trade** over 28,456
boundaries. Full record:
[`PATHD_OMAR_DRIFT_ROBUST_REVALIDATION_RESULTS_2026_08_04.md`](../research/PATHD_OMAR_DRIFT_ROBUST_REVALIDATION_RESULTS_2026_08_04.md).

## 2. What is preserved, at zero cost

- **Execution plane.** IBKR paper guard and executor, proven by a guarded round trip on `DU***40`. The
  contract-identity bridge is closed: a live-qualified `SPXW  260804C07775000` matches the training
  corpus OSI format exactly. Strategy-agnostic — it does not care what the signal is.
- **Research machinery.** `pathd_model_gate.py` (4 rejection tests + 4 charter diagnostics),
  `pathd_research_loop.py` (bounded waves, maxT family, semantic dedup, prior-art blocking),
  `pathd_phase1_replay.py` (four-box, negative controls, bootstrap LCB). All validated against known-bad
  artifacts where they must fail.
- **The measured friction bar: ≥ 0.358 ES points per trade** ($17.9176 ÷ $50), horizon-independent in
  points. ES spread measured at **1.0397 ticks** (1.0734 in the top volatility quartile).
- **Owned data.** 261 sessions GLBX ES OHLCV-1m (full year), 251 sessions Databento OPRA SPXW CBBO-1m/1s,
  505 SPX 1m, 338 VIX 1m, 20 sessions GLBX ES bbo-1s.
- **The do-not-retest ledger**, now carrying every closure of this programme.

## 3. Correction — the protected holdout is SPENT, not unspent

The closure report stated that an **unspent** protected holdout survives. **That is incorrect, and it is
corrected here rather than carried forward**, because a future restart that believes it holds a
confirmation firewall it does not hold would be badly misled.

The 36-session protected holdout was **opened once**, on 2026-08-02, for the `signed18` entry-model
confirmation — which was subsequently **invalidated** by a 60-second SPX look-ahead. Evidence:
`v4/audit/autoresearch/autoresearch_v2_entry_model_confirmation_2026_08_02_attempt001/holdout_access_receipt.json`
and `confirmation_result.json`, both recording **`holdout_open_count: 1`**. Eighteen separate documents
state the holdout is SPENT.

The per-run receipts asserting `protected_holdout_opened: false` are correct and are **not** in conflict:
they record that *that particular run* did not open it. The two are easy to conflate; they are different
claims.

**Consequence.** There is no unspent confirmation firewall. The only remaining confirmation path for any
future hypothesis is **fresh live-paper observation**, or a newly reserved holdout built from data not used
in any prior campaign.

## 4. Live-state audit

| Item | State |
|---|---|
| `paper_orders_enabled` | **`false`** — already disarmed, verified in `v4/runtime/protocol101_paper_order_enablement.json` |
| Track-A Databento capture launchd jobs | **gone** — no labels loaded, no plists |
| crontab | empty |
| Long-running processes | none |
| **`com.autoresearch.protocol101.shadowasof.snapshot`** | **LOADED**, fires every **900 s** |
| **`com.autoresearch.protocol101.shadowasof.ledger`** | **LOADED**, fires every **60 s** |

The two `shadowasof` jobs read only a local capture directory
(`~/.autoresearch-trading/live_runtime/ibkr_capture`); they contact no broker and no vendor, and their logs
have been **zero bytes since 2026-07-18**, so they are producing nothing. They cost nothing but they are
**parity-recording research apparatus, not the execution plane**, and Option C stands the research
programme down.

**Owner runs this — the executing agent does not**, following the precedent set by the Track-A stand-down
in [`PATHD_BUILD_ORDER_AMENDMENT_2026_08_04.md`](PATHD_BUILD_ORDER_AMENDMENT_2026_08_04.md) §2:

```bash
launchctl bootout gui/$(id -u)/com.autoresearch.protocol101.shadowasof.snapshot
launchctl bootout gui/$(id -u)/com.autoresearch.protocol101.shadowasof.ledger
```

The plists may be left in place; they will not run once booted out. The unused
`com.autoresearch.protocol101.parityrecorder.*` plists in `~/Library/LaunchAgents/` are **not loaded** and
need no action.

## 5. What stays closed

- **0DTE long-premium class** — closed, structural. No feature set, learner, exit law, or execution tweak
  reaches it.
- **`omar` as a tradable directional signal** — closed on raw economics.
- **Phase-0 feature certification queue** — frozen by signed amendment; 8 ADMITTED / 75 BARRED, and the
  fail-closed enforcement stays exactly as-is.
- **Option B (longer-tenor options data)** — hard stop.

## 6. What would justify a restart

Not a feature association. Specifically:

1. **A mechanism, not a correlation** — a reason the edge should exist that survives being stated out loud.
2. **Economics computed on raw data first**, clearing the measured friction bar for its instrument
   (0.358 points for ES) *before* any null, model, or governance apparatus is built.
3. **A confirmation path that exists** — a newly reserved holdout from unused data, or a pre-committed
   live-paper test. The old firewall is gone.
4. **Nulls that pass a known-answer gate** before their verdicts are believed (§7).

## 7. Durable lessons

**Compute the economics on raw data first.** The number that closed this programme — −0.324 points per
trade — was computable from owned data with no surrogate machinery at any point. It sat inside a committed
artifact for most of a day while three studies argued about whether an information coefficient was real.
Use nulls to *explain* a number that is already interesting, not to decide whether to be interested.

**A null is a hypothesis and must be tested like one.** This programme designed nulls one at a time, each
repairing the previous one's flaw after seeing the data — session-shuffle, then block bootstrap, then wild
bootstrap. That is the same multiple-comparison error as searching over features, moved up one level: the
search was over nulls until one produced a result. The remedy is cheap and worked: **make the null pass a
test with a known answer before believing what it says about your hypothesis.** The gate caught the null
that had produced the headline result — it manufactured a perfectly monotone surrogate profile
(rho 1.0, spread 10.121) where the real data is flat (rho 0.2, spread −0.507), so its p-value of 1e-06 was
measuring its own defect.

**An association statistic is not a tradable one.** A within-session Spearman IC ranks against the whole
session, including boundaries that have not happened. It is a legitimate measure of association and never a
measure of what a trader can capture.

**`NO_EDGE` is a successful outcome.** Per the signed Charter. Five negatives bought cheaply, with the
machinery honest enough to catch its own errors repeatedly — a 60-second look-ahead, a 119× clock error, a
1000× microsecond clock error, a re-run of closed experiments, a fee figure wrong by 2.4×, and a headline
result produced by its own null — is a better outcome than a sixth campaign built on an assumption.

*Signed: Claude Opus 5 — 2026-08-04. Owner decision: stand down, 2026-08-04.*
