# Mechanical Baseline — Implementation Plan

**Thesis source:** [strategy_card_opening_reversion.md](strategy_card_opening_reversion.md)
**Feature source:** [feature_shortlist_opening_reversion.md](feature_shortlist_opening_reversion.md)
**Audit source:** [feature_schema.md](feature_schema.md)

This file translates the strategy card into a decision-complete, CPU-only
mechanical backtest. Nothing below reopens the thesis. Every spec is pinned
to a number, a column, or a file path — if something is ambiguous in the
strategy card, this doc picks a value and records the rationale.

## Hypothesis

Opening-structure reversion has edge in SPX/SPXW 0DTE long premium **only**
when all three conditions hold:

1. Price first **overextends away from VWAP** — the move has reached a
   point where continuation has become improbable.
2. Price then **reclaims or rejects back into opening structure** — the
   failure of continuation is confirmed, not just hypothesized.
3. The **chosen near-ATM premium on that side is actually executable** and
   not obviously overpaying for convexity — the thesis expression is not
   killed by spread or premium richness.

Each condition maps to a specific gate in the trigger and selection logic
below:

| condition | gate |
|---|---|
| 1. overextension | `min(vwap_dist[t-10 .. t-1]) ≤ -10 bps` (bull) / `max ≥ +10 bps` (bear) |
| 2. reclaim/reject into structure | `vwap_dist[t]` flips sign AND `first15_acceptance[t]` aligned AND `bar_delta[t]` aligned |
| 3. executable premium | delta band `[0.45, 0.55]` + dual spread gate (context + per-contract) |

If the baseline fails, the first diagnostic question is **which** of the
three conditions is not discriminating — not whether the thesis as a whole
is dead. Per-trade records capture the values at each gate so that
post-mortem can attribute failure to a specific condition.

## Scope

- One strategy: the v1 opening-structure reversion baseline.
- One evaluator: a pure mechanical trade loop, no ML, no learned policy.
- One output shape: per-trade CSV + per-fold `EvalReport` JSON + summary
  JSON + control-comparison JSON.
- No GPU. No training. No sidecar rebuild. No changes to the shipped
  simulator, schema, or feature pipeline.
- Two controls: a random-bar same-day control and a same-bar random-day
  control.

## Architecture (single pass)

```
for fold in folds 0..4:
    for day in fold.test_days:                 # held-out, per walkforward
        load sidecar(day)
        reconstruct trigger features(day)      # raw, from bar data
        for bar in bars[15 .. 120]:
            if in_cooldown: continue
            if already_traded_today: continue
            detect bull_trigger / bear_trigger
            if no trigger: continue
            pick contract (delta band, spread gate)
            if no eligible contract: record skip; continue
            compute spot-driven exit bar
            simulate_trade_spot_exit(entry, exit_bar, contract)
            record trade
    aggregate fold
write per-fold EvalReport + per-trade CSV
compute controls
compute falsification verdict
```

One implementation file, no new library modules. All helpers inline unless a
shared utility already exists.

**Target location:** `v2/analysis/mechanical_baseline_opening_reversion.py`
(matches existing CPU-analysis convention; `v2/strategies/fork_a1_random_control.py`
is the closest prior art but only covers MFE/MAE at fixed horizons, not full
trades).

## Feature inputs (exact)

The trigger runs on **raw** (non-normalized) spot-state values, computed
from the same inputs the pipeline uses. Do not denormalize from the stored
`X` tensor. The functions in [v2/pipeline/compute_features.py](../pipeline/compute_features.py)
are the reference; the baseline reimplements only the quantities it needs,
keyed by day.

Required per-bar inputs (per day):

| quantity | source | computation |
|---|---|---|
| `bar_of_day` | sidecar or recomputed | `0..389` per session |
| `session_vwap[t]` | raw SPX close + SPY volume | cumulative `(Σ p·v) / Σ v` from bar 0 of the day ([compute_features.py:311-318](../pipeline/compute_features.py#L311-L318)) |
| `first15_high`, `first15_low`, `first15_close` | raw SPX high/low/close | max/min/close of bars `[0..14]` — known from bar 15 onward |
| `bar_delta[t]` | raw SPX open/high/low/close | `clip((c - o) / max(h - lo, 1), -1, 1)` ([compute_features.py:301-304](../pipeline/compute_features.py#L301-L304)) |
| `vwap_dist[t]` | `c` and `session_vwap` | `(c - vwap) / c` |
| `vwap_reclaim_state[t]` | `c` and `vwap` (prior 3 bars) | `+1` if `c > vwap` and any of prior 3 bars had `c ≤ vwap`; `-1` symmetric ([compute_features.py:830-843](../pipeline/compute_features.py#L830-L843)) |
| `option_spread_pct[t]` | near-ATM call high/low/mid from chain | `(call_high - call_low) / mid` — context-level spread gate ([compute_features.py:993-1000](../pipeline/compute_features.py#L993-L1000)) |

Raw SPX/SPY/VIX bars are available through `_load_market_cache` in
[build_v2_dataset.py:~1016](../pipeline/build_v2_dataset.py) (or via the
same source files it reads). The baseline uses the same loader. VIX is only
used for post-hoc regime tagging, not for the trigger.

The 12 core features in the shortlist are **not directly consumed** by the
mechanical trigger — the trigger uses a small spot-reduction of them. The
schema table and shortlist remain the forward-looking feature set for the
ML stage that will follow this baseline.

## PIT guards (cross-checked against [feature_schema.md](feature_schema.md))

| guard | rationale | enforced by |
|---|---|---|
| `bar_of_day ≥ 15` | `first15_*` values are broadcast day-wide; meaningful only after bar 15 | outer loop lower bound |
| `bar_of_day ≤ 120` | thesis trade-window cutoff (11:30 ET) | outer loop upper bound |
| `bar_of_day ≥ 1` for `vwap_reclaim_state` | needs at least one prior bar | trigger short-circuits before bar 1 (moot given `≥ 15`) |
| `session_vwap` warmup | cumulative; valid from bar 0 once volume arrives | guard against `vwap` being NaN — skip bars where `isnan(vwap)` |
| no IB features used | `ib_break`, `ib_extension_pct`, `breakout_confirmation` are off-thesis and unsafe before bar 30 — explicitly excluded | not referenced in trigger |
| no `iv_percentile` used | v1 has no premium-rank gate — deferred to gate-first supporting | not referenced in trigger |

## Trigger logic (exact)

Two triggers, both boolean. Only one side can fire per bar. Parameters taken
verbatim from the strategy card.

```
VWAP_BPS        = 0.0010     # 10 bps
LOOKBACK        = 10         # bars
BAR_LO, BAR_HI  = 15, 120    # session window
```

### Bull call

All must be true at bar `t`:

1. `BAR_LO ≤ bar_of_day[t] ≤ BAR_HI`
2. `min(vwap_dist[t-LOOKBACK .. t-1]) ≤ -VWAP_BPS`
   (price was ≥ 10 bps below VWAP at some point in the prior 10 bars)
3. `vwap_dist[t] > 0`
   (current bar closes back above VWAP)
4. `first15_acceptance[t] ≥ 0`
5. `bar_delta[t] > 0`

### Bear put

All must be true at bar `t`:

1. `BAR_LO ≤ bar_of_day[t] ≤ BAR_HI`
2. `max(vwap_dist[t-LOOKBACK .. t-1]) ≥ +VWAP_BPS`
3. `vwap_dist[t] < 0`
4. `first15_acceptance[t] ≤ 0`
5. `bar_delta[t] < 0`

If both fire simultaneously (should not, given condition 2 and the direction
of condition 3), the bar is skipped and logged as `ambiguous_trigger`.

### Entry deduplication

- **One trade per day, total.** The first fired trigger on the day produces
  the trade. Subsequent triggers the same day are logged as
  `skipped_same_day`.
- **No intraday overlap.** Only one open position ever.
- **No cooldown between days** — the day is the natural unit.
- Rationale: the strategy card is a short-horizon test; stacking trades
  inside a day complicates attribution and doubles exposure to the same
  single-day regime. A future variant can enable a cooldown-based multi-entry
  mode; v1 is single-entry-per-day for interpretability.

## Contract selection (exact)

At the entry bar `gi`, load the day's sidecar and slice the executable
contracts:

```python
start, end = int(sc['bar_ptrs'][local_i]), int(sc['bar_ptrs'][local_i + 1])
row_features = sc['row_features'][start:end]          # (n, 22)
contract_ids = sc['row_contract_idx'][start:end]      # (n,)
```

Row-feature column indices are fixed in [chain_data.py:31-54](../core/chain_data.py#L31-L54):

- `[0]` `contract_valid` — require `> 0.5`
- `[2]` `right_is_put` — `1.0` for puts, `0.0` for calls
- `[3]` `mid` — entry price candidate
- `[4]` `spread_fraction` — per-contract spread (NOT the gate — see below)
- `[8]` `delta` — signed; puts are negative
- `[14]` `quality_flag` — require `≥ QUALITY_VALID` (2)

### Delta band and side

- Bull call: `right_is_put == 0 and 0.45 ≤ delta ≤ 0.55`
- Bear put: `right_is_put == 1 and 0.45 ≤ |delta| ≤ 0.55` (i.e. `-0.55 ≤ delta ≤ -0.45`)

### Spread gate (dual)

Two hard gates must both pass for entry. The rationale is direct from the
hypothesis: condition 3 ("chosen premium is actually executable") is not
satisfied by either gate alone.

1. **Context gate** — `option_spread_pct[t] ≤ 0.20` (from the strategy
   card). This is a near-ATM call high/low/mid proxy — effectively a
   call-side spread estimator. It flags when the market-wide premium book
   is wide, but can be asymmetrically favorable on put entries if the
   put side is structurally wider.
2. **Selected-contract gate** — `row_features[selected, 4] ≤ 0.20`
   (the per-contract `spread_fraction` from [chain_data.py:36](../core/chain_data.py#L36)).
   This enforces that the actual tradeable instrument is executable, not
   just that the call-side market-wide proxy is.

Both gates use the same `0.20` threshold (matches the strategy card's
quality cap). Keeping both closes the call-side / put-side asymmetry.

Skip codes: `skipped_context_spread` and `skipped_contract_spread` record
which gate rejected, so attribution CSV can distinguish the two failure
modes.

### Tiebreaker

Among eligible contracts in band:

1. Smallest `|delta - 0.50|` (closest to ATM by Black-Scholes delta).
2. If tied, lowest `spread_fraction` (column 4).
3. If still tied, lowest `abs(moneyness_pct)` (column 11).

If no contract satisfies the delta band, log `skipped_no_delta_contract`
and record the bar as a triggered-but-not-traded event (needed for the
skip-rate falsification check).

## Exit logic (exact)

Spot-driven. The built-in simulator exits on option prices only
(`simulator.py:172-242`); it cannot honor VWAP re-cross or first-15
boundary touches. Therefore the baseline runs its own exit loop on the
contract mid series.

### Exit conditions (evaluated in priority order at each bar `τ > t`)

1. **Hard stop — VWAP re-cross**
   - Bull: first `τ` where `close[τ] ≤ vwap[τ]` (re-crosses down through VWAP)
   - Bear: first `τ` where `close[τ] ≥ vwap[τ]`
   - Exit reason: `stop_vwap`

2. **Profit target — first-15 boundary touch**
   - Bull: first `τ` where `high[τ] ≥ first15_high`
   - Bear: first `τ` where `low[τ] ≤ first15_low`
   - Exit reason: `target_first15`

3. **Time stop**
   - Whichever fires first: `bar_of_day[τ] == bar_of_day[t] + 30` OR
     `bar_of_day[τ] == 120` (11:30 ET).
   - Exit reason: `time_stop`

4. **EOD safety**
   - If none of the above fires, force exit at `bar_of_day == 389` with
     reason `eod`. Should not occur given the 11:30 cap but kept for
     correctness.

Exits fire at bar close. Same-bar-stop-and-target ties resolve to the
hard-stop (conservative). If stop and target fire on the same bar as the
entry (`τ == t+1`), record with `one_bar_trade` tag for audit.

### P&L and cost accounting

The baseline does **not** call `simulate_trade`. It computes P&L directly:

```python
entry_mid   = row_features[selected, 3]               # contract mid at t
exit_mid    = sc['contract_mid'][contract_id, local_exit]
bars_held   = local_exit - local_i
gross_pct   = (exit_mid - entry_mid) / entry_mid
spread_bps  = compute_adaptive_spread_bps(mtc_at_entry, vix_regime, is_otm=False)
spread_cost = 2 * spread_bps / 10000.0                # round-trip
net_pct     = gross_pct - spread_cost
```

- Uses [compute_adaptive_spread_bps](../core/features.py#L174) so cost
  accounting matches the rest of the system.
- `is_otm=False` because entries are in the 0.45-0.55 delta band — not
  far-OTM.
- Commission: zero for v1. Add later if needed; the call is to keep the
  first result interpretable against the existing simulator's output.
- VIX regime at entry is read from raw VIX bucket boundaries — same
  formula as `compute_features.py:693-696`.

### Output record per trade

Pydantic-light dict with:

```
date, bar_entry (bod), bar_exit (bod), side ("C"/"P"),
strike, right, delta_at_entry, spread_at_entry, entry_mid, exit_mid,
bars_held, gross_pct, spread_cost_pct, net_pct,
net_pnl_dollars (= net_pct * entry_mid * 100),
exit_reason, trigger_vwap_dist_min_prior, trigger_vwap_dist_now,
first15_acceptance_at_entry, vix_regime_at_entry
```

## Control baselines (exact)

Both controls reuse the exact same contract-selection, spread gate, and
exit logic. They differ only in **when** they enter.

### Control A — same-day, random-bar, same side as strategy

For each day that the strategy triggered:

- Draw one random bar uniformly from `[BAR_LO, BAR_HI]`.
- **Preserve the strategy's fired side** (bull if the paired strategy trade
  went bull, bear if bear). A fair-coin side flip would confound timing
  with direction; this control isolates the timing question only.
- Enter, exit by the same rules (contract band, dual spread gate, spot
  exits).

Tests: "does the opening-reversion trigger add value over any random-bar
entry on the same side in the same window?"

### Control B — same-bar, random-day

For each strategy trade `(date, bar)`:

- Pick a random other day from the same fold's test window.
- Enter at the same `bar_of_day` on that day, with the same side.
- Exit by the same rules.

Tests: "is this a time-of-day effect rather than a signal effect?"

Both controls share the strategy's contract band and spread gate. Random
seed is fixed per fold (`fold_idx * 10007 + 1`) for reproducibility.

## Falsification criteria (concrete)

From [strategy_card_opening_reversion.md:118-124](strategy_card_opening_reversion.md).
The strategy card names the qualitative rules; this plan pins them to
numbers.

Let:

- `N` = number of strategy trades across all test days, all folds
- `N_target` / `N_stop` / `N_time` = exits by reason
- `mean_net_pct` = mean of `net_pct` over N
- `dollar_pf` = `Σ net_pnl_dollars | net_pnl_dollars > 0` / `|Σ net_pnl_dollars | net_pnl_dollars < 0|`
- `ctrl_A_mean_net_pct`, `ctrl_B_mean_net_pct` = same metric on controls

**Baseline is falsified if ANY of:**

1. `N_target / N ≤ N_stop / N`
2. `mean_net_pct ≤ 0`
3. `ctrl_A_mean_net_pct ≥ mean_net_pct - 0.5 * std_err(mean_net_pct)` OR
   `ctrl_B_mean_net_pct ≥ mean_net_pct - 0.5 * std_err(mean_net_pct)`
4. `N < 20` (insufficient sample to speak at all — report as
   `inconclusive`, not `falsified` — but treat as a fail for "is this
   thesis expressible in the current dataset")

Success requires all four to be false on the aggregate test set. Per-fold
breakdowns are reported alongside but do not individually gate the verdict.
The `0.5 * std_err` margin is deliberately modest — we are looking for a
clean separation, not a statistically significant edge on a small sample.

The strategy card also notes that weak results do not imply "0DTE long
premium is dead"; they only falsify this specific reversion expression.
The output summary restates this explicitly.

## V1A → V1B escalation

The baseline runs in two tiers. Only **V1A** (this plan) is built initially.
**V1B** is built *only* if V1A returns inconclusive or near-miss — not if
it passes cleanly or fails clearly. The escalation is bounded: no V1C.

### V1A — structure-only (this plan)

The mechanical baseline exactly as specified above. No premium-sanity
gates beyond the dual spread gate. This is the cleanest test of the
three-part hypothesis on its own terms: if the structural trigger has
edge, we want to see it before adding regime filters.

### V1B — V1A plus premium-sanity gates (conditional)

| V1A result | action |
|---|---|
| All four falsification criteria false | **accept V1A**; no V1B needed |
| `N < 20` (inconclusive) and signal frequency is the primary issue | rebuild dataset scope first; V1B only if scope is already maximal |
| Near-miss: `mean_net_pct > 0` but control margin breached | **build V1B** |
| Near-miss: `N_target / N ≤ N_stop / N` by ≤ 3 percentage points | **build V1B** |
| Clear fail: `mean_net_pct ≤ -0.01` (−1% per trade or worse) | **do NOT build V1B.** Thesis as structured is falsified; reopen the hypothesis, do not rescue with filters |

V1B adds two no-trade gates from the shortlist's gate-first supporting
set ([feature_shortlist_opening_reversion.md](feature_shortlist_opening_reversion.md)):

- `iv_percentile[t]` ceiling — skip entries when ATM IV is in the top
  percentile band of its 60-day history (premium unusually rich).
- `vrp[t]` ceiling — skip entries when IV² meaningfully exceeds
  realized-vol² (premium overpaid relative to recent realized expansion).

**Thresholds are deliberately not pre-committed here.** They are picked
from V1A's attribution CSV, specifically from the `iv_percentile` / `vrp`
distribution on losing trades vs winners. Picking them ex-post from
V1A-test-day data is acceptable *only because* V1B is itself a new
specification evaluated against the same test-day set and judged against
the same falsification criteria; it is not a hyperparameter search within
V1A.

Rationale for the two-tier bound: avoid "rescue by filter" behavior where
a failing thesis is kept alive by adding increasingly specific no-trade
conditions. If V1B also fails, the next move is to reopen the thesis —
not to add a third tier.

## Output contract

All under `v2/artifacts/mechanical_baseline_opening_reversion/`.

```
summary.json                     # aggregate + per-fold metrics, falsification verdict
trades.csv                       # every strategy trade (all folds)
trades_fold{0..4}.csv            # per-fold
skips.csv                        # triggered-but-skipped bars + reason
controls.json                    # control A and B metrics, per-fold
report_fold{0..4}.json           # EvalReport per fold (reusing EvalReport.from_replay)
```

`EvalReport` fields ([eval_report.py:26-62](../core/eval_report.py#L26-L62))
that apply:

- `trades`: serialized trade dicts as defined above
- `dollar_pf`, `dollar_net_pnl`, `dollar_gross_profit`, `dollar_gross_loss`
- `daily_equity_curve`, `daily_dates`
- `metrics`: reconstructed-enough `ReplayMetrics` with zeros for
  ML-specific fields
- `baselines`: `{"control_A_mean_net_pct": ..., "control_B_mean_net_pct": ...}`
- `experiment_id`: `"mechbase_opening_reversion_v1"`
- `dataset_fingerprint`: pulled from `data.pt.sha256`
- `model_fingerprint`: `""` (no model)

## Implementation checklist (work items, in order)

1. Create `v2/analysis/mechanical_baseline_opening_reversion.py` with
   argument parsing: `--fold {0-4|all}`, `--out <dir>`, `--seed <int>`,
   `--controls/--no-controls`.
2. Load raw SPX / SPY / VIX bars and day boundaries (same loader that
   `build_v2_dataset.py` uses).
3. Implement `reconstruct_trigger_features(day)` that returns the six
   per-bar quantities from the feature-inputs table.
4. Implement `iter_signals(day_features) -> list[(bar_idx, side)]` with
   exact trigger logic.
5. Implement `select_contract(sidecar, local_i, side, spot) -> row | None`
   with delta band, spread gate, tiebreaker.
6. Implement `run_exit(sc, contract_id, entry_local_i, side, raw_bars) ->
   (exit_local_i, exit_reason, exit_mid)` with the four exit conditions.
7. Implement `compute_pnl(entry_mid, exit_mid, mtc_at_entry, vix_regime)`
   using `compute_adaptive_spread_bps`.
8. Implement `run_strategy_fold(fold_spec)` and `run_control(control,
   fold_spec, seed)`.
9. Implement `write_outputs(fold_results, control_results, out_dir)` —
   CSV + EvalReport JSON per fold + summary.
10. Implement `falsification_verdict(summary) -> {passed, failed, reason}`.
11. Add a top-of-file docstring referencing this plan, the strategy card,
    and the schema audit.
12. Run on fold 0 only as a smoke test; inspect `trades_fold0.csv` for
    shape, exit-reason distribution, sample bar sequences.
13. Only after smoke passes: run on all 5 folds, commit artifacts under
    `v2/artifacts/mechanical_baseline_opening_reversion/`, log in
    `v2/lab_notebook.md`.

## What is explicitly NOT in v1

- No ML scoring or ranking.
- No `simulate_trade` integration — spot-driven exits are incompatible.
- No multiple entries per day.
- No `iv_percentile` / `vrp` no-trade gate (these are supporting,
  gate-first features; revisit if the baseline needs filtering).
- No surface / dealer-hedging features.
- No trailing stops, scaling, or partial exits.
- No position sizing (each trade records `net_pnl_dollars` at 1-contract
  notional; PF metrics are invariant to uniform sizing).
- No live / paper execution.

## Locked decisions (2026-04-19)

All open decisions from the first-pass plan are resolved:

| decision | locked value |
|---|---|
| Trades per day | **1 total** (single-entry, no intraday cooldown needed) |
| Contract tiebreaker | **`|delta − 0.50|` nearest**, then lowest `spread_fraction`, then lowest `|moneyness_pct|` |
| Control construction | **two controls** — A (same-day, random-bar, strategy-side-preserved) + B (same-bar, random-day) |
| Fold choice | **test_days only** (held-out per walkforward) |
| Spread gate | **dual** — context `option_spread_pct ≤ 0.20` AND per-contract `spread_fraction ≤ 0.20` |
| Escalation | **V1A first**; V1B only on inconclusive / near-miss; no V1C |

No remaining thesis-level ambiguities. Implementation proceeds against
the V1A checklist.
