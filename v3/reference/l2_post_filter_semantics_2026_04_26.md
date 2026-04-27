---
date: 2026-04-26
parent: phase1_filter_validation_2026_04_26.md
status: SEMANTICS DOCUMENTED — existing gates use rescan-after-veto at the bar→day level; veto layer will match
---

# Phase 2a: L2 post-selection veto runtime semantics

## Question

The Phase C 1664-trade sample was constructed from `chosen_trades.pkl`,
which represents the OUTPUT of `_select_daily_trades` — one chosen
trade per day. When Phase 1 vetoed trades from this sample, it used
**drop-only** semantics: the vetoed trade simply disappeared, no
replacement.

The live runtime uses `_select_daily_trades` from
[v3/layer2/train_unified_policy.py](../layer2/train_unified_policy.py#L439-L468).
Does that path use drop-only or rescan-after-veto?

## Code reference

```python
# v3/layer2/train_unified_policy.py:439-468
def _select_daily_trades(pred_df, decision_margin=None, *, abstention_policy=None):
    if abstention_policy is not None:
        decision_margin = float(abstention_policy.get("decision_margin", 0.0))
        min_win_prob = float(abstention_policy.get("min_win_prob", 0.0))
        max_stopout_prob = float(abstention_policy.get("max_stopout_prob", 1.0))
        ...
    eligible_mask = (
        (pred_df["chosen_action_id"] > 0)
        & np.isfinite(pred_df["best_nonflat_score"])
        & (pred_df["decision_margin"] >= decision_margin)
        & (pred_df["pred_win_prob"].fillna(0.0) >= min_win_prob)
        & (pred_df["pred_stopout_risk"].fillna(1.0) <= max_stopout_prob)
    )
    eligible = pred_df[eligible_mask].copy()
    if eligible.empty:
        return eligible
    idx = eligible.groupby("day")["best_nonflat_score"].idxmax()
    return eligible.loc[idx].sort_values(["day", "bar_index"]).reset_index(drop=True)
```

The eligibility gate is applied to `pred_df` — a frame containing **all
~91 candidate (day, bar_index) bars per day** (verified on
`oof_predictions.pkl` for seed 42: mean 90.9 bars/day, std 0.8). After
gate filtering, the daily best-scored bar is selected.

**This is rescan-after-veto at the bar→day level.** If the bar that
would have won the daily contest is rejected by the gate, the
next-best-scored eligible bar in the same day is picked. The day only
goes to "no trade" if every candidate bar fails the gate.

## Champion's actual gate thresholds

`v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed42/seed_42/window_00/calibration.json`:

```json
{
  "decision_margin": 0.0,
  "min_win_prob": 0.0,
  "max_stopout_prob": 1.0
}
```

All three thresholds are neutral on the spx_combined_3seed_001 champion
— the eligibility gate currently rejects no candidates beyond the
trivial `chosen_action_id > 0` and `np.isfinite(best_nonflat_score)`
checks. The daily best-scored bar always wins. (This itself is a Phase
3 finding: the operational champion runs without any score-quality
gating.)

## Decision for the veto layer

**The veto layer adopts rescan-after-veto semantics.** It is plumbed
into `_select_daily_trades` at the same site as `decision_margin` —
augmenting `eligible_mask` with `~apply_avoid_filters_mask(pred_df)`.

Practical consequences:
- If a day's highest-scored bar matches an avoid rule, the next-best
  eligible bar in that day is selected instead. Trade count is
  approximately preserved, but the chosen bar may differ.
- If every bar in a day matches an avoid rule, no trade that day
  (rare under the current rules — they target specific cells / sides /
  bar windows).
- The Phase 1 drop-only result is a **pessimistic** approximation of
  the rescan result for any avoid rule that hits the daily winner: in
  rescan mode a replacement candidate is found, which may reduce or
  eliminate the trade-count delta and add (typically lower-scored)
  trades back into the kept distribution.

## Sensitivity check (drop-only vs rescan, in-sample)

`scripts/research_phase2a_rescan_sensitivity.py` re-applies the
avoid-union under rescan semantics on `oof_predictions.pkl` per seed
and compares to drop-only on the same matched subset. Per-seed ΔPF on
the hl-matched trades:

| seed | base_pf | drop_pf | drop ΔPF | rescan_pf | rescan ΔPF | days replaced | days dropped (no replacement) |
|---|---|---|---|---|---|---|---|
| 42 | 2.386 | 2.768 | +0.382 | 2.612 | +0.226 | 148 | 8 |
| 43 | 2.124 | 2.563 | +0.439 | 2.493 | +0.369 | 170 | 3 |
| 44 | 1.464 | 1.943 | +0.479 | 2.055 | +0.592 | 170 | 2 |
| 45 | 1.801 | 2.253 | +0.451 | 2.137 | +0.336 | 159 | 4 |
| 46 | 1.632 | 1.968 | +0.336 | 1.997 | +0.365 | 174 | 7 |

- **Mean ΔPF drop-only:** +0.417 per seed
- **Mean ΔPF rescan:** +0.377 per seed (about 90% of drop-only)
- **Mean days replaced:** 164.2 / ~715 base days (~23%)
- **Mean days dropped (no replacement):** 4.8 (~0.7% of days)

The directions agree on every seed; magnitudes differ by ≤0.16 with no
sign flips. Seed 44 actually exceeds drop-only under rescan (+0.59 vs
+0.48), suggesting the replacement bars on losing-days can be
constructive. Net: rescan is a mild conservative shrink of drop-only,
not a sign-flipping divergence — the Phase 1 drop-only result is a
reasonable proxy for runtime behavior.

The forward-walk evaluation in Phase 2b uses rescan mode end-to-end
(since it goes through `_select_daily_trades`), and is the
authoritative OOS test.

## Acceptance

The veto layer in `v3/layer2/post_filters.py` will:
1. Expose `apply_avoid_filters_mask(pred_df) -> bool array` — one boolean
   per row of the per-bar prediction frame, True for rows to veto
2. Be invoked inside `_select_daily_trades` AFTER the existing
   `eligible_mask`, identical AND-composition pattern
3. Use the frozen Phase 0 spec (`phase_c_gpt55_v0_2026_04_26`) — no
   inline rule definitions
4. Emit per-rule veto counts to a log so post-FW analysis can
   attribute kept-distribution changes to specific rules
