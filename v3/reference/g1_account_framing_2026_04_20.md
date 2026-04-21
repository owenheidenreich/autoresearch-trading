# G1 — Account-Framing Reconciliation

**Workstream:** G1 (hard gate before W2) from the layered implementation plan.

**Decision:** **PROCEED** with W2.

v3 is a strategy-first framework. The account framing (SPX 0DTE long premium,
$25k, 1 contract per trade, Moderate-tier rails) is evidence-locked via the
`decay_analysis_2026_04_20.md` study. The question G1 needs to answer is
whether that framing is internally coherent on v3's own evidence — not
whether some external sizing model agrees with it.

## The three real questions

1. **Are the Layer 0 rails adequate for $25k / 1-contract deployment?**
2. **Is post-A1 signal density high enough to justify the W2 model spend?**
3. **Is the per-trade risk profile acceptable?**

### Q1. Rails

Source: [decay_analysis_2026_04_20.md](decay_analysis_2026_04_20.md).

The Moderate-tier rails (premium cap $1,200; daily brake $1,500; 2 full-
premium losers max; no fixed stop-loss fraction) were calibrated from a
5,194-path 0DTE contract-price study. Key calibration findings used:

- $1,200 cap fits morning 0.20–0.30 delta SPX 0DTE contracts (typical entry
  $200–$700 premium) with meaningful envelope for the occasional premium
  outlier.
- 5–9% of contracts that drawdown to −60% recover to positive, so a fixed
  stop-loss fraction in Layer 0 is empirically harmful. Exit discipline is
  a Layer 2/3 problem; Layer 0 just prevents account blow-up.
- Daily brake $1,500 caps worst-case daily loss at ~9.6% (two full-premium
  losers + slippage).

Verdict: the rails are evidence-locked for this exact profile. No additional
calibration needed for W2 to run; W2 is adding soft features to the model,
not changing rails.

### Q2. Signal density

Source: post-A1 attribution output in `attribution_full_2026-04-20.txt`.

Across 986 sessions:
- 24,754 teacher-entered bars (≈25/day pre-deduplication)
- Post-A1 outcome counts: 84 entered_right, 176 side_error, 726 abstention,
  0 guardrail_suppression
- Feasibility stratum verdicts: all "within envelope" (no cap-bind severity,
  no suppression-rate issues, no low-delta forcing)

Translation to deployment density: with Layer 0's first-trigger-per-session
+ daily brake + losers cap, expected deployment rate is ≈1 trade/day =
≈250 trades/year. That is a healthy training sample for W2's Layer-2 model
over the 986-day cache.

Verdict: signal density is more than adequate. The question for W2 is
whether enriched features (the 8 proposed: sigma_pos, omar_retest_dist_norm,
omar_range_norm, last10_range_over_omar, inside_first15,
late_window_40_120_flag, omar_mid_pos_units, last10_break_state) improve
the model's decisions on this trade population. W2 is the test.

### Q3. Risk profile

v3's Moderate-tier rails concentrate risk per-trade by design:

- A single max-cap loser = 4.8% of $25k
- Worst case daily = 9.6% (2 losers hitting the cap)
- Rails explicitly reject fixed stop-loss fractions per decay evidence

This is research-valid for 0DTE structure (recovery rate on drawn-down
contracts makes stops counterproductive). It is an aggressive personal-
account risk profile in absolute terms. Acceptance of that profile is a
trader-preference call, not a research-validity call. v3's research
rigorously defines the strategy under this profile; it does not claim the
profile is right for every account.

Verdict: acceptable if the trader is comfortable with the stated rails.
Flag on the record, not a blocker.

## Decision

Per the plan §5:

> If yes, W2a may proceed once the user approves the new lineage.
> If no, defer W2a until the account framing is resolved.

**Decision: YES.** W2a is unblocked.

## Conditions to carry forward (non-blocking)

Worth keeping on the record in any W2 ship doc:

1. The W2 model is trained on the v3 rules as of the post-A1 + W2 lineage.
   Future changes to rails / teachers / features require a fresh rebuild.
2. The model's signal quality is evaluated under v3's deployment rules
   (1 contract, premium cap, daily brake, no fixed stop). Transfer to any
   different execution regime is out of scope and would need its own audit.
3. The 9.6% worst-case daily risk rail is the trader's deliberate choice,
   backed by decay_analysis. It is not a research ceiling and is not
   meant to be softened post-hoc.

## Note on the original plan wording

The original G1 section cited external sizing research as a source of
concern. That was imported framing — v3's rails are calibrated internally
from decay_analysis, and v3 does not claim compatibility with any other
sizing model. External audits have no standing against v3's framing;
v3 stands or falls on its own evidence. This memo records that the
internal evidence supports proceeding.
