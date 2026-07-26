# Protocol101 Scoped Synchronization Decision - SIGNED FINAL

Prepared: 2026-07-25
Signed: 2026-07-25

## Decision

Protocol101 synchronization is accepted as **passed for offline Stage-1
research on the exact 17-feature contract below**. This does not claim that
Databento/ThetaData and IBKR are identical feeds, and it does not certify every
available field.

The evidence supports stopping the search for perfect full-day raw recorder
captures as a prerequisite for offline hill climbing. A selected model must
still prove candidate-specific decision transfer in no-order shadow mode before
paper validation.

## Authorized Stage-1 Alpha

The initial model-facing alpha list is exactly:

1. `spx_vwap_gap_points`
2. `spx_vwap_gap_bps`
3. `spx_vwap_gap_over_session_range`
4. `session_range_bps`
5. `momentum_5m_bps`
6. `momentum_15m_bps`
7. `momentum_5m_over_session_range`
8. `momentum_15m_over_session_range`
9. `omar_clipped_neg3_pos3`
10. `vwap_side_alignment_flag`
11. `omar_side_alignment_flag`
12. `momentum15_side_alignment_flag`
13. `D.near_atm.straddle_mid_spot_bps`
14. `D.near_atm.put_call_mid_ratio`
15. `D.near_atm.side_smile_slope_bps_per_5pt`
16. `E.bs.delta`
17. `E.bs.gamma`

The D-family composites use quantized minute mids and near-ATM aggregate inputs.
The E-family values are computed internally from the canonical quantized mid,
spot, strike, time to expiry, and frozen Black-Scholes constants. Vendor Greeks
are not model inputs.

## Evidence

The governing packet is:

`v4/audit/autoresearch/protocol101_clean_window_hill_climb_readiness_2026_07_25/summary.json`

Its recorded decision is
`governed_hill_climbing_ready_on_parity_stable_subset`.

Supporting facts:

- 864 option-eligible minutes across five newly recorded dates:
  2026-07-15, 2026-07-17, 2026-07-20, 2026-07-21, and 2026-07-22.
- Family D source-discriminator HGB AUC: `0.502582`.
- Family E source-discriminator HGB AUC: `0.500492`.
- Stable-probe minimum action agreement:
  straddle expansion `0.995690`, put/call skew `0.992908`, and internal-delta
  geometry `1.000000`.
- The 12 non-VIX context features retain their earlier three-full-day parity
  certification.
- A 15-complete-minute runtime freshness guard and a lightweight no-order
  decision-shadow logger exist.

## Explicitly Not Authorized As Initial Alpha

- Direct per-slot option-price paths (Family C): discriminator AUC `0.550524`
  exceeded the frozen `0.55` ceiling.
- Internal-IV expansion/compression: minimum action agreement `0.961538`.
- VIX 5m/15m changes: insufficient paired pre-window history.
- Raw bid, ask, spread, sizes, quote age, volume, open interest, and
  vendor-computed Greeks.

Raw quote fields remain available for guards, fills, labels, PnL, and audit.
They may not silently enter the model feature matrix.

## Scope Boundary

This decision authorizes **offline governed model research only** after the
training design and trader charter are signed and the exact-contract runner
preflight passes. It does not authorize:

- broker/API calls;
- paper order submission;
- promotion/default changes;
- paid downloads;
- runtime or launchd edits;
- protected-holdout access;
- real-money trading.

Before paper validation, a frozen selected candidate must pass the no-order
decision-shadow transfer battery, including the clean-window reset rule and
frozen action/slot agreement gates.

## Owner Checklist

- [ x] I accept that synchronization passed for this 17-feature Stage-1 scope,
      not for every market-data field.
- [ x] I accept the quarantined-feature list above.
- [ x] I approve ending perfect-full-day recorder collection as a blocker for
      offline hill climbing.
- [ x] I require candidate-specific no-order shadow transfer before paper
      validation.
- [ x] I understand that paper-submit remains a separate approval.

Owner signature: Owen Heidenreich

Date: 07/25/2026
