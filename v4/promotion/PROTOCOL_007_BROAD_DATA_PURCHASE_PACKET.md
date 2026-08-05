# Protocol 007 Broad-Data-Purchase Packet

## Status

Superseded by the fresh Q3 2025 validation audit.

The original packet approved a narrow validation purchase. That purchase was completed for Q3 2025. The locked Protocol 007 hypothesis failed that fresh holdout, so this packet should no longer be read as approval for further broad-data purchases.

Fresh audit:

```text
v4/audit/autoresearch/v4_aplus_value_veto_protocol_008_q3_fresh_locked/report.md
```

Fresh Q3 result for the locked `bce_cost25__gamma_theta_ge_0.0025` hypothesis:

```text
Q3 median PnL: 0
Q3 median trades: 0
Q3 +50 stress median PnL: 0
positive seed fraction: 0.33
```

Only one of three seeds traded Q3 after the locked permission and gamma/theta filters. That fails the validation gate.

## Original Decision

Approve a narrow additional validation data purchase for the locked Protocol 007 hypothesis.

This is not live-trading approval. This is not approval to spend the remaining credit balance on broad training data. The next spend should buy unseen validation months, run one frozen audit, and then decide whether the signal deserves expansion.

## Locked Hypothesis

Trade one SPXW 0DTE contract at a time using the A+ permission policy with a deterministic contract-quality veto:

```text
base surface model: surface_structure_aplus_huber
policy: policy1
permission model: bce_cost25
value veto: feature_gamma_theta_ratio_scaled >= 0.0025
execution labels: ask entry, bid exit, broker commission excluded
settlement universe: SPXW PM-settled only
selection mode: domain-priority
```

The veto can only remove trades from the already simulated one-contract permission stream. It cannot add overlapping positions or create new entries outside the policy.

## Evidence

Report:

```text
v4/audit/autoresearch/v4_aplus_value_veto_protocol_007_domain_priority/report.md
```

Locked champion:

| Split | Median PnL | Median PF | Trades | Positive Seeds | Edge vs Random | +50 PnL/PF |
|---|---:|---:|---:|---:|---:|---:|
| selection | 3747 | 1.862 | 29 | 1.00 | 3870 | 2297 / 1.658 |
| March 2026 | 2816 | 1.440 | 29 | 1.00 | 5119 | 2466 / 1.298 |
| Q4 2025 | 1222 | 1.771 | 95 | 1.00 | 5317 | 772 / 1.165 |

The locked policy clears the broad-data-purchase gate:

```text
positive selection, March, and frozen Q4 medians
March and Q4 beat matched random by median
March and Q4 survive +$50/trade stress
March and Q4 have at least 2/3 positive seeds
at least 8 median trades in selection, March, and Q4
```

## 1s Path Audit

Report:

```text
v4/audit/autoresearch/v4_aplus_value_veto_protocol_007_domain_priority/one_second_path_audit_march/report.md
```

Result:

```text
input trades: 91
audited trades: 16
coverage: 18%
audited sessions: 2026-03-06, 2026-03-13, 2026-03-20, 2026-03-27
sign flip fraction: 0.00
1s - 1m PnL sum: -60
median diff: 0
p95 absolute diff: 62.5
```

The audit supports the current 1-minute label path on available slices, but coverage is still limited.

## Caveats

Q4 +$100 stress fails.

Q4 +$50 median survives, but one seed loses -4040 under the same stress. March +$50 also has one slightly negative seed at -65.

The policy has now been examined through several related audits on the same frozen Q4 block. The next block must be treated as fresh validation and scored once before any further tuning.

## Recommended Purchase

Buy one modest unseen validation block first:

```text
primary block: 2025-07-01 through 2025-09-30
required schemas: definition, cbbo-1m, ohlcv-1m, statistics
symbols: SPXW.OPT filtered to 0DTE PM-settled contracts
index context: SPX and VIX 1-minute bars
audit slice: CBBO-1s/tcbbo for ATM +/- $50 on 8-10 selected validation days
```

Do not train on this block before the first audit. The first run should only answer:

```text
Does locked Protocol 007 survive unseen months after +$50/trade stress?
```

## Promotion Gate For The New Block

The locked protocol should remain eligible only if the new validation block shows:

```text
median PnL > 0
median PF >= 1.15
matched random edge > 0
+$50/trade median stress PnL > 0
+$50/trade median stress PF >= 1.05
positive seed fraction >= 2/3
1s audit has no material sign-flip problem
```

If it fails, stop buying data and return to model design.
