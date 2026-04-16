# Handoff: exp_153 Screening Results (Pre-Audit)

## Summary

exp_153 was the first supervised model trained on the full-day regime (bars 30-270). The screening ran on 2026-04-15, **before** the adversarial pipeline audit documented in `hashed-tinkering-yeti.md`. The audit subsequently found two critical gate failures (G1: harness eval trailing tiers, G7: stale documentation) and one high-severity data contamination issue (6.3% zero-Greeks contract rows).

**The exp_153 results are real but were not properly validated.** The training and replay machinery is mechanically sound (audit Pillar I passes), but the safety gate that was supposed to verify label-simulation parity was itself broken. This handoff contextualizes what we can and cannot trust from this experiment.

---

## What exp_153 showed

**Screening (1-fold, fold 0, test period 2025-12-05 to 2026-03-04):**

| Metric | Value |
|--------|-------|
| Trades | 673 |
| Trades/day | 12.9 |
| Traded days | 52/60 |
| PF | 0.745 |
| Win rate | 45.8% |
| Direction | 513C / 160P (76%/24%) |
| Drawdown | 108% (gate failure) |
| Net P&L | -$10,805 |
| Beats baselines | No (all 4 baselines also gate-failed) |
| Training | 17 epochs, best at epoch 3, 308s |
| Gate accuracy | 53.5% |
| Direction accuracy | 49.1% |
| Side accuracy | 48.7% |

---

## What we can trust from these results

**The core data-to-model path is sound** (confirmed by audit Pillar I):
- Feature names and indices are correctly aligned (Gates G2, G3: PASS)
- Sidecars match the current policy and schema (Gate G4: PASS)
- Dataset fingerprint verified (Gate G5: PASS)
- Replay uses the same simulation parameters as the oracle
- Sign flip, normalization, masking, batch construction all work as intended

**Therefore these observations are real:**
- The model engages with the full-day opportunity set (673 trades, 52 traded days). It did not collapse to abstention.
- The model makes directional decisions (76/24 call/put, not 100/0).
- Win rate at 45.8% is meaningful, not random.
- The model overtrades aggressively (12.9 trades/day), leading to catastrophic drawdown.
- All four baselines also gate-failed at -0.2 on the full-day regime — the opportunity set is harder for naive strategies too.

---

## What we cannot trust

### 1. The pre-run gate gave false confidence

**Audit finding (G1 FAIL):** `harness_eval.py` calls `simulate_trade()` without `breakeven_trigger_pct` or `extra_trailing_tiers`, defaulting to breakeven at +30% unrealized. The oracle labels use +15% (from `DEFAULT_POLICY`). For any trade with MFE between +15% and +30%, the harness simulates a different exit than what the model was trained on.

**Impact on exp_153:** The pre-run gate "passed" before the screening run. But the harness eval's trade-level fidelity checks were comparing against wrong trailing stop behavior. We cannot claim the labels were validated by the gate — they weren't.

**What this does NOT mean:** The oracle labels themselves are correct (they use the right policy). The training data is sound. The gate just couldn't verify it.

### 2. ~6.3% of contract rows have zero Greeks

**Audit finding:** 6.3% of contract-bar rows (sampled on 50 days) have `contract_valid=1.0` but `iv=0.0, delta=0.0, gamma=0.0, theta=0.0, vega=0.0, charm=0.0`. These are IV solver failures written as zero. The model sees these as "valid contract with zero volatility" — an economically impossible state.

**Impact on exp_153:** The model trained on ~6% noise rows where contract features are meaningless. These rows also contaminate per-bar z-score normalization in `forward()`, slightly distorting all contracts in affected bars. The model may have learned spurious patterns from these impossible states.

**What this means:** The 45.8% win rate and 0.745 PF were achieved despite this contamination, not because of it. Cleaning the contamination would likely improve results, not degrade them.

### 3. Float16 sidecar precision was verified on only 2% of corpus

**Audit finding:** The float16 downcast in `strip_sidecars.py` was verified on 20 of 986 days (2%). The 2.97% max relative error on charm was measured pre-normalization; post-normalization amplification was not measured.

**Impact on exp_153:** This experiment used float16 stripped sidecars on GPU. The training data may have minor precision differences from the full sidecars used to build data.pt. The effect is almost certainly negligible (charm values are ±0.001, and the error is in the least-significant digits), but it has not been rigorously bounded.

---

## What this experiment tells us about the regime reset

Despite the validation gaps, the exp_153 results answer the primary question the regime reset was designed to ask:

**Q: Can the supervised model engage with the full-day opportunity set (30-270)?**

**A: Yes.** The model traded 673 times across 52 days with a non-random win rate. It did not collapse, did not fully abstain, and maintained directional diversity. The problem is not engagement — it is selectivity. The model overtrades because it has not learned when to stand down.

**Q: Is the 108% drawdown a structural failure?**

**A: Not necessarily.** At 12.9 trades/day over 52 days with PF 0.745, the cumulative loss is mechanical: many slightly-losing trades compound. The old regime (60-105) masked this by limiting the opportunity window to 45 bars. The model's actual decision quality may be similar — it's just being tested on 5x more decisions now.

**Q: Should we revert to 60-105?**

**A: No.** The audit confirmed the 60-105 regime was a legacy filtering choice. The model's overtrading under 30-270 is information, not a reason to hide from it.

---

## What must happen before the next GPU experiment

Per the audit findings:

1. **Fix harness_eval.py** — pass `breakeven_trigger_pct` and `extra_trailing_tiers` from `DEFAULT_POLICY` to both `simulate_trade()` calls. Rebuild suite. Re-run gate.
2. **Fix or quarantine documentation** — `feature_schema.md` indices are wrong from position 28 onward. `data_contract.md` claims 47 features and wrong schema version.
3. **Run full-corpus zero-Greeks census** — confirm whether Gate G6 passes (<10%) or fails (>=10%). If fails, sidecars must be rebuilt with `contract_valid=0` for solver-failed rows.
4. **Update RuntimeConfig.max_contracts_per_bar** — currently 100, actual is 285.
5. **Add post-upload sidecar hash verification** — current check is directory existence only.

---

## Regime status

The full-day regime (30-270) remains the active mainline. The regime is frozen:
- No evaluator changes
- No baseline changes
- No window changes
- No feature/schema changes
- No promotion-path changes

exp_153 is logged as `regime: full_day_30_270`, status `screening`, in the lab notebook. It is not entered in `results.tsv` (screening runs are not official results).

The next experiment should be run after the audit blockers are resolved. It can use the same experiment ID (exp_153) for the official 5-fold, or a new ID if code changes are made to address audit findings.
