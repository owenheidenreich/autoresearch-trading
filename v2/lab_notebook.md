# v2 Lab Notebook

## Data Audit (2026-04-03)

Full pipeline audit revealed 5 critical problems. See `v2/docs/data_audit_findings.md`.

Key findings:
- Oracle labels had 100% win rate (zero losers in training). PF=388 was an artifact.
- Moneyness drift: 84% of bars had >5pt drift from opening ATM. 58% of "OTM5 call" were actually ITM.
- Narrow grid: only 14 of 209 available strikes downloaded from Polygon.
- Spreads estimated incorrectly (old lookup table off by 4-20x).
- Simple momentum signals had no edge (PF=0.49-0.54).

## Signal Scan (2026-04-03)

Tested all 39 original features as direction signals against always-put baseline (PF=1.106).

Edge found in volatility-regime features:
- `option_spread_width`: PF=1.855 (+0.748 vs baseline)
- `session_range_pct`: PF=1.435 (+0.328)
- `rsi_7`: PF=1.219, `bollinger_position`: PF=1.216
- `atm_iv`: PF=1.182, `atm_gamma`: PF=1.176

**Pattern:** High vol/range/gamma = calls win (gamma convexity). Low vol = puts win (theta decay). Multiple features confirm independently.

Momentum (ret_6) has NO edge (PF=0.81).

## Data Rebuild (2026-04-03)

### Wide-grid download
- ATM +/- 100pt (82 contracts per day) from Polygon flat files
- 994 days, 2.2 GB, avg 41 strikes per day
- After 50pt intraday move, still 50pt OTM coverage

### Enriched features (16 new)
- `current_moneyness_pct`, `intraday_drift_pct`, `near_atm_moneyness_pct`
- `near_atm_call_volume`, `near_atm_put_volume`, `call_put_flow_ratio`
- `log_total_volume`, `chain_call_put_ratio`, `log_chain_volume`
- `volume_zero_flag`, `call_hl_range_pct`, `near_atm_call/put_price_norm`
- `theta_acceleration`, `near_atm_transactions`

All features are RELATIVE (moneyness %, normalized prices) so patterns at SPX 4300 transfer to SPX 6500.

### Risk-grid labels
- 64 combos (4 stops x 4 targets x 4 holds) searched per bar
- Direction from session_range_pct volatility regime (causal, no lookahead)
- Same-contract simulation (no switching artifacts)
- Gate=True only when best combo is profitable
- 87K signal bars, 62K gate=True (72%), 24K gate=False (28%)
- Risk diversity: 4 distinct stops, 4 targets, 4 holds

### NaN handling
- 27% of bars had NaN (0DTE options stop trading near close)
- Root cause: no trades in Polygon flat file = no price
- Fix: forward-fill within day (matches live IBKR behavior -- stale quotes)
- Remaining NaN (start-of-day): filled with 0

### Cost model
- Commission: $1.30 round trip (0.13% on $10 option, negligible)
- Bid-ask spread: $0.30 round trip estimate (conservative)
- Total cost per trade: ~$1.60

## Smoke Test (2026-04-03)

1-epoch local test on CPU with new data:
- train_loss=1.11, val_loss=1.04
- gate_acc=79.4% (learning -- baseline 72%)
- dir_acc=67.4% (learning -- baseline 50%)
- No NaN in loss. Model trains.

## Experiment: honest_baseline on Akash H100 (2026-04-03)

**First training with honest data.** 25 epochs on H100, 305s training, 46s replay.

**Training metrics:**
- val_loss: 1.06 -> 0.60 (converging)
- dir_acc: 58.8% -> 88.3% (direction head LEARNED)
- gate_acc: 79.4% -> 79.8% (gate head DID NOT LEARN -- stuck at majority class baseline)

**Replay on promote_mask (60 held-out days):**
- 52 trades total, 6 traded days (way too few)
- ALL 52 trades are puts (call_count=0) -- direction collapse
- WR=98.1%, PF=102 -- looks good but only 52 trades on 6 days
- Score: -0.5 (gate failure: too_few_traded_days, 6 < 15 minimum)

**Diagnosis:**
1. **Gate head stuck.** 72% of training labels are gate=True. The model learns to always
   predict True (gets 72% accuracy). The gate loss (BCE with pos_weight=0.3) doesn't give
   enough gradient to learn selectivity. Need to either:
   - Increase pos_weight (penalize false positives more)
   - Use a different gate architecture (separate classifier)
   - Add the P&L as a weighting signal (higher loss for confident-but-wrong predictions)

2. **Direction collapse to puts.** The volatility-regime signal (session_range_pct) during
   the promote_mask period (Dec 2025 - Mar 2026) is consistently below the training median,
   so it always says "put." The model learned this correctly but it means no directional
   diversity. Need to either:
   - Use a more balanced direction signal
   - Force a minimum call fraction in replay
   - Use multiple direction features (not just session_range_pct)

3. **Gate threshold too strict.** The default policy gate_threshold=0.5 filters almost
   everything because the gate output is near 0.5 (barely learned). Lower the threshold
   for initial experiments, then let the model learn to be more decisive.

**Key insight:** The DATA pipeline is honest and working. The MODEL needs tuning.
This is what the ART² loop is for.

**Next experiments to try:**
- Increase GATE_POS_WEIGHT from 0.3 to 1.0 or higher
- Lower gate_threshold from 0.5 to 0.3 in policy.py
- Add more direction features (not just session_range_pct median split)
- Try weighting gate loss by |label_pnl| so confident trades matter more
