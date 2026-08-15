# Environment Diagnostics

Policy: `ask_to_bid_stop65_target150_hold45m`

Jan-Mar 2026 is real market truth, but only one environment slice. The goal is to identify causal environment descriptors the model can learn from, then require validation-selected patterns to survive holdout scoring.

## Validation-Selected Rules

| Rule | Val Trades | Val PnL | Val PF | Test Trades | Test PnL | Test PF |
|---|---:|---:|---:|---:|---:|---:|
| `P|range_bucket=mid` | 30 | 6760 | 1.677 | 37 | -3274 | 0.819 |
| `C|above_vwap=False|momentum_15m_sign=negative|time_bucket=post_open_morning` | 30 | 6740 | 1.549 | 35 | 250 | 1.018 |
| `C|momentum_15m_sign=positive|time_bucket=post_open_morning` | 35 | 6680 | 1.887 | 44 | -2818 | 0.850 |
| `C|above_vwap=False|time_bucket=post_open_morning` | 31 | 5678 | 1.456 | 37 | 546 | 1.035 |
| `C|above_vwap=True|momentum_15m_sign=flat|time_bucket=post_open_morning` | 23 | 4824 | 2.109 | 27 | -2624 | 0.784 |
| `C|above_vwap=True|momentum_15m_sign=negative|time_bucket=post_open_morning` | 22 | 4606 | 2.223 | 24 | -5038 | 0.616 |
| `C|above_vwap=True|time_bucket=post_open_morning` | 28 | 4594 | 1.717 | 38 | -3936 | 0.772 |
| `C|momentum_15m_sign=negative` | 127 | 4561 | 1.109 | 153 | -18171 | 0.692 |
| `C|vix_bucket=high` | 122 | 4386 | 1.107 | 175 | -25130 | 0.673 |
| `C|momentum_15m_sign=negative|time_bucket=post_open_morning` | 36 | 4268 | 1.322 | 42 | -524 | 0.972 |
| `C|range_bucket=low` | 26 | 4038 | 1.390 | 25 | -7720 | 0.528 |
| `C|above_vwap=False` | 98 | 3999 | 1.113 | 133 | -18996 | 0.669 |
| `C|mean_reversion_side=True` | 98 | 3999 | 1.113 | 133 | -18996 | 0.669 |
| `C|above_vwap=True|momentum_15m_sign=positive|time_bucket=post_open_morning` | 28 | 2794 | 1.458 | 37 | -3624 | 0.778 |
| `P|range_bucket=low` | 26 | 2438 | 1.189 | 25 | -2990 | 0.817 |
| `C|momentum_15m_sign=positive|time_bucket=late_afternoon` | 43 | 2054 | 1.241 | 57 | -4044 | 0.807 |
| `C|trend_aligned=False` | 145 | 1815 | 1.041 | 169 | -23313 | 0.662 |
| `P|above_vwap=True|momentum_15m_sign=positive|time_bucket=midday` | 35 | 1610 | 1.188 | 39 | -138 | 0.987 |
| `P|momentum_15m_sign=positive|time_bucket=midday` | 48 | 1564 | 1.120 | 58 | -3096 | 0.814 |
| `P|above_vwap=False|momentum_15m_sign=positive|time_bucket=midday` | 22 | 1486 | 1.227 | 31 | -3122 | 0.687 |

## Test-Ranked Survivors

| Rule | Val PnL | Test Trades | Test PnL | Test PF |
|---|---:|---:|---:|---:|
| `P|above_vwap=True|momentum_15m_sign=positive|time_bucket=late_afternoon` | 1253 | 32 | 3776 | 1.440 |
| `P|above_vwap=True|momentum_15m_sign=flat|time_bucket=late_afternoon` | 330 | 25 | 1815 | 1.302 |
| `C|momentum_15m_sign=flat|time_bucket=post_open_morning` | 1090 | 43 | 894 | 1.048 |
| `P|above_vwap=False|momentum_15m_sign=negative|time_bucket=midday` | 360 | 40 | 650 | 1.052 |
| `C|above_vwap=False|time_bucket=post_open_morning` | 5678 | 37 | 546 | 1.035 |
| `C|above_vwap=False|momentum_15m_sign=negative|time_bucket=post_open_morning` | 6740 | 35 | 250 | 1.018 |
| `P|above_vwap=False|time_bucket=midday` | 136 | 44 | 42 | 1.003 |
| `P|above_vwap=True|momentum_15m_sign=positive|time_bucket=midday` | 1610 | 39 | -138 | 0.987 |
| `C|momentum_15m_sign=negative|time_bucket=post_open_morning` | 4268 | 42 | -524 | 0.972 |
| `C|time_bucket=post_open_morning` | 1304 | 44 | -1038 | 0.948 |
| `C|above_vwap=True|momentum_15m_sign=flat|time_bucket=late_afternoon` | 765 | 25 | -2395 | 0.711 |
| `C|above_vwap=True|momentum_15m_sign=negative|time_bucket=midday` | 1250 | 28 | -2596 | 0.702 |
| `C|above_vwap=True|momentum_15m_sign=flat|time_bucket=post_open_morning` | 4824 | 27 | -2624 | 0.784 |
| `C|momentum_15m_sign=positive|time_bucket=post_open_morning` | 6680 | 44 | -2818 | 0.850 |
| `P|range_bucket=low` | 2438 | 25 | -2990 | 0.817 |
| `P|momentum_15m_sign=positive|time_bucket=midday` | 1564 | 58 | -3096 | 0.814 |
| `P|above_vwap=False|momentum_15m_sign=positive|time_bucket=midday` | 1486 | 31 | -3122 | 0.687 |
| `P|range_bucket=mid` | 6760 | 37 | -3274 | 0.819 |
| `P|momentum_15m_sign=negative` | 1116 | 153 | -3396 | 0.939 |
| `C|above_vwap=True|momentum_15m_sign=positive|time_bucket=post_open_morning` | 2794 | 37 | -3624 | 0.778 |
