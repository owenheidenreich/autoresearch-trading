# 0DTE Contract Decay Analysis — 2026-04-20

> Research trail: **[this doc, #1]** → [attribution_full_2026-04-20.txt](attribution_full_2026-04-20.txt) → [stage1_research_findings_2026_04_20.md](stage1_research_findings_2026_04_20.md) → [omar_findings_2026_04_20.md](omar_findings_2026_04_20.md). See [INDEX.md](INDEX.md) for the full reading order.

## Purpose

Empirical basis for the v3 Layer 0 guardrail parameters. Replaces intuition-based
premium-cap numbers with evidence from actual 0DTE contract price paths.

## Method

- Sampled 40 days evenly from 2022-04-11 through 2026-01-21 (986-day cache)
- For each entry bar in {09:45, 10:00, 10:30, 11:00}, traced every observable
  contract's mid from entry to 15:45 ET (bar 375), skipping the last-15-minute
  chaos
- 5,194 contract-entry paths analyzed
- For each path: max_drawdown_pct, max_gain_pct, final_pnl_pct, plus two flags
  (`went_to_zero` = final ≤ -90%, `recovered` = drawdown ≤ -60% AND final > 0)
- Script location: `v3/analysis/decay_analysis.py`

## Findings — by entry premium tier

| Tier          | n     | med_drawdown | med_final | %wiped (≤-90%) | %recovered from -60% |
|---------------|-------|-------------|-----------|----------------|----------------------|
| <$100         | 646   | -25%        | -21%      | **0%**         | —                    |
| $100-300      | 949   | -64%        | -59%      | 0%             | 3.5%                 |
| $300-700      | 782   | -79%        | -65%      | 9%             | 6.5%                 |
| **$700-1500** | **835** | **-65%**  | **-37%**  | **16%**        | **7.4%**             |
| $1500-3000    | 908   | -46%        | -16%      | 7%             | 8.9%                 |
| ≥$3000        | 1,074 | -27%        | -2%       | 2%             | 2.9%                 |

## Findings — by entry abs(delta)

| Tier          | n     | med_drawdown | %wiped |
|---------------|-------|-------------|--------|
| 0.05-0.15     | 1,041 | -58%        | **0%** |
| 0.15-0.25     | 561   | -77%        | 7%     |
| **0.25-0.35** | **421** | **-76%**  | **15%** |
| 0.35-0.50     | 551   | -67%        | 14%    |
| 0.50-0.75     | 940   | -54%        | 8%     |
| 0.75-0.99     | 876   | -31%        | 3%     |

## Findings — wipe rate by premium × entry minute

(entry minute 15 = 09:45 has no data; v2 sidecar coverage starts at bar 30)

| premium        | 10:00  | 10:30  | 11:00  |
|----------------|--------|--------|--------|
| <$100          | 0.0%   | 0.0%   | 0.0%   |
| $100-300       | 0.0%   | 0.0%   | 0.0%   |
| $300-700       | 10.8%  | 8.2%   | 7.5%   |
| **$700-1500**  | **18.3%** | **16.1%** | **14.0%** |
| $1500-3000     | 8.7%   | 5.4%   | 5.7%   |
| ≥$3000         | 2.7%   | 1.5%   | 0.3%   |

Time-of-day effect is modest — later entries are ~25% safer in wipe rate than
10:00 entries in the mid-premium tier, consistent with less forward-theta
exposure, but the difference is not dramatic.

## Findings — recovery-after-drawdown (Pickles case)

| premium        | n hit ≤-60% | % recovered to positive |
|----------------|-------------|-------------------------|
| $100-300       | 574         | 3.5%                    |
| $300-700       | 527         | 6.5%                    |
| **$700-1500**  | 448         | **7.4%**                |
| $1500-3000     | 361         | **8.9%**                |
| ≥$3000         | 209         | 2.9%                    |

5-9% of contracts that hit -60% end positive. Real but rare. A fixed stop-loss
rule would exit all of these. A learned state-dependent exit model could
plausibly identify the recoverable subset.

## Implications for the guardrail

1. **<$100 tier never wipes over 646 entries** — cheap deep-OTM contracts
   mechanically can't go to zero. But they lose on average (median final -21%),
   so they're not worth systematically trading. Premium floor stays at $100 to
   exclude the far-OTM garbage.

2. **$700-$1500 tier has the highest wipe rate (16%)** — and this is exactly
   the zone of morning 0.20-0.30 delta SPX 0DTE contracts that we want to
   access. This is where real stop discipline matters, but Layer 0 does not
   assume any specific stop fraction. The DAILY BRAKE catches bad days.

3. **≥$3000 tier rarely wipes (2%)** — deep ITM with intrinsic-value floor.
   But these are inefficient 0DTE trades (low gamma per dollar). No reason to
   include them; capping premium at $1200 keeps us out.

4. **Fixed stop fractions are incompatible with the data** — 5-9% of -60%
   drawdown contracts recover. A 35% or 50% stop would cut all of them. Layer
   0 stays assumption-free; exit discipline is Layer 2/3.

## Chosen guardrail parameters — "Moderate" tier

- `max_premium_per_trade = min($1200, 4.8% × equity)` = $1200 at $25k
- `premium_floor = $100`
- `delta_floor = abs(delta) ≥ 0.12`
- `spread_fraction_cap = 0.10`
- `daily_loss_brake = min($1500, 6% × equity)` = $1500 at $25k
- `max_full_premium_losers_per_day = 2`

Worst-case single day: one full-premium wipe at $1200 = -4.8%; a second
significant loser triggers either the dollar brake or the 2-losers rule; halt
with the account at roughly -6% to -10% of starting equity depending on
whether the second loss was also a wipe. Survivable.
