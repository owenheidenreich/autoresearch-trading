# Horizon vs the Charter's Big-Loss Discipline (2026-08-04)

**Result: no horizon from 5 to 240 minutes satisfies the signed Charter.** The strategy class fails on
charter grounds independently of every edge argument, using data already owned and no fitting.

## Why this was run

"Hold longer" was the one conclusion that survived every attack, including Codex's adversarial review:
friction is roughly fixed per round trip while the move being chased grows with horizon, so the required
win rate falls from 116%/80% at one minute to ~54% at sixty.

But holding longer also lets each loser run further, and
[`PROTOCOL101_TRADER_CHARTER.md`](../contracts/PROTOCOL101_TRADER_CHARTER.md) (SIGNED 2026-07-25) caps the
big-loss bucket at **2%** — *"the killer discipline is the bottom row staying under 2%."*

Two constraints moving in opposite directions. Nobody had checked whether a window exists between them.

## Result

Returns are on premium paid (charter definition), frozen $3.00 round-trip fees, full-length holds only —
no forced-flat truncation. 1,031 OOF trajectories.

| Hold | n | mean net $ | big win % | scratch % | **big loss %** | charter ok |
|---|---|---|---|---|---|---|
| 5m | 1031 | −44.84 | 26.8 | 15.4 | **23.1** | ✗ |
| 15m | 1031 | −54.06 | 30.1 | 10.7 | **39.1** | ✗ |
| 30m | 1031 | −60.06 | 31.7 | 5.9 | **49.1** | ✗ |
| 60m | 852 | −62.81 | 30.6 | 3.1 | **56.3** | ✗ |
| 90m | 845 | −35.08 | 30.7 | 3.7 | **57.8** | ✗ |
| 120m | 667 | −42.81 | 30.1 | 3.1 | **61.3** | ✗ |
| 180m | 490 | −30.20 | 28.2 | 1.6 | **65.1** | ✗ |
| 240m | 315 | −88.79 | 26.0 | 1.9 | **68.9** | ✗ |

Charter target for comparison: big win 18%, scratch 73%, small loss 8%, **big loss 1.6%**.

## What this says

**1. The big-loss share is monotonically increasing in horizon** — 23.1% → 68.9%. The "hold longer" fix
makes the charter violation *worse*, not better. The two constraints do not merely fail to overlap; they
diverge.

**2. Even the shortest horizon is 11.5× over the limit.** At 5 minutes, big-loss share is 23.1% against a
2% cap. This does not depend on the long-horizon subsample.

**3. The outcome distribution is a barbell, not the Pickles profile.** The charter wants **73% scratches**;
this class delivers **3–15%**. It wants ≤2% big losses; this class delivers 23–69%. Big wins are actually
*fine* at 26–32% versus a target of 18%.

That third point is the substantive one. The Pickles profile depends on an exit discipline that
"scratches anything that stops working" — the 73% scratch column is *"risk management wearing a win's
clothing."* **SPXW 0DTE long options do not offer scratches.** They are too high-gamma to sit still: a
position either resolves large-favourable or large-adverse. There is no quiet middle for a scratch engine
to operate in.

This is a structural incompatibility between the instrument and the signed risk profile, not a modelling
shortfall. No entry model, exit model, feature family, or horizon repairs it, because the charter's
central risk mechanism requires an outcome bucket this instrument does not produce.

## Limitations

- `n` falls at longer horizons (852 → 315) because full-length holds were required rather than truncating
  at forced flat. That biases long-horizon rows toward earlier entries. The conclusion does not rest on
  them: the 5-minute row uses the full 1,031 and is already 11.5× over.
- Returns use the frozen `FILL_LAW` entry price. The owner's real fees are $0.65/side rather than $1.50,
  but fees move the mean by ~$1.70 and cannot move a 23–69% big-loss share to 2%.
- These are the existing OOF trajectories, so entry selection is the Phase-1 policy. A different entry
  would change *which* trades, not the instrument's gamma.

## Reproduction

`horizon_charter_sweep.py` (session scratchpad). Inputs: `reports/phase1_four_box/trajectory_outcomes.parquet`
and `exit_features/session=*/*.parquet` on the SSD. Bucket boundaries come from
`v4/research/pathd_model_gate.py` (`SCRATCH_BAND`, `BIG_LOSS_RETURN`, `MAX_BIG_LOSS_SHARE`).

*Signed: Claude Opus 5 — 2026-08-04.*
