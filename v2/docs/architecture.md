# ART2 System Architecture

Seven independent sections. Each can be analyzed, tested, and improved in isolation.
Data flows top-to-bottom. Each section has a clear contract with its neighbors.

```
                         [1. DATA ACQUISITION]
                          Polygon S3 flat files
                          Raw OHLCV per strike
                                  |
                                  v
                         [2. FEATURE ENGINEERING]
                          71 normalized features
                          Dual-direction P&L labels
                          4-way temporal split
                                  |
                                  v
                              data.pt
                                  |
                     +------------+------------+
                     |                         |
                     v                         v
            [3. MODEL]                [4. SIMULATOR]
            Architecture              Fill model
            Loss function             Exit logic
            Training loop             Spread costs
                  |                   Trailing stops
                  v                        |
              model.pt                     |
                  |                        |
                  +----------+-------------+
                             |
                             v
                    [5. EVALUATION]
                    Replay engine
                    Baselines (3x)
                    Score formula
                    Hard gates
                             |
                             v
                      ReplayMetrics
                             |
                    +--------+--------+
                    |                 |
                    v                 v
          [6. EXPERIMENT LOOP]   [7. LIVE TRADING]
          ART2 orchestration     IBKR execution
          Keep/revert            Order lifecycle
          Session limits         Paper/shadow/live
          Akash GPU ops
          Artifact bundles
```

---

## Section 1: Data Acquisition

**What it does:** Downloads raw SPX 0DTE option data from Polygon.

**Files:**
- `v2/pipeline/download_wide_grid.py` -- Polygon S3 flat file downloader
- `v2/pipeline/extract_raw.py` -- raw data extraction and cleaning
- `~/.cache/autoresearch-trading/data/spxw_wide/` -- local cache

**Input:** Polygon API credentials, date range
**Output:** Raw pickle files with per-strike OHLCV (82 contracts/day, 994 days)

**Key parameters:**
- ATM +/- 100 points (41 strikes per side)
- 5-point strike grid
- Full OHLCV per bar per strike (not just close prices)

**Questions an audit should answer:**
- Are the raw prices correct? Spot-check against a known source.
- Are there days with missing/corrupt data?
- What's the minimum option price in the raw data? (penny option issue starts here)
- How many strikes per day actually have meaningful liquidity?

---

## Section 2: Feature Engineering & Labels

**What it does:** Transforms raw data into the training dataset (features + labels + splits).

**Files:**
- `v2/pipeline/build_v2_dataset.py` -- main dataset builder (800 lines)
- `v2/core/labels.py` -- oracle label generator (350 lines)
- `v2/core/features.py` -- feature spec and validation (400 lines)
- `v2/core/candidates.py` -- candidate contract generation (130 lines)

**Input:** Raw Polygon cache + existing feature arrays
**Output:** `v2/data.pt` -- 387,990 bars x 71 features + labels + masks

**Features (71 total):**
- 39 market: returns, volume, VIX, Greeks, spreads, momentum
- 16 option-enriched: moneyness %, flow ratios, per-strike volume
- 16 volume/moneyness: normalized prices, call/put ratios, chain stats

**Labels:**
- `label_call_pnl` / `label_put_pnl`: forward P&L for each direction
- Fixed risk params: stop=0.30, target=0.50, hold=30 bars
- `label_trade = True` when max(call_pnl, put_pnl) > 0 (79.7% of bars!)
- `label_direction`: whichever side had higher P&L

**Splits (temporal, no leakage):**
- train: 859 days, val: 60 days, promote: 60 days, shadow: 20 days

**Questions an audit should answer:**
- The labels say 80% of bars have a profitable trade if you pick the right direction. Is that realistic or an artifact of the label methodology?
- How are P&L labels computed? Are they using the same fill/cost model as the simulator?
- Do the labels include realistic spread costs? Or only commission?
- What option prices are used for P&L computation? Mid? Bid? Ask?
- Are penny options ($0.01-$0.50) included in labels? Should they be?
- What happens when both call and put P&L are negative (20% of bars)? Is the no-trade signal correct?

---

## Section 3: Model

**What it does:** Neural network that predicts call_pnl and put_pnl for each bar.

**Files:**
- `v2/train.py` -- model architecture + training loop (460 lines, MUTABLE)
- `v2/core/policy.py` -- trading decision parameters (84 lines, MUTABLE)

**Input:** (batch, 60, 71) -- 60-bar lookback window of 71 features
**Output:** call_pnl, put_pnl (regression), risk params, derived gate/direction

**Architecture:**
- Linear(71->64) -> LayerNorm -> PositionalEncoding
- TransformerEncoder (3 layers, 4 heads, causal mask)
- RegimeEncoder: last bar features -> 16-dim FiLM conditioning
- call_pnl_head, put_pnl_head, risk_head (each FiLM-conditioned)

**Training:**
- Loss: Huber on both P&L predictions + Huber on risk targets
- Optimizer: AdamW, cosine annealing, 30 epochs or 300s budget
- Best checkpoint by val_loss

**Inference-time derivations:**
- gate_logit = max(call_pnl, put_pnl) * 5.0
- direction = argmax(call_pnl, put_pnl)
- strike = always ATM (center class)

**Questions an audit should answer:**
- The model trains on labels capped at [-33%, +49%]. But at replay time, penny options produce 50,000% returns. Is the model actually learning P&L prediction or just learning which bars have cheap options?
- gate_acc is 66% -- what does this mean? Is the model beating a "always trade" baseline on gate decisions?
- dir_acc is 64% -- on which bars? All bars or only gated bars?
- Does the model differentiate between a $0.03 option and a $30 option? The features don't directly encode option price.

---

## Section 4: Simulator

**What it does:** Executes TradeIntents against historical price bars. Produces SimulatedTrades.

**Files:**
- `v2/core/simulator.py` -- trade simulation engine (300 lines, IMMUTABLE)
- `v2/core/schema.py` -- TradeIntent and SimulatedTrade contracts (71 lines, IMMUTABLE)

**Input:** TradeIntent + historical OHLCV bars
**Output:** SimulatedTrade with P&L, exit reason, MFE/MAE

**Fill model:**
- Entry: next bar's ask price
- Exit: bar's bid price
- Spread cost: adaptive (time-of-day, VIX, moneyness) + $1.30 commission

**Exit logic (TRAILING mode):**
- Stop loss: fixed % below entry
- Take profit: fixed % above entry
- Trailing stop tiers: lock in at +30/+50/+80/+120% unrealized
- Max hold: bars limit
- EOD: forced exit at 15:59 ET

**Questions an audit should answer:**
- How does the simulator handle penny options ($0.01-$0.50)?
- What spread is applied to a $0.03 option? Is it realistic?
- Can a $0.03 put actually be FILLED at that price? What's the minimum lot/tick?
- STOP_LOSS exits show +3,734% average P&L and 86% win rate. How?
- Is net_pnl_pct computed relative to entry_price? If a $0.03 option goes to $0.10 then hits stop at $0.07, that's +133% "stop loss win."
- The trailing stop tiers (+30/+50/+80/+120%) -- are these relative to entry or to some moving reference?
- Are the 940 trades held <= 2 bars (33% of all trades) realistic? Can you enter and exit a 0DTE option in 2 minutes profitably after costs?

---

## Section 5: Evaluation

**What it does:** Runs trained model on held-out data, computes score, compares to baselines.

**Files:**
- `v2/replay.py` -- replay engine + baselines (400 lines, IMMUTABLE)
- `v2/core/metrics.py` -- scoring formula + metrics (250 lines, IMMUTABLE)

**Input:** model.pt + data.pt + promote_mask + policy
**Output:** ReplayMetrics (50+ fields) including promotion score

**Replay flow:**
1. Batch inference on all promote_mask bars
2. Convert model outputs to TradeIntents (via policy thresholds)
3. Simulate each TradeIntent (via simulator)
4. Aggregate into daily P&L, build equity curve
5. Compute score = min(sortino, 6.0) * positive_day_rate * dd_mult

**Hard gates (fail = negative score):**
- Min 30 trades, min 15 traded days
- Direction balance >= 15% minority
- Max drawdown <= 20%

**Baselines (model must beat all three):**
- Random: random direction, 2% of bars
- ATM-always: always buy ATM call, 1x/day
- Simple-rules: momentum direction signal

**Questions an audit should answer:**
- The equity curve uses dollar P&L = net_pnl_pct * entry_price * 100 * qty. A $0.03 option with +50,000% pnl = $0.03 * 500 * 100 = $1,500 per trade. Is this realistic for a $10K account?
- Does position sizing account for the option price? Or is it always 1 contract regardless of cost?
- Should the score formula weight trades by capital at risk rather than treating a $0.03 trade the same as a $30 trade?
- The baselines are very weak (random, ATM-always, simple-rules). Would a smarter baseline (e.g., buy ATM puts in high-vol, calls in low-vol) be harder to beat?

---

## Section 6: Experiment Loop (ART2)

**What it does:** Autonomous research loop. Hypothesis -> edit -> train -> evaluate -> keep/revert.

**Files:**
- `v2/ops/run_experiment.py` -- single experiment runner (200 lines, IMMUTABLE)
- `v2/ops/inner_loop.py` -- session orchestrator (225 lines)
- `v2/ops/artifact.py` -- reproducibility bundles (200 lines)
- `v2/ops/deploy.sh` -- Akash H100 lifecycle (850 lines)
- `v2/ops/monitor.py` -- dashboard (300 lines)
- `v2/program.md` -- protocol definition
- `v2/COMMANDS.md` -- command reference
- `v2/lab_notebook.md` -- experiment journal
- `v2/results.tsv` -- experiment history

**Session limits:**
- 50 experiments, 6 hours, 8 no-improve streak, 3hr plateau, 3 crashes

**Keep/revert logic:**
- KEEP if score > best AND beats all 3 baselines
- REVERT otherwise (restore train.py, policy.py, model.pt from best artifact)

**Questions an audit should answer:**
- Is the keep/revert logic correct? (Bug was found: model.pt wasn't being restored on revert)
- Are session limits appropriate?
- Does the artifact system correctly preserve reproducibility?
- Is the experiment loop optimizing for the right thing? (Currently: score formula)

---

## Section 7: Live Trading

**What it does:** Execute model decisions as real IBKR orders. Currently stubs.

**Files:**
- `v2/live/market.py` -- market data streaming (stub)
- `v2/live/decision.py` -- model inference to TradeIntent (stub)
- `v2/live/execution.py` -- IBKR order lifecycle (stub)
- `v2/live/service.py` -- orchestrator (stub)
- `v2/live/adoption.py` -- model promotion to live (stub)
- `v2/docs/execution.md` -- order state machine spec

**Reference implementation (v1, archived):**
- `archive/v1/live/execution.py` -- working IBKR order engine
- `archive/v1/live/decision.py` -- working inference engine
- `archive/v1/live/features.py` -- working feature computation
- `archive/v1/live/service.py` -- working market data streaming

**Contract:** TradeIntent flows identically through train -> replay -> live.

**Questions an audit should answer:**
- Can the current model's TradeIntent be executed on IBKR without modification?
- What's the gap between simulated fills and real IBKR fills?
- The model trades 47x/day -- is that executable with IBKR rate limits?
- Penny options ($0.03) -- does IBKR even allow orders at that price? What's the minimum?

---

## Cross-Section Issues (discovered in this session)

These problems span multiple sections and need coordinated fixes:

1. **Penny option exploitation** (Sections 2, 4, 5): Options priced $0.01-$0.50 produce 100-100,000% returns in the simulator. These drive the 80% WR and PF 350. Not realistic. Needs fixing in labels (exclude penny options?) or simulator (realistic spread on cheap options?) or evaluation (cap returns? weight by capital?).

2. **Label-simulator disconnect** (Sections 2, 4): Labels are capped at [-33%, +49%] but the simulator produces uncapped returns. The model trains on one distribution and is evaluated on a completely different one.

3. **Trade frequency** (Sections 3, 5): 47 trades/day with max_concurrent=1 and cooldown=5 bars means the model fires on nearly every eligible bar. This is spray-and-pray, not selective trading.

4. **dead_loss_cap_pct** (Sections 4, 6): The daily loss cap in policy.py is never enforced by the simulator or replay.
