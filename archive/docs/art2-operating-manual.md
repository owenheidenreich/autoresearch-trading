# ART² Operating Manual

Single source of truth. If anything conflicts with this file, this file wins.

---

## 1. Mission

Build a profitable SPX 0DTE **long options** trading bot.

**v17 approach:** Pure SPX prediction model. The model predicts where SPX goes (15/30/60 bar horizons). Trading logic (strike selection, stops, sizing) is separate code. No P&L in training. No hindsight.

**Ground truth hierarchy:**
1. **Paper trading P&L (IBKR)** -- ultimate truth
2. **Prediction accuracy (direction, rank correlation)** -- model quality
3. **Replay backtest PF** -- trading logic validation
4. **Training score** -- proxy only

**You (Claude Code) are the strategic brain.** No daemon, no subprocess, no API calls. You run experiments, analyze results, make decisions, and iterate -- all directly in this interactive session.

---

## 2. Router

When the user gives an instruction, route to the correct protocol:

| User says | Go to |
|-----------|-------|
| "train", "run experiments", "run ART²" | → §3 TRAIN |
| "PBT", "hyperparameter sweep" | → §4 PBT |
| "paper trade", "go live", "market open" | → §5 LIVE |
| "replay", "backtest", "validate" | → §6 VALIDATE |
| "analyze", "research", "what's wrong" | → §7 RESEARCH |
| "what should we change", "brainstorm" | → §7 RESEARCH then §8 DECIDE |
| "fresh start", "new version" | → §9 FRESH START |
| "status", "where are we" | → §10 STATUS |

**At any time of day:** If market is open and user hasn't explicitly said "train", default to §5 LIVE. If market is closed, default to whatever the user asked.

---

## 3. TRAIN (Sequential Experiments)

### 3a. Preflight

```bash
cat training/.best_score                             # current best
ls training/best_model.pt 2>/dev/null || echo "No model"
python3 -m pytest tests/test_version_consistency.py -v  # version gate
```

**GPU check:** If no `.deploy-state` or SSH fails → boot GPU:
```bash
./infra/deploy.sh boot && ./infra/deploy.sh start
```

If GPU already up but code changed → re-upload:
```bash
./infra/deploy.sh stop -y && ./infra/deploy.sh boot && ./infra/deploy.sh start
```

If GPU already up and no code changed → reuse as-is.

### 3b. Run experiments

Run 5 experiments (one batch). Each takes ~7 minutes.

**Autoresearch mode (preferred):** Propose a mutation to train.py, test it, keep/revert.
```bash
# 1. Read current train.py and lab_notebook.md
# 2. Propose ONE small change (hyperparameter, loss weight, threshold)
# 3. Write mutated train.py to /tmp/mutation.py
# 4. Test it:
python3 tools/inner_loop.py experiment --mutation /tmp/mutation.py --summary "hypothesis: what and why"
```

**Baseline mode:** Run current train.py unchanged (for establishing a baseline or warm-start compounding).
```bash
python3 tools/inner_loop.py experiment --summary "baseline (no mutation)"
```

After each, note: score, PF, WR, verdict (KEEP/REVERT). Stop early if:
- 3+ consecutive reverts with same reason → model is stuck, go to §7 RESEARCH
- Training crash → diagnose and fix before continuing

### 3c. Checkpoint (after 5 experiments)

```bash
python3 tools/art2.py analyze
python3 training/replay.py --backtest --model training/best_model.pt --output-dir results/art2/cycle-NNN/replay
python3 tools/art2.py report
```

Read the briefing. Go to §7 RESEARCH to analyze, then §8 DECIDE.

### 3d. Loop

Based on §8 decision:
- **Let it cook** → run 5 more (back to §3b)
- **Make a change** → apply change, decide fresh/warm start (§9), teardown GPU if code changed, back to §3a
- **Unsure** → present options to user, ask

---

## 4. PBT (Population-Based Training)

Use PBT when: model at stable baseline, score stuck after 10+ experiments, want to tune hyperparameters (loss weights, LR, etc.) without code changes.

Do NOT use PBT when: model has structural problems (wrong labels, train/eval mismatch, architecture issues). PBT tunes knobs — it can't fix broken pipes.

```bash
python3 tools/inner_loop.py pbt-init --population 6 --generations 3
python3 tools/inner_loop.py pbt-run
```

PBT runs to completion autonomously. After it finishes:
1. Check results: `python3 tools/inner_loop.py status`
2. Replay backtest the best model
3. Compare against pre-PBT baseline
4. If improved → keep. If not → revert to pre-PBT model.

PBT always warm-starts (same code, different hyperparameters).

---

## 5. LIVE (Paper Trading)

**Prerequisites:** IB Gateway running on port 4002. Market hours: 9:30-16:00 ET.

```bash
# 1. Refresh context
python3 tools/paper_live.py --context-only --port 4002

# 2. Start paper trader
python3 tools/paper_live.py --paper-auto --port 4002 --client-id 80 --kill-switch results/live/kill_switch

# 3. Dashboard
python3 tools/live_dash.py    # localhost:8421

# 4. After market close
python3 tools/export_trades_csv.py --date $(date +%Y-%m-%d)
```

**Safety:** Paper account guard (DU* only), 5% daily loss limit, 10% hard kill, EOD flatten at 4PM ET. Kill switch: `echo kill > results/live/kill_switch`.

**After each trading day:** Compare paper trades against replay backtest. If divergence > 20%, investigate execution realism before training more.

---

## 6. VALIDATE (Replay Backtest)

Run anytime to assess model quality without GPU cost.

```bash
# Standard validation set
python3 training/replay.py --backtest --model training/best_model.pt

# Full period (train + val)
python3 training/replay.py --backtest --model training/best_model.pt --all-dates

# Save to specific directory for comparison
python3 training/replay.py --backtest --model training/best_model.pt --output-dir /tmp/replay_test
```

**Analyze trades:**
```python
import csv
trades = list(csv.DictReader(open('backtest_output/backtest_trades.csv')))
# Key columns: bars_held, pnl_pct, entry_gate_prob, exit_reason_codes, direction, strike
```

Key metrics to check: WR, PF, total P&L, hold duration distribution, exit reason breakdown, time-of-day performance, conviction vs outcome correlation.

---

## 7. RESEARCH (Analysis & Hypothesis Formation)

### 7a. Read the data

1. **Experiment metrics** — scores, PF, WR across recent experiments
2. **Trade-level replay data** — time-of-day, hold duration, exit reasons, strike selection
3. **Domain knowledge** — `docs/domain/pickles-trading-knowledge.md`
4. **What's been tried** — `training/lab_notebook.md` (37 anti-patterns), `docs/journal/art2-notebook.md` (strategic dead ends)

### 7b. Diagnose

Priority order — stop at first YES:

1. **Train/eval mismatch?** (training PF diverges >20% from replay PF) → Fix mismatch in prepare.py/train.py/replay.py
2. **All experiments crashed?** → Fix infrastructure bug in train.py
3. **Inner loop stuck?** (0% accept, 5+ experiments) → Structural change needed
4. **Model improving?** (kept > 0, score trending up) → Let it cook
5. **Sharp optimum?** (high baseline, nothing beats it) → Analyze trades for execution-layer improvements
6. **Feature gap?** (model blind to known pattern) → Add feature, rebuild data.pt

### 7c. Hypothesize

Form 2-3 candidate ideas. For each:
- **Observation** from the data
- **Domain knowledge** prediction
- **Proposed change** (specific files and edits)
- **How to validate** (expected metric change, zero-GPU test if possible)

### 7d. Validate cheaply first

Before any GPU spend, check if the hypothesis can be validated on existing data:
```python
# Filter existing trades by hold duration, time, conviction, etc.
# Compute counterfactual metrics
```

If the pre-validation fails → hypothesis is wrong, don't waste GPU.

---

## 8. DECIDE

After research, pick ONE action:

| Action | When | GPU needed? |
|--------|------|-------------|
| **Let it cook** | Model improving, no issues found | No change |
| **Hyperparameter tweak** | Same code, adjust loss weights/LR | Warm start, keep GPU |
| **Code change** | Fix mismatch, change labels, add feature | May need fresh start, teardown GPU |
| **Execution filter** | Model good but trades poorly filtered | No GPU — replay.py change |
| **PBT sweep** | Stable baseline, want to optimize knobs | → §4 PBT |
| **Ask human** | Multiple competing paths, big design change | Present options clearly |

**Before any code change:**
1. Run version consistency: `python3 -m pytest tests/test_version_consistency.py -v`
2. Determine fresh vs warm start (§9)
3. Sync best_train.py: `cp training/train.py training/best_train.py`

**After any code change:** Document what changed and why in `docs/journal/project-chronicle.md` (2-4 sentences).

---

## 9. FRESH START vs WARM START

**Warm start test — ALL must be true:**
1. Model architecture unchanged (same heads, dimensions, feature count)
2. Loss function computes same gradients for same inputs (no semantic change)
3. Training data (data.pt) unchanged
4. No bugs fixed that changed what model sees or how loss is computed

If ALL pass → **warm start**. If ANY fail → check tables below.

### MUST Fresh Start
| Change | Why |
|--------|-----|
| Feature count changed | Input projection weights wrong shape |
| Position state dims changed | Position projection misaligned |
| Architecture changed (D_MODEL, DEPTH, N_HEADS) | Weight shapes incompatible |
| Head added/removed | Random weights produce noise |
| Head output dims changed | Old weights predict wrong outputs |

### SHOULD Fresh Start
| Change | Why |
|--------|-----|
| Loss function semantics changed | Gradients push wrong direction |
| Label generation changed | Supervision signal means different thing |
| data.pt rebuilt with different features/labels | Data doesn't match learned weights |
| Multiple structural changes at once | Can't attribute improvement to any one change |

### Safe to Warm Start
| Change | Why |
|--------|-----|
| Hyperparameter-only (loss weights, LR, dropout) | Same model, same loss, same data |
| Bug fix in non-training code (replay.py, monitor.py) | Training loop unaffected |
| Evaluation-only changes (score formula, replay logic) | Model weights still valid |

### Fresh Start Procedure
```bash
rm training/best_model.pt
echo "-5.0" > training/.best_score
rm training/.inner_loop_state.json training/.pbt_state.json
cp training/train.py training/best_train.py
```

**If uncertain, fresh start.** Cost of poisoned warm start > cost of retraining.

---

## 10. STATUS (Quick Health Check)

```bash
python3 tools/art2.py time                              # time + market
cat training/.best_score                                 # best score
python3 tools/inner_loop.py status                       # training state
cat .deploy-state 2>/dev/null && ./infra/deploy.sh status  # GPU
python3 -m pytest tests/test_version_consistency.py -v   # version gate
```

---

## 11. Model Reference (v17)

```
Input: (batch, 120 bars, 38 features)
  → Transformer (d=64, heads=4, depth=3, Pre-LN, causal)
  → 3 heads: Return(3), Confidence(1), Action(3)

Account state: 5 dims (growth, consec_losses, daily_pnl, win_rate, drawdown)
Labels: forward SPX returns (15/30/60 bars) + realized vol + action target
Loss: Huber(returns) + MSE(confidence) + MSE(action)
Score: direction_accuracy * (1 + rank_correlation)
```

**Full architecture contract:** `training/program.md`
**Anti-patterns (37 things that don't work):** `training/lab_notebook.md`
**Strategic dead ends:** `docs/journal/art2-notebook.md`
**Domain knowledge:** `docs/domain/pickles-trading-knowledge.md`

---

## 12. Key Invariants

- `best_model.pt`, `best_train.py`, `.best_score` are **always in sync**
- Version consistency gate blocks training if satellite files drift from train.py
- Score comparisons are only valid within the same data.pt
- NEVER compare 70-day era scores (5-42) with 298-day era scores (0.05-1.0)

---

## 13. Commands

| Task | Command |
|------|---------|
| Deploy GPU | `./infra/deploy.sh boot && ./infra/deploy.sh start` |
| Run experiment | `python3 tools/inner_loop.py experiment --summary "hypothesis"` |
| PBT sweep | `python3 tools/inner_loop.py pbt-init --population 6 && pbt-run` |
| Replay backtest | `python3 training/replay.py --backtest --model training/best_model.pt` |
| Rebuild data | `python3 training/prepare.py --use-spx` |
| Version check | `python3 -m pytest tests/test_version_consistency.py -v` |
| Monitor dashboard | `python3 tools/monitor.py` (localhost:8420) |
| Paper trade | `python3 tools/paper_live.py --paper-auto --port 4002 --client-id 80 --kill-switch results/live/kill_switch` |
| Live dashboard | `python3 tools/live_dash.py` (localhost:8421) |
| Export trades | `python3 tools/export_trades_csv.py [--date YYYY-MM-DD]` |
| Teardown GPU | `./infra/deploy.sh stop -y` |
| Fresh start | `rm training/best_model.pt; echo -5.0 > training/.best_score; rm training/.inner_loop_state.json; cp training/train.py training/best_train.py` |

---

## 14. Policies

- **Repair over workaround.** Fix root cause. No TODO/HACK/FIXME without fix.
- **Minimal documentation.** Don't create new doc files. Use existing: chronicle, art2-notebook, lab_notebook, program.md, this file.
- **Time awareness.** User is PT. Market is ET (9:30-16:00).
- **Research before GPU.** Always analyze existing data and form hypotheses before spending GPU time.
- **One change at a time.** Make one testable change, run 5 experiments, measure. Never bundle multiple structural changes.
- **Validate cheaply first.** If a hypothesis can be tested on existing trades (filtering, counting), do that before training.

---

*This file is the complete ART² protocol. There is no other protocol document to consult.*
