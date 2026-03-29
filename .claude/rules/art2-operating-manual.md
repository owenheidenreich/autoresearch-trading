# ART² Operating Manual

Single source of truth. If anything conflicts with this file, this file wins.

---

## 1. Mission

Build a profitable SPX 0DTE **long options** trading bot. Ground truth hierarchy:
1. **Paper trading P&L (IBKR)** — ultimate truth
2. **Replay backtest PF** — historical validation
3. **Training score** — proxy only, NEVER optimize directly

## 2. The Loop

ART² is an autonomous research loop (inspired by Karpathy's autoresearch). It researches and hypothesizes FIRST (no GPU), then trains to test the hypothesis (GPU), then validates. The loop runs via `art2.py daemon`.

```
art2.py daemon
  │
  ├─ Sentinels: STOP / PAUSED / REVIEW (results/art2/)
  ├─ Budget + market awareness
  │
  └─ Per cycle (research-first):
     │
     ├─ PHASE A: RESEARCH & DECIDE (no GPU — free)
     │  ├─ RESEARCH ─ trade-level analysis vs domain knowledge
     │  ├─ REPORT ─── generate briefing.md from prior results
     │  ├─ DECIDE ─── Opus reads briefing + domain knowledge → hypothesis
     │  ├─ SNAPSHOT ─ save pipeline state before strategic changes
     │  └─ APPLY ──── execute decision (file edits to train.py, etc.)
     │
     └─ PHASE B: TRAIN & VALIDATE (GPU — costs money)
        ├─ TRAIN ──── inner_loop.py experiments on Akash H100
        ├─ ANALYZE ── collect metrics from experiments.v2.jsonl
        ├─ REPLAY ─── backtest best_model.pt on validation dates
        ├─ COMPARE ── outer loop: compare replay PF against snapshot
        ├─ DIAGNOSE ─ train vs replay mismatch detection
        ├─ VIABILITY  profitability verdict (VIABLE / NOT VIABLE)
        └─ REVIEW ─── pause for human approval (remove REVIEW sentinel)
```

**Outer loop keep/revert:** When Opus makes strategic changes (Actions B-F), the daemon snapshots the pipeline (train.py, best_train.py, best_model.pt, .best_score, replay_metrics). After the next training cycle, it compares the new replay PF against the snapshot. If replay PF regressed >10%, the pipeline is automatically reverted. History tracked in `state.json["outer_loop_history"]`.

## 3. Three Tiers

| Tier | Who | Reads | Decides | Prohibited |
|------|-----|-------|---------|------------|
| **Strategist** (Opus) | Claude Code | Briefings, domain knowledge, lab_notebook.md | Architecture, features, training mode, fresh start | score_config, run_loop.py safety |
| **Code Writer** (Sonnet) | Claude agent | program.md, lab_notebook.md | How to implement Opus's hypothesis in train.py | New nn.Module, architecture changes, score formula |
| **Mechanical** (inner_loop.py) | Python code | Nothing — no AI reasoning | Keep if score > best AND no critical flags | Strategic decisions |

**Sonnet agents are used in sequential mode only.** PBT mode is purely evolutionary (no AI in the loop).

## 4. Fresh Start vs Warm Start

A **warm start** loads `best_model.pt` and continues training from learned weights. A **fresh start** removes `best_model.pt`, resets `.best_score` to -5.0, and trains from random initialization.

Warm starts compound learning across cycles. Fresh starts throw away all learned weights. Getting this wrong wastes GPU hours either way — warm-starting a corrupted model poisons all future experiments, while fresh-starting unnecessarily loses weeks of compounded learning.

### MUST Fresh Start (non-negotiable)
These changes make old weights incompatible or semantically wrong. Warm-starting will produce garbage.

| Change | Why | Historical evidence |
|--------|-----|---------------------|
| Feature count changed (e.g., 32→37) | Input projection weights have wrong shape | v3 fresh start (37 features) |
| Position state dimensions changed (e.g., 5→7 dims) | Position projection weights misaligned | v4→v5: zero-padded 5→7 dims, value head fired 87% garbage exits |
| Model architecture changed (D_MODEL, DEPTH, N_HEADS) | Transformer weight shapes incompatible | |
| New head added or removed | New head has random weights that produce noise | v5 added value head — `strict=False` loading created Frankenstein model |
| Head output dimensions changed (e.g., value head 3→1) | Old weights predict wrong number of outputs | v7 value head MSE→BCE changed output semantics |
| Action space changed | Direction head outputs have different meaning | |

**Procedure:** Archive current model to `archive/models/vN/`, then:
1. Delete `training/best_model.pt`
2. Reset `training/.best_score` to `-5.0`
3. Delete `training/.inner_loop_state.json` (forces inner loop re-init with correct best_score)
4. Sync `best_train.py` from `train.py`
5. Run `inner_loop.py init` before first experiment

If running via `art2.py`, the fresh_start flag handles steps 1-3 automatically.

### SHOULD Fresh Start (strong default, override with justification)
The weights are technically loadable but were optimized for a different objective. They'll converge to the old behavior.

| Change | Why | Historical evidence |
|--------|-----|---------------------|
| Loss function semantics changed | Gradients push in different direction than what weights learned | v7 exit labels 98%→66.5% + BCE: old weights learned "always exit" |
| Exit/entry label generation changed | Supervision signal means something different now | v7 take-profit labels, underwater filter |
| data.pt rebuilt with different features or labels | Training data doesn't match what model learned | v3 added market structure features |
| Multiple structural changes at once | Can't attribute improvement/regression to any single change | Cycle 021: 3 changes + fresh start |
| Train/val PF divergence >50% | Model memorized training set, weights are overfit garbage | Cycle pre-v3: train PF=5.26 vs val PF=0.13 |
| Score is a stochastic outlier (can't reproduce after 10+ experiments) | Warm-starting from an unreproducible peak wastes cycles chasing a phantom | v6 score 41.92: 40 experiments couldn't improve it |

### MUST Warm Start (do NOT fresh start)
These conditions mean old weights are valuable and a fresh start wastes GPU hours for no benefit.

| Condition | Why | Example |
|-----------|-----|---------|
| Hyperparameter-only changes (loss weights, LR, batch size, dropout, weight decay) | Same model, same loss, same data — weights are optimal starting point | Tuning ENTROPY_COEFF from 0.02→0.05 |
| PBT sweeps (env-var overrides only) | By definition: same code, different hyperparameters | `inner_loop.py pbt-run` |
| Regularization knob changes (entropy coeff, temporal smooth) | Additive regularization on same objective | REG_TEMPORAL_SMOOTH adjustment |
| Score improving and no code changes planned | Compounding learning is the whole point of warm start | Score trending up across experiments |

**Warm start test (ALL must be true):**
1. Model architecture unchanged (same heads, same dimensions, same feature count)
2. Loss function computes the same gradients for the same inputs (no semantic change)
3. Training data (data.pt) unchanged
4. No bugs were fixed that changed what the model sees or how loss is computed

If ANY of these fail → it's NOT a warm start situation, check fresh start tables above.

### SHOULD NOT Fresh Start (but often mistaken for fresh-start triggers)
| Change | Why it's safe to warm start |
|--------|---------------------------|
| Bug fix in non-training code (replay.py, prepare.py eval, monitor.py) | Training loop unaffected — model weights are still valid |
| Documentation or config file changes | No effect on training |
| Label smoothing value adjustments | Minor perturbation on same supervision signal |

### MUST Fresh Start After Bug Fixes That Change Loss Landscape
Bug fixes are NOT automatically safe to warm start. If a bug fix changes any of these, it's a SHOULD Fresh Start:

| Bug fix type | Why fresh start | Historical evidence |
|-------------|----------------|---------------------|
| NaN/sentinel fill value changed (e.g., -999→0) | Old weights learned to avoid/exploit the sentinel value | v12: NaN fill -999 caused loss=359, model learned wrong gradients |
| P&L clamping/scaling changed (e.g., adding tanh) | Gradient magnitudes completely different — old weights tuned to unclamped signal | v12: uncapped 3000% OTM returns dominated loss |
| Feature normalization changed | Input distribution shifted — all learned weights miscalibrated | |
| Evaluation logic fixed in training loop | Model optimized against wrong objective | v12: MAX_TRADE_RETURN cap missing in training eval |

### Before Every Start Decision
1. **Check `archive/models/`** — is the current model worth preserving? If yes, archive it as `archive/models/vN/` with model + train code + score.
2. **Apply the warm start test** (4 conditions above). If all pass → warm start. If any fail → check MUST/SHOULD fresh start tables.
3. **State the decision and rationale** in `decision.md` — "Fresh start because [specific reason from table above]" or "Warm start because [all 4 conditions pass]."
4. **If uncertain, fresh start.** The cost of a wasted warm start (poisoned experiments, days of debugging) is higher than the cost of a fresh start (retraining from scratch).

## 5. Training Modes

**Sequential** — `inner_loop.py experiment --summary "hypothesis"`
- Trains current train.py, scores the result, KEEP (promote model) or REVERT.
- Each experiment warm-starts from last kept model.
- **inner_loop.py has no AI** — it's purely mechanical: validate → upload → train → score → keep/revert.
- When run via `art2.py daemon`, Claude Code edits train.py BEFORE calling experiment. When run manually, no AI — trains whatever train.py has as-is.
- Use for: fresh starts (early convergence), code changes, one-at-a-time testing.

**PBT** — `inner_loop.py pbt-init --population N && inner_loop.py pbt-run`
- Evolutionary parameter search. No AI, no code changes — just env var overrides (GATE_W, EXIT_W, LR, etc.).
- Creates N members with different hyperparameter configs, trains all from same baseline, promotes best, mutates, repeats for M generations.
- Use for: multi-parameter optimization once model has stable baseline.

**How to tell which is running:** `inner_loop.py status` shows `training_mode` field. monitor.py dashboard shows "Mode: sequential" or "Mode: pbt". PBT experiment records have `pbt_generation` and `pbt_member` fields.

**When to switch:** Fresh start → sequential until stable. Score stuck after 5+ experiments → PBT. PBT stagnation → back to sequential with structural changes.

## 5. Model (v14 Four-Head — Exact v10 Restoration, 38 Features)

```
Input: (batch, 120 bars, 38 features)
  → Transformer (d=64, heads=4, depth=3, Pre-LN, causal)
  → 4 heads:
     A. Gate      (2) softmax → [NO_TRADE, TRADE] (entry/exit signal)
     B. Direction (14) softmax → CALL/PUT × ATM/OTM5/10/15/20/25/30
     C. Value     (1) scalar  → expected remaining P&L (exit intelligence)
     D. Risk      (3) mixed   → stop_pct [0.15-0.60], size_frac [0-1], conviction [-1,+1]

Position state (7 dims): in_trade, bars_held, unrealized_pnl, account_health,
                          loss_streak, best_pnl_since_entry, bars_since_pnl_high
Account state  (4 dims): growth_ratio, log_account_size, daily_pnl_frac, win_rate_20
```

**Exit logic:** While holding, gate=NO_TRADE → EXIT. While flat, NO_TRADE → stay flat. Stop-loss, max-hold, EOD exits unchanged.

**Loss:** Multi-component sniper_loss (proven in v10):
- Gate: cross-entropy on `(best_pnl > 0)` binary labels + learned time-of-day weighting
- Direction: KL-divergence against P&L-weighted soft targets (return-weighted strike selection)
- PnL alignment: trade_prob × (dir_probs × stopped_pnl).sum() — direct reward signal
- Confidence: penalize high confidence on losers
- DIRECTION_ENTROPY_BONUS = 0.20 (hardcoded, prevents direction collapse)

**38 features:** v10 feature set (indices 0-37). v12's 6 raw candle features removed (redundant with existing features, no proven value).

**v14 changes from v13:** Fixed hidden gate label bug — v13 silently ran v11's AND-gate logic (regime & setup & pnl_ok ≈ 1.6% TRADE) via data.pt. Reverted to pure `pnl_ok.long()` (~50% TRADE). Removed 6 raw candle features (44→38). Restored lookback=120. Warm start from v10 weights.

**v13 (failed, score -0.27):** Intended v10 restoration but ran v11 AND-gate labels due to data.pt containing setup/regime masks. Direction collapsed to 100% PUT, 11.8% WR.

**v12 (failed, score 1.16):** Unified 15-class action head + EV loss. 48 experiments (24 sequential + 24 PBT). Single EV loss too weak — model collapsed to 1 strike (71.6%), 13% win rate, 84% stop-loss exits. Archived at `archive/models/v12/`.

**v11 (failed, score 0.48):** Regime+setup gate labels. Circular — relabeling with lagging features doesn't add information.

**v10 baseline:** Replay PF 1.74, score 16.73 **(on 70 val days — NOT comparable to current 298-day scores)**. When re-evaluated on the current 298-day val set, v10 scores 0.064. V14 (0.138) is the actual best. Archived at `archive/models/v10/`.

## 5b. Validation Set & Data Provenance

**Validation set:** 298 days (30% of trading days from 2022-03-16 to 2026-03-23). Temporal split — training on first 70% of dates, validation on last 30%. No overlap.

**Score expectations on 298-day set:**
- Scores in the **0.05-1.0 range are normal**. Do NOT compare with historical scores from the 70-day era (5-42 range).
- Score = `profit_factor × trade_sharpe × freq_mult`. More val days → more trades → higher P&L variance → lower trade_sharpe → lower score.
- Focus on **replay PF** and **paper trading P&L** as ground truth, not training score.

**Data provenance (v14+):**
- `data.pt` includes `_provenance` metadata: feature names, val day count, date range, creation timestamp
- `data.pt.sha256` sidecar file for upload integrity verification
- Model checkpoints store `data_fingerprint`, `num_val_days`, `val_date_range`
- `promoted/history.jsonl` records `data_fingerprint`, `num_val_days`, `num_features`, `lookback`
- Feature count mismatch between data.pt and code is **FATAL** (no silent truncation)

**NEVER compare scores across different data.pt files.** If data.pt is rebuilt, all prior scores are invalidated. Reset `.best_score` and start fresh.

## 6. Files That Matter

### Code-consumed config (read by Python at runtime)
| File | Read by | Purpose |
|------|---------|---------|
| `tools/opus-prompt.md` | art2.py `_invoke_opus()` | System prompt for programmatic Opus decisions |
| `docs/domain/0dte-domain-knowledge.md` | art2.py `_invoke_opus()` | Domain knowledge injected into Opus prompt |
| `docs/domain/pickles-trading-knowledge.md` | art2.py `_invoke_opus()` | Domain knowledge injected into Opus prompt |
| `training/lab_notebook.md` | art2.py `cmd_report()` | Sections extracted into briefing (What Fails, Best Runs, Next Priorities) |
| `docs/journal/art2-notebook.md` | art2.py `cmd_report()` | Outer loop memory included in briefing |
| `docs/journal/project-chronicle.md` | art2.py `_write_chronicle_entry()` | Prepended with dated entries each cycle |

### Agent instructions (auto-loaded by Claude Code framework)
| File | Purpose |
|------|---------|
| `.claude/rules/art2-operating-manual.md` | **This file.** NOT read by Python code. |

### Agent contracts (referenced, not code-consumed)
| File | Purpose |
|------|---------|
| `training/program.md` | Sonnet agent contract — model architecture, constraints, what can/cannot change |
| `training/lab_notebook.md` | Also serves as Sonnet context — dead ends, best runs, priorities |

### Non-essential (docs/misc/)
Archived docs not used during cycles. Do not update.

## 7. Decision Framework

Priority order — stop at first YES:

1. **Train/eval mismatch?** (PF diverges >20%) → Fix mismatch in prepare.py/train.py/replay.py
2. **Inner loop stuck?** (0% accept, tunnel vision) → Steer lab_notebook.md with domain knowledge
3. **Model improving?** (kept > 0, score trending up) → Let it cook
4. **Feature gap?** (blind to known pattern) → Add feature, rebuild data.pt, fresh-start
5. **Paper trading diverges from backtest?** → Investigate execution realism

## 8. Key Invariant

`best_model.pt`, `best_train.py`, and `.best_score` are **always in sync**.
- Model downloads to temp file, only promoted on KEEP
- REVERT restores train.py from backup, model unchanged
- Any experiment can crash without corrupting state

## 9. Domain Knowledge (condensed)

- **Theta:** ~1/sqrt(T). Long options bleed past noon.
- **Gamma:** ATM 0.02-0.04 morning, spikes 0.10-0.20 by 3pm.
- **Time-of-day:** 9:35-10:30 trends. 11:30-13:30 chop (avoid). 15:30+ extreme gamma.
- **VIX:** <15 tight, 15-20 normal, 20-30 wide stops, >30 crisis. Transitions most dangerous.
- **Exits > Entries.** ATM preferred. "Always take profits off the table."
- Full reference: `docs/domain/0dte-domain-knowledge.md`, `docs/domain/pickles-trading-knowledge.md`

## 10. Policies

**Repair over workaround.** Every change fixes root cause. No TODO/HACK/FIXME without repair in same commit.

**Minimal documentation.** Do NOT create new doc files. All knowledge fits in existing files:
- `docs/journal/project-chronicle.md` — human narrative (every cycle)
- `docs/journal/art2-notebook.md` — machine memory (every cycle)
- `training/lab_notebook.md` — inner loop memory (every cycle)
- `training/program.md` — agent contract (when constraints change)
- This file — process/policies (when process changes)

**Time awareness.** User is PT. Market is ET (9:30-16:00). Run `python3 tools/art2.py time` before operations.

**Version consistency gate.** `validate_version_consistency()` in `run_loop.py` blocks training if satellite files drift from `train.py` (source of truth). Checked at two gates: `validate_safety()` (before upload) and `_preflight_validate()` (before GPU boot). Standalone tests: `pytest tests/test_version_consistency.py`. When changing architecture, features, params, or heads — run this test first. It catches:
- PBT param space (`inner_loop.py _PARAM_SPACE`) referencing params train.py doesn't read
- `monitor.py` displaying stale param names
- Satellite files (`replay.py`, `prepare.py`) reading dead `TRAIN_*` env vars
- Architecture constant mismatches (`NUM_ACTION_CLASSES`, `POSITION_STATE_DIM`, `ACCOUNT_STATE_DIM`) between `train.py`, `best_train.py`, and `replay.py`
- Model head drift (`action_head`/`risk_head` vs old `gate_head`/`value_head`)
- Feature count/group mismatches across files

**Warm-start vs fresh-start:**
- MUST fresh-start: feature count change, position state dim change, d_model/depth/heads change, new module
- SHOULD fresh-start: score formula change, evaluation mechanics change
- Safe to warm-start: hyperparameters, loss weights, bias tuning

## 11. Loss Hyperparameters (v14)

| Parameter | Default | Env Var | Notes |
|-----------|---------|---------|-------|
| Gate | 0.95 | TRAIN_GATE_W | Binary entry/exit signal (v7.2 PBT winner ~0.97) |
| Direction | 1.5 | TRAIN_DIR_W | Strike selection soft targets |
| PnL Alignment | 1.5 | TRAIN_PNL_W | trade_prob × dir_probs × P&L (primary signal) |
| Exit | 0.15 | TRAIN_EXIT_W | Exit label override strength (PBT drove toward 0) |
| Confidence | 0.05 | TRAIN_CONF_W | Calibration loss (minimal) |
| Value | 0.0 | TRAIN_VALUE_W | Disabled (proven destructive in v9) |
| Risk | 0.2 | TRAIN_RISK_W | Stop + size + conviction |
| DIRECTION_ENTROPY_BONUS | 0.20 | (hardcoded) | Prevents direction collapse (10x v12's 0.02) |

**Unchanged from v10:** All loss weights identical to the v10 baseline that scored 16.73.

**Kept unchanged:** LR, BATCH_SIZE, DROPOUT, WEIGHT_DECAY, GRAD_CLIP, WARMUP_RATIO, COOLDOWN_RATIO, D_MODEL, DEPTH, N_HEADS, FF_MULT, REG_TEMPORAL_SMOOTH.

## 12. Commands

| Task | Command |
|------|---------|
| Full cycle | `python3 tools/art2.py daemon --max-cycles N --minutes M` |
| Deploy GPU | `./infra/deploy.sh boot && ./infra/deploy.sh start` (auto-mints ACT if insufficient) |
| Run experiment | `python3 tools/inner_loop.py experiment --summary "hypothesis"` |
| PBT sweep | `python3 tools/inner_loop.py pbt-init --population 6 && pbt-run` |
| Check status | `python3 tools/inner_loop.py status` |
| Replay backtest | `python3 training/replay.py --backtest --model training/best_model.pt` |
| Rebuild data | `python3 training/prepare.py` |
| IBKR analysis | `python3 tools/art2.py ibkr-analyze --report` |
| Monitor | `python3 tools/monitor.py` (localhost:8420) |
| Paper trade (full day) | `python3 tools/paper_live.py --paper-auto --port 4002 --client-id 80 --kill-switch results/live/kill_switch` |
| Paper trade (dry-run) | `python3 tools/paper_live.py --dry-run --port 4002 --max-minutes 5` |
| Context refresh | `python3 tools/paper_live.py --context-only --port 4002` |
| Live dashboard | `python3 tools/live_dash.py` (localhost:8421) |
| Export trades CSV | `python3 tools/export_trades_csv.py [--date YYYY-MM-DD]` |
| Version consistency | `python -m pytest tests/test_version_consistency.py -v` |
| Teardown | `./infra/deploy.sh stop -y` |

## 13. Daily Paper Trading Operations

**Status: VERIFIED 2026-03-26.** Full end-to-end confirmed: IB Gateway → context refresh → model inference → bracket order (BUY LMT + STP + LMT TP) → CBOE fill → position tracking → EOD flatten → audit trail + trade CSV + daily summary. Account DUP440540, real fills on CBOE.

**Prerequisites:** IB Gateway running on port 4002 (paper account DU*). Context bundle refreshed.

**Manual daily flow:**
1. Start IB Gateway, log in with paper credentials
2. Refresh context: `python3 tools/paper_live.py --context-only --port 4002`
3. Start paper trader: `python3 tools/paper_live.py --paper-auto --port 4002 --client-id 80 --kill-switch results/live/kill_switch`
4. Start dashboard: `python3 tools/live_dash.py` → localhost:8421
5. After market close: `python3 tools/export_trades_csv.py --date $(date +%Y-%m-%d)`
6. Compare `results/live/trades.csv` against IBKR paper trade history

**Automated daily flow (launchd):**
```bash
cp infra/daily_paper_trade.plist ~/Library/LaunchAgents/com.trinity.daily-paper-trade.plist
launchctl load ~/Library/LaunchAgents/com.trinity.daily-paper-trade.plist
```
Triggers Mon-Fri at 6:30 AM PT. Requires IB Gateway already running. Logs to `results/live/paper_trade.log`.

**Output files:**
| File | Contents |
|------|----------|
| `results/live/audit.jsonl` | Full audit trail (every bar, inference, order, fill) |
| `results/live/trades.jsonl` | Closed trade records (entry/exit price, P&L, reason) |
| `results/live/trades.csv` | CSV export for IBKR comparison |
| `results/live/daily/YYYY-MM-DD.json` | Daily summary (P&L, win rate, profit factor) |
| `results/live/kill_switch` | Write "kill" to halt trading immediately |

**Safety:**
- Paper account guard: rejects non-DU* accounts
- Daily loss limit: 5% (blocks new entries)
- Hard kill: 10% session loss auto-activates kill switch
- Kill switch file: write `kill` to `results/live/kill_switch` to stop immediately
- EOD flatten: all positions closed at 4:00 PM ET
