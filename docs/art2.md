# ART² — AutoResearch Trading Squared

> **Operating manual** (lifecycle, roles, policies): [.claude/rules/art2-operating-manual.md](../.claude/rules/art2-operating-manual.md)

Mechanical orchestrator for the outer meta-loop. Handles subprocess management, data collection, and verification. Strategic decisions are made by Claude Code (Opus) reading the briefing output.

## Overview

ART² wraps the inner autoresearch loop (`run_loop.py` on Akash GPU) with:
- **Market awareness** — trains when market is closed, monitors paper trading when open
- **IBKR compatibility gate** — validates model before/after every training cycle
- **Automated analysis** — collects metrics, runs replay backtests, compares training vs replay
- **Decision support** — generates briefings with recommended actions for Claude Code
- **API budget tracking** — monitors cumulative spend against tier limits

### Architecture

```
  Claude Code (Opus) ◄──── briefing.md ──────────┐
       │                 (includes research        │
       │ Strategic        findings + domain         │
       │ decisions        knowledge context)        │
       ▼                                            │
  art2.py cycle / daemon                            │
       │                                            │
       ├── train ───► Akash H100 ───► run_loop.py (Sonnet)
       │                                  │
       ├── analyze ◄── results download ──┘
       │
       ├── replay ──► backtest best model
       │
       ├── diagnose ──► training vs replay comparison
       │
       ├── research ──► trade-level analysis vs domain knowledge
       │                 (time-of-day, exits, strikes, direction)
       │
       └── report ──► briefing.md ──────────────────┘
```

## Subcommand Reference

### `train --minutes N [--deposit-akt N] [--budget N] [--dry-run]`

Full training cycle on Akash:
1. Preflight validation (syntax check, data contract, budget check)
2. Boot Akash deployment (1 AKT boot deposit)
3. Fund deployment (auto-calculated from minutes, ~8 AKT/hour for H100)
4. Upload code + data.pt, start inner loop
5. Poll until training completes or timeout
6. Stop deployment, download results
7. Record API spend

**Flags:**
- `--minutes` — Training time budget (default: 130 ≈ 20 experiments)
- `--deposit-akt` — Override AKT deposit (default: auto from minutes)
- `--budget` — Monthly API budget override (default: $1,000)
- `--dry-run` — Print what would happen without deploying

### `analyze [--dry-run]`

Collect metrics from the latest training run into `analysis.json`:
- Total/kept/failed experiment counts
- Best/worst/mean scores
- API cost + cache hit percentage
- Paper trading P&L data
- Anti-gaming checks
- Inner loop health assessment

### `replay`

Run backtest on `best_model.pt` against validation dates:
- Trade-level metrics (PF, win rate, avg winner/loser)
- Time-of-day bucketing (morning/midday/afternoon/power hour)
- Hold time distribution

### `diagnose`

Compare training metrics vs replay backtest:
- Metric divergence: profit_factor, stop_loss_rate, trades_per_day, win_rate
- Time-of-day PF alignment
- Tunnel vision detection (inner loop fixated on one approach)
- Score trend analysis
- Priority-ranked recommendations

### `research`

Deep analysis of replay trade data against domain knowledge:
- Time-of-day P&L breakdown (morning/midday/afternoon/power hour)
- Stop-loss clustering analysis (when and where stops hit)
- Exit quality distribution (model_exit vs dynamic_stop vs max_hold vs EOD)
- Strike performance (ATM vs OTM5 vs OTM10)
- Direction bias (call vs put P&L)
- Hold time analysis and consecutive loss patterns
- Each finding generates a hypothesis grounded in domain knowledge
- Output: `research.json` + `research.md` in cycle directory

### `report`

Generate a single-message briefing for Claude Code:
- Inner loop results (top kept, recent failures)
- Health assessment + gaming alerts
- Research findings (time-of-day, exits, strikes, direction)
- Paper trading status
- Previous cycle decisions
- Lab notebook excerpts
- Recommended action with decision tree

### `verify --dates N`

Out-of-sample validation:
- Replay battery on N recent weekdays
- IBKR connectivity probe
- Pre/post metric comparison
- Gaming detection (score up but PF down)

### `cycle --minutes N [--deposit-akt N] [--skip-train]`

Full pipeline: train → analyze → replay → diagnose → research → report
- `--skip-train` — Skip training, analyze existing results

### `daemon --minutes N [--max-cycles N] [--deposit-akt N] [--dry-run]`

Outer loop with human review gate. Runs the complete 9-phase lifecycle:
- SETUP → TRAIN → TEARDOWN → ANALYZE → RESEARCH → IMPROVE → DOCUMENT → REVIEW → REPEAT
- **High-confidence action A:** Auto-decides without Opus call
- **All other actions:** Invokes Opus with domain knowledge + briefing
- Opus returns `file_edits` + `chronicle_entry` JSON → applied to whitelisted paths
- Both human (`project-chronicle.md`) and machine docs updated every cycle
- **REVIEW gate:** Daemon pauses after each decision. Human removes `results/art2/REVIEW` to approve.
- Supports `rebuild_data`, `fresh_start`, `needs_human` flags
- Loops until max-cycles or manual interrupt

**Control files:**
- `results/art2/REVIEW` — remove to approve decision and continue
- `results/art2/STOP` — graceful exit after current phase
- `results/art2/PAUSED` — remove to resume after infrastructure pause

### `autonomous --minutes N [--max-cycles N] [--deposit-akt N] [--dry-run]`

Market-aware continuous loop:
- **Market closed + >2h to open:** Run training cycle (auto-sized minutes)
- **Pre-market (<2h to open):** Stop training, run readiness check
- **Market open:** Monitor paper trading (check PID, audit trail)
- Loops until max-cycles or manual interrupt

### `market`

Check current market status:
- Open/closed, reason (weekend, holiday, pre-market, after-hours)
- Next open/close times
- Run all 4 IBKR compatibility gate checks

### `status`

Print current ART² state + recent cycle history + API budget summary.

### `init`

Initialize `results/art2/` directory structure and `art2_notebook.md`.

## IBKR Compatibility Gate

Runs before and after every training cycle. All 4 checks must pass:

| Check | What it validates | How |
|-------|-------------------|-----|
| model_exists | `best_model.pt` present | File exists + SHA256 hash |
| model_checkpoint | Gate head=2 outputs, Dir head=6 outputs | Load state_dict, check tensor shapes |
| feature_parity | 32 features, `val_start_idx` present | Load data.pt, verify dimensions |
| ibkr_probe | TWS connectivity + market data | Run `ib_probe.py` (only valid during market hours) |

## Decision Tree

Automated action recommendations based on training results:

| Step | Condition | Action |
|------|-----------|--------|
| 0 | No experiments completed | **F) Fix infrastructure** |
| 1 | All experiments crashed | **F) Fix infrastructure** |
| 1 | Accept rate ≥33% | **A) Let it cook** — inner loop is productive |
| 2 | Gaming detected (score↑ PF↓) | **F) Fix infrastructure** |
| 2 | >80% safety-blocked | **F) Fix infrastructure** |
| 3 | Paper trading divergence >30% | **E) Rebuild data** |
| 3b | Stall (<5% accept, flat scores) | **B) Steer inner loop** — edit lab_notebook.md |
| 4 | Accept rate ≥15% | **A) Let it cook** |
| 4 | Default | **B) Steer inner loop** |

**Action definitions:**
- **A) Let it cook** — Inner loop is making progress, no changes needed
- **B) Steer** — Edit `training/lab_notebook.md` priorities to redirect the inner loop
- **C) Change constraints** — Edit `training/program.md` to unlock/modify what the inner loop can do
- **D) Change features** — Edit `training/prepare.py` feature engineering
- **E) Rebuild data** — Run `prepare.py` with new parameters
- **F) Fix infrastructure** — Address deploy/API/data issues
- **G) Modify train.py directly** — Make architectural or loss function changes that the inner loop cannot

## State Management

State persisted at `results/art2/state.json`:

```json
{
  "cycle": 13,
  "phase": "trained",
  "run_name": "run-2026-03-22-064555",
  "best_score": 8.234,
  "last_action": "A",
  "timestamp": "2026-03-22T00:15:00"
}
```

Per-cycle artifacts in `results/art2/cycle-NNN/`:
- `analysis.json` — collected metrics
- `research.json` + `research.md` — trade-level analysis vs domain knowledge
- `briefing.md` — generated report (includes research findings)
- `decision.md` — strategic decision with rationale
- `review.md` — human-readable review summary (REVIEW phase)
- `verify_results.json` — OOS validation results
- `replay/` — backtest results, trade log, equity curve
- `logs/` — command logs (train, stop, fund, etc.)

## API Budget Tracking

- Tracked in `~/.cache/autoresearch-trading/api_spend_tracker.json`
- Auto-resets on month change
- Pre-flight check blocks sessions that would exceed 150% of remaining budget
- Displayed in `status` and `report` outputs

**Typical costs (Sonnet 4):**
- Per experiment: ~$0.15-0.30
- 20-experiment run: ~$3-6
- Monthly capacity at tier 3 ($1K): ~3,000-6,000 experiments

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `fund FAILED` | Wrong escrow deposit syntax | Fixed in deploy.sh (2026-03-22) |
| SSH timeout during start | Intermittent sshpass auth failure | Retry — usually works on second attempt |
| Stop doesn't close on-chain | `die()` in download kills script | Fixed: download runs in subshell (2026-03-22) |
| Training stuck at experiment N | Claude API credit exhaustion | Auto-shuts down after 3 failures. Top up credits. |
| Replay metrics diverge from training | Feature parity mismatch | Run `live_feature_parity_report.py` |
| Score gaming detected | Agent inflating score without PF improvement | Score config is locked. Check keep/reject gates. |
