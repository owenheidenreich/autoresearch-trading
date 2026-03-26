# ART² Operating Manual

This is the single authoritative reference for ART². If anything conflicts with this file, this file wins.

## 0. Time Awareness (MANDATORY)

**Before any ART² operation**, run `python3 tools/art2.py time` to establish correct time context. Never trust metadata dates — always check the real clock.

- The **user is in Pacific Time (PT)**. All communication should reference PT.
- The **market runs on Eastern Time (ET)**. All market logic uses ET internally.
- Market hours: **9:30 AM - 4:00 PM ET** (6:30 AM - 1:00 PM PT).
- Do NOT assume the date from system metadata. The `time` subcommand is the source of truth.

## 1. Mission

Build a profitable SPX 0DTE **long options** trading bot. Paper trading P&L on IBKR is ground truth — not training score, not backtest PF. Every decision must trace back to improving live paper trading results.

## 2. ART² Owns the Infinite Loop

ART² is responsible for the end-to-end lifecycle. No phase is optional. Every cycle runs all nine phases sequentially:

```
SETUP → TRAIN → TEARDOWN → ANALYZE → RESEARCH → IMPROVE → DOCUMENT → REVIEW → REPEAT
```

### Phase 1: SETUP
- Preflight: syntax check train.py, validate data.pt contract, check API budget
- IBKR compatibility gate: model exists, checkpoint shapes (gate=2, dir=6, value=1, risk=3), 37 features (v3), 7-dim position state, 4-dim account state, IBKR probe
- Duration sizing: improving (90 min) / stable (65 min) / stuck (40 min) / after fix (65 min fresh)
- Fresh-start vs warm-start decision (see docs/operations/reference.md "Warm-Start vs Fresh-Start")
- Verify `.best_score` and promoted history are clean (no inflated baselines)

### Phase 2: TRAIN
Opus IS the loop. No API costs — uses Max subscription Sonnet agents.

**Architecture:** Opus (strategist) → Sonnet agent (code writer) → inner_loop.py (mechanical)

**Flow per experiment:**
1. Opus reads history.jsonl + metrics, diagnoses trading behavior
2. Opus calls Sonnet agent with: program.md, current train.py, last N results, specific hypothesis
3. Sonnet proposes targeted edits to train.py (not full rewrites)
4. `inner_loop.py experiment --mutation FILE --summary "hypothesis"` — atomic:
   - Validates (syntax + safety)
   - Backs up train.py, applies mutation (--mutation optional; omit for baseline)
   - Uploads train.py + best_model.pt to Akash (ensures warm start uses correct weights)
   - Runs training, downloads model to temp file
   - Parses METRICS_JSON + trade diagnostics
   - Scores (anomaly detection + score comparison)
   - **KEEP:** promotes model_candidate.pt → best_model.pt, train.py → best_train.py, updates .best_score (all three always in sync)
   - **REVERT:** restores train.py from backup, best_model.pt unchanged
   - Writes status.json + experiments.v2.jsonl (monitor.py compatible)
5. Opus reads result JSON, reasons about trading behavior, loops to step 1

**PBT mode** (`pbt-init` / `pbt-run` / `pbt-status`):
- Population of N members competing per generation with env-var overrides
- Selection: elite carry-forward, exploit top-25%, explore top-50%
- Anti-stagnation: inject random members after 2 stalled generations
- State: `training/.pbt_state.json` (resumable mid-generation)

**Key invariant:** `best_model.pt`, `best_train.py`, and `.best_score` are always in sync.
- Model downloads to temp file (`artifacts/exp-N/model_candidate.pt`), only promoted on KEEP
- On REVERT, `best_model.pt` unchanged — still matches `best_train.py`
- Any experiment can crash/timeout without corrupting state

**SSH transport:** All Akash communication via sshpass SCP/SSH.
- Connection from `.deploy-state` (SSH_HOST, SSH_PORT), written by `deploy.sh boot`
- Password: `SSH_PASS` env var (default: `autoresearch2026`)
- 3 retries on SCP failure, SSH timeout = TIME_BUDGET + 240s

**Validation pipeline** (runs locally before upload):
- `run_loop.validate_syntax()` — ast.parse() catches syntax errors
- `run_loop.validate_safety()` — blocks os.system/subprocess/exec/eval, score config mutations

**Scoring:** `score = PF × trade_sharpe × freq_mult × penalties × bonuses`
- Penalties: consecutive loss (15% per beyond 3), drawdown (0.5 × |DD| above 0.10), stop rate (above 0.30)
- Score config is LOCKED — mutation-guarded in run_loop.py

**Loop detection:** Tracks consecutive identical revert reasons. Logs warning after 3+ repeats.

**Setup:**
- Deploy Akash GPU: `./infra/deploy.sh boot && ./infra/deploy.sh start` (compute only)
- Initialize run: `python3 tools/inner_loop.py init`
- Monitor: `python3 tools/monitor.py` (localhost:8420)
- **Note:** `start_loop.sh` runs run_loop.py (a library, not executable). For Opus-driven experiments, ignore loop failure — drive via inner_loop.py from local machine.

**Key files:**
- `tools/inner_loop.py` — mechanical layer (SSH, validation, scoring, PBT)
- `training/run_loop.py` — utility library (validation functions, anomaly detection, metric parsing)
- `training/program.md` — contract (injected into Sonnet's prompt by Opus)

**Artifact layout** (per experiment):
- `artifacts/exp-N/train_before.py` — backup before mutation
- `artifacts/exp-N/train_candidate.py` — proposed mutation
- `artifacts/exp-N/model_candidate.pt` — downloaded model (promoted on KEEP)
- `artifacts/exp-N/metrics.json` — parsed training metrics
- `artifacts/exp-N/train_output.log` — full Akash stdout/stderr

**Full architecture reference:** `docs/architecture/INNER-LOOP-ARCHITECTURE.md`

### Phase 3: TEARDOWN
- `./infra/deploy.sh stop -y` — downloads results, stops Akash lease
- **CRITICAL:** deploy.sh stop will NOT overwrite train.py if it has uncommitted local changes
- Record API spend to cycle artifacts
- Verify best_model.pt exists and matches best_train.py (both updated only on KEEP)

### Phase 4: ANALYZE
- `python3 tools/art2.py analyze` — parse experiments.v2.jsonl, collect metrics
- `python3 tools/art2.py replay` — backtest best_model.pt, get trade-level output
- `python3 tools/art2.py diagnose` — compare training vs replay metrics
- `python3 tools/art2.py ibkr-analyze` — parse IBKR paper trading audit, produce metrics for briefing
- **Red flags:** PF diverges >20%, stop rate diverges >10%, tunnel vision (>80% same approach), IBKR-vs-replay divergence

### Phase 4.5: RESEARCH
- `python3 tools/art2.py research` — deep analysis of replay trade data against domain knowledge
- Cross-references: time-of-day P&L, stop clustering, exit quality, strike selection, direction bias
- Each finding generates a hypothesis grounded in domain knowledge (theta decay, gamma dynamics, etc.)
- Output: `research.json` + `research.md` in cycle directory
- **Deep research is REQUIRED before every strategic change.** Every hypothesis must cite specific domain knowledge.

### Phase 4.7: VIABILITY
- `python3 tools/art2.py viability` — structured profitability assessment ("Can this model make money?")
- Runs validation-only AND full-period backtests, then computes:
  - Train/val split comparison (overfit detection, PF divergence)
  - Survivorship concentration (top-N trades as % of total profit)
  - Direction & strike diversity (one-dimensional trading detection)
  - Statistical significance (t-test on trade P&Ls, 95% CI)
  - Exit quality breakdown (model_exit vs stop_loss vs max_hold)
  - Time-of-day P&L decomposition
- Output: `viability.json` + `viability.md` in cycle directory
- Verdict: VIABLE / PROMISING / INCONCLUSIVE / NOT VIABLE (with confidence level)
- Included in the briefing (Section 2.7) so Opus sees the verdict when making strategic decisions

### Phase 5: IMPROVE
- Read the briefing (`results/art2/cycle-NNN/briefing.md`) which now includes research findings
- **Multiple strategic changes are allowed per cycle** when they address independent concerns (e.g., fixing exit loss + adding position state to random batches). Use judgment: if changes interact or compound risk, split them across cycles. Log the rationale for bundling.
- Each change must be grounded in research findings and domain knowledge
- Apply the Repair-Over-Workaround Policy (see Section 6)
- Log all decisions + rationale to `cycle-NNN/decision.md`

### Phase 6: DOCUMENT
- **REQUIRED after every cycle.** Two types of documentation are maintained:
- **Human documentation** (`docs/journal/project-chronicle.md`): Narrative, reverse-chronological project log. Written for the project owner. Updated via `chronicle_entry` in every Opus decision (including action A). Auto-formatted with dated headers.
- **Machine documentation** (all other docs below): Technical reference for AI agents. Updated via `doc_edits` in Opus decisions.
- Update project-level docs that were affected by this cycle's changes:
  - `docs/journal/project-chronicle.md` — **always** update with narrative chronicle entry (auto-handled by art2.py)
  - `docs/architecture/ARCHITECTURE.md` — if architecture, pipeline, or data flow changed
  - `docs/operations/reference.md` — if key files, constants, subcommands, design rules, or daily pipeline changed
  - `docs/journal/art2-notebook.md` — always update with cycle decision + outcome
  - `training/lab_notebook.md` — always update dead ends + best runs
  - `training/program.md` — if model contract, loss policy, or constraints changed
- Documentation must be accurate to the current codebase. Stale docs are worse than no docs.
- Keep docs concise. Update what changed, don't rewrite what didn't.

### Phase 7: REVIEW
- **Human feedback gate.** The daemon pauses here for human approval before starting the next training run.
- `review.md` is written to the cycle directory with: decision summary, chronicle entry, what happens next.
- A `REVIEW` sentinel file is created at `results/art2/REVIEW`.
- The daemon polls every 60 seconds until the sentinel is removed.
- **Human options during REVIEW:**
  - **Approve:** Remove `results/art2/REVIEW` → daemon continues to REPEAT
  - **Redirect:** Edit `lab_notebook.md`, `program.md`, or `train.py` before removing REVIEW
  - **Stop:** Create `results/art2/STOP` → daemon exits gracefully

### Phase 8: REPEAT
- Adapt next session duration based on model health
- Launch next cycle: `python3 tools/art2.py daemon --max-cycles 1 --minutes M`

## 3. Role Definitions

### Outer Loop — Claude Code (Opus)
**Owns:** Any file in the project. Can modify train.py, prepare.py, program.md, lab_notebook.md, art2.py, deploy.sh, infrastructure — whatever is needed to improve paper trading P&L.
**Decides:** What to change, when to rebuild data.pt, when to steer inner loop, when to fresh-start, when to modify train.py directly
**Consults:** Domain knowledge files when diagnosing model behavior or proposing changes
**Prohibited:** Modifying run_loop.py safety checks, score formula (_score_config)

### Inner Loop — Sonnet via Claude Code Agent
**Owns:** train.py hyperparameters and training dynamics only (within the guardrails set by program.md)
**Role:** Code writer, not strategist. Receives specific hypothesis from Opus, outputs targeted edits.
**Decides:** How to implement Opus's hypothesis in train.py (_env_float values, bias tuning, sample weighting)
**Prohibited:** New nn.Module subclasses, new loss terms, architecture changes, score formula changes

### Mechanical Layer — art2.py
**Owns:** Subprocess management, data collection, IBKR compatibility checks
**Does NOT make:** Strategic decisions. It generates briefings; Claude Code (Opus) decides.

## 4. Domain Knowledge Integration

When diagnosing model behavior, evaluating feature gaps, or proposing strategic changes, **ALWAYS read:**
- `docs/domain/0dte-domain-knowledge.md` — Greeks behavior, theta decay, gamma dynamics, dealer mechanics, volatility regimes, formulas
- `docs/domain/pickles-trading-knowledge.md` — Practical entry/exit rules, VWAP framework, time-of-day patterns, risk management, anti-patterns

### Condensed Quick Reference (always in context)
- **Theta:** ~1/sqrt(T). Doubles when remaining time quarters. Long options bleed past noon.
- **Gamma:** ATM 0.02-0.04 morning, spikes to 0.10-0.20 by 3pm. The 0DTE opportunity AND risk.
- **Time-of-day:** 9:35-10:30 strongest trends. 11:30-13:30 lunch chop (avoid). 15:30+ extreme gamma.
- **VIX regimes:** <15 tight ranges, 15-20 normal, 20-30 wide stops, >30 crisis. TRANSITIONS most dangerous.
- **Exits > Entries:** The edge is in exit timing. Gate head + value head cooperate on exits.
- **ATM preferred:** OTM backtest cumulative -601%. ATM has highest gamma, most responsive.
- **"Always take profits off the table"** — Pickles' Holy Gospel. The model must learn this.
- **Charm flows (afternoon):** Delta decays via time, dealers unwind hedges. Creates predictable PM flows.
- **No credit spreads after noon on 0DTE.** Premium decayed too much for the risk.

### When to Consult Domain Knowledge
- Model bleeds in afternoon → Read theta decay + charm flow sections
- High stop-loss rate → Read gamma dynamics + stop methodology sections
- Wrong strike selection → Read strike selection + gamma band sections
- Poor VIX regime performance → Read volatility regimes + Pickles VIX mechanics
- Low win rate despite good entries → Read exit rules + time-based exit patterns

## 5. Decision Framework

Priority order — stop at the first YES:

1. **Train/eval mismatch?** (PF diverges >20%) → Fix the mismatch in prepare.py/train.py/replay.py. Nothing else matters if training and evaluation play different games.
2. **Inner loop stuck?** (0% accept, tunnel vision) → Steer lab_notebook.md. Consult domain knowledge for fresh hypotheses the agent hasn't tried.
3. **Model improving?** (kept > 0, score trending up) → Let it cook. Extend next session duration.
4. **Feature gap?** (model blind to known pattern) → Consult domain knowledge, add feature to prepare.py, rebuild data.pt, fresh-start.
5. **Paper trading diverges from backtest?** → Investigate execution model realism (spreads, slippage, fill quality).

## 6. Repair-Over-Workaround Policy (MANDATORY)

- Every code change MUST be a proper fix at the root cause, not a workaround
- If a quick fix is needed urgently, the permanent repair MUST happen in the same cycle
- Workarounds without repair plans are **forbidden** — they accumulate into systemic debt
- All changes logged with: **what broke**, **root cause**, **fix description**
- If the same bug appears twice → the first fix was a workaround. Escalate and fix properly.
- No `# TODO`, `# HACK`, `# FIXME` without an accompanying repair in the same commit

## 7. Documentation & Tracking Standards

### Project Documentation (updated every cycle in Phase 6: DOCUMENT)
Project docs must always reflect the current state of the codebase. Stale documentation is a liability — it misleads future decisions. The DOCUMENT phase is mandatory, not optional.

| Doc | What to update | When |
|-----|---------------|------|
| `docs/journal/project-chronicle.md` | Human-readable narrative chronicle entry | **Every cycle** (auto via art2.py) |
| `docs/architecture/ARCHITECTURE.md` | System diagrams, data flow, component descriptions | Architecture/pipeline changes |
| `docs/operations/reference.md` | Key files, constants, subcommands, design rules, daily pipeline, IBKR ops | Any significant change |
| `docs/journal/art2-notebook.md` | Strategic changes tried, paper P&L, dead ends | Every cycle |
| `training/lab_notebook.md` | Best runs, dead ends, next priorities | Every cycle |
| `training/program.md` | Model contract, loss policy, constraints | Inner loop constraint changes |
| `.claude/rules/art2-operating-manual.md` | Lifecycle, roles, policies | Process changes |

### Per-Cycle Artifacts (in `results/art2/cycle-NNN/`)
- `analysis.json` — structured metrics from the training run
- `research.json` + `research.md` — trade-level analysis vs domain knowledge
- `viability/viability.json` + `viability.md` — profitability assessment with verdict
- `briefing.md` — generated report for strategic decision
- `decision.md` — the strategic decision with rationale
- `review.md` — human-readable review summary (generated during REVIEW phase)
- `replay/` — backtest results, trade log, equity curve

### Persistent Memory
- `docs/journal/project-chronicle.md` — **Human documentation:** narrative project log, reverse-chronological, for the project owner
- `docs/journal/art2-notebook.md` — **Machine documentation:** outer loop memory (strategic changes, paper P&L, dead ends)
- `training/lab_notebook.md` — **Machine documentation:** inner loop memory (improvements, dead ends, priorities — injected into Sonnet prompt)

### Rules
- No silent changes. Every modification has a written rationale.
- Multiple strategic changes per cycle allowed when addressing independent concerns. Log rationale for bundling.
- Full audit trail. Every experiment preserved in `results/`.
- **Documentation is not optional.** Phase 6 (DOCUMENT) runs after every IMPROVE phase.
- **Human review is not optional.** Phase 7 (REVIEW) pauses the daemon for human approval before every training run.

## 8. Ground Truth Hierarchy

1. **Paper trading P&L (IBKR)** — Ultimate truth. If backtest says PF=4 but paper trading loses money, the model is wrong.
2. **Replay backtest PF** — Historical validation on held-out data. Useful but not gospel.
3. **Training score** — Proxy only. NEVER optimize the score metric itself. Improve actual trading behavior (PF, win rate, drawdown).

## 9. Tooling Quick Reference

| Task | Command |
|------|---------|
| Full autonomous cycle | `python3 tools/art2.py daemon --max-cycles N --minutes M` |
| Single phase cycle | `python3 tools/art2.py cycle --minutes M` |
| Check deployment | `./infra/deploy.sh status` |
| Monitor dashboard | `python3 tools/monitor.py` (localhost:8420) |
| Replay backtest | `python3 training/replay.py --backtest --model training/best_model.pt` |
| Viability check | `python3 tools/art2.py viability` |
| IBKR session analysis | `python3 tools/art2.py ibkr-analyze --report` |
| Rebuild data.pt | `python3 training/prepare.py` |
| Subcommand reference | `docs/operations/reference.md` |
| Domain knowledge | `docs/domain/0dte-domain-knowledge.md`, `docs/domain/pickles-trading-knowledge.md` |
| Init experiment run | `python3 tools/inner_loop.py init` |
| Run experiment | `python3 tools/inner_loop.py experiment --summary "hypothesis"` (--mutation FILE optional) |
| PBT sweep | `python3 tools/inner_loop.py pbt-init --population 6 && python3 tools/inner_loop.py pbt-run` |
| Check run status | `python3 tools/inner_loop.py status` |
