# ART² Operating Manual

This is the single authoritative reference for ART². If anything conflicts with this file, this file wins.

## 1. Mission

Build a profitable SPX 0DTE **long options** trading bot. Paper trading P&L on IBKR is ground truth — not training score, not backtest PF. Every decision must trace back to improving live paper trading results.

## 2. ART² Owns the Infinite Loop

ART² is responsible for the end-to-end lifecycle. No phase is optional. Every cycle runs all nine phases sequentially:

```
SETUP → TRAIN → TEARDOWN → ANALYZE → RESEARCH → IMPROVE → DOCUMENT → REVIEW → REPEAT
```

### Phase 1: SETUP
- Preflight: syntax check train.py, validate data.pt contract, check API budget
- IBKR compatibility gate: model exists, checkpoint shapes (gate=2, dir=6), 32 features, IBKR probe
- Duration sizing: improving (90 min) / stable (65 min) / stuck (40 min) / after fix (65 min fresh)
- Fresh-start vs warm-start decision (see docs/CLAUDE.md "Warm-Start vs Fresh-Start Rules")
- Verify `.best_score` and promoted history are clean (no inflated baselines)

### Phase 2: TRAIN
- Deploy to Akash H100 via `python3 tools/art2.py train --minutes M`
- Inner loop (Sonnet) mutates train.py → trains → scores → keeps or reverts
- Monitor via `./infra/deploy.sh status` or `tools/monitor.py` (localhost:8420)
- Auto-sync downloads results as experiments complete

### Phase 3: TEARDOWN
- `./infra/deploy.sh stop -y` — downloads results, stops Akash lease
- **CRITICAL:** deploy.sh stop will NOT overwrite train.py if it has uncommitted local changes
- Record API spend to cycle artifacts
- Verify best_model.pt was downloaded (if not, restore from backup)

### Phase 4: ANALYZE
- `python3 tools/art2.py analyze` — parse experiments.v2.jsonl, collect metrics
- `python3 tools/art2.py replay` — backtest best_model.pt, get trade-level output
- `python3 tools/art2.py diagnose` — compare training vs replay metrics
- **Red flags:** PF diverges >20%, stop rate diverges >10%, tunnel vision (>80% same approach)

### Phase 4.5: RESEARCH
- `python3 tools/art2.py research` — deep analysis of replay trade data against domain knowledge
- Cross-references: time-of-day P&L, stop clustering, exit quality, strike selection, direction bias
- Each finding generates a hypothesis grounded in domain knowledge (theta decay, gamma dynamics, etc.)
- Output: `research.json` + `research.md` in cycle directory
- **Deep research is REQUIRED before every strategic change.** Every hypothesis must cite specific domain knowledge.

### Phase 5: IMPROVE
- Read the briefing (`results/art2/cycle-NNN/briefing.md`) which now includes research findings
- Make **ONE** strategic change per cycle. No compounding changes.
- The change must be grounded in research findings and domain knowledge
- Apply the Repair-Over-Workaround Policy (see Section 6)
- Log decision + rationale to `cycle-NNN/decision.md`

### Phase 6: DOCUMENT
- **REQUIRED after every cycle.** Two types of documentation are maintained:
- **Human documentation** (`docs/project-chronicle.md`): Narrative, reverse-chronological project log. Written for the project owner. Updated via `chronicle_entry` in every Opus decision (including action A). Auto-formatted with dated headers.
- **Machine documentation** (all other docs below): Technical reference for AI agents. Updated via `doc_edits` in Opus decisions.
- Update project-level docs that were affected by this cycle's changes:
  - `docs/project-chronicle.md` — **always** update with narrative chronicle entry (auto-handled by art2.py)
  - `docs/ARCHITECTURE.md` — if architecture, pipeline, or data flow changed
  - `docs/art2.md` — if subcommands, decision tree, or state management changed
  - `docs/CLAUDE.md` — if key files, constants, design rules, or lessons learned changed
  - `docs/art2-notebook.md` — always update with cycle decision + outcome
  - `training/lab_notebook.md` — always update dead ends + best runs
  - `training/program.md` — if model contract, loss policy, or constraints changed
  - `docs/daily-pipeline.md` — if daily automation was modified
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

### Inner Loop — Sonnet via run_loop.py
**Owns:** train.py hyperparameters and training dynamics only (within the guardrails set by program.md)
**Decides:** _env_float values (LR, DROPOUT, WEIGHT_DECAY, etc.), bias tuning, sample weighting
**Prohibited:** New nn.Module subclasses, new loss terms, architecture changes, score formula changes

### Mechanical Layer — art2.py
**Owns:** Subprocess management, data collection, IBKR compatibility checks
**Does NOT make:** Strategic decisions. It generates briefings; Claude Code (Opus) decides.

## 4. Domain Knowledge Integration

When diagnosing model behavior, evaluating feature gaps, or proposing strategic changes, **ALWAYS read:**
- `docs/0dte-domain-knowledge.md` — Greeks behavior, theta decay, gamma dynamics, dealer mechanics, volatility regimes, formulas
- `docs/pickles-trading-knowledge.md` — Practical entry/exit rules, VWAP framework, time-of-day patterns, risk management, anti-patterns

### Condensed Quick Reference (always in context)
- **Theta:** ~1/sqrt(T). Doubles when remaining time quarters. Long options bleed past noon.
- **Gamma:** ATM 0.02-0.04 morning, spikes to 0.10-0.20 by 3pm. The 0DTE opportunity AND risk.
- **Time-of-day:** 9:35-10:30 strongest trends. 11:30-13:30 lunch chop (avoid). 15:30+ extreme gamma.
- **VIX regimes:** <15 tight ranges, 15-20 normal, 20-30 wide stops, >30 crisis. TRANSITIONS most dangerous.
- **Exits > Entries:** The edge is in exit timing. Gate head must learn when to take profits.
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
| `docs/project-chronicle.md` | Human-readable narrative chronicle entry | **Every cycle** (auto via art2.py) |
| `docs/ARCHITECTURE.md` | System diagrams, data flow, component descriptions | Architecture/pipeline changes |
| `docs/art2.md` | Subcommands, decision tree, state management | ART² tooling changes |
| `docs/CLAUDE.md` | Key files, constants, design rules, lessons learned | Any significant change |
| `docs/art2-notebook.md` | Strategic changes tried, paper P&L, dead ends | Every cycle |
| `training/lab_notebook.md` | Best runs, dead ends, next priorities | Every cycle |
| `training/program.md` | Model contract, loss policy, constraints | Inner loop constraint changes |
| `docs/daily-pipeline.md` | Daily automation stages, schedules | Pipeline changes |
| `.claude/rules/art2-operating-manual.md` | Lifecycle, roles, policies | Process changes |

### Per-Cycle Artifacts (in `results/art2/cycle-NNN/`)
- `analysis.json` — structured metrics from the training run
- `research.json` + `research.md` — trade-level analysis vs domain knowledge
- `briefing.md` — generated report for strategic decision
- `decision.md` — the strategic decision with rationale
- `review.md` — human-readable review summary (generated during REVIEW phase)
- `replay/` — backtest results, trade log, equity curve

### Persistent Memory
- `docs/project-chronicle.md` — **Human documentation:** narrative project log, reverse-chronological, for the project owner
- `docs/art2-notebook.md` — **Machine documentation:** outer loop memory (strategic changes, paper P&L, dead ends)
- `training/lab_notebook.md` — **Machine documentation:** inner loop memory (improvements, dead ends, priorities — injected into Sonnet prompt)

### Rules
- No silent changes. Every modification has a written rationale.
- One strategic change per cycle. No compounding (impossible to attribute improvement).
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
| Rebuild data.pt | `python3 training/prepare.py` |
| Subcommand reference | `docs/art2.md` |
| Domain knowledge | `docs/0dte-domain-knowledge.md`, `docs/pickles-trading-knowledge.md` |
