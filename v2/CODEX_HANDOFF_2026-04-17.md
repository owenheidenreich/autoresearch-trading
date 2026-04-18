# Codex handoff — 2026-04-17

**Branch:** `codex/deploy-runtime-fix` (5 commits ahead of remote, unpushed).
**Supersedes:** the older [CODEX_BUNDLE.md](CODEX_BUNDLE.md) (which predates the harness repair and still names exp_165 as "best"). Read the bundle for project-level context on the instrument, architecture, and earlier dead-ends; then read this doc for everything since.

## 1. One-paragraph state

exp_171 is the official post-harness-repair baseline (PF 0.721, DD 457%, all 5 folds gate-fail). Three subsequent training hypotheses were falsified: **exp_172** (val_replay checkpoint selection), **exp_173** (strict-mask selection target), **exp_174/174b** (continuous PnL supervision via soft_pnl + max_pnl). Between exp_173 and exp_174, a local two-phase diagnostic (Phase A drawdown decomposition + Phase B signal viability + B3 audit) reshaped the problem: the true oracle ceiling is **PF 34-188 / WR 93-94%** uniform across vix regimes, context features CAN predict oracle_pnl (plain 52-feature LR achieves Spearman ρ 0.11 / AUC 0.615), but the trained transformer produces ρ ≈ 0 and its top-1-per-day picks are *anti-selected* in 4/5 folds. The training objective is the bottleneck, not features or architecture capacity. exp_174 tried to fix the objective (replace one-hot CE + binary BCE with continuous PnL regression) and failed the primary prediction, though secondary signals are present: call-bias dropped 82.9% → 65% in exp_174b and opp_std grew across epochs, so the gate-head supervision IS working — just not enough to close ρ from 0 to 0.11.

## 2. New since CODEX_BUNDLE.md

### 2a. Harness repair (context)

The harness was repaired before exp_171 — see lab_notebook entries dated 2026-04-15/16 for details (pipeline audit memory: `project_pipeline_audit_apr15.md`). All results before the repair (exp_150–exp_169 series) are not directly comparable to the current regime. `run_screen` and `run_one` commands were removed; the new CLI is `run_cv / run_screen_latest / run_screen_mini`.

### 2b. Diagnostic tooling added (commit 24e7b09)

Three new analyzers + a patch to replay.py:

- [v2/analysis/dd_decomposition.py](analysis/dd_decomposition.py) — per-fold equity-curve / tail-concentration / expectancy attribution from cv_report.json alone. Run: `python3 -m v2.analysis.dd_decomposition --cv-report v2/artifacts/exp_171/cv_report.json`.
- [v2/analysis/signal_viability.py](analysis/signal_viability.py) — Spearman ρ, decile tables, quantile gating, oracle top-K ceiling, side-bias × regime — from replay_traces.csv files. Run: `python3 -m v2.analysis.signal_viability --traces <csv> [<csv> ...]`.
- [v2/replay.py](replay.py) gained `--trace-out PATH`, `--date-range YYYY-MM-DD:YYYY-MM-DD`, `--skip-baselines`, and `--mask train`. Required to regenerate per-fold traces because the global `promote_mask` only covers one fold's test window.
- 5 per-fold replay_traces.csv under `v2/artifacts/exp_171/folds/<wid>/replay_traces.csv` + `v2/artifacts/exp_171/signal_viability.md`.

### 2c. Key diagnostic findings (commit 24e7b09, lab_notebook sections starting ~line 1990)

- **DD is not tail-driven.** Worst-5-day loss contribution is 22-35% across folds; capital-aware training would not address the binding constraint.
- **Expectancy is negative by construction.** avg_win ≈ |avg_loss| ≈ 0.26-0.29 with WR 42-48% and 6-11 TPD compounds to 60-100% DD inside each 60-day fold window.
- **Label/sim paths converge exactly.** On 119 matched selections (model selected oracle contract), |Δ(model_pnl − oracle_pnl)| = 0.0. No policy drift between label-build and replay.
- **Oracle ceiling is enormous.** Trading the best contract at every eligible bar: PF 34-188, WR 93-94%, DD 0%. Regime-uniform. The instrument admits profitable strategies. (Earlier "oracle ceiling broken" claim was an artifact of restricting oracle-top-K to bars the model chose to trade.)
- **Signal is learnable from context.** 52-feature LR: Spearman(LR_pred, oracle_pnl) = 0.113, AUC 0.615 on oracle_pnl>0, AUC 0.628 on oracle_side=Call. No individual feature has |ρ| > 0.05; the signal is distributed/weak but extractable in combination.
- **Trained transformer rank ρ ≈ 0.** Exp_171 model-top-1-per-day has PF 0.33-0.75 in 4/5 folds — the head's highest-confidence picks are anti-selected among the model's trades.

### 2d. exp_174 / exp_174b (commits 6f90556 / af49dbd / ebf29f8)

Added four opt-in env flags to [v2/train.py](train.py) (all default to existing behavior):
- `SEL_TARGET_MODE=soft_pnl` — selection KL with no ambiguity filter (reverted in v1b, not recommended).
- `GATE_TARGET_MODE=max_pnl` — replace opp_logit BCE with MSE on per-bar `max(row_labels)`.
- `GATE_PNL_THRESHOLD` (default 0.2) — subtract from MSE target so inference gate threshold 0 splits profitable-ceiling vs marginal bars.
- `GATE_PNL_LOSS_SCALE` (default 10.0) — multiplies the max_pnl MSE to restore gradient parity with BCE.

All falsifiable predictions failed both runs. Secondary signal: exp_174b reduced call bias 82.9% → 65% (a +17.9pp shift away from the historical call-skew), decile-10 PF reached 1.31, opp_std grew 0.03 → 0.05 across epochs. The gate head IS learning input-conditional signal; the selection head is not catching up. Training saturated at ~5 min/fold under the 300s TIME_BUDGET and val_replay checkpoint selection picked epoch 1 across tied replay scores.

## 3. Open decisions for the next ACT spend

Two plausible directions. Pick one; do not attempt both simultaneously.

### Direction A — extend training budget (cheap test first)

Evidence for: training was early-stopped at epoch 1 across tied val_replay scores; val_loss was still falling at epoch 12; loss was not converged. The loss-shape mechanics in exp_174b are mostly right (comp_loss=0.53, opp_std growing, call-bias dropping) — they may just need more gradient steps.

Proposed test:
```bash
# Single-fold screen first, ~15-25 min
DEPOSIT_ACT=1 ./v2/ops/deploy.sh boot
./v2/ops/deploy.sh start
TRAIN_ENV='CKPT_SELECTION_MODE=val_replay SEL_TARGET_MODE=default SOFT_TEMP=0.5 GATE_TARGET_MODE=max_pnl GATE_PNL_THRESHOLD=0.2 GATE_PNL_LOSS_SCALE=10.0 TIME_BUDGET=1200 OPP_W=1.0' ./v2/ops/deploy.sh run_screen_latest exp_174c
```

Kill-criterion: if fold 4 rank ρ (via `python3 -m v2.analysis.signal_viability --traces <fold_4_trace>` after downloading) is still below +0.05, longer training did not help and Direction B becomes correct.

### Direction B — architectural pivot (bigger change, higher EV)

Evidence for: a plain 52-feature linear regression achieves ρ 0.11 and AUC 0.615 — the transformer's trained head produces ρ ≈ 0, which means the encoder is *destroying* signal that a linear model can extract from raw features. This echoes the earlier Wave 1 finding ([memory project_exp160_findings.md]) that "trained encoder causes overtrading".

Proposed test: frozen-encoder + shallow (linear or 1-hidden-layer) head. Specifically:
1. Keep the encoder from exp_171 (or fresh random init — both worth trying).
2. Freeze its weights.
3. Train only the contract-selection and opportunity heads via the exp_174b continuous supervision path.
4. If that still produces ρ ≈ 0, the heads are the problem; replace with direct MLP on raw 52-feature context + per-contract 22-feature embedding.

This is a larger surgery than Direction A. Requires a new code path in train.py (freeze toggle) and probably a new run_experiment_wf config. Possibly better to first do a **local no-GPU diagnostic**: load exp_171 fold 4 model, extract context embeddings on all eligible bars, fit LR on embeddings → oracle_pnl. If LR(embeddings) has ρ << LR(raw_features), the encoder is provably destroying signal before anyone spends an ACT. See Section 5.

## 4. Commands / files you'll actually touch

### Run another training experiment
```bash
# Pre-flight
python3 -m py_compile v2/train.py
python3 -m v2.ops.pre_run_gate --data v2/data.pt

# Boot + upload + ship
./v2/ops/deploy.sh boot          # default 5 ACT deposit
./v2/ops/deploy.sh start         # rsync workspace to remote
TRAIN_ENV='...' ./v2/ops/deploy.sh run_screen_latest exp_NNN  # 1-fold triage, no artifact
TRAIN_ENV='...' ./v2/ops/deploy.sh run_cv exp_NNN             # 5-fold official

# Post
./v2/ops/deploy.sh download       # pulls artifacts
./v2/ops/deploy.sh stop -y        # close lease
```

### Re-run local diagnostics on any fold model
```bash
# Patch a new trace (if needed — traces under v2/artifacts/exp_171/folds/<wid>/replay_traces.csv already exist for baseline)
python3 -m v2.replay --model <model.pt> --mask {train,val,promote} \
    --date-range YYYY-MM-DD:YYYY-MM-DD \
    --traces --trace-out <path> --skip-baselines

# Signal viability
python3 -m v2.analysis.signal_viability --traces <csv> [<csv> ...] --out <md>

# DD decomposition (needs cv_report.json)
python3 -m v2.analysis.dd_decomposition --cv-report v2/artifacts/exp_NNN/cv_report.json
```

### Key files

Tier 1 (must read before changing behavior):
- [v2/train.py](train.py) — env flags at lines 32-56, selection loss block around 764-807, gate-MSE path at 715-735, total-loss composition at 906-917.
- [v2/core/policy.py](core/policy.py) — DEFAULT_POLICY (stop_pct 0.35, target_pct 0.50, max_hold 120, TRAILING).
- [v2/replay.py](replay.py) — recently patched for per-fold tracing.

Tier 2 (read if touching labels or policy):
- [v2/pipeline/build_v2_dataset.py](pipeline/build_v2_dataset.py) lines 614-617 — label generation calls `simulate_trade` with `DEFAULT_POLICY`. Sync point with replay.
- [v2/core/simulator.py](core/simulator.py) — `simulate_trade()`, trailing tiers.

Analyzers (read to understand diagnostic methodology):
- [v2/analysis/dd_decomposition.py](analysis/dd_decomposition.py)
- [v2/analysis/signal_viability.py](analysis/signal_viability.py)
- [v2/analysis/competence_score_analysis.py](analysis/competence_score_analysis.py) (older, pre-harness-repair, but useful reference for decile/threshold patterns)

Artifacts:
- [v2/artifacts/exp_171/](artifacts/exp_171/) — baseline: cv_report.json, 5 fold models, 5 per-fold replay_traces.csv, signal_viability.md.
- [v2/artifacts/exp_174_screen_mini/](artifacts/exp_174_screen_mini/) — 3 fold models (0, 2, 4) + traces.
- [v2/artifacts/exp_174b_screen_latest/](artifacts/exp_174b_screen_latest/) — fold 4 model + trace.

Narrative:
- [v2/lab_notebook.md](lab_notebook.md) — Phase A (line ~1990), Phase B (line ~2035), B3 (line ~2085), exp_174/174b (line ~2212).

Memory files (user-level, `~/.claude/projects/.../memory/`):
- `project_exp171_rebaseline.md` — baseline context.
- `project_exp173_strict_target.md` — strict-mask falsification.
- `project_oracle_ceiling_broken.md` — B3 audit summary (note: title says "broken" but content was revised post-audit — ceiling is actually enormous; title retained for continuity).
- `project_exp174_falsified.md` — exp_174/174b falsification + next directions.

## 5. Suggested no-GPU next step (cheapest, most diagnostic)

Before picking Direction A or B, the following local experiment would be decisive:

**Test whether the trained encoder destroys signal that is present in raw features.**

```python
# Pseudocode — local CPU, ~5 min
1. Load exp_171 fold 4 model.
2. For each eligible bar in fold 4 test window:
   - Compute context embedding = encoder forward on 30-bar lookback.
   - Compute raw feature vector = X[bar_idx] (52-d).
   - Compute oracle_pnl_max = max(row_labels[valid_contracts_for_bar]).
3. Split eligible bars 80/20.
4. Fit LinearRegression(embedding → oracle_pnl_max) on train, measure Spearman ρ on test.
5. Fit LinearRegression(raw_features → oracle_pnl_max) on same split, measure Spearman ρ on test.
6. Compare.
```

If (embedding → oracle) ρ << (raw → oracle) ρ, the encoder is provably the problem → Direction B (pivot) is correct, no ACT needed to confirm.
If they're similar, the encoder preserves signal and the heads are the problem → Direction A (longer training on heads) is correct.
If both are low, the fold 4 window is just hard (unlikely given LR baseline of 0.11 on the pooled 5-fold sample).

This test has no downside and replaces one full ACT spend with ~5 min of local compute. I'd run it before booting the GPU.

## 6. What NOT to do (falsified priors)

- Do not reshape the selection CE target shape alone. Three experiments (exp_171/172/173) and two continuous-supervision variants (exp_174/174b) have now failed on this axis.
- Do not propose capital-aware training / trajectory reward based on tail-day concentration. DD is not tail-driven (worst-5 contributes 22-35% of total loss).
- Do not spend ACT to regenerate the exp_171 baseline. It's canonical and already committed under [v2/artifacts/exp_171/](artifacts/exp_171/).
- Do not touch the sidecar labels (`row_labels`). They were verified correct against `simulate_trade` with zero delta on matched selections. If you think they're wrong, you're about to waste hours.
- Do not push to origin without explicit user confirmation. Branch is `codex/deploy-runtime-fix`, 5 commits ahead.
- Do not skip lab_notebook entries. Every experiment must log; monitor.py + downstream analysis depend on it.

## 7. Contact points

- Human user: Owen Heidenreich — prefers direct, terse updates; no trailing summaries; root-cause thinking over symptom patching ([memory feedback files](file:///Users/gduby/.claude/projects/-Users-gduby-Documents-autoresearch-trading/memory/)).
- CLAUDE.md at repo root has operating directives: document precedence, experiment pipeline, hard rules. Read before first modification.
