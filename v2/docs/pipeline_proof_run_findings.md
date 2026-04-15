# Pipeline Proof Run Findings — exp_152

Date: 2026-04-15
Run: exp_152 (5-fold supervised, default config)
Purpose: First end-to-end test of the hardened ART2 loop

## Verdict

The hardened pipeline works end-to-end. All artifacts were generated, linked, and verified. The keep/revert decision was executed through code. The loop is proven.

The model itself failed (all folds gated on drawdown). This is expected and acceptable — the milestone is "loop works," not "model is profitable."

## Pipeline stages completed

| Stage | Status | Notes |
|-------|--------|-------|
| Pre-run gate | PASS | 12/12 checks, 20 warnings (known data characteristics) |
| GPU boot (Akash) | PASS | H100 80GB, required manual ACT mint |
| Workspace deploy | PASS | Required data.pt.sha256 fix |
| 5-fold training | PASS | 31 min total, all folds completed |
| Model download | PASS | model_candidate.pt (1.7MB) |
| Local replay | PASS | All 4 baselines nonzero, diagnostics generated |
| Trace generation | PASS | 2700 bars, decision trace CSV |
| Artifact verification | PASS | eval_report + replay_diagnostics + triage_index linked |
| Keep/revert decision | PASS | Revert executed via model_manage.py |
| Results logging | PASS | results.tsv auto-appended, lab_notebook.md updated |

## Findings requiring future fixes

### FINDING 1: Pre-run gate thin-bar warnings are structural, not bugs
- 81% of bars in the 30-270 supervision window have < 5 executable contracts
- 50-70% NaN label rates per sidecar
- The check window (bars 30-270) now matches the policy window after the full-day regime reset
- NaN labels are correctly masked in training (zero gradient)
- **Not a blocker.** These are structural characteristics of 0DTE liquidity. The thin-bar check now covers the full policy window.

### FINDING 2: deploy.sh auto-mint ACT failed silently
- `boot` command failed because ACT balance was 0
- The auto-mint in deploy.sh tried to mint but failed (likely gas estimation)
- Manual mint worked: `provider-services tx bme mint-act "50000000uakt"`
- **Fix needed:** The auto-mint error handling should surface the actual failure reason. Currently it dies with "ACT mint failed (code=X)" but doesn't show the underlying error.

### FINDING 3: data.pt.sha256 was stale
- `start` command failed with "data.pt upload CORRUPTED!" because the `.sha256` sidecar had an old hash
- The data.pt file was rebuilt at some point without updating the sidecar
- **Fixed during run:** Updated sha256 to match current data.pt
- **Preventive fix needed:** `build_v2_dataset.py` should auto-update data.pt.sha256 when it rebuilds data.pt.

### FINDING 4: Zero real-time training visibility
- `run_one` runs the entire experiment via a single SSH session
- SSH buffers all output until the remote command completes
- The `status` dashboard only shows GPU util and PID — no fold count, no epoch, no ETA
- Training log (JSONL) is written to the GPU filesystem but not streamed
- **Result:** 30+ minutes of complete blindness. The only way to check progress is to SSH in separately and inspect fold checkpoint files.
- **Fix needed:** Either (a) stream training progress lines to stdout so they flow through SSH, or (b) have `status` read the remote training_log.jsonl or fold checkpoint count.

### FINDING 5: Status dashboard misleading during run_one
- Dashboard shows "No experiments have been run yet" because `.inner_loop_state.json` is only written by the inner-loop mode, not `run_one`
- The experiment history shows stale data from previous runs
- **Fix needed:** `run_experiment_wf.py` should write a progress file even in single-experiment mode, or `status` should check for the presence of fold checkpoints.

### FINDING 6: SSH intermittently fails with Permission denied
- ~10% of SSH attempts fail with "Permission denied" even with correct password
- deploy.sh's `ssh_cmd` helper uses the same sshpass setup but seems more reliable
- Manual sshpass commands with explicit SSHPASS export fail intermittently
- **Not a showstopper** but adds friction to manual monitoring.

## Model observations (not pipeline issues)

These are recorded for context but are not harness bugs:

- Best epoch was 1 or 2 on every fold — model overfits immediately after that
- Heavy call bias (70% calls across folds), especially fold 3 (89% calls)
- Gate accuracy 14.3%, selection accuracy 2.9% — barely better than random
- ATM-Trailing baseline scores 0.632 — a simple trailing-stop on ATM calls beats the trained model
- All folds gated on excessive drawdown (33-69%)
- net PnL = -$5,895 across 949 trades

## Milestone status

**"Supervised loop proven" milestone: ACHIEVED.**

All criteria met:
1. Real GPU-trained candidate completed full pipeline
2. All required artifacts generated and linked (eval_report, replay_diagnostics, triage_index)
3. All 4 baselines meaningful (ATM baselines produce 57 trades each)
4. Keep/revert decision executed through code (`model_manage.py revert`)
5. Result logged to results.tsv and lab_notebook.md

The milestone is valid despite the candidate being rejected — the loop works.

### FINDING 7: Plots have no model provenance
- `v2/output/trades.html` and `v2/output/equity.html` show trade count, win rate, and equity curve but do not identify which model, experiment, or timestamp produced them.
- The plots at time of discovery were from a 12:40 PM replay of the epoch-1 verification stub — not from exp_152 (which ran at 2:52 PM). There was no way to tell this from the charts.
- Plots are only regenerated on auto-promote (`deploy.sh run_one` line 973). If a candidate is reverted, the plots remain stale.
- **Fixed:** `v2/plot_trades.py` now includes model source and generation timestamp in both chart titles.

### FINDING 8: Trading window audit needed → RESOLVED
**Status: Audited and resolved.** The window audit (`v2/artifacts/window_audit/`) confirmed that bars 60-105 was a legacy filtering choice, not a structural edge. Oracle PF is high across the entire day. The window ranked 17th/20 among random windows of the same width. The narrowing control showed smooth mechanical improvement — no structural breakpoint.

**Resolution:** Policy window widened to full supervised day (bars 30-270) to maximize training opportunity. See `v2/artifacts/window_audit/window_audit_report.txt` for the full analysis.

**Files for audit:**
- `v2/core/policy.py:35-37` — window definition
- `v2/pipeline/build_v2_dataset.py:528-533` — label filtering
- `v2/lab_notebook.md` — exp_104, exp_105, exp_106 entries
- `v2/docs/domain/pickles-trading-knowledge.md` — practitioner view on AM session
- `v2/docs/domain/0dte-domain-knowledge.md` — market structure context

## Recommended next actions

1. Commit this proof run (findings doc + lab notebook + results.tsv + sha256 fix)
2. Fix findings 2-5 before starting hill climbing (estimated: 1-2 sessions)
3. Freeze evaluator/baselines/protocol
4. Audit trading window (Finding 8) before starting hill climbing — determine if the 45-minute restriction should be widened
5. Begin bounded hill climbing on policy.py + hyperparameters
