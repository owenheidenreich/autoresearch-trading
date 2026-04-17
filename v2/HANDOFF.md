# Handoff: Screen exp_169 (Unified Scorer)

## Immediate task

Run screening experiment `exp_169` on GPU. The code is committed and ready. The GPU is booted (DSEQ=26418658) but `deploy.sh start` did not complete — `torch` is not installed on the remote. Re-run `start`, then screen.

```bash
./v2/ops/deploy.sh start
TRAIN_ENV="SOFT_TEMP=0.08" ./v2/ops/deploy.sh run_screen_latest exp_169
```

*(Harness-integrity repair 2026-04-17 renamed `run_screen` → `run_screen_latest` / `run_screen_mini`, and split `run_one` into `run_cv` + `run_final_train`. See `v2/COMMANDS.md`.)*

Use `DEPOSIT_ACT=1` if re-booting GPU (1 ACT is enough for screening).

## What exp_169 changes

**One change:** Replaces dual call/put score heads with a single unified `score_head` in `v2/train.py`. Everything else (loss, replay, policy) is identical to exp_165 baseline.

- Commit: `7fc1e75`
- Env: `SOFT_TEMP=0.08` only. No SIDE_W, no SIDE_MODE, no ALPHA_SIDE.
- Baseline: exp_165 fold-0 PF=0.854, DD=53.5%, 482 trades, ~66% calls.

## Why

The dual heads learn fundamentally different scales (call head mean +0.117, put head mean -0.643), producing 98% call bias at the raw level. Per-side centering corrects to 73%, but can't fully fix it. A unified head eliminates this by forcing all contracts onto one learned scale. Greek sign flip already normalizes put features to look like calls.

## After screening completes

Run required diagnostics before deciding next branch. See decision tree in `v2/plans/typed-herding-globe.md` (or `~/.claude/plans/typed-herding-globe.md`).

**Required post-run diagnostics:**
1. Download the fold-0 model from GPU
2. Run `python3 -m v2.analysis.side_bias_audit --model <model_path>` — check if raw call% collapsed from 98%
3. Measure within-call and within-put chosen rank (per-side ranking quality)
4. Measure per-side PF (call-only PF, put-only PF)
5. Branch based on diagnostics per the decision tree, not just headline PF

**No-regression gate:** PF < 0.80 on fold 0 = material regression. DD > 65% = fail.
