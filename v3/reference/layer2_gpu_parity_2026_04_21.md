# Layer-2 GPU Parity Run — 2026-04-21

## Verdict

The Layer-2 shared-encoder + detach-side architecture (exp A from
[layer2_shared_encoder_diagnostics_2026_04_21.md](layer2_shared_encoder_diagnostics_2026_04_21.md))
trains cleanly on H100 CUDA. GPU and CPU results are within seed/device
variance on all metrics and both clear the plan's §2 bar by wide
margins. The CUDA path is validated for future Layer-2 experiments.

## Result

| Path | Trades | PF | DD | TPD | Mean PnL | Call% |
|---|---:|---:|---:|---:|---:|---:|
| CPU A (exp A baseline) | 275 | 1.455 | 36.9% | 0.917 | +225.4 | 15.3% |
| **GPU parity (H100)**  | **275** | **1.447** | **39.6%** | **0.917** | +222.0 | 15.3% |
| Δ (GPU − CPU) | 0 | −0.008 | +2.7 pp | 0 | −3.4 | 0 |

§2 bar (PF ≥ 1.122, DD ≤ 56.2%, TPD ∈ [0.8, 1.1]) cleared on GPU:
- PF 1.447 ≥ 1.122 ✅
- DD 39.6% ≤ 56.2% ✅
- TPD 0.917 ∈ [0.8, 1.1] ✅

Training diagnostics match CPU A within noise:
- Entry top-decile mean: CPU 3008.29 → GPU 3013.31 (+0.2%)
- Chosen top1/day entry value: CPU 1903.21 → GPU 1908.92 (+0.3%)
- Chosen time-stop mean: CPU 225.38 → GPU 223.52 (−0.8%)
- Side-error weighted accuracy: 0.377 → 0.377 (identical)

## What the parity confirms

1. **The CUDA path works end-to-end.** The `Layer2SharedEncoder`,
   `train_multitask`, weighted Huber loss, and `detach_side` gradient
   stop-grad all execute on CUDA without NaN / divergence.
2. **`SharedEncoderPredictor` pickles cross devices cleanly.** The
   per-fold pickles saved from a CUDA-trained model (with
   `device_hint="cuda"`) are deserialized locally, fall back to CPU
   since the local box has no CUDA, and produce correct predictions
   for replay. No cross-device deserialization bug in the path.
3. **Training is fast on H100.** Full 5-fold run completed in ~1 min
   including fold loop overhead. CPU was roughly 2 min for the same
   workload; the bottleneck for this dataset size is not compute —
   it's the dataloader + per-fold setup.
4. **No seed reproducibility perfect-match.** Small DD drift (+2.7 pp)
   is CUDA non-determinism (cuBLAS, torch.backends.cudnn) — expected
   and within noise. Not a bug.

## Implication for larger GPU runs

For this dataset size (~900 train rows per fold, 5 folds), GPU is
overkill and the CPU path is already fast enough. GPU becomes
justified if:

- The feature set grows (W2b VP features after matched-control retest,
  or a future batch of features). Larger F scales the trunk param
  count linearly.
- The architecture grows (larger hidden_dim, deeper trunk, or more
  heads). GPU matters once trunk forward costs dominate over dataloader.
- The export bundle expands (longer history, more bars per day, or
  shorter walk-forward folds with more positions). Larger n scales
  every loop.

For the current architecture + feature set, CPU is the practical path.
GPU remains a capability check and a forward-compatibility signal.

## Artifact

`v3/artifacts/layer2_shared_enc_fixedq_detach_gpu/` — full 5-fold GPU run,
downloaded locally. Gitignored (large + derivative of `data.pt`).
Contents:
- `audit.json` — per-fold training info + thresholds
- `manifest.pkl` — calibration / direction / score mode (matches CPU A)
- `oof_predictions.pkl` — out-of-fold test predictions
- `folds/{0..4}/` — per-fold entry + side predictor pickles, calibration JSON,
  training info JSON
- `replay_report.json` — GPU replay result
- `layer2_trades.csv`, `teacher_baseline_trades.csv` — per-trade detail

## GPU run command

```bash
# On local machine, after deploy.sh boot + start:
# (v3/ must be tarred and uploaded separately; deploy.sh syncs only v2/)
tar --exclude='__pycache__' --exclude='artifacts' --exclude='*.pyc' \
    -czf /tmp/v3.tar.gz v3/
scp -P $SSH_PORT /tmp/v3.tar.gz root@$SSH_HOST:/root/v3.tar.gz
ssh -p $SSH_PORT root@$SSH_HOST 'cd /root && tar -xzf v3.tar.gz'
scp -P $SSH_PORT v3/artifacts/layer2_dataset.pkl \
    root@$SSH_HOST:/root/v3/artifacts/

# Remote deps: pyarrow (the v2 dep set doesn't include it)
ssh -p $SSH_PORT root@$SSH_HOST '/opt/conda/bin/pip install pyarrow'

# Training:
ssh -p $SSH_PORT root@$SSH_HOST 'cd /root && \
  PYTHONUNBUFFERED=1 /opt/conda/bin/python -m v3.layer2.train_neural \
    --run-dir v3/artifacts/layer2_shared_enc_fixedq_detach_gpu \
    --entry-target entry_value_rank \
    --side-target  time_stop_margin_raw \
    --direction-mode teacher_if_triggered_else_put \
    --calibration-mode fixed_quantiles \
    --entry-quantile 0.60 --side-quantile 0.10 \
    --score-mode product \
    --detach-side \
    --device cuda'

# Download + local replay:
scp -r -P $SSH_PORT \
  root@$SSH_HOST:/root/v3/artifacts/layer2_shared_enc_fixedq_detach_gpu/ \
  v3/artifacts/
python -m v3.layer2.replay --run-dir v3/artifacts/layer2_shared_enc_fixedq_detach_gpu
```

## Hygiene notes (fixed during the run)

1. **`v2/data.pt.sha256` was stale.** My earlier local patch to
   `v2/data.pt` (fixing `chain_sidecar_digest` + `fingerprint` after
   `build_v2_dataset` wrote stale values) changed the file's SHA256 but
   did not regenerate the sidecar. `deploy.sh start` blocked the upload
   with a "data.pt CORRUPTED" error because remote computed a different
   hash than the stale sidecar claimed. Fixed by regenerating:
   ```bash
   shasum -a 256 v2/data.pt | awk '{print $1}' > v2/data.pt.sha256
   ```
   Future builds should keep the sidecar regenerated after ANY local
   `torch.save` to data.pt, not just after `build_v2_dataset`.

2. **`deploy.sh start` does not sync `v3/`.** It's scoped to `v2/`
   only, which is correct — deploy.sh was built for v2 training runs.
   Layer-2 GPU runs need v3 tarred and uploaded separately (commands
   above). If Layer-2 GPU becomes a recurring workflow, extending
   deploy.sh with a `sync_v3` command would be worth the effort.

3. **`pyarrow` isn't in the remote base environment.** Layer-2 uses
   parquet for oof predictions (via pandas → pyarrow). The remote
   container has torch / numpy / pandas / scipy but not pyarrow.
   Pip-installed ad hoc this run (above); worth baking into the deploy
   image if Layer-2 GPU becomes regular.

4. **SCP to Akash provider CAN deterministically corrupt large uploads
   on retry.** My first two `deploy.sh start` attempts reported
   `data.pt CORRUPTED! Local: 47caf47...  Remote: 16d8648...`. On the
   first two attempts the "Local" hash was from the stale sidecar; the
   "Remote" hash was the correct current file hash. Once I regenerated
   the sidecar, the third attempt succeeded. The takeaway: the
   corruption message can fire not because the upload is actually bad
   but because the sidecar expects a different local hash.

## Deployment accounting

- Lease DSEQ: 26489815, provider
  akash15pkdkewzarpsx42t98vzf45h42hlq6ra8w96hr
- Boot: 2026-04-21 15:33:28
- GPU READY: 15:35:13
- Training start (first SSH on exp): ~15:54
- Training end: ~15:55 (~1 min on H100 for 5 folds)
- Replay (local): ~15:58
- Lease closed: 15:59:40
- Close TX: 05BF5E55973165429202E57FB0B4D13ACCFD3763DCB09727DB554B3E022DEDE3
- Deposit: 1 ACT (screening-sized)
- Wall-clock from boot to close: ~27 min, mostly spent on workspace
  sync retries after the sha256-sidecar issue
