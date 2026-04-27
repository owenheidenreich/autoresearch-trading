#!/usr/bin/env bash
# L2_combined_side_balance_calibration_exit_exposure_v1
#
# Driver script for the locked Phase-3-prescribed retrain. See
# v3/reference/exp_l2_combined_v1_contract_2026_04_26.md for the
# frozen contract and pass/fail gates.
#
# This script has THREE explicit phases. The first two are CPU-only
# and safe to run automatically; phase 3 spends GPU credits and must
# be invoked manually after explicit user OK.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXP_ID="combined_v1"
DATASET="v3/artifacts/layer2_action_surface_dataset.pkl"
SEEDS=(42 43 44 45 46)
SMOKE_MAX_DAYS=50  # for phase 1 only; full build has --max-days 0

ORACLE_OUT_DIR="v3/artifacts"
ORACLE_PREFIX="simulated_l3_oracle_spx_live_0945_1130_seed"
ORACLE_SUFFIX="_${EXP_ID}.npz"

LOG_DIR="v3/artifacts/exp_${EXP_ID}_logs"
mkdir -p "$LOG_DIR"

phase=${1:-help}

case "$phase" in
  smoke)
    # Phase 1: 1-seed CPU smoke of the L3 oracle build to validate the
    # candidate_surface mode works end-to-end on a small slice.
    # Uses scripts.build_l3_oracle_with_trailing wrapper to add FW-day
    # coverage (maps trailing days post 2026-02-24 to W12's model).
    echo "[phase smoke] Building L3 oracle for seed 42 with --max-days $SMOKE_MAX_DAYS"
    PYTHONPATH=. python3 -m scripts.build_l3_oracle_with_trailing \
      --seed 42 \
      --dataset "$DATASET" \
      --l3-training-source candidate_surface \
      --candidate-train-max-per-day 4 \
      --target peak \
      --max-days "$SMOKE_MAX_DAYS" \
      --output "${ORACLE_OUT_DIR}/${ORACLE_PREFIX}42_${EXP_ID}_SMOKE.npz" \
      2>&1 | tee "$LOG_DIR/smoke_oracle_seed42.log"
    echo "[phase smoke] OK"
    ;;

  oracles)
    # Phase 2: full CPU L3 oracle build for all 5 seeds.
    # Uses scripts.build_l3_oracle_with_trailing wrapper so the L3 oracle
    # also covers forward-walk days (FW days are mapped to W12's model).
    # Without this wrapper, FW rows get NaN predictions and the FW gate is
    # un-evaluable on the with-oracle metric (see commit 963fe0e3 +
    # `Oracle restoration recovers offline PF on forward walk` memory).
    # Rough cost: ~1-1.5 hr/seed sequential on a quiet 10-core box.
    for seed in "${SEEDS[@]}"; do
      out="${ORACLE_OUT_DIR}/${ORACLE_PREFIX}${seed}${ORACLE_SUFFIX}"
      log="$LOG_DIR/oracle_seed${seed}.log"
      if [[ -f "$out" ]]; then
        echo "[phase oracles] seed $seed: oracle exists at $out (skip)"
        continue
      fi
      echo "[phase oracles] Building L3 oracle for seed $seed -> $out"
      PYTHONPATH=. python3 -m scripts.build_l3_oracle_with_trailing \
        --seed "$seed" \
        --dataset "$DATASET" \
        --l3-training-source candidate_surface \
        --candidate-train-max-per-day 4 \
        --target peak \
        --output "$out" \
        2>&1 | tee "$log"
    done
    echo "[phase oracles] All 5 seeds complete."
    ;;

  train)
    # Phase 3: GPU spend phase. DO NOT run without explicit user OK.
    # Requires CUDA; will RuntimeError on CPU per the promotion-tier guard.
    #
    # train_unified_policy.py takes ONE --simulated-l3-oracle path and uses
    # it for all seeds in its internal seed loop, so we invoke it 5 times
    # (once per seed) to get per-seed oracle pairing.
    echo "[phase train] *** GPU SPEND PHASE *** ($((${#SEEDS[@]})) seeds × ~12 epochs)"
    echo "[phase train] Confirmed user authorization? Set GPU_GO=yes to proceed."
    if [[ "${GPU_GO:-no}" != "yes" ]]; then
      echo "[phase train] Aborting — set GPU_GO=yes to confirm."
      exit 1
    fi
    for seed in "${SEEDS[@]}"; do
      oracle="${ORACLE_OUT_DIR}/${ORACLE_PREFIX}${seed}${ORACLE_SUFFIX}"
      run_dir="${ORACLE_OUT_DIR}/layer2_unified_policy_spx_${EXP_ID}_seed${seed}"
      log="$LOG_DIR/train_seed${seed}.log"
      if [[ ! -f "$oracle" ]]; then
        echo "[phase train] FATAL: oracle missing for seed $seed: $oracle"
        exit 2
      fi
      echo "[phase train] Training seed $seed -> $run_dir (oracle: $oracle)"
      PYTHONPATH=. python3 -m v3.layer2.train_unified_policy \
        --seed "$seed" \
        --run-dir "$run_dir" \
        --tier promotion \
        --seeds "$seed" \
        --dataset "$DATASET" \
        --simulated-l3-oracle "$oracle" \
        --utility-target hybrid_live \
        --w-ranking 0.5 \
        --w-clean 0.6 \
        --w-stopout 0.6 \
        --w-win 0.4 \
        --w-regression 0.5 \
        --w-dollar 0.25 \
        --w-return 0.25 \
        --w-side-contrastive 0.75 \
        --side-balance-weight 0.75 \
        --max-epochs 12 \
        --patience 4 \
        2>&1 | tee "$log"
    done
    echo "[phase train] All 5 seeds complete."
    ;;

  fw)
    # Phase 4: forward-walk evaluation (CPU; no GPU spend).
    out="v3/artifacts/forward_walk/${EXP_ID}.json"
    PYTHONPATH=. python3 -m v3.analysis.forward_walk \
      --champion-dir v3/artifacts \
      --champion-name "spx_${EXP_ID}" \
      --seeds "${SEEDS[@]}" \
      --dataset "$DATASET" \
      --oracle-pattern "${ORACLE_OUT_DIR}/${ORACLE_PREFIX}%s${ORACLE_SUFFIX}" \
      --forward-walk-after 2026-02-24 \
      --out "$out" \
      2>&1 | tee "$LOG_DIR/fw.log"
    echo "[phase fw] FW result at $out"
    ;;

  *|help)
    cat <<'USAGE'
Usage: scripts/exp_l2_combined_v1_run.sh <phase>

Phases:
  smoke    1-seed CPU smoke (50 days, ~2 min) to validate plumbing
  oracles  Full 5-seed L3 oracle rebuild on CPU (~25 min total)
  train    L2 5-seed training on GPU (~6-8 GPU-hours, ~$200-300 Akash)
           Requires GPU_GO=yes env var.
  fw       Forward-walk evaluation on the trained models (CPU, ~5 min)

Spec contract: v3/reference/exp_l2_combined_v1_contract_2026_04_26.md
USAGE
    ;;
esac
