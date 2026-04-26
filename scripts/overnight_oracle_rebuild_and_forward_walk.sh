#!/bin/bash
# Overnight job:
#  1. Rebuild L3 oracle for each of 5 seeds in parallel against the EXTENDED
#     action_surface dataset (now covering through 2026-04-24).
#  2. After all 5 complete, re-run forward walk using the now-full oracles
#     (predictions available on every forward-walk row).
#  3. Write a summary line to a result file.
#
# Logs land in /tmp/oracle_rebuild_*.log; final forward-walk JSON lands in
# v3/artifacts/forward_walk/spx_combined_3seed_001_with_oracle.json.
#
# Started in background via nohup; survives session disconnect.

set -u
cd /Users/gduby/Documents/autoresearch-trading

PY=.venv/bin/python
DATASET=v3/artifacts/layer2_action_surface_dataset.pkl
ORACLE_PATTERN_FRESH="v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed%s_balanced_fresh.npz"
SEEDS="42 43 44 45 46"
LOG_DIR=/tmp
RESULT_DIR=v3/artifacts/forward_walk

mkdir -p "$RESULT_DIR"

# Phase 1: parallel oracle builds, OMP_NUM_THREADS=2 each (10 cores total).
echo "$(date '+%H:%M:%S') ===== Phase 1: Parallel L3 oracle rebuild ====="
PIDS=()
for S in $SEEDS; do
  out_path="v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed${S}_balanced_fresh.npz"
  log_path="${LOG_DIR}/oracle_rebuild_seed${S}.log"
  echo "$(date '+%H:%M:%S') launching seed ${S} -> ${out_path}"
  OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
    "$PY" -u -m v3.layer2.build_simulated_l3_oracle \
      --dataset "$DATASET" \
      --l3-training-source candidate_surface \
      --candidate-train-max-per-day 4 \
      --output "$out_path" \
      --seed "$S" \
      --progress-every-days 100 \
      > "$log_path" 2>&1 &
  PIDS+=($!)
done

echo "$(date '+%H:%M:%S') waiting for ${#PIDS[@]} oracle builds..."

# Wait for all to finish; report individual exit codes
FAIL_SEEDS=""
SUCCESS_SEEDS=""
for IDX in "${!PIDS[@]}"; do
  pid=${PIDS[$IDX]}
  seed=$(echo $SEEDS | cut -d' ' -f $((IDX+1)))
  if wait $pid; then
    echo "$(date '+%H:%M:%S') seed ${seed} build OK"
    SUCCESS_SEEDS="${SUCCESS_SEEDS} ${seed}"
  else
    rc=$?
    echo "$(date '+%H:%M:%S') seed ${seed} build FAILED (exit ${rc})"
    FAIL_SEEDS="${FAIL_SEEDS} ${seed}"
  fi
done

if [ -n "$FAIL_SEEDS" ]; then
  echo "$(date '+%H:%M:%S') WARNING: failed seeds:$FAIL_SEEDS"
fi
if [ -z "$SUCCESS_SEEDS" ]; then
  echo "$(date '+%H:%M:%S') ABORT: no oracles built successfully"
  exit 1
fi
echo "$(date '+%H:%M:%S') Phase 1 done. successful seeds:$SUCCESS_SEEDS"

# Phase 2: forward walk on the extended bundle, using the FRESH oracles.
# These oracles will have predictions on every forward-walk row (the
# previous _extended npzs were NaN-padded; these are real predictions).
echo "$(date '+%H:%M:%S') ===== Phase 2: Forward walk with fresh oracles ====="
PYTHONUNBUFFERED=1 "$PY" -u -m v3.analysis.forward_walk \
  --dataset "$DATASET" \
  --oracle-pattern "$ORACLE_PATTERN_FRESH" \
  --out "${RESULT_DIR}/spx_combined_3seed_001_with_oracle.json" \
  > "${LOG_DIR}/forward_walk_with_oracle.log" 2>&1
FW_RC=$?
echo "$(date '+%H:%M:%S') forward walk exit ${FW_RC}"

# Phase 3: short summary
echo "$(date '+%H:%M:%S') ===== Phase 3: Summary ====="
"$PY" -c "
import json
with open('${RESULT_DIR}/spx_combined_3seed_001_with_oracle.json') as f:
    out = json.load(f)
fw = out.get('forward_walk_first_day'), out.get('forward_walk_last_day')
print(f'Forward walk: {fw[0]} -> {fw[1]} ({out.get(\"forward_walk_unique_days\")} days)')
print()
print('Per-seed forward-walk PFs (with oracle):')
print(f\"  {'seed':>5} {'n':>4} {'pf_oracle':>10} {'dd_oracle':>10} {'pf_no_oracle':>13} {'cal_pf':>8}\")
for s in out.get('seeds', []):
    m = out.get('per_seed', {}).get(str(s)) or out.get('per_seed', {}).get(s, {})
    if not m or m.get('n_trades', 0) == 0:
        print(f'  {s:>5}  empty')
        continue
    o = m.get('with_oracle', {})
    n_o = m.get('no_oracle_dataset_label', {})
    cal = m.get('calibration_pf')
    print(f'  {s:>5} {o.get(\"n\", 0):>4} {o.get(\"pf\", 0):>10.3f} {o.get(\"dd_pct\", 0):>10.2f} {n_o.get(\"pf\", 0):>13.3f} {cal if cal else 0:>8.3f}')
print()
xs = out.get('cross_seed_no_filter', {})
print(f'Cross-seed (no consensus filter): mean PF with oracle = {xs.get(\"mean_pf_with_oracle\", 0):.3f}, no oracle = {xs.get(\"mean_pf_no_oracle\", 0):.3f}')
" >> "${LOG_DIR}/forward_walk_with_oracle.log"

cat "${LOG_DIR}/forward_walk_with_oracle.log" | tail -30

echo "$(date '+%H:%M:%S') ===== DONE ====="
echo "Result JSON: ${RESULT_DIR}/spx_combined_3seed_001_with_oracle.json"
echo "Logs: ${LOG_DIR}/oracle_rebuild_seed*.log + ${LOG_DIR}/forward_walk_with_oracle.log"
