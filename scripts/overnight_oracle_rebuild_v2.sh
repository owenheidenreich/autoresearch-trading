#!/bin/bash
# v2: with the day_to_window trailing patch so forward-walk days get
# predictions from W12's trained model.
set -u
cd /Users/gduby/Documents/autoresearch-trading

PY=.venv/bin/python
DATASET=v3/artifacts/layer2_action_surface_dataset.pkl
ORACLE_PATTERN_FRESH="v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed%s_balanced_fresh.npz"
SEEDS="42 43 44 45 46"
LOG_DIR=/tmp
RESULT_DIR=v3/artifacts/forward_walk

mkdir -p "$RESULT_DIR"

# Remove the bad fresh npzs from the previous run so we can detect failures.
for S in $SEEDS; do
  rm -f "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed${S}_balanced_fresh.npz"
done

echo "$(date '+%H:%M:%S') ===== Phase 1: Parallel L3 oracle rebuild WITH TRAILING PATCH ====="
PIDS=()
for S in $SEEDS; do
  out_path="v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed${S}_balanced_fresh.npz"
  log_path="${LOG_DIR}/oracle_rebuild_v2_seed${S}.log"
  echo "$(date '+%H:%M:%S') launching seed ${S} -> ${out_path}"
  OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
    "$PY" -u -m scripts.build_l3_oracle_with_trailing \
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
SUCCESS_SEEDS=""
FAIL_SEEDS=""
for IDX in "${!PIDS[@]}"; do
  pid=${PIDS[$IDX]}
  seed=$(echo $SEEDS | cut -d' ' -f $((IDX+1)))
  if wait $pid; then
    SUCCESS_SEEDS="${SUCCESS_SEEDS} ${seed}"
    echo "$(date '+%H:%M:%S') seed ${seed} build OK"
  else
    rc=$?
    FAIL_SEEDS="${FAIL_SEEDS} ${seed}"
    echo "$(date '+%H:%M:%S') seed ${seed} build FAILED (exit ${rc})"
  fi
done

if [ -z "$SUCCESS_SEEDS" ]; then
  echo "$(date '+%H:%M:%S') ABORT: no oracles built"
  exit 1
fi
echo "$(date '+%H:%M:%S') Phase 1 done. successful seeds:$SUCCESS_SEEDS"

echo "$(date '+%H:%M:%S') ===== Phase 2: Forward walk with fresh oracles ====="
PYTHONUNBUFFERED=1 "$PY" -u -m v3.analysis.forward_walk \
  --dataset "$DATASET" \
  --oracle-pattern "$ORACLE_PATTERN_FRESH" \
  --out "${RESULT_DIR}/spx_combined_3seed_001_with_oracle.json" \
  > "${LOG_DIR}/forward_walk_with_oracle_v2.log" 2>&1

echo "$(date '+%H:%M:%S') ===== Phase 3: Summary ====="
"$PY" -c "
import json
with open('${RESULT_DIR}/spx_combined_3seed_001_with_oracle.json') as f:
    out = json.load(f)
print('Forward walk:', out.get('forward_walk_first_day'), '->', out.get('forward_walk_last_day'),
      f'({out.get(\"forward_walk_unique_days\")} days)')
print()
print('Per-seed forward-walk PFs (with oracle restored):')
header = f\"  {'seed':>5} {'n':>4} {'pf_oracle':>10} {'dd_oracle':>10} {'pf_no_oracle':>13} {'cal_pf':>8}\"
print(header)
oracle_pfs = []
for s in out.get('seeds', []):
    m = out.get('per_seed', {}).get(str(s)) or out.get('per_seed', {}).get(s, {})
    if not m:
        continue
    o = m.get('with_oracle', {})
    n_o = m.get('no_oracle_dataset_label', {})
    cal = m.get('calibration_pf')
    n = o.get('n', 0)
    pf = o.get('pf', 0)
    print(f'  {s:>5} {n:>4} {pf:>10.3f} {o.get(\"dd_pct\",0):>10.2f} {n_o.get(\"pf\",0):>13.3f} {cal if cal else 0:>8.3f}')
    if n > 0:
        oracle_pfs.append(pf)
if oracle_pfs:
    print()
    print(f'Cross-seed mean WITH ORACLE: {sum(oracle_pfs)/len(oracle_pfs):.3f}  (offline claim: 1.881)')
" >> "${LOG_DIR}/forward_walk_with_oracle_v2.log"

cat "${LOG_DIR}/forward_walk_with_oracle_v2.log" | tail -30

echo "$(date '+%H:%M:%S') ===== DONE ====="
