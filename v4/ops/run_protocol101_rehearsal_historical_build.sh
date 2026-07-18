#!/usr/bin/env bash
# Protocol101 rehearsal: historical-side build + battery, post-ThetaData-renewal.
#
# PRECONDITION: ThetaData Index Standard/Pro subscription active (SPX/VIX
# index_history_ohlc). Databento OPRA raw for 2026-07-10/13/14 is already
# downloaded. IBKR-side replays already built under
# v4/audit/autoresearch/protocol101_canonical_rehearsal_ibkr_capture_replay_*.
#
# Runs: ThetaData SPX/VIX download -> source-aligned dataset build ->
# historical replays -> rehearsal battery (runner already validated to
# reproduce attempt005 exactly in burned_validation mode).

set -euo pipefail
cd "$(dirname "$0")/../.."

PY="$HOME/.autoresearch-trading/runtime-venv/bin/python"
export PYTHONPATH="$PWD"
export V4_PAID_DATA_APPROVAL_TEXT="APPROVED: paid vendor download of Databento OPRA SPXW (definition, cbbo-1m, ohlcv-1m, statistics) and ThetaData SPX/VIX 1m index bars for sessions 2026-07-10, 2026-07-13, 2026-07-14 for the canonical v1.4 rehearsal battery."
MANIFEST="v4/promotion/PROTOCOL101_REHEARSAL_2026_07_JULY_DOWNLOAD_MANIFEST.json"
REHEARSAL_INPUTS="v4/audit/autoresearch/protocol101_canonical_rehearsal_inputs"

echo "== 1/4 ThetaData SPX/VIX index bars =="
"$PY" v4/scripts/download_thetadata_index_bars.py \
  --start-date 2026-07-10 --end-date 2026-07-14 --symbols SPX VIX \
  --approval-manifest "$MANIFEST"

echo "== 2/4 source-aligned dataset build =="
"$PY" v4/scripts/build_databento_neural_dataset.py \
  --start-date 2026-07-10 --end-date 2026-07-14 \
  --feature-contract protocol101-live-v2-microstructure-masked \
  --context-mode official \
  --official-spx-dir data/vendor/thetadata/index/spx_1m \
  --official-vix-dir data/vendor/thetadata/index/vix_1m \
  --normalized-dir "$REHEARSAL_INPUTS/source_aligned_normalized" \
  --processed-dir "$REHEARSAL_INPUTS/source_aligned_processed" \
  --summary-out "$REHEARSAL_INPUTS/source_aligned_historical_summary.json" \
  --compute-live-policy-labels

echo "== 3/4 historical replays =="
for d in 2026-07-10 2026-07-13 2026-07-14; do
  "$PY" v4/scripts/run_protocol101_fair_contract_dataset_replay.py \
    --input-pkl "$REHEARSAL_INPUTS/source_aligned_processed/$d.pkl" \
    --session "$d" \
    --out-dir "v4/audit/autoresearch/protocol101_canonical_rehearsal_historical_replay_${d//-/_}"
done

echo "== 4/4 rehearsal battery =="
"$PY" v4/scripts/run_protocol101_canonical_v1_4_rehearsal_battery.py \
  --mode rehearsal \
  --eval-sessions 2026-07-10 2026-07-13 2026-07-14 \
  --eval-prefix protocol101_canonical_rehearsal \
  --out-dir v4/audit/autoresearch/protocol101_canonical_v1_4_rehearsal_battery_attempt001 \
  --force

echo "DONE — read v4/audit/autoresearch/protocol101_canonical_v1_4_rehearsal_battery_attempt001/report.md"
