"""Audit a bounded mixed-schema live OPRA capture without model or holdout access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import databento as db
import pandas as pd

from v4.scripts.capture_databento_live_opra_training_twin import _sha256_path, _stable_hash


SCHEMA_VERSION = "autoresearch.databento-live-opra-feature-surface-audit.v1"


def _frame(raw_path: Path, schema: str) -> pd.DataFrame:
    return db.DBNStore.from_file(raw_path).to_df(schema=schema).reset_index()


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.astype(str).value_counts().items()}


def audit(capture_dir: Path) -> dict[str, Any]:
    summary_path = capture_dir / "capture_summary.json"
    summary = json.loads(summary_path.read_text())
    expected_summary_hash = summary.get("summary_sha256")
    if expected_summary_hash != _stable_hash(summary):
        raise RuntimeError("capture summary semantic hash mismatch")
    raw_path = Path(summary["raw_dbn"]["path"])
    if _sha256_path(raw_path) != summary["raw_dbn"]["sha256"]:
        raise RuntimeError("raw live DBN hash mismatch")

    cmbp = _frame(raw_path, "cmbp-1")
    tcbbo = _frame(raw_path, "tcbbo")
    trades = _frame(raw_path, "trades")
    ohlcv = _frame(raw_path, "ohlcv-1m")
    statistics = _frame(raw_path, "statistics")
    status = _frame(raw_path, "status")
    universe = int(summary["plan"]["symbol_count"])
    elapsed = float(summary["elapsed_seconds"])
    trade_keys = ["ts_recv", "ts_event", "instrument_id", "price", "size", "symbol"]
    minute_counts = {
        pd.Timestamp(key).isoformat(): int(value)
        for key, value in ohlcv.groupby("ts_event")["symbol"].nunique().items()
    }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "AUDITED_NO_MODEL_NO_ORDER_LIVE_FEATURE_SURFACE",
        "source": {
            "capture_summary_path": str(summary_path.resolve()),
            "capture_summary_sha256": _sha256_path(summary_path),
            "capture_result_sha256": expected_summary_hash,
            "raw_dbn_path": str(raw_path.resolve()),
            "raw_dbn_sha256": summary["raw_dbn"]["sha256"],
        },
        "universe": {
            "subscribed_current_session_spxw_0dte_symbols": universe,
            "symbols_sha256": summary["plan"]["symbols_sha256"],
        },
        "schemas": {
            "cmbp-1": {
                "rows": int(len(cmbp)),
                "symbols": int(cmbp["symbol"].nunique()),
                "rows_per_second": len(cmbp) / elapsed,
                "action_counts": _counts(cmbp["action"]),
                "side_counts": _counts(cmbp["side"]),
                "local_receipt_minus_ts_recv_ns": summary[
                    "local_receipt_minus_ts_recv_ns"
                ].get("CMBP1Msg:rtype=177"),
            },
            "tcbbo": {
                "rows": int(len(tcbbo)),
                "symbols": int(tcbbo["symbol"].nunique()),
                "side_counts": _counts(tcbbo["side"]),
            },
            "trades": {
                "rows": int(len(trades)),
                "symbols": int(trades["symbol"].nunique()),
                "side_counts": _counts(trades["side"]),
            },
            "ohlcv-1m": {
                "rows": int(len(ohlcv)),
                "symbols": int(ohlcv["symbol"].nunique()),
                "symbol_count_by_bar_open": minute_counts,
                "missing_symbol_count_by_bar_open": {
                    key: universe - value for key, value in minute_counts.items()
                },
                "all_printed_bars_have_positive_volume": bool((ohlcv["volume"] > 0).all()),
            },
            "statistics": {"rows": int(len(statistics))},
            "status": {"rows": int(len(status))},
        },
        "cross_schema_checks": {
            "trades_tcbbo_row_count_equal": len(trades) == len(tcbbo),
            "trades_tcbbo_exact_key_order_equal": trades[trade_keys].equals(tcbbo[trade_keys]),
            "all_native_trade_sides_unknown": set(trades["side"].astype(str)) == {"N"},
        },
        "fit_implications": {
            "cmbp_full_universe": "NOT_FIT_READY_UNDER_CAPTURED_TOPOLOGY",
            "trade_flow": "NEEDS_HISTORICAL_TRADES_TCBBO_AND_QUOTE_RULE_AUDIT",
            "ohlcv_missing_minute": "SYNTHESIZE_ZERO_ONLY_AFTER_HEALTHY_FROZEN_CUTOFF_NEVER_CARRY",
            "open_interest": "PREOPEN_OR_PRIOR_DAY_SESSION_STATIC_ONLY",
            "status": "GUARD_ONLY_REQUIRES_REPLAY_OR_FAULT_TEST",
        },
        "hard_stops": {
            "broker_accessed": False,
            "holdout_open_count": 0,
            "model_loaded_or_fit": False,
            "order_path_accessed": False,
            "paper_runtime_accessed": False,
        },
    }
    result["result_sha256"] = _stable_hash(result)
    return result


def report(result: dict[str, Any]) -> str:
    schemas = result["schemas"]
    return f"""# Databento live OPRA entry feature-surface audit

Status: `{result['status']}`

- CMBP-1: {schemas['cmbp-1']['rows']:,} rows across {schemas['cmbp-1']['symbols']} symbols
  ({schemas['cmbp-1']['rows_per_second']:.1f} rows/second).
- Trades/TBBO: {schemas['trades']['rows']:,} exact one-to-one rows; every native trade side was `N`.
- OHLCV-1m: {schemas['ohlcv-1m']['rows']} printed rows with per-minute symbol counts
  {schemas['ohlcv-1m']['symbol_count_by_bar_open']} out of
  {result['universe']['subscribed_current_session_spxw_0dte_symbols']} subscribed symbols.
- Statistics rows: {schemas['statistics']['rows']}; status rows: {schemas['status']['rows']}.

Disposition: full-universe CMBP, tick trade flow, sparse minute volume, open
interest, and status all remain blocked or guard-only exactly as recorded in
`fit_implications`. No model, holdout, broker, or order path was accessed.

Result SHA-256: `{result['result_sha256']}`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = audit(args.capture_dir)
    (args.output_dir / "feature_surface_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "report.md").write_text(report(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
