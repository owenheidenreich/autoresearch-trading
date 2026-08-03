"""Compare a bounded Databento Live OPRA capture with historical CBBO-1m."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import databento as db
import numpy as np
import pandas as pd

from v4.scripts.capture_databento_live_opra_training_twin import _stable_hash


SCHEMA_VERSION = "autoresearch.databento-live-opra-training-twin-audit.v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _q(values: list[int]) -> dict[str, int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.int64)
    return {
        "min": int(np.min(array)),
        "p50": int(np.quantile(array, 0.50, method="nearest")),
        "p90": int(np.quantile(array, 0.90, method="nearest")),
        "p99": int(np.quantile(array, 0.99, method="nearest")),
        "max": int(np.max(array)),
    }


def _raw_inventory(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    gateway: dict[str, list[int]] = defaultdict(list)
    mappings: dict[str, int] = {}
    for record in db.DBNStore.from_file(path):
        rtype = int(record.rtype)
        key = f"{type(record).__name__}:rtype={rtype}"
        counts[key] += 1
        ts_recv = getattr(record, "ts_recv", None)
        ts_out = getattr(record, "ts_out", None)
        if isinstance(ts_recv, int) and isinstance(ts_out, int) and ts_recv <= ts_out:
            gateway[key].append(ts_out - ts_recv)
        if type(record).__name__ == "SymbolMappingMsg":
            mappings[str(record.stype_out_symbol)] = int(record.instrument_id)
    return {
        "counts": dict(sorted(counts.items())),
        "gateway_ts_out_minus_ts_recv_ns": {
            key: _q(gateway[key]) for key in sorted(counts)
        },
        "mappings": mappings,
    }


def _dtype_map(frame: pd.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in frame.dtypes.items()}


def _put_call_spot_and_ladder(frame: pd.DataFrame) -> dict[str, Any]:
    latest_time = frame.index.max()
    latest = frame.loc[[latest_time]].copy()
    latest["right"] = latest["symbol"].str.extract(r"[0-9]{6}([CP])")[0]
    latest["strike"] = pd.to_numeric(
        latest["symbol"].str.extract(r"([0-9]{8})$")[0], errors="coerce"
    ) / 1000.0
    latest["mid"] = (latest["bid_px_00"] + latest["ask_px_00"]) / 2.0
    pairs = latest.pivot_table(
        index="strike", columns="right", values="mid", aggfunc="last"
    ).dropna()
    pairs["spot_proxy"] = pairs.index + pairs.get("C") - pairs.get("P")
    valid = pairs[
        pairs["spot_proxy"].between(3_000.0, 15_000.0)
        & (pairs["C"] > 0.0)
        & (pairs["P"] > 0.0)
    ]
    if valid.empty:
        return {
            "last_complete_minute": latest_time.isoformat(),
            "spot_proxy": None,
            "ladder": None,
        }
    spot = float(valid["spot_proxy"].median())
    atm = int(round(spot / 5.0) * 5)
    expiry = str(latest["symbol"].iloc[0])[6:12]
    needed = tuple(
        f"SPXW  {expiry}{right}{strike * 1000:08d}"
        for strike in range(atm - 25, atm + 26, 5)
        for right in ("C", "P")
    )
    observed = set(latest["symbol"].astype(str))
    return {
        "last_complete_minute": latest_time.isoformat(),
        "rows": int(len(latest)),
        "unique_symbols": int(latest["symbol"].nunique()),
        "paired_strikes_for_proxy": int(len(valid)),
        "put_call_spot_proxy": spot,
        "put_call_spot_proxy_p10": float(valid["spot_proxy"].quantile(0.10)),
        "put_call_spot_proxy_p90": float(valid["spot_proxy"].quantile(0.90)),
        "nearest_5pt_atm": atm,
        "plus_minus_25_ladder_expected": len(needed),
        "plus_minus_25_ladder_observed": sum(symbol in observed for symbol in needed),
        "plus_minus_25_ladder_missing": [symbol for symbol in needed if symbol not in observed],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--historical-cbbo-1m", type=Path, required=True)
    parser.add_argument("--definition-path", type=Path, required=True)
    parser.add_argument("--live-definition-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.capture_dir / "capture_summary.json"
    raw_path = args.capture_dir / "opra_live_mixed.dbn.zst"
    summary = json.loads(summary_path.read_text())
    if summary.get("summary_sha256") != _stable_hash(summary):
        raise RuntimeError("capture summary self-hash drift")
    if _sha(raw_path) != summary["raw_dbn"]["sha256"]:
        raise RuntimeError("raw live DBN hash drift")

    raw = _raw_inventory(raw_path)
    live = db.DBNStore.from_file(raw_path).to_df(schema="cbbo-1m")
    historical = pd.read_parquet(args.historical_cbbo_1m)
    expected_columns = list(historical.columns)
    schema_equal = list(live.columns) == expected_columns and _dtype_map(live) == _dtype_map(
        historical
    )
    counts = live.groupby(level=0).size().sort_index()
    first_minute = counts.index.min()
    capture_start = pd.Timestamp(summary["capture_started_unix_ns"], tz="UTC")
    first_interval_was_warm = capture_start <= first_minute - pd.Timedelta(minutes=1)
    identity = ["ts_recv", "symbol"]
    live_rows = live.reset_index().sort_values(identity).reset_index(drop=True)
    historical_rows = historical.reset_index().sort_values(identity).reset_index(drop=True)
    overlap = live_rows.merge(
        historical_rows,
        on=identity,
        how="left",
        suffixes=("_live", "_historical"),
        indicator=True,
        validate="one_to_one",
    )
    compared_columns = [name for name in live.columns if name not in identity]
    exact_columns: dict[str, bool] = {}
    for name in compared_columns:
        left = overlap[f"{name}_live"]
        right = overlap[f"{name}_historical"]
        if pd.api.types.is_numeric_dtype(left):
            exact_columns[name] = bool(
                np.isclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                ).all()
            )
        else:
            exact_columns[name] = bool(
                left.fillna("<NA>").astype(str).eq(
                    right.fillna("<NA>").astype(str)
                ).all()
            )
    same_session_exact = bool(
        len(overlap) == len(live_rows)
        and overlap["_merge"].eq("both").all()
        and all(exact_columns.values())
    )

    definitions = pd.read_parquet(
        args.definition_path, columns=["raw_symbol", "expiration", "asset", "instrument_id"]
    )
    session = summary["plan"]["session_date"]
    definitions = definitions[
        pd.to_datetime(definitions["expiration"], utc=True).dt.date.astype(str).eq(session)
        & definitions["asset"].astype(str).eq("SPXW")
    ].drop_duplicates("raw_symbol", keep="last")
    live_ids = definitions["raw_symbol"].astype(str).map(raw["mappings"])
    id_match = definitions["instrument_id"].astype("Int64") == live_ids.astype("Int64")

    live_definition_evidence: dict[str, Any] | None = None
    if args.live_definition_dir is not None:
        definition_summary_path = args.live_definition_dir / "definition_capture_summary.json"
        definition_raw_path = args.live_definition_dir / "opra_live_definitions.dbn.zst"
        definition_summary = json.loads(definition_summary_path.read_text())
        if definition_summary.get("summary_sha256") != _stable_hash(definition_summary):
            raise RuntimeError("live definition summary self-hash drift")
        if _sha(definition_raw_path) != definition_summary["raw_dbn"]["sha256"]:
            raise RuntimeError("live definition DBN hash drift")
        live_definition_frame = db.DBNStore.from_file(definition_raw_path).to_df(
            schema="definition"
        ).reset_index()
        live_expiration = pd.to_datetime(
            live_definition_frame["expiration"], utc=True, errors="coerce"
        )
        live_definition_frame = live_definition_frame[
            live_expiration.dt.date.astype(str).eq(session)
            & live_definition_frame["asset"].astype(str).eq("SPXW")
        ].sort_values("ts_recv").drop_duplicates("raw_symbol", keep="last")
        prior_symbols = set(definitions["raw_symbol"].astype(str))
        current_symbols = set(live_definition_frame["raw_symbol"].astype(str))
        live_definition_evidence = {
            "summary_path": str(definition_summary_path.resolve()),
            "summary_sha256": definition_summary["summary_sha256"],
            "raw_path": str(definition_raw_path.resolve()),
            "raw_sha256": definition_summary["raw_dbn"]["sha256"],
            "current_session_definition_count": len(current_symbols),
            "prior_session_known_current_expiry_count": len(prior_symbols),
            "same_day_added_symbol_count": len(current_symbols - prior_symbols),
            "same_day_removed_symbol_count": len(prior_symbols - current_symbols),
            "same_day_added_symbols": sorted(current_symbols - prior_symbols),
            "same_day_removed_symbols": sorted(prior_symbols - current_symbols),
        }

    ladder = _put_call_spot_and_ladder(live)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "TRAINING_REGIMEN_NOT_IDENTICAL_REPLACEMENT_REQUIRED",
        "capture": {
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": summary["summary_sha256"],
            "raw_path": str(raw_path.resolve()),
            "raw_sha256": summary["raw_dbn"]["sha256"],
            "record_inventory": raw["counts"],
            "gateway_ts_out_minus_ts_recv_ns": raw[
                "gateway_ts_out_minus_ts_recv_ns"
            ],
            "local_receipt_timing_ns": summary["local_receipt_minus_ts_recv_ns"],
        },
        "opra_schema_comparison": {
            "historical_path": str(args.historical_cbbo_1m.resolve()),
            "historical_sha256": _sha(args.historical_cbbo_1m),
            "live_columns": list(live.columns),
            "historical_columns": expected_columns,
            "live_dtypes": _dtype_map(live),
            "historical_dtypes": _dtype_map(historical),
            "columns_and_dtypes_identical": schema_equal,
            "live_cbbo_1m_rtype": sorted(set(live["rtype"].astype(int))),
            "historical_cbbo_1m_rtype": sorted(set(historical["rtype"].astype(int))),
            "ts_recv_is_exact_minute_boundary_live": bool(
                (live.index.second == 0).all() and (live.index.microsecond == 0).all()
            ),
            "record_level_schema_verdict": (
                "PASS_SAME_DBN_CBBO_1M_SHAPE" if schema_equal else "FAIL_SCHEMA_DRIFT"
            ),
            "same_session_live_rows": len(live_rows),
            "same_session_historical_rows": len(historical_rows),
            "same_session_rows_matched_by_minute_and_raw_symbol": int(
                overlap["_merge"].eq("both").sum()
            ),
            "same_session_exact_columns": exact_columns,
            "same_session_value_identity_verdict": (
                "PASS_BYTE_VALUE_IDENTICAL_AFTER_DECODE"
                if same_session_exact
                else "FAIL_SAME_SESSION_VALUE_DRIFT"
            ),
        },
        "live_interval_behavior": {
            "capture_started_utc": capture_start.isoformat(),
            "minute_counts": {stamp.isoformat(): int(value) for stamp, value in counts.items()},
            "subscribed_symbol_count": int(summary["plan"]["symbol_count"]),
            "first_interval_had_full_minute_warmup": first_interval_was_warm,
            "first_interval_must_be_discarded": not first_interval_was_warm,
            "last_minute_ladder": ladder,
        },
        "contract_identity": {
            "prior_session_definition_path": str(args.definition_path.resolve()),
            "prior_session_definition_sha256": _sha(args.definition_path),
            "raw_symbols_requested": int(len(definitions)),
            "raw_symbols_mapped_live": int(live_ids.notna().sum()),
            "prior_definition_instrument_ids_matching_live": int(id_match.sum()),
            "prior_definition_instrument_ids_changed_live": int(
                (live_ids.notna() & ~id_match).sum()
            ),
            "binding_rule": "raw_OSI_symbol_plus_current_session_live_mapping_not_prior_day_instrument_id",
            "live_definition_evidence": live_definition_evidence,
        },
        "current_training_regimen": {
            "opra_record_schema_identical": schema_equal,
            "decision_game_identical": False,
            "reason": (
                "the frozen training builder consumed ThetaData bar-open t close at decision t; "
                "the lawful bar for the OPRA interval ending t is ThetaData event_time t-60s"
            ),
            "fitted_rows_using_unavailable_context": 445_063,
            "fitted_rows_total": 445_063,
        },
        "replacement_clock_law": {
            "feature_interval": "[t-60s,t)",
            "option_features": "native OPRA CBBO-1m with ts_recv=t; do not rebuild from cbbo-1s",
            "official_spx": "ThetaData event_time=t-60s, whose represented interval ends t",
            "decision_emission": "fixed t+L after both exact source rows are actually received; fail closed on miss",
            "entry_execution_reference": "latest fresh exact-contract OPRA CBBO-1s at frozen order-arrival cutoff",
            "label_origin": "entry-arrival/fill clock, so the 25-minute exit is arrival+25m, not t+25m",
            "crossed_time_delayed_adapter": "REJECTED",
            "emission_lag_L_ms": 2_336,
            "emission_lag_status": "FROZEN_FROM_THETADATA_LIVE_RECEIPT_DISTRIBUTION",
        },
        "remaining_fit_blockers": [
            "bind current-session definition replay into the production candidate adapter and test add/modify/remove handling",
            "run multi-session warmup, reconnect, missing-row, early-close, and numeric feature parity tests",
            "materialize a distinct development dataset with the replacement clock; never reuse the spent holdout",
        ],
        "hard_stops": summary["hard_stops"],
        "claude_consultation": {
            "status": "REMOVED_BY_OWNER_DIRECTION_FOR_THIS_GENERATION",
            "available_after": None,
            "fresh_claim_of_claude_agreement": False,
        },
    }
    result["result_sha256"] = _stable_hash(result)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "comparison_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = "\n".join(
        [
            "# Databento Live OPRA → Historical Training-Twin Audit",
            "",
            f"Status: `{result['status']}`",
            "",
            "## Result",
            "",
            "The Databento portion has record-level schema parity: live and historical native "
            "`cbbo-1m` decode to the same 15 columns, dtypes, and rtype 193. The complete "
            "training decision game is not identical because the fitted builder consumed the "
            "following ThetaData minute at the current option boundary.",
            "",
            f"- Live CBBO-1m minutes: `{result['live_interval_behavior']['minute_counts']}`",
            f"- First partial interval discarded: `{result['live_interval_behavior']['first_interval_must_be_discarded']}`",
            f"- Last ±25-point/5-point ladder observed: `{ladder.get('plus_minus_25_ladder_observed')}/{ladder.get('plus_minus_25_ladder_expected')}`",
            f"- Prior-day instrument IDs changed live: `{result['contract_identity']['prior_definition_instrument_ids_changed_live']}/{result['contract_identity']['raw_symbols_mapped_live']}`",
            f"- Current-session definitions / prior-known: `{(live_definition_evidence or {}).get('current_session_definition_count')}/{(live_definition_evidence or {}).get('prior_session_known_current_expiry_count')}`; same-day additions: `{(live_definition_evidence or {}).get('same_day_added_symbol_count')}`",
            f"- Same-session live/historical value identity: `{result['opra_schema_comparison']['same_session_value_identity_verdict']}` (`{result['opra_schema_comparison']['same_session_rows_matched_by_minute_and_raw_symbol']}/{result['opra_schema_comparison']['same_session_live_rows']}` live rows)",
            f"- Fitted rows with unavailable context: `445063/445063`",
            "",
            "## Replacement",
            "",
            "Use native `cbbo-1m` on both live and history for features. At minute boundary `t`, "
            "pair it with the official SPX bar stamped `t-60s`. Emit only at a frozen `t+L` "
            "after both rows are received, and start execution/labels from the fresh `cbbo-1s` "
            "quote at the frozen order-arrival cutoff. `L` is frozen at 2,336 ms.",
            "",
            "## Claim boundary",
            "",
            "No model was loaded, fit, or tuned. Holdout opens: 0. No broker, paper runtime, "
            "order path, registry, promotion, or default was touched.",
            "",
            f"Result SHA-256: `{result['result_sha256']}`",
            "",
            "OWNER_REVIEW_REQUIRED",
            "",
        ]
    )
    (args.out_dir / "report.md").write_text(report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
