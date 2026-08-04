"""Guarded ES BBO-1s acquisition and time-weighted spread measurement.

The downloader accepts only a manifest whose per-session windows resolve to
09:30-16:00 America/New_York.  It cost-checks every named session, invokes the
paid-data guard immediately before every range call, and refuses to add to an
existing output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.checks.paid_data_guard import DEFAULT_APPROVAL_ENV_VAR, require_paid_data_approval
from v4.research.pathd_matched_feasibility_gate import _es_files, _roll_exclusion


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
TICK_POINTS = 0.25
TICK_DOLLARS = 12.50
COMMISSIONS_DOLLARS = 4.50
HORIZONS = (15, 30, 60)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    scope = payload.get("authorized_scope", {})
    expected = {
        "dataset": "GLBX.MDP3",
        "symbols": ["ES.FUT"],
        "stype_in": "parent",
        "schema": "bbo-1s",
        "paid_cap_usd": 3.0,
        "session_count": 20,
    }
    for key, value in expected.items():
        if scope.get(key) != value:
            raise SystemExit(f"manifest scope mismatch for {key}: {scope.get(key)!r}")
    sessions = scope.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 20 or len(set(sessions)) != 20:
        raise SystemExit("manifest must name exactly 20 unique sessions")
    return payload


def _session_windows(scope: dict[str, Any]) -> dict[str, tuple[datetime, datetime]]:
    sessions = [str(item) for item in scope["sessions"]]
    declared = scope.get("session_windows_utc")
    windows: dict[str, tuple[datetime, datetime]] = {}
    if isinstance(declared, dict):
        if set(declared) != set(sessions):
            raise SystemExit("session_windows_utc must cover exactly the named sessions")
        for session in sessions:
            item = declared[session]
            start = datetime.fromisoformat(str(item["start"]).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(item["end"]).replace("Z", "+00:00"))
            windows[session] = (start, end)
    else:
        item = scope.get("session_window_utc")
        if not isinstance(item, dict):
            raise SystemExit("manifest needs session_windows_utc or session_window_utc")
        for session in sessions:
            windows[session] = (
                datetime.fromisoformat(f"{session}T{item['start']}:00+00:00"),
                datetime.fromisoformat(f"{session}T{item['end']}:00+00:00"),
            )
    for session, (start, end) in windows.items():
        local_start = start.astimezone(NY)
        local_end = end.astimezone(NY)
        if local_start.date().isoformat() != session or local_end.date().isoformat() != session:
            raise SystemExit(f"window date drift: {session}")
        if (local_start.hour, local_start.minute, local_end.hour, local_end.minute) != (9, 30, 16, 0):
            raise SystemExit(
                f"manifest window is not full New York RTH for {session}: "
                f"{local_start:%H:%M %Z}-{local_end:%H:%M %Z}"
            )
        if (end - start).total_seconds() != 6.5 * 3600:
            raise SystemExit(f"manifest window is not 6.5 hours: {session}")
    return windows


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if not len(values):
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(values[int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))])


def _spread_summary(frame: pd.DataFrame) -> dict[str, Any]:
    ticks = frame["spread_ticks"].to_numpy(float)
    weights = frame["duration_seconds"].to_numpy(float)
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("spread frame has no positive time weight")
    return {
        "time_weight_seconds": total,
        "mean_ticks": float(np.average(ticks, weights=weights)),
        "median_ticks": _weighted_quantile(ticks, weights, 0.5),
        "share_one_tick": float(weights[ticks < 1.5].sum() / total),
        "share_two_ticks": float(weights[(ticks >= 1.5) & (ticks < 2.5)].sum() / total),
        "share_three_plus_ticks": float(weights[ticks >= 2.5].sum() / total),
    }


def _front_instrument_id(es_raw: Path, session: str) -> int:
    matches = sorted(es_raw.glob(f"{session}*.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one ES OHLCV file for {session}, got {len(matches)}")
    frame = pd.read_parquet(matches[0], columns=["instrument_id"])
    ids = frame["instrument_id"].dropna().astype(int).unique()
    if len(ids) != 1:
        raise RuntimeError(f"continuous ES instrument identity is not unique: {session}")
    return int(ids[0])


def _rv15(es_raw: Path, session: str) -> pd.DataFrame:
    path = next(iter(sorted(es_raw.glob(f"{session}*.parquet"))), None)
    if path is None:
        raise RuntimeError(f"missing ES OHLCV for {session}")
    frame = pd.read_parquet(path, columns=["close"]).sort_index()
    returns = np.log(frame["close"].astype(float)).diff()
    rv = np.sqrt(returns.pow(2).rolling(15, min_periods=15).sum())
    # OHLCV index is minute-open; this value becomes available at minute-close.
    return pd.DataFrame({"rv_available": frame.index + pd.Timedelta(minutes=1), "rv15": rv}).dropna()


def _measurement_rows(
    parquet_paths: dict[str, Path],
    windows: dict[str, tuple[datetime, datetime]],
    es_raw: Path,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for session, path in sorted(parquet_paths.items()):
        frame = pd.read_parquet(path).reset_index()
        timestamp = "ts_recv" if "ts_recv" in frame.columns else frame.columns[0]
        frame[timestamp] = pd.to_datetime(frame[timestamp], utc=True)
        start, end = windows[session]
        front = _front_instrument_id(es_raw, session)
        frame = frame[
            (frame["instrument_id"].astype("Int64") == front)
            & (frame[timestamp] >= start)
            & (frame[timestamp] < end)
        ].copy()
        frame = frame.sort_values(timestamp, kind="mergesort").drop_duplicates(timestamp, keep="last")
        valid = (
            np.isfinite(frame["bid_px_00"].to_numpy(float))
            & np.isfinite(frame["ask_px_00"].to_numpy(float))
            & (frame["bid_px_00"].to_numpy(float) > 0.0)
            & (frame["ask_px_00"].to_numpy(float) > frame["bid_px_00"].to_numpy(float))
        )
        frame = frame.loc[valid].copy()
        if frame.empty:
            raise RuntimeError(f"no actionable front ES BBO rows: {session}")
        next_time = frame[timestamp].shift(-1).fillna(pd.Timestamp(end))
        frame["duration_seconds"] = (next_time - frame[timestamp]).dt.total_seconds().clip(lower=0.0)
        frame["spread_ticks"] = (frame["ask_px_00"] - frame["bid_px_00"]) / TICK_POINTS
        rv = _rv15(es_raw, session).sort_values("rv_available")
        frame = pd.merge_asof(
            frame.sort_values(timestamp),
            rv,
            left_on=timestamp,
            right_on="rv_available",
            direction="backward",
        )
        frame["session"] = session
        frame["window_seconds"] = (end - start).total_seconds()
        rows.append(frame[["session", timestamp, "duration_seconds", "spread_ticks", "rv15", "window_seconds"]].rename(columns={timestamp: "ts_recv"}))
    return pd.concat(rows, ignore_index=True)


def _mean_abs_es_moves(es_raw: Path) -> tuple[dict[str, float], list[str], list[dict[str, Any]]]:
    files = _es_files(es_raw)
    excluded, details = _roll_exclusion(files)
    moves: dict[int, list[np.ndarray]] = {horizon: [] for horizon in HORIZONS}
    for session, _, frame in files:
        if session in excluded:
            continue
        close = frame["close"].to_numpy(float)
        for horizon in HORIZONS:
            if len(close) > horizon:
                moves[horizon].append(np.abs((close[horizon:] - close[:-horizon]) * 50.0))
    return (
        {str(horizon): float(np.concatenate(moves[horizon]).mean()) for horizon in HORIZONS},
        sorted(excluded),
        details,
    )


def _download(
    *,
    client: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    windows: dict[str, tuple[datetime, datetime]],
    output_dir: Path,
    approval_env_var: str,
) -> tuple[dict[str, Path], dict[str, Any]]:
    scope = manifest["authorized_scope"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to add to or repeat an existing acquisition: {output_dir}")
    costs: dict[str, float] = {}
    for session, (start, end) in sorted(windows.items()):
        costs[session] = float(
            client.metadata.get_cost(
                dataset=scope["dataset"], schema=scope["schema"], symbols=scope["symbols"],
                stype_in=scope["stype_in"], start=start, end=end,
            )
        )
    total = float(sum(costs.values()))
    if total > float(scope["paid_cap_usd"]):
        raise SystemExit(f"exact cost ${total:.6f} exceeds cap ${scope['paid_cap_usd']:.2f}")
    # Validate the human-supplied text before creating even an empty acquisition
    # directory.  The same guard is repeated immediately before every range call.
    require_paid_data_approval(
        manifest_path=manifest_path,
        approval_text=None,
        approval_env_var=approval_env_var,
        operation="Databento GLBX ES bbo-1s acquisition",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    parquet_paths: dict[str, Path] = {}
    files: list[dict[str, Any]] = []
    for session, (start, end) in sorted(windows.items()):
        dbn_path = output_dir / f"{session}.es_fut.bbo-1s.dbn.zst"
        parquet_path = output_dir / f"{session}.es_fut.bbo-1s.parquet"
        require_paid_data_approval(
            manifest_path=manifest_path,
            approval_text=None,
            approval_env_var=approval_env_var,
            operation=f"Databento GLBX ES bbo-1s download for {session}",
        )
        store = client.timeseries.get_range(
            dataset=scope["dataset"], schema=scope["schema"], symbols=scope["symbols"],
            stype_in=scope["stype_in"], start=start, end=end, path=dbn_path,
        )
        frame = store.to_df()
        frame.to_parquet(parquet_path, index=True)
        parquet_paths[session] = parquet_path
        files.append(
            {
                "session": session,
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
                "estimated_cost_usd": costs[session],
                "dbn_bytes": dbn_path.stat().st_size,
                "parquet_bytes": parquet_path.stat().st_size,
                "dbn_sha256": _sha256(dbn_path),
                "parquet_sha256": _sha256(parquet_path),
                "rows": int(len(frame)),
            }
        )
    receipt = {
        "schema_version": "pathd.es-bbo1s-acquisition-receipt.v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "request": {key: scope[key] for key in ("dataset", "symbols", "stype_in", "schema")},
        "session_count": len(windows),
        "estimated_cost_usd": total,
        "paid_cap_usd": float(scope["paid_cap_usd"]),
        "files": files,
        "guard_invoked_before_each_range_call": True,
    }
    (output_dir / "acquisition_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return parquet_paths, receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    _load_env(args.env_file)
    manifest = _manifest(args.manifest)
    windows = _session_windows(manifest["authorized_scope"])
    try:
        import databento as db
    except ImportError as exc:
        raise SystemExit("databento package is required") from exc
    client = db.Historical()
    parquet_paths, receipt = _download(
        client=client,
        manifest_path=args.manifest,
        manifest=manifest,
        windows=windows,
        output_dir=args.output_dir,
        approval_env_var=args.approval_env_var,
    )
    states = _measurement_rows(parquet_paths, windows, args.es_ohlcv_root)
    unconditional = _spread_summary(states)
    rv_threshold = _weighted_quantile(
        states["rv15"].to_numpy(float), states["duration_seconds"].to_numpy(float), 0.75
    )
    elevated = states[states["rv15"] >= rv_threshold]
    conditional = _spread_summary(elevated)
    session_summaries = {
        session: {
            **_spread_summary(frame),
            "coverage_fraction": float(frame["duration_seconds"].sum() / frame["window_seconds"].iloc[0]),
        }
        for session, frame in states.groupby("session", sort=True)
    }
    mean_moves, roll_excluded, roll_details = _mean_abs_es_moves(args.es_ohlcv_root)
    friction = COMMISSIONS_DOLLARS + conditional["mean_ticks"] * TICK_DOLLARS
    hurdles = {
        horizon: 100.0 * (0.5 + friction / (2.0 * mean_move))
        for horizon, mean_move in mean_moves.items()
    }
    credible = any(value <= 56.6 for value in hurdles.values())
    result = {
        "schema_version": "pathd.es-spread-measurement.v1",
        "acquisition_receipt": receipt,
        "time_weighting": "each actionable BBO state weighted until next state or RTH end",
        "elevated_volatility": {
            "definition": "top time-weighted quartile of causal trailing-15-completed-minute root-sum-squared ES log returns",
            "rv15_threshold": rv_threshold,
        },
        "unconditional": unconditional,
        "elevated_rv15": conditional,
        "session_summaries": session_summaries,
        "measured_primary_friction_dollars": friction,
        "friction_formula": "$4.50 commissions + elevated-RV time-weighted mean spread ticks * $12.50",
        "mean_abs_move_dollars_roll_excluded": mean_moves,
        "roll_excluded_sessions": roll_excluded,
        "roll_change_details": roll_details,
        "mean_payoff_hurdles_pct": hurdles,
        "passive_0dte_comparison_pct": {"low": 53.9, "high": 56.6},
        "verdict": "ES_REMAINS_CREDIBLE_BRANCH" if credible else "ES_BRANCH_CLOSED_BY_SPREAD",
        "protected_holdout_opened": False,
        "model_fit_executed": False,
    }
    result_path = args.output_dir / "spread_measurement_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("v4/audit/autoresearch/protocol101_pathd_data_acquisition/paid_data_approval_manifest_es_bbo1s_2026_08_03.json"),
    )
    parser.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    parser.add_argument("--approval-env-var", default=DEFAULT_APPROVAL_ENV_VAR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31/raw/databento/glbx_es_bbo_1s_measurement_2026_08_03"),
    )
    parser.add_argument(
        "--es-ohlcv-root",
        type=Path,
        default=Path("/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31/raw/databento/glbx_es_ohlcv_1m"),
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"verdict": result["verdict"], "friction": result["measured_primary_friction_dollars"]}, sort_keys=True))


if __name__ == "__main__":
    main()
