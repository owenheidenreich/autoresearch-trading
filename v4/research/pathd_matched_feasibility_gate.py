"""Execute the frozen 2026-08-03 matched Path-D feasibility gate.

This module fits no model, reads no protected-holdout partition, and contacts no
external service.  It turns the prose pre-registration into deterministic
mechanics and records the interpretations that the prose left operationally
implicit.  In particular, option fills use the primary one-tick-penetration
counterfactual directly; they do not apply the previously measured average
saving as a universal constant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from v4.research.autoresearch_v2.statistics import paired_summary, session_blocked_max_t


NY = ZoneInfo("America/New_York")
UTC = timezone.utc
MINUTE_NS = 60_000_000_000
SECOND_NS = 1_000_000_000
OPTION_FEE_DOLLARS = 3.0
ES_FRICTION = {1: 17.0, 2: 29.5, 3: 42.0, 4: 54.5}
HORIZONS_OPTION = (30, 60)
HORIZONS_ES_MOMENTUM = (15, 30, 60)
HORIZONS_ES_REVERSION = (15, 30)
MAXT_PERMUTATIONS = 20_000
MAXT_SEED = 101


@dataclass(frozen=True)
class Paths:
    root: Path
    scores: Path
    option_raw: Path
    es_raw: Path


def _tick(price: float) -> float:
    return 0.05 if price < 3.0 else 0.10


def _session_terminal_ns(session: str) -> int:
    day = datetime.fromisoformat(session).date()
    return int(datetime.combine(day, time(15, 55), tzinfo=NY).astimezone(UTC).timestamp() * 1e9)


def _stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:16], 16) % (2**32)


def _raw_option_session(path: Path, symbols: Iterable[str]) -> pd.DataFrame:
    dataset = pads.dataset(path, format="parquet")
    table = dataset.to_table(
        columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"],
        filter=pads.field("symbol").isin(sorted(set(symbols))),
    )
    frame = table.to_pandas(ignore_metadata=False).reset_index()
    if "ts_recv" not in frame.columns:
        raise RuntimeError(f"missing ts_recv after source read: {path}")
    frame["ts_ns"] = pd.to_datetime(frame["ts_recv"], utc=True).astype("int64")
    frame = frame.drop_duplicates(["symbol", "ts_ns"], keep="last")
    return frame.sort_values(["symbol", "ts_ns"], kind="mergesort")


def _option_candidate_outcomes(paths: Paths, scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, candidates in scores.groupby("session", sort=True):
        source = paths.option_raw / f"{session}.cbbo-1s.parquet"
        if not source.is_file():
            raise RuntimeError(f"missing owned option source: {source}")
        quotes = _raw_option_session(source, candidates["raw_symbol"].unique())
        quote_paths: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for symbol, group in quotes.groupby("symbol", sort=False):
            valid = (
                np.isfinite(group["bid_px_00"].to_numpy(float))
                & np.isfinite(group["ask_px_00"].to_numpy(float))
                & (group["bid_px_00"].to_numpy(float) > 0.0)
                & (group["ask_px_00"].to_numpy(float) > group["bid_px_00"].to_numpy(float))
            )
            clean = group.loc[valid]
            quote_paths[str(symbol)] = (
                clean["ts_ns"].to_numpy(np.int64),
                clean["bid_px_00"].to_numpy(float),
                clean["ask_px_00"].to_numpy(float),
            )
        terminal_ns = _session_terminal_ns(str(session))
        for item in candidates.itertuples(index=False):
            path = quote_paths.get(str(item.raw_symbol))
            base = {
                "candidate_uid": str(item.candidate_uid),
                "session": str(session),
                "outer_fold": int(item.outer_fold),
                "entry_arrival_ns": int(item.entry_arrival_ns),
            }
            local = pd.Timestamp(int(item.entry_arrival_ns), unit="ns", tz="UTC").tz_convert(NY)
            base["h1_window"] = bool(local.hour == 10 and local.minute >= 30)
            if path is None:
                for horizon in HORIZONS_OPTION:
                    rows.append({**base, "horizon_min": horizon, "filled": False, "pnl": 0.0, "reverse_pnl": 0.0, "invalid_source": True})
                continue
            times, bids, asks = path
            arrival = int(item.entry_arrival_ns)
            arrival_index = int(np.searchsorted(times, arrival, side="right") - 1)
            actionable = arrival_index >= 0 and arrival - int(times[arrival_index]) <= 2 * SECOND_NS
            if not actionable:
                for horizon in HORIZONS_OPTION:
                    rows.append({**base, "horizon_min": horizon, "filled": False, "pnl": 0.0, "reverse_pnl": 0.0, "invalid_source": True})
                continue
            limit = float(bids[arrival_index])
            penetration = limit - _tick(limit)
            fill_start = int(np.searchsorted(times, arrival, side="left"))
            fill_stop = int(np.searchsorted(times, arrival + 60 * SECOND_NS, side="right"))
            fill_relative = np.flatnonzero(asks[fill_start:fill_stop] <= penetration + 1e-12)
            if not len(fill_relative):
                for horizon in HORIZONS_OPTION:
                    rows.append({**base, "horizon_min": horizon, "filled": False, "pnl": 0.0, "reverse_pnl": 0.0, "invalid_source": False})
                continue
            fill_index = fill_start + int(fill_relative[0])
            fill_time = int(times[fill_index])
            for horizon in HORIZONS_OPTION:
                target = min(fill_time + horizon * MINUTE_NS, terminal_ns)
                exit_index = int(np.searchsorted(times, target, side="right") - 1)
                if exit_index < fill_index:
                    rows.append({**base, "horizon_min": horizon, "filled": False, "pnl": 0.0, "reverse_pnl": 0.0, "invalid_source": True})
                    continue
                move = (float(bids[exit_index]) - limit) * 100.0
                rows.append(
                    {
                        **base,
                        "horizon_min": horizon,
                        "filled": True,
                        "pnl": move - OPTION_FEE_DOLLARS,
                        "reverse_pnl": -move - OPTION_FEE_DOLLARS,
                        "invalid_source": False,
                        "limit": limit,
                        "fill_time_ns": fill_time,
                        "exit_time_ns": int(times[exit_index]),
                    }
                )
    return pd.DataFrame(rows)


def _es_files(es_raw: Path) -> list[tuple[str, Path, pd.DataFrame]]:
    output: list[tuple[str, Path, pd.DataFrame]] = []
    for path in sorted(es_raw.glob("*.parquet")):
        session = path.name[:10]
        frame = pd.read_parquet(path, columns=["instrument_id", "close"])
        if frame.empty:
            continue
        output.append((session, path, frame))
    return output


def _roll_exclusion(es_files: list[tuple[str, Path, pd.DataFrame]]) -> tuple[set[str], list[dict[str, Any]]]:
    changes: list[int] = []
    details: list[dict[str, Any]] = []
    ids = [int(frame["instrument_id"].iloc[0]) for _, _, frame in es_files]
    for index in range(1, len(es_files)):
        if ids[index] != ids[index - 1]:
            changes.append(index)
            details.append(
                {
                    "session": es_files[index][0],
                    "prior_instrument_id": ids[index - 1],
                    "new_instrument_id": ids[index],
                }
            )
    excluded_indexes = {
        neighbor
        for index in changes
        for neighbor in range(max(0, index - 1), min(len(es_files), index + 2))
    }
    return {es_files[index][0] for index in excluded_indexes}, details


def _es_trade_frames(
    paths: Paths, fold_map: dict[str, int]
) -> tuple[dict[tuple[str, int], pd.DataFrame], set[str], list[dict[str, Any]]]:
    files = _es_files(paths.es_raw)
    excluded, roll_details = _roll_exclusion(files)
    output: dict[tuple[str, int], pd.DataFrame] = {}
    for session, _, frame in files:
        if session not in fold_map or session in excluded or len(frame) < 2:
            continue
        close = frame["close"].to_numpy(float)
        for horizon in sorted(set(HORIZONS_ES_MOMENTUM) | set(HORIZONS_ES_REVERSION)):
            starts = np.arange(horizon, len(close) - horizon, horizon, dtype=int)
            if not len(starts):
                continue
            prior = close[starts] - close[starts - horizon]
            future = (close[starts + horizon] - close[starts]) * 50.0
            signal = np.sign(prior)
            output[(session, horizon)] = pd.DataFrame(
                {
                    "session": session,
                    "outer_fold": fold_map[session],
                    "trade_index": np.arange(len(starts)),
                    "momentum_signal": signal,
                    "future_dollars": future,
                }
            )
    return output, excluded, roll_details


def _session_shuffle_signals(
    trade_frames: dict[tuple[str, int], pd.DataFrame], horizon: int, label: str
) -> dict[str, np.ndarray]:
    by_length: dict[int, list[str]] = {}
    for (session, item_horizon), frame in trade_frames.items():
        if item_horizon == horizon:
            by_length.setdefault(len(frame), []).append(session)
    output: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(_stable_seed(f"session-shuffle|{label}|{horizon}"))
    for _, sessions in sorted(by_length.items()):
        targets = sorted(sessions)
        sources = targets.copy()
        rng.shuffle(sources)
        if len(sources) > 1 and any(target == source for target, source in zip(targets, sources, strict=True)):
            sources = sources[1:] + sources[:1]
        for target, source in zip(targets, sources, strict=True):
            output[target] = trade_frames[(source, horizon)]["momentum_signal"].to_numpy(float)
    return output


def _es_outcome(
    trade_frames: dict[tuple[str, int], pd.DataFrame],
    *,
    horizon: int,
    direction: int,
    friction: float,
    shuffled: bool = False,
    constant: bool = False,
    label: str,
) -> pd.DataFrame:
    shuffled_signals = _session_shuffle_signals(trade_frames, horizon, label) if shuffled else {}
    rows: list[pd.DataFrame] = []
    for (session, item_horizon), frame in sorted(trade_frames.items()):
        if item_horizon != horizon:
            continue
        if constant:
            signal = np.ones(len(frame), dtype=float)
        elif shuffled:
            signal = shuffled_signals[session] * direction
        else:
            signal = frame["momentum_signal"].to_numpy(float) * direction
        active = signal != 0.0
        pnl = signal * frame["future_dollars"].to_numpy(float) - np.where(active, friction, 0.0)
        item = frame[["session", "outer_fold", "trade_index"]].copy()
        item["pnl"] = pnl
        rows.append(item)
    return pd.concat(rows, ignore_index=True)


def _session_values(frame: pd.DataFrame) -> dict[str, float]:
    return frame.groupby("session", sort=True)["pnl"].mean().astype(float).to_dict()


def _fold_means(frame: pd.DataFrame) -> dict[str, float]:
    return {str(int(key)): float(value) for key, value in frame.groupby("outer_fold", sort=True)["pnl"].mean().items()}


def _concentration(frame: pd.DataFrame) -> dict[str, Any]:
    session = pd.Series(_session_values(frame), dtype=float)
    if len(session) < 2:
        return {"leave_one_session_out_all_positive": False, "leave_one_day_of_week_out_all_positive": False}
    loo_session = [float(session.drop(index).mean()) for index in session.index]
    weekdays = pd.Series({item: pd.Timestamp(item).day_name() for item in session.index})
    loo_weekday = [float(session[weekdays != day].mean()) for day in sorted(set(weekdays))]
    return {
        "leave_one_session_out_all_positive": bool(all(value > 0.0 for value in loo_session)),
        "leave_one_session_out_min_mean": float(min(loo_session)),
        "leave_one_day_of_week_out_all_positive": bool(all(value > 0.0 for value in loo_weekday)),
        "leave_one_day_of_week_out_min_mean": float(min(loo_weekday)),
    }


def _base_summary(frame: pd.DataFrame) -> dict[str, Any]:
    session = _session_values(frame)
    paired = paired_summary(session)
    folds = _fold_means(frame)
    return {
        "n_opportunities": int(len(frame)),
        "n_sessions": int(frame["session"].nunique()),
        "pooled_net_ev_dollars": float(frame["pnl"].mean()) if len(frame) else None,
        "session_mean_net_ev_dollars": paired["mean"],
        "raw_p_one_sided": paired["p_one_sided"],
        "fold_means_dollars": folds,
        "positive_folds": int(sum(value > 0.0 for value in folds.values())),
        "concentration": _concentration(frame),
    }


def _control_clears(summary: dict[str, Any]) -> bool:
    concentration = summary["concentration"]
    return bool(
        summary["pooled_net_ev_dollars"] is not None
        and summary["pooled_net_ev_dollars"] > 0.0
        and summary["positive_folds"] >= 4
        and summary["raw_p_one_sided"] is not None
        and summary["raw_p_one_sided"] <= 0.05
        and concentration["leave_one_session_out_all_positive"]
        and concentration["leave_one_day_of_week_out_all_positive"]
    )


def _option_tests(option: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    tests: dict[str, pd.DataFrame] = {}
    controls: dict[str, dict[str, pd.DataFrame]] = {}
    sanity_source = pd.read_parquet(
        "/Volumes/AR_TRADING_DATA/artifacts/entry_v2/oof_scores.parquet",
        columns=["session", "entry_arrival_ns", "outer_fold", "entry_feature_bid", "exit_bid"],
    )
    local = pd.to_datetime(sanity_source["entry_arrival_ns"], unit="ns", utc=True).dt.tz_convert(NY)
    sanity = sanity_source[(local.dt.hour == 10) & (local.dt.minute >= 30)].copy()
    sanity["gross"] = (sanity["exit_bid"] - sanity["entry_feature_bid"]) * 100.0
    sanity_result = {
        "candidates": int(len(sanity)),
        "sessions": int(sanity["session"].nunique()),
        "mean_gross_dollars": float(sanity["gross"].mean()),
        "fold_means_dollars": {str(int(k)): float(v) for k, v in sanity.groupby("outer_fold")["gross"].mean().items()},
        "positive_folds": int((sanity.groupby("outer_fold")["gross"].mean() > 0.0).sum()),
    }
    for branch, mask_column in (("H1", "h1_window"), ("H2", None)):
        for horizon in HORIZONS_OPTION:
            name = f"{branch}_{horizon}m"
            current = option[option["horizon_min"] == horizon]
            if mask_column is not None:
                current = current[current[mask_column]]
            main = current[["session", "outer_fold", "pnl"]].copy()
            reverse = current[["session", "outer_fold", "reverse_pnl"]].rename(columns={"reverse_pnl": "pnl"})
            all_time = option[option["horizon_min"] == horizon][["session", "outer_fold", "pnl"]].copy()
            tests[name] = main
            controls[name] = {
                "sign_reversed": reverse,
                # A fixed time mask/all-time mask is invariant to session shuffling.
                "session_shuffled": main.copy(),
                "constant": all_time,
            }
    return tests, controls, sanity_result


def _es_tests(
    trade_frames: dict[tuple[str, int], pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, Any]]]:
    tests: dict[str, pd.DataFrame] = {}
    controls: dict[str, dict[str, pd.DataFrame]] = {}
    bands: dict[str, dict[str, Any]] = {}
    for branch, horizons, direction in (
        ("H3", HORIZONS_ES_MOMENTUM, 1),
        ("H4", HORIZONS_ES_REVERSION, -1),
    ):
        for horizon in horizons:
            name = f"{branch}_{horizon}m"
            band: dict[str, Any] = {}
            for ticks, friction in ES_FRICTION.items():
                frame = _es_outcome(trade_frames, horizon=horizon, direction=direction, friction=friction, label=name)
                band[str(ticks)] = _base_summary(frame)
                if ticks == 2:
                    tests[name] = frame
            bands[name] = band
            controls[name] = {
                "sign_reversed": _es_outcome(trade_frames, horizon=horizon, direction=-direction, friction=ES_FRICTION[2], label=f"{name}|reverse"),
                "session_shuffled": _es_outcome(trade_frames, horizon=horizon, direction=direction, friction=ES_FRICTION[2], shuffled=True, label=f"{name}|shuffle"),
                "constant": _es_outcome(trade_frames, horizon=horizon, direction=direction, friction=ES_FRICTION[2], constant=True, label=f"{name}|constant"),
            }
    return tests, controls, bands


def run(paths: Paths, output_dir: Path) -> dict[str, Any]:
    scores = pd.read_parquet(paths.scores)
    session_fold = scores[["session", "outer_fold"]].drop_duplicates()
    if session_fold["session"].duplicated().any() or session_fold["outer_fold"].nunique() != 5:
        raise RuntimeError("entry OOF session/fold identity is not unique five-fold data")
    fold_map = {str(row.session): int(row.outer_fold) for row in session_fold.itertuples(index=False)}

    option = _option_candidate_outcomes(paths, scores)
    option_tests, option_controls, sanity = _option_tests(option)
    es_trade_frames, roll_excluded, roll_details = _es_trade_frames(paths, fold_map)
    es_tests, es_controls, es_bands = _es_tests(es_trade_frames)
    tests = {**option_tests, **es_tests}
    controls = {**option_controls, **es_controls}
    if len(tests) != 9:
        raise RuntimeError(f"frozen family must contain nine tests, got {sorted(tests)}")

    family = session_blocked_max_t(
        {name: _session_values(frame) for name, frame in tests.items()},
        permutations=MAXT_PERMUTATIONS,
        seed=MAXT_SEED,
    )
    results: dict[str, Any] = {}
    accepted_control = False
    for name, frame in sorted(tests.items()):
        summary = _base_summary(frame)
        summary["maxT_p_one_sided"] = family[name]["maxT_p_one_sided"]
        control_summaries: dict[str, Any] = {}
        for control_name, control_frame in controls[name].items():
            control_summary = _base_summary(control_frame)
            control_summary["clears_unadjusted_gate"] = _control_clears(control_summary)
            accepted_control = accepted_control or control_summary["clears_unadjusted_gate"]
            control_summaries[control_name] = control_summary
        concentration = summary["concentration"]
        criteria = {
            "positive_primary_net_ev": bool(summary["pooled_net_ev_dollars"] > 0.0),
            "positive_at_least_4_of_5_folds": bool(summary["positive_folds"] >= 4),
            "family_maxT_p_le_0_05": bool(summary["maxT_p_one_sided"] <= 0.05),
            "all_negative_controls_fail": bool(not any(item["clears_unadjusted_gate"] for item in control_summaries.values())),
            "not_single_session_concentrated": bool(concentration["leave_one_session_out_all_positive"]),
            "not_day_of_week_concentrated": bool(concentration["leave_one_day_of_week_out_all_positive"]),
        }
        summary["criteria"] = criteria
        summary["verdict"] = "FEASIBLE" if all(criteria.values()) else "NOT_FEASIBLE"
        summary["negative_controls"] = control_summaries
        if name in es_bands:
            summary["friction_band_1_to_4_ticks"] = es_bands[name]
        results[name] = summary

    overall = "INVALID" if accepted_control else (
        "FEASIBLE_BRANCH_PRESENT" if any(item["verdict"] == "FEASIBLE" for item in results.values()) else "NO_FEASIBLE_BRANCH"
    )
    payload = {
        "schema_version": "pathd.matched-feasibility-gate.v1",
        "status": overall,
        "frozen_family_size": len(tests),
        "maxT": {"method": "session_blocked_sign_flip_maxT", "permutations": MAXT_PERMUTATIONS, "seed": MAXT_SEED},
        "interpretations_fixed_before_result_inspection": {
            "option_primary_fill": "arrival bid; fill at first ask <= bid-one_tick inside 60s; no-fill=0 per opportunity; hold from fill; force flat 15:55 ET; $3 fees",
            "option_controls": "sign reversal flips post-fill price move; shuffle preserves invariant fixed time masks; constant is all-time passive long",
            "es_sampling": "strict nonoverlap: first decision after one prior horizon, hold one horizon, then next decision",
            "es_primary_friction": "two ticks = $29.50 round trip",
            "folds": "entry-v2 OOF session fold mapped to both branches",
            "concentration": "mean remains positive after removing each session and after removing each weekday",
        },
        "h1_sanity_check": sanity,
        "option_source": {
            "rows": int(len(option)),
            "invalid_source_rows": int(option["invalid_source"].sum()),
            "fill_rate_by_horizon": {str(int(k)): float(v) for k, v in option.groupby("horizon_min")["filled"].mean().items()},
        },
        "es_source": {
            "roll_excluded_sessions": sorted(roll_excluded),
            "roll_change_details": roll_details,
            "eligible_session_count": len({session for session, _ in es_trade_frames}),
        },
        "tests": results,
        "protected_holdout_opened": False,
        "model_fit_executed": False,
        "paid_data_used": False,
    }
    semantic = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["result_sha256"] = hashlib.sha256(semantic).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matched_feasibility_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    option.to_parquet(output_dir / "option_primary_counterfactuals.parquet", index=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/Volumes/AR_TRADING_DATA"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/AR_TRADING_DATA/reports/pathd_matched_feasibility_2026_08_03"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(
        root=args.root,
        scores=args.root / "artifacts/entry_v2/oof_scores.parquet",
        option_raw=args.root / "vendor/pathd_2025-08-01_2026-07-31/raw/databento/opra_spxw_cbbo_1s",
        es_raw=args.root / "vendor/pathd_2025-08-01_2026-07-31/raw/databento/glbx_es_ohlcv_1m",
    )
    payload = run(paths, args.output_dir)
    print(json.dumps({"status": payload["status"], "result_sha256": payload["result_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
