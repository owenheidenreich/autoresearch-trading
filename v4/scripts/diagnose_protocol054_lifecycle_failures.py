"""Diagnose Protocol 054 clipped winners and Q2 model-exit-loss behavior.

This is a no-paid-data, lifecycle-only diagnostic. It does not train or select
anything. It reconstructs same-contract paths from normalized quote files after
the selected lifecycle exit to answer: did the trade recover, keep decaying, or
only look bad because of the 1-minute lifecycle rule?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.model.environment_diagnostics import time_bucket


_NY = ZoneInfo("America/New_York")
_CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-trades",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/"
            "v4_aplus_hypothesis_054_protocol052_lifecycle_10seed_validation/"
            "selected_trades_with_lifecycle_exits.json"
        ),
    )
    parser.add_argument("--normalized-dir", type=Path, default=Path("v4/normalized_official_context"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/"
            "v4_aplus_hypothesis_056_protocol054_lifecycle_failure_diagnosis"
        ),
    )
    parser.add_argument("--max-hold-minutes", type=int, default=25)
    parser.add_argument("--forced-flat-time", default="15:55")
    return parser.parse_args()


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _deadline(decision_time: pd.Timestamp, *, max_hold_minutes: int, forced_flat_time: str) -> pd.Timestamp:
    max_hold = decision_time + pd.Timedelta(minutes=max_hold_minutes)
    hour, minute = [int(part) for part in forced_flat_time.split(":", 1)]
    local_day = decision_time.tz_convert(_NY).date()
    forced = pd.Timestamp(local_day).replace(hour=hour, minute=minute, tzinfo=_NY).tz_convert("UTC")
    return min(max_hold, forced)


def _normalized_session_path(normalized_dir: Path, session: str) -> Path | None:
    preferred = sorted(normalized_dir.glob(f"*{session}*official_context.parquet"))
    if preferred:
        return preferred[0]
    fallback = sorted(normalized_dir.glob(f"*{session}*.parquet"))
    return fallback[0] if fallback else None


def _load_selected(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"no selected trades found in {path}")
    frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["local_time"] = frame["decision_ts"].dt.tz_convert(_NY).dt.strftime("%H:%M")
    frame["time_bucket"] = frame["decision_ts"].map(time_bucket)
    numeric_columns = [
        "baseline_pnl",
        "dynamic_pnl",
        "hold_minutes",
        "mfe",
        "mae",
        "giveback",
        "giveback_fraction",
        "current_pnl",
        "predicted_headroom",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["lifecycle_delta"] = frame["dynamic_pnl"] - frame["baseline_pnl"]
    return frame


def _load_normalized_session(path: Path, contracts: set[str]) -> pd.DataFrame:
    columns = [
        "quote_time",
        "contract_id",
        "bid",
        "ask",
        "mid",
        "delta",
        "gamma",
        "theta",
        "iv",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["contract_id"] = frame["contract_id"].astype(str)
    frame = frame[frame["contract_id"].isin(contracts)].copy()
    frame["quote_time"] = pd.to_datetime(frame["quote_time"], utc=True)
    for column in ["bid", "ask", "mid", "delta", "gamma", "theta", "iv"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["contract_id", "quote_time"]).reset_index(drop=True)


def _finite_float(value: object, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _row_at_or_before(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    rows = frame[frame["quote_time"] <= timestamp]
    if rows.empty:
        return None
    return rows.iloc[-1]


def _path_stats(
    trade: pd.Series,
    contract_rows: pd.DataFrame,
    *,
    max_hold_minutes: int,
    forced_flat_time: str,
) -> dict:
    decision_time = _utc_timestamp(trade["decision_time"])
    hold_minutes = _finite_float(trade.get("hold_minutes"))
    if hold_minutes is None:
        return {"path_status": "missing_hold_minutes"}
    exit_time = decision_time + pd.Timedelta(minutes=hold_minutes)
    deadline = _deadline(decision_time, max_hold_minutes=max_hold_minutes, forced_flat_time=forced_flat_time)
    entry = _row_at_or_before(contract_rows, decision_time)
    if entry is None:
        return {"path_status": "missing_entry_quote"}
    entry_ask = _finite_float(entry.get("ask"))
    if entry_ask is None or entry_ask <= 0:
        return {"path_status": "invalid_entry_ask"}

    path = contract_rows[
        (contract_rows["quote_time"] > decision_time)
        & (contract_rows["quote_time"] <= deadline)
        & contract_rows["bid"].notna()
    ].copy()
    if path.empty:
        return {"path_status": "missing_future_path", "entry_ask": entry_ask}
    path["path_pnl"] = (path["bid"] - entry_ask) * _CONTRACT_MULTIPLIER
    through_exit = path[path["quote_time"] <= exit_time]
    if through_exit.empty:
        exit_row = path.iloc[0]
    else:
        exit_row = through_exit.iloc[-1]
    post_exit = path[path["quote_time"] > exit_row["quote_time"]]
    dynamic_pnl = _finite_float(trade.get("dynamic_pnl"), 0.0) or 0.0
    baseline_pnl = _finite_float(trade.get("baseline_pnl"), 0.0) or 0.0
    exit_path_pnl = _finite_float(exit_row.get("path_pnl"), dynamic_pnl) or dynamic_pnl

    if post_exit.empty:
        post_max = exit_path_pnl
        post_min = exit_path_pnl
        post_final = exit_path_pnl
        post_max_time = exit_row["quote_time"]
        post_min_time = exit_row["quote_time"]
    else:
        max_idx = post_exit["path_pnl"].idxmax()
        min_idx = post_exit["path_pnl"].idxmin()
        post_max = float(post_exit.loc[max_idx, "path_pnl"])
        post_min = float(post_exit.loc[min_idx, "path_pnl"])
        post_final = float(post_exit.iloc[-1]["path_pnl"])
        post_max_time = post_exit.loc[max_idx, "quote_time"]
        post_min_time = post_exit.loc[min_idx, "quote_time"]

    exit_delta = _finite_float(exit_row.get("delta"))
    exit_gamma = _finite_float(exit_row.get("gamma"))
    exit_theta = _finite_float(exit_row.get("theta"))
    entry_delta = _finite_float(entry.get("delta"))
    entry_gamma = _finite_float(entry.get("gamma"))
    entry_theta = _finite_float(entry.get("theta"))
    return {
        "path_status": "ok",
        "entry_ask": entry_ask,
        "path_exit_time": pd.Timestamp(exit_row["quote_time"]).isoformat(),
        "exit_path_pnl": exit_path_pnl,
        "exit_path_vs_recorded": exit_path_pnl - dynamic_pnl,
        "post_exit_max_pnl": post_max,
        "post_exit_min_pnl": post_min,
        "post_exit_final_pnl": post_final,
        "post_exit_max_time": pd.Timestamp(post_max_time).isoformat(),
        "post_exit_min_time": pd.Timestamp(post_min_time).isoformat(),
        "post_exit_max_delta": post_max - dynamic_pnl,
        "post_exit_min_delta": post_min - dynamic_pnl,
        "post_exit_final_delta": post_final - dynamic_pnl,
        "recovered_to_baseline_after_exit": bool(baseline_pnl > dynamic_pnl and post_max >= baseline_pnl),
        "recovered_positive_after_exit": bool(dynamic_pnl < 0.0 and post_max > 0.0),
        "kept_falling_after_exit": bool(post_min < dynamic_pnl),
        "entry_delta": entry_delta,
        "exit_delta": exit_delta,
        "delta_change_to_exit": None if entry_delta is None or exit_delta is None else exit_delta - entry_delta,
        "entry_gamma": entry_gamma,
        "exit_gamma": exit_gamma,
        "gamma_change_to_exit": None if entry_gamma is None or exit_gamma is None else exit_gamma - entry_gamma,
        "entry_theta": entry_theta,
        "exit_theta": exit_theta,
        "theta_change_to_exit": None if entry_theta is None or exit_theta is None else exit_theta - entry_theta,
        "path_points": int(len(path)),
        "post_exit_points": int(len(post_exit)),
    }


def _enrich_with_paths(frame: pd.DataFrame, *, normalized_dir: Path, max_hold_minutes: int, forced_flat_time: str) -> pd.DataFrame:
    enriched_rows: list[dict] = []
    for session, session_group in frame.groupby("session", dropna=False):
        session = str(session)
        path = _normalized_session_path(normalized_dir, session)
        if path is None:
            for _, trade in session_group.iterrows():
                enriched_rows.append({**trade.to_dict(), "path_status": "missing_normalized_session"})
            continue
        contracts = set(session_group["contract_id"].astype(str))
        normalized = _load_normalized_session(path, contracts)
        by_contract = {contract_id: group for contract_id, group in normalized.groupby("contract_id")}
        for _, trade in session_group.iterrows():
            contract_rows = by_contract.get(str(trade["contract_id"]))
            if contract_rows is None or contract_rows.empty:
                stats = {"path_status": "missing_contract_path"}
            else:
                stats = _path_stats(
                    trade,
                    contract_rows,
                    max_hold_minutes=max_hold_minutes,
                    forced_flat_time=forced_flat_time,
                )
            enriched_rows.append({**trade.to_dict(), **stats})
    out = pd.DataFrame(enriched_rows)
    out["baseline_pnl"] = pd.to_numeric(out["baseline_pnl"], errors="coerce").fillna(0.0)
    out["dynamic_pnl"] = pd.to_numeric(out["dynamic_pnl"], errors="coerce").fillna(0.0)
    out["lifecycle_delta"] = out["dynamic_pnl"] - out["baseline_pnl"]
    out["post_exit_max_delta"] = pd.to_numeric(out.get("post_exit_max_delta"), errors="coerce")
    out["post_exit_min_delta"] = pd.to_numeric(out.get("post_exit_min_delta"), errors="coerce")
    out["post_exit_final_delta"] = pd.to_numeric(out.get("post_exit_final_delta"), errors="coerce")
    out["hold_minutes"] = pd.to_numeric(out["hold_minutes"], errors="coerce")
    out["mfe"] = pd.to_numeric(out.get("mfe"), errors="coerce")
    out["giveback_fraction"] = pd.to_numeric(out.get("giveback_fraction"), errors="coerce")
    out["clipped_winner"] = (out["baseline_pnl"] > 0.0) & (out["dynamic_pnl"] < out["baseline_pnl"])
    out["q2_model_exit_loss"] = (out["split"] == "q2_2025") & (out["exit_reason"] == "model_exit_loss")
    out["recoverability_bucket"] = np.select(
        [
            out["recovered_to_baseline_after_exit"].fillna(False),
            (out["post_exit_max_delta"].fillna(0.0) >= 200.0),
            (out["post_exit_max_delta"].fillna(0.0) > 0.0),
        ],
        ["recovered_to_baseline", "large_partial_recovery", "small_partial_recovery"],
        default="no_recovery",
    )
    out["baseline_sign_bucket"] = np.where(out["baseline_pnl"] > 0.0, "baseline_winner", "baseline_loser")
    out["dynamic_sign_bucket"] = np.where(out["dynamic_pnl"] > 0.0, "dynamic_winner", "dynamic_loser")
    out["hold_bucket"] = np.select(
        [out["hold_minutes"] <= 5, out["hold_minutes"] <= 15, out["hold_minutes"] < 25],
        ["hold_00_05", "hold_06_15", "hold_16_24"],
        default="hold_25_flat",
    )
    out["mfe_bucket"] = np.select(
        [out["mfe"] < 100, out["mfe"] < 300, out["mfe"] < 700],
        ["mfe_lt_100", "mfe_100_300", "mfe_300_700"],
        default="mfe_700p",
    )
    out["giveback_bucket"] = np.select(
        [
            out["giveback_fraction"].fillna(0.0) < 0.25,
            out["giveback_fraction"].fillna(0.0) < 0.50,
            out["giveback_fraction"].fillna(0.0) < 0.75,
        ],
        ["gbfrac_lt_25", "gbfrac_25_50", "gbfrac_50_75"],
        default="gbfrac_75p",
    )
    return out


def _metrics(group: pd.DataFrame) -> dict:
    if group.empty:
        return {
            "trades": 0,
            "baseline_pnl": 0.0,
            "dynamic_pnl": 0.0,
            "lifecycle_delta": 0.0,
            "post_exit_max_delta_sum": 0.0,
            "recovered_to_baseline_fraction": 0.0,
        }
    return {
        "trades": int(len(group)),
        "baseline_pnl": float(group["baseline_pnl"].sum()),
        "dynamic_pnl": float(group["dynamic_pnl"].sum()),
        "lifecycle_delta": float(group["lifecycle_delta"].sum()),
        "avg_lifecycle_delta": float(group["lifecycle_delta"].mean()),
        "median_lifecycle_delta": float(group["lifecycle_delta"].median()),
        "post_exit_max_delta_sum": float(group["post_exit_max_delta"].fillna(0.0).sum()),
        "post_exit_max_delta_median": float(group["post_exit_max_delta"].fillna(0.0).median()),
        "post_exit_final_delta_sum": float(group["post_exit_final_delta"].fillna(0.0).sum()),
        "recovered_to_baseline_fraction": float(group["recovered_to_baseline_after_exit"].fillna(False).mean()),
        "recovered_positive_fraction": float(group["recovered_positive_after_exit"].fillna(False).mean()),
        "kept_falling_fraction": float(group["kept_falling_after_exit"].fillna(False).mean()),
    }


def _group_metrics(frame: pd.DataFrame, columns: list[str], *, sort_by: str = "lifecycle_delta") -> list[dict]:
    rows = []
    for keys, group in frame.groupby(columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({column: key for column, key in zip(columns, keys)} | _metrics(group))
    return sorted(rows, key=lambda row: (str(row.get("split", "")), row.get(sort_by, 0.0)))


def _examples(frame: pd.DataFrame, *, ascending: bool, limit: int = 30) -> list[dict]:
    columns = [
        "split",
        "seed",
        "session",
        "local_time",
        "right",
        "offset",
        "exit_reason",
        "hold_minutes",
        "baseline_pnl",
        "dynamic_pnl",
        "lifecycle_delta",
        "post_exit_max_pnl",
        "post_exit_max_delta",
        "post_exit_final_pnl",
        "recoverability_bucket",
        "mfe",
        "mae",
        "giveback",
        "giveback_fraction",
        "contract_id",
    ]
    available = [column for column in columns if column in frame.columns]
    subset = frame.sort_values("lifecycle_delta", ascending=ascending).head(limit)
    return subset[available].to_dict(orient="records")


def _interpretation(payload: dict) -> str:
    clipped = payload["clipped_winner_summary"]
    q2_loss = payload["q2_model_exit_loss_summary"]
    recoverable_clip = next(
        (row for row in payload["clipped_by_recoverability"] if row.get("recoverability_bucket") == "recovered_to_baseline"),
        None,
    )
    q2_recoverable = next(
        (
            row
            for row in payload["q2_model_exit_loss_by_recoverability"]
            if row.get("recoverability_bucket") == "recovered_to_baseline"
        ),
        None,
    )
    text = (
        f"Clipped winners are the main lifecycle cost: {clipped['trades']} trades, "
        f"{clipped['lifecycle_delta']:.0f} lifecycle delta, with "
        f"{clipped['recovered_to_baseline_fraction']:.2f} recovering to the frozen baseline after exit. "
    )
    if recoverable_clip:
        text += (
            f"Recoverable clipped winners alone account for {recoverable_clip['lifecycle_delta']:.0f} "
            f"of clipped-winner delta across {recoverable_clip['trades']} trades. "
        )
    text += (
        f"Q2 model_exit_loss is smaller but important: {q2_loss['trades']} trades, "
        f"{q2_loss['lifecycle_delta']:.0f} lifecycle delta, and "
        f"{q2_loss['recovered_to_baseline_fraction']:.2f} later recovered to the frozen baseline. "
    )
    if q2_recoverable:
        text += (
            f"Q2 recovered-to-baseline model_exit_loss rows contributed {q2_recoverable['lifecycle_delta']:.0f} "
            f"across {q2_recoverable['trades']} trades. "
        )
    text += (
        "The next lifecycle hypothesis should focus on exit confirmation or recovery risk after model exits, "
        "not entry selection."
    )
    return text


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Protocol 054 Lifecycle Failure Diagnosis",
        "",
        "No paid data was downloaded. No model or entry-side change was made.",
        "",
        "## Headline",
        "",
        payload["interpretation"],
        "",
        "## Path Coverage",
        "",
        "| Status | Trades |",
        "|---|---:|",
    ]
    for status, count in payload["path_status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines += [
        "",
        "## Clipped Winners By Split",
        "",
        "| Split | Trades | Baseline | Dynamic | Delta | Post-Exit Max Delta | Recovered To Baseline | Kept Falling |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["clipped_by_split"]:
        lines.append(
            f"| {row['split']} | {row['trades']} | {row['baseline_pnl']:.0f} | {row['dynamic_pnl']:.0f} | "
            f"{row['lifecycle_delta']:.0f} | {row['post_exit_max_delta_sum']:.0f} | "
            f"{row['recovered_to_baseline_fraction']:.2f} | {row['kept_falling_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Clipped Winners By Exit Reason",
        "",
        "| Split | Exit Reason | Trades | Delta | Post-Exit Max Delta | Recovered To Baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["clipped_by_exit_reason"]:
        lines.append(
            f"| {row['split']} | {row['exit_reason']} | {row['trades']} | {row['lifecycle_delta']:.0f} | "
            f"{row['post_exit_max_delta_sum']:.0f} | {row['recovered_to_baseline_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Clipped Winners By Recoverability",
        "",
        "| Recoverability | Trades | Baseline | Dynamic | Delta | Post-Exit Max Delta | Kept Falling |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["clipped_by_recoverability"]:
        lines.append(
            f"| {row['recoverability_bucket']} | {row['trades']} | {row['baseline_pnl']:.0f} | "
            f"{row['dynamic_pnl']:.0f} | {row['lifecycle_delta']:.0f} | "
            f"{row['post_exit_max_delta_sum']:.0f} | {row['kept_falling_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Q2 Model Exit Loss",
        "",
        "| Bucket | Trades | Baseline | Dynamic | Delta | Post-Exit Max Delta | Recovered To Baseline | Kept Falling |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["q2_model_exit_loss_by_recoverability"]:
        lines.append(
            f"| {row['recoverability_bucket']} | {row['trades']} | {row['baseline_pnl']:.0f} | "
            f"{row['dynamic_pnl']:.0f} | {row['lifecycle_delta']:.0f} | {row['post_exit_max_delta_sum']:.0f} | "
            f"{row['recovered_to_baseline_fraction']:.2f} | {row['kept_falling_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Q2 Model Exit Loss By Side/Time",
        "",
        "| Right | Time Bucket | Trades | Delta | Post-Exit Max Delta | Recovered To Baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["q2_model_exit_loss_by_side_time"]:
        lines.append(
            f"| {row['right']} | {row['time_bucket']} | {row['trades']} | {row['lifecycle_delta']:.0f} | "
            f"{row['post_exit_max_delta_sum']:.0f} | {row['recovered_to_baseline_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Q2 Model Exit Loss By Outcome Sign",
        "",
        "| Baseline Sign | Dynamic Sign | Trades | Baseline | Dynamic | Delta | Post-Exit Max Delta | Recovered To Baseline |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["q2_model_exit_loss_by_sign"]:
        lines.append(
            f"| {row['baseline_sign_bucket']} | {row['dynamic_sign_bucket']} | {row['trades']} | "
            f"{row['baseline_pnl']:.0f} | {row['dynamic_pnl']:.0f} | {row['lifecycle_delta']:.0f} | "
            f"{row['post_exit_max_delta_sum']:.0f} | {row['recovered_to_baseline_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Worst Clipped Winner Examples",
        "",
        "| Split | Session | Time | Side | Offset | Exit | Base | Dynamic | Delta | Post-Exit Max | Recovery |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["worst_clipped_examples"][:15]:
        lines.append(
            f"| {row['split']} | {row['session']} | {row['local_time']} | {row['right']} | {float(row['offset']):.0f} | "
            f"{row['exit_reason']} | {float(row['baseline_pnl']):.0f} | {float(row['dynamic_pnl']):.0f} | "
            f"{float(row['lifecycle_delta']):.0f} | {float(row.get('post_exit_max_pnl') or 0.0):.0f} | "
            f"{row.get('recoverability_bucket')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    selected = _load_selected(args.selected_trades)
    enriched = _enrich_with_paths(
        selected,
        normalized_dir=args.normalized_dir,
        max_hold_minutes=args.max_hold_minutes,
        forced_flat_time=args.forced_flat_time,
    )
    clipped = enriched[enriched["clipped_winner"]].copy()
    q2_loss = enriched[enriched["q2_model_exit_loss"]].copy()
    payload = {
        "selected_trades": str(args.selected_trades),
        "normalized_dir": str(args.normalized_dir),
        "path_status_counts": {str(k): int(v) for k, v in enriched["path_status"].value_counts(dropna=False).items()},
        "overall_summary": _metrics(enriched),
        "clipped_winner_summary": _metrics(clipped),
        "clipped_by_split": _group_metrics(clipped, ["split"]),
        "clipped_by_exit_reason": _group_metrics(clipped, ["split", "exit_reason"]),
        "clipped_by_side_time": _group_metrics(clipped, ["split", "right", "time_bucket"]),
        "clipped_by_recoverability": _group_metrics(clipped, ["recoverability_bucket"]),
        "clipped_by_hold_bucket": _group_metrics(clipped, ["split", "hold_bucket"]),
        "clipped_by_mfe_bucket": _group_metrics(clipped, ["split", "mfe_bucket"]),
        "q2_model_exit_loss_summary": _metrics(q2_loss),
        "q2_model_exit_loss_by_recoverability": _group_metrics(q2_loss, ["recoverability_bucket"]),
        "q2_model_exit_loss_by_side_time": _group_metrics(q2_loss, ["right", "time_bucket"]),
        "q2_model_exit_loss_by_sign": _group_metrics(q2_loss, ["baseline_sign_bucket", "dynamic_sign_bucket"]),
        "q2_model_exit_loss_by_hold_bucket": _group_metrics(q2_loss, ["hold_bucket"]),
        "q2_model_exit_loss_by_mfe_bucket": _group_metrics(q2_loss, ["mfe_bucket"]),
        "worst_clipped_examples": _examples(clipped, ascending=True),
        "worst_q2_model_exit_loss_examples": _examples(q2_loss, ascending=True),
        "best_saved_loss_examples": _examples(enriched[(enriched["baseline_pnl"] < 0) & (enriched["lifecycle_delta"] > 0)], ascending=False),
    }
    payload["interpretation"] = _interpretation(payload)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    md_path = args.out_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(md_path, payload)
    print(json.dumps({k: payload[k] for k in ["path_status_counts", "clipped_winner_summary", "q2_model_exit_loss_summary"]}, indent=2, allow_nan=True))
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
