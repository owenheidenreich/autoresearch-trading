"""Protocol 162: serial lifecycle replay for May 2026 Protocol101 entries.

Protocol 161 intentionally answered only whether frozen Protocol101 would emit
historical entry signals on May 19-20, 2026. That was useful for diagnosing the
live startup gap, but it was not a tradable PnL replay because independent
entry signals can overlap while the real bot can hold only one contract.

This runner converts Protocol 161 entry signals into a one-position serial
historical replay:

* one contract per trade
* max concurrency 1
* enter at ask
* exit at bid
* frozen Protocol081 lifecycle artifact
* no paid data downloads, no broker endpoints, no model retraining
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.live.protocol066_inference import (
    Protocol066Artifact,
    load_protocol066_artifact,
    prediction_for_step,
    predict_protocol066_sequence,
)
from v4.scripts.build_lifecycle_sequence_dataset import (
    CAUSAL_STEP_FEATURE_COLUMNS,
    _build_for_trade,
    _load_session,
    _normalized_session_path,
)
from v4.scripts.run_protocol081_live_shadow_router import DEFAULT_PROTOCOL081_MANIFEST


LOOP_ID = "v4_aplus_hypothesis_162_may2026_serial_lifecycle_replay"
DEFAULT_PROTOCOL161_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_161_may2026_historical_replay")
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
FORCED_FLAT_TIME = "15:55"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol161-dir", type=Path, default=DEFAULT_PROTOCOL161_DIR)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--protocol081-manifest", type=Path, default=DEFAULT_PROTOCOL081_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default=FORCED_FLAT_TIME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entries = _load_protocol161_entries(args.protocol161_dir / "replay_decisions.csv")
    artifact = load_protocol066_artifact(args.protocol081_manifest)
    candidate_paths = _build_candidate_paths(
        entries,
        normalized_dir=args.normalized_dir,
        artifact=artifact,
        forced_flat_time=str(args.forced_flat_time),
    )
    serial_trades, skipped = _serial_replay(candidate_paths)
    independent = _independent_summary(candidate_paths)
    serial_summary = _trade_summary(serial_trades)
    payload = {
        "protocol": "162_may2026_serial_lifecycle_replay",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "source_protocol161_dir": str(args.protocol161_dir),
        "normalized_dir": str(args.normalized_dir),
        "protocol081_manifest": str(args.protocol081_manifest),
        "protocol081_fold": artifact.fold,
        "protocol081_seed": artifact.seed,
        "protocol081_override_threshold": artifact.selected_override_threshold
        if math.isfinite(artifact.selected_override_threshold)
        else "inf",
        "candidate_signals": int(len(candidate_paths)),
        "candidate_path_status": _count_values(row["path_status"] for row in candidate_paths),
        "independent_protocol081_summary": independent,
        "serial_protocol081_summary": serial_summary,
        "skipped_due_to_open_position": _skipped_summary(skipped),
        "answer": _answer(candidate_paths, serial_trades, skipped),
        "decision": _decision(serial_summary, skipped),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(candidate_paths).to_csv(args.out_dir / "candidate_lifecycle_paths.csv", index=False)
    pd.DataFrame(serial_trades).to_csv(args.out_dir / "serial_lifecycle_trades.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.out_dir / "skipped_entry_signals.csv", index=False)
    _write_report(args.out_dir / "report.md", payload, candidate_paths, serial_trades, skipped)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "answer": payload["answer"],
                "serial_summary": payload["serial_protocol081_summary"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


def _load_protocol161_entries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing Protocol 161 replay decisions: {path}")
    frame = pd.read_csv(path)
    entries = frame[frame["action"].eq("enter")].copy()
    if entries.empty:
        return pd.DataFrame(
            columns=[
                "session",
                "decision_time",
                "contract_id",
                "right",
                "offset",
                "edge",
                "ask",
                "source_row",
                "decision_ts",
                "canonical_entry_uid",
                "trade_uid",
            ]
        )
    entries["source_row"] = np.arange(len(entries), dtype=np.int64)
    entries["decision_ts"] = pd.to_datetime(entries["decision_time"], utc=True)
    entries["contract_id"] = entries["selected_contract_id"].astype(str)
    entries["right"] = entries["selected_right"].astype(str)
    entries["offset"] = pd.to_numeric(entries["selected_offset_points"], errors="coerce")
    entries["edge"] = pd.to_numeric(entries["selected_edge"], errors="coerce")
    entries["ask"] = pd.to_numeric(entries["selected_ask"], errors="coerce")
    entries["fold"] = "protocol162_may2026"
    entries["split"] = "may2026"
    entries["seed"] = 1
    entries["baseline_pnl"] = 0.0
    entries["dynamic_pnl"] = 0.0
    entries["hold_minutes"] = np.nan
    entries["exit_reason"] = "protocol081_fallback_unknown"
    entries["predicted_headroom"] = np.nan
    entries["canonical_entry_uid"] = [
        _uid(row.session, row.decision_time, row.contract_id) for row in entries.itertuples(index=False)
    ]
    entries["trade_uid"] = [
        _uid("protocol162", row.session, row.decision_time, row.contract_id, row.source_row)
        for row in entries.itertuples(index=False)
    ]
    return entries.sort_values(["session", "decision_ts", "contract_id"]).reset_index(drop=True)


def _build_candidate_paths(
    entries: pd.DataFrame,
    *,
    normalized_dir: Path,
    artifact: Protocol066Artifact,
    forced_flat_time: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if entries.empty:
        return rows
    for session, group in entries.groupby("session", sort=True):
        session_path = _normalized_session_path(normalized_dir, str(session))
        if session_path is None:
            for _, trade in group.iterrows():
                rows.append(_missing_candidate(trade, path_status="missing_normalized_session"))
            continue
        normalized = _load_session(session_path, set(group["contract_id"].astype(str)))
        by_contract = {contract_id: part for contract_id, part in normalized.groupby("contract_id", sort=False)}
        for _, trade in group.iterrows():
            contract_rows = by_contract.get(str(trade["contract_id"]))
            if contract_rows is None or contract_rows.empty:
                rows.append(_missing_candidate(trade, path_status="missing_contract_path"))
                continue
            trade_row, step_rows = _build_for_trade(trade, contract_rows, forced_flat_time=forced_flat_time)
            if str(trade_row.get("path_status")) != "ok" or not step_rows:
                rows.append(
                    {
                        **_base_candidate(trade),
                        **{key: _jsonable(value) for key, value in trade_row.items() if key not in _base_candidate(trade)},
                        "path_status": str(trade_row.get("path_status")),
                    }
                )
                continue
            steps = pd.DataFrame(step_rows)
            prepared = _prepare_steps(steps, artifact)
            exit_info = _protocol081_exit(prepared, artifact)
            rows.append(
                {
                    **_base_candidate(trade),
                    "path_status": "ok",
                    "entry_quote_time": str(trade_row.get("entry_quote_time")),
                    "entry_ask": _finite_float(trade_row.get("entry_ask")),
                    "entry_bid": _finite_float(trade_row.get("entry_bid")),
                    "entry_mid": _finite_float(trade_row.get("entry_mid")),
                    "entry_spread": _finite_float(trade_row.get("entry_spread")),
                    "entry_underlying_price": _finite_float(trade_row.get("entry_underlying_price")),
                    "entry_iv": _finite_float(trade_row.get("entry_iv")),
                    "entry_delta": _finite_float(trade_row.get("entry_delta")),
                    "entry_gamma": _finite_float(trade_row.get("entry_gamma")),
                    "entry_theta": _finite_float(trade_row.get("entry_theta")),
                    "path_points": int(trade_row.get("path_points") or len(prepared)),
                    "path_max_pnl": _finite_float(trade_row.get("path_max_pnl")),
                    "path_min_pnl": _finite_float(trade_row.get("path_min_pnl")),
                    "path_final_pnl": _finite_float(trade_row.get("path_final_pnl")),
                    "baseline_exit_time": str(trade_row.get("baseline_exit_time")),
                    "baseline_exit_reason": str(trade_row.get("baseline_exit_reason")),
                    "baseline_path_pnl": _finite_float(trade_row.get("baseline_path_pnl")),
                    "protocol054_exit_time": str(trade_row.get("protocol054_exit_time")),
                    "protocol054_path_pnl": _finite_float(trade_row.get("protocol054_path_pnl")),
                    **exit_info,
                }
            )
    return sorted(rows, key=lambda row: (str(row.get("session")), pd.Timestamp(row.get("decision_time"))))


def _prepare_steps(steps: pd.DataFrame, artifact: Protocol066Artifact) -> pd.DataFrame:
    out = steps.sort_values("step_idx").reset_index(drop=True).copy()
    out["quote_ts"] = pd.to_datetime(out["quote_time"], utc=True)
    quote_gap = out["quote_ts"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
    for column in artifact.feature_columns:
        if column not in out.columns:
            out[column] = 0.0
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        bad = ~np.isfinite(values.to_numpy(dtype=float))
        if column == "quote_gap_seconds":
            values = values.mask(bad, quote_gap)
        else:
            values = values.mask(bad, 0.0)
        out[column] = values.astype(float)
    return out


def _protocol081_exit(steps: pd.DataFrame, artifact: Protocol066Artifact) -> dict[str, Any]:
    value, recovery, decay = predict_protocol066_sequence(artifact, steps)
    selected_idx = len(steps) - 1
    selected_prediction = None
    for idx, row in steps.iterrows():
        prediction = prediction_for_step(
            step_index=int(idx),
            value=value,
            recovery=recovery,
            decay=decay,
            override_threshold=artifact.selected_override_threshold,
            step_row=row,
        )
        selected_prediction = prediction
        if prediction.action in {"exit", "stop", "forced_flat"}:
            selected_idx = int(idx)
            break
    row = steps.iloc[selected_idx]
    if selected_prediction is None:
        selected_prediction = prediction_for_step(
            step_index=int(selected_idx),
            value=value,
            recovery=recovery,
            decay=decay,
            override_threshold=artifact.selected_override_threshold,
            step_row=row,
        )
    return {
        "candidate_exit_time": pd.Timestamp(row["quote_time"]).isoformat(),
        "candidate_exit_step": int(row["step_idx"]),
        "candidate_pnl": _finite_float(row["current_pnl"], 0.0),
        "candidate_exit_reason": selected_prediction.reason,
        "candidate_action": selected_prediction.action,
        "predicted_continuation_value": selected_prediction.predicted_continuation_value,
        "predicted_recovery_probability": selected_prediction.predicted_recovery_probability,
        "predicted_decay_probability": selected_prediction.predicted_decay_probability,
        "mfe_to_exit": _finite_float(row.get("mfe_to_now"), 0.0),
        "mae_to_exit": _finite_float(row.get("mae_to_now"), 0.0),
        "future_max_delta_at_exit": _finite_float(row.get("future_max_delta"), 0.0),
        "future_min_delta_at_exit": _finite_float(row.get("future_min_delta"), 0.0),
    }


def _serial_replay(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    open_until_by_session: dict[str, pd.Timestamp] = {}
    open_trade_by_session: dict[str, str] = {}
    for row in candidates:
        if row.get("path_status") != "ok":
            skipped.append({**row, "skip_reason": str(row.get("path_status"))})
            continue
        session = str(row["session"])
        decision_time = pd.Timestamp(row["decision_time"])
        open_until = open_until_by_session.get(session)
        if open_until is not None and decision_time < open_until:
            skipped.append(
                {
                    **row,
                    "skip_reason": "open_position",
                    "blocking_trade_uid": open_trade_by_session.get(session),
                    "blocking_until": open_until.isoformat(),
                }
            )
            continue
        trades.append(
            {
                **row,
                "quantity": 1,
                "max_concurrent_positions": 1,
                "serial_status": "taken",
            }
        )
        open_until_by_session[session] = pd.Timestamp(row["candidate_exit_time"])
        open_trade_by_session[session] = str(row["trade_uid"])
    return trades, skipped


def _independent_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in candidates if row.get("path_status") == "ok"]
    return _trade_summary(ok)


def _trade_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_finite_float(row.get("candidate_pnl"), 0.0) for row in rows]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    gross_win = float(sum(wins))
    gross_loss = float(-sum(losses))
    by_session: dict[str, Any] = {}
    for session in sorted({str(row.get("session")) for row in rows}):
        subset = [row for row in rows if str(row.get("session")) == session]
        by_session[session] = _trade_summary_no_session(subset)
    return {
        "trades": int(len(rows)),
        "total_pnl": float(sum(pnls)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float(len(wins) / len(rows)) if rows else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "max_winner": float(max(pnls)) if pnls else 0.0,
        "max_loser": float(min(pnls)) if pnls else 0.0,
        "by_session": by_session,
        "by_side": _group_total(rows, "right"),
        "by_exit_reason": _group_total(rows, "candidate_exit_reason"),
    }


def _trade_summary_no_session(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_finite_float(row.get("candidate_pnl"), 0.0) for row in rows]
    return {
        "trades": int(len(rows)),
        "total_pnl": float(sum(pnls)),
        "wins": int(sum(pnl > 0 for pnl in pnls)),
        "losses": int(sum(pnl < 0 for pnl in pnls)),
    }


def _skipped_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "skipped": int(len(rows)),
        "by_reason": _count_values(row.get("skip_reason") for row in rows),
        "would_have_pnl": float(sum(_finite_float(row.get("candidate_pnl"), 0.0) for row in rows if row.get("path_status") == "ok")),
        "would_have_winners": int(
            sum(_finite_float(row.get("candidate_pnl"), 0.0) > 0.0 for row in rows if row.get("path_status") == "ok")
        ),
        "would_have_losers": int(
            sum(_finite_float(row.get("candidate_pnl"), 0.0) < 0.0 for row in rows if row.get("path_status") == "ok")
        ),
    }


def _answer(candidates: list[dict[str, Any]], trades: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> str:
    if not candidates:
        return "no Protocol101 entries existed in the Protocol161 replay"
    if not trades:
        return "serial replay took no trades because candidate paths were unavailable"
    first = trades[0]
    skipped_open = [row for row in skipped if row.get("skip_reason") == "open_position"]
    skipped_pnl = sum(_finite_float(row.get("candidate_pnl"), 0.0) for row in skipped_open)
    first_pnl = _finite_float(first.get("candidate_pnl"), 0.0)
    if first_pnl < 0 and skipped_pnl > 0:
        return (
            "yes: independent May20 entry labels were misleading for live one-position behavior; "
            "the first serial trade was a loser and it occupied the slot while later independent winners appeared"
        )
    if skipped_open:
        return "partly: serial constraints skipped overlapping entry signals, so independent-entry totals overstate tradable opportunity"
    return "no overlap conflict found; the serial replay took every Protocol101 entry signal"


def _decision(summary: dict[str, Any], skipped: list[dict[str, Any]]) -> str:
    if summary["trades"] == 0:
        return "blocked_no_serial_trades"
    if summary["total_pnl"] <= 0.0:
        return "reject_independent_entry_pnl_as_evidence: serial replay is not profitable"
    if any(row.get("skip_reason") == "open_position" for row in skipped):
        return "keep_only_as_serial_replay_evidence: independent entry pnl overstated by overlap"
    return "keep: serial replay remains positive without overlap conflict"


def _write_report(
    path: Path,
    payload: dict[str, Any],
    candidates: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> None:
    serial = payload["serial_protocol081_summary"]
    independent = payload["independent_protocol081_summary"]
    lines = [
        "# Protocol 162: May 2026 Serial Lifecycle Replay",
        "",
        "Frozen Protocol101 entries from Protocol161 were replayed with one-position serial accounting and frozen Protocol081 lifecycle exits. No model was retrained, no paid data was downloaded, and no broker endpoint was called.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Answer: {payload['answer']}",
        f"- Protocol081 artifact: `{payload['protocol081_manifest']}`",
        f"- Candidate entry signals: `{payload['candidate_signals']}`",
        f"- Candidate path status: `{payload['candidate_path_status']}`",
        "",
        "## Independent vs Serial",
        "",
        "| replay | trades | total PnL | wins | losses | win rate | profit factor |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _summary_row("independent entries", independent),
        _summary_row("serial one-position", serial),
        "",
        "## Serial Trades",
        "",
        "| session | entry ET | exit ET | side | contract | ask | exit reason | PnL | MFE | MAE |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|",
    ]
    for row in trades:
        lines.append(
            f"| {row['session']} | {_et(row['decision_time'])} | {_et(row['candidate_exit_time'])} | "
            f"{row['right']} | `{row['contract_id']}` | {_fmt(row.get('entry_ask'))} | "
            f"{row['candidate_exit_reason']} | {_fmt(row.get('candidate_pnl'))} | "
            f"{_fmt(row.get('mfe_to_exit'))} | {_fmt(row.get('mae_to_exit'))} |"
        )
    lines.extend(
        [
            "",
            "## Skipped Entry Signals",
            "",
            f"- Skipped signals: `{payload['skipped_due_to_open_position']['skipped']}`",
            f"- Skipped by reason: `{payload['skipped_due_to_open_position']['by_reason']}`",
            f"- Skipped open-position candidate PnL: `{payload['skipped_due_to_open_position']['would_have_pnl']:.0f}`",
            "",
            "| session | entry ET | side | contract | reason | blocking until ET | candidate PnL |",
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for row in skipped:
        lines.append(
            f"| {row.get('session')} | {_et(row.get('decision_time'))} | {row.get('right')} | "
            f"`{row.get('contract_id')}` | {row.get('skip_reason')} | {_et(row.get('blocking_until'))} | "
            f"{_fmt(row.get('candidate_pnl'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Protocol161 remains useful as a live-startup diagnostic, but it should not be used as a profitability artifact. The source of truth for tradable replay must be serial/account-aware because overlapping entry signals can only be realized by a one-position bot if the earlier position exits first.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Candidate paths: `{path.parent / 'candidate_lifecycle_paths.csv'}`",
            f"- Serial trades: `{path.parent / 'serial_lifecycle_trades.csv'}`",
            f"- Skipped signals: `{path.parent / 'skipped_entry_signals.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _summary_row(label: str, row: dict[str, Any]) -> str:
    return (
        f"| {label} | {row['trades']} | {_fmt(row['total_pnl'])} | {row['wins']} | {row['losses']} | "
        f"{row['win_rate']:.2f} | {_fmt(row['profit_factor'])} |"
    )


def _base_candidate(trade: pd.Series) -> dict[str, Any]:
    return {
        "trade_uid": str(trade["trade_uid"]),
        "canonical_entry_uid": str(trade["canonical_entry_uid"]),
        "source_row": int(trade["source_row"]),
        "session": str(trade["session"]),
        "decision_time": pd.Timestamp(trade["decision_ts"]).isoformat(),
        "contract_id": str(trade["contract_id"]),
        "right": str(trade["right"]),
        "offset": _finite_float(trade.get("offset")),
        "edge": _finite_float(trade.get("edge")),
        "protocol101_selected_ask": _finite_float(trade.get("ask")),
    }


def _missing_candidate(trade: pd.Series, *, path_status: str) -> dict[str, Any]:
    return {**_base_candidate(trade), "path_status": path_status}


def _group_total(rows: list[dict[str, Any]], column: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(column))
        item = out.setdefault(key, {"trades": 0, "total_pnl": 0.0})
        item["trades"] += 1
        item["total_pnl"] += _finite_float(row.get("candidate_pnl"), 0.0)
    return out


def _count_values(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def _uid(*parts: object) -> str:
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:20]


def _finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _et(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return ""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York").strftime("%H:%M")


def _fmt(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return ""
    if abs(number) == float("inf"):
        return "inf"
    return f"{number:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
