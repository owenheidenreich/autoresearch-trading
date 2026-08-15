from __future__ import annotations

import argparse
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
from v4.live.shadow_parity import ShadowParityConfig, summarize_shadow_parity


DEFAULT_SEQUENCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_060_lifecycle_sequence_dataset")
DEFAULT_ARTIFACT_MANIFEST = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_069_protocol066_artifact_persistence/"
    "model_artifacts/train_q2_q3_q4_test_q1/seed_1/manifest.json"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_071_protocol066_offline_shadow_rehearsal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", default="protocol066")
    parser.add_argument("--protocol-label", default="Protocol 066")
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_ARTIFACT_MANIFEST)
    parser.add_argument("--vix-dir", type=Path, default=Path("data/raw/index/vix_1m"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", default="q1_2026")
    parser.add_argument("--session-start", default="2026-03-01")
    parser.add_argument("--max-trades", type=int, default=50)
    parser.add_argument("--max-shadow-rows", type=int, default=1500)
    parser.add_argument("--max-quote-age-ms", type=int, default=1500)
    parser.add_argument("--max-context-age-ms", type=int, default=5000)
    return parser.parse_args()


def _finite(value: Any) -> bool:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def _timestamp_ms(value: Any) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def _read_sequence(sequence_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades_path = sequence_dir / "protocol054_lifecycle_trades.parquet"
    steps_path = sequence_dir / "protocol054_lifecycle_steps.parquet"
    if not trades_path.exists() or not steps_path.exists():
        raise SystemExit(f"missing Protocol 060 sequence files under {sequence_dir}")
    trades = pd.read_parquet(trades_path)
    steps = pd.read_parquet(steps_path)
    trades["session"] = trades["session"].astype(str)
    trades["trade_uid"] = trades["trade_uid"].astype(str)
    steps["session"] = steps["session"].astype(str)
    steps["trade_uid"] = steps["trade_uid"].astype(str)
    steps["quote_ts"] = pd.to_datetime(steps["quote_time"], utc=True)
    return trades, steps


def _select_trade_uids(trades: pd.DataFrame, *, split: str, session_start: str, max_trades: int) -> list[str]:
    frame = trades[(trades["split"].eq(split)) & (trades["session"].ge(session_start)) & (trades["path_status"].eq("ok"))].copy()
    frame = frame.sort_values(["session", "decision_time", "seed", "trade_uid"])
    if max_trades > 0:
        frame = frame.head(max_trades)
    return frame["trade_uid"].astype(str).tolist()


def _vix_path(vix_dir: Path, session: str) -> Path | None:
    for pattern in (f"{session}.official_vix.parquet", f"{session}*.official_vix.parquet", f"{session}.derived_spxw_atm_iv.parquet", f"{session}*.parquet"):
        matches = sorted(vix_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_vix_lookup(vix_dir: Path, sessions: set[str]) -> dict[str, pd.DataFrame]:
    lookup: dict[str, pd.DataFrame] = {}
    for session in sorted(sessions):
        path = _vix_path(vix_dir, session)
        if path is None:
            continue
        frame = pd.read_parquet(path)
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["event_time", "close"]).sort_values("event_time").reset_index(drop=True)
        frame["source_path"] = str(path)
        lookup[session] = frame
    return lookup


def _vix_at(vix_lookup: dict[str, pd.DataFrame], session: str, quote_ts: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None, str | None]:
    frame = vix_lookup.get(session)
    if frame is None or frame.empty:
        return None, None, None
    idx = frame["event_time"].searchsorted(quote_ts, side="right") - 1
    if idx < 0:
        return None, None, str(frame["source_path"].iloc[0])
    row = frame.iloc[int(idx)]
    return float(row["close"]), pd.Timestamp(row["event_time"]), str(row["source_path"])


def _prepare_trade_steps(frame: pd.DataFrame, artifact: Protocol066Artifact) -> tuple[pd.DataFrame, dict[str, int]]:
    out = frame.sort_values("step_idx").reset_index(drop=True).copy()
    quote_gap = out["quote_ts"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
    fill_counts: dict[str, int] = {}
    for column in artifact.feature_columns:
        if column not in out.columns:
            out[column] = 0.0
            fill_counts[column] = len(out)
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        bad = ~np.isfinite(values.to_numpy(dtype=float))
        if column == "quote_gap_seconds":
            values = values.mask(bad, quote_gap)
        else:
            values = values.mask(bad, 0.0)
        if bool(bad.any()):
            fill_counts[column] = int(bad.sum())
        out[column] = values.astype(float)
    return out, fill_counts


def _feature_dict(row: pd.Series, feature_columns: list[str]) -> dict[str, float]:
    return {column: float(row[column]) for column in feature_columns}


def _shadow_observation(
    *,
    protocol_id: str,
    artifact: Protocol066Artifact,
    row: pd.Series,
    prediction,
    vix_value: float,
    vix_ts: pd.Timestamp,
    vix_source: str,
) -> dict[str, Any]:
    quote_ts = pd.Timestamp(row["quote_ts"])
    quote_ms = _timestamp_ms(quote_ts)
    return {
        "protocol_id": protocol_id,
        "timestamp_ms": quote_ms,
        "decision_time": quote_ts.isoformat(),
        "entry_decision_time": str(row["decision_time"]),
        "session": str(row["session"]),
        "position_state": "holding",
        "intended_size": 1,
        "contract_id": str(row["contract_id"]),
        "trade_uid": str(row["trade_uid"]),
        "sequence_step_index": int(row["step_idx"]),
        "nbbo": {
            "bid": float(row["bid"]),
            "ask": float(row["ask"]),
            "timestamp_ms": quote_ms,
        },
        "context": {
            "spx": float(row["underlying_price"]),
            "vix": float(vix_value),
            "source": f"offline_rehearsal:{vix_source}",
            "timestamp_ms": _timestamp_ms(vix_ts),
        },
        "features": _feature_dict(row, artifact.feature_columns),
        "model": {
            "artifact": artifact.artifact_ref,
            "fold": artifact.fold,
            "seed": artifact.seed,
            "predicted_continuation_value": prediction.predicted_continuation_value,
            "predicted_recovery_probability": prediction.predicted_recovery_probability,
            "predicted_decay_probability": prediction.predicted_decay_probability,
            "override_threshold": prediction.override_threshold if math.isfinite(prediction.override_threshold) else 1e18,
        },
        "decision": {
            "action": prediction.action,
            "reason": prediction.reason,
        },
        "order_intent": None,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    parity = payload["shadow_parity"]
    protocol_label = payload["protocol_label"]
    lines = [
        f"# {protocol_label} Offline Shadow Rehearsal",
        "",
        "No paid data was downloaded. No broker endpoint was called.",
        "",
        "## Purpose",
        "",
        f"This rehearsal checks whether persisted {protocol_label} artifacts can be loaded, run over causal lifecycle feature sequences, emitted as no-order shadow observations, and passed through the shadow parity validator.",
        "",
        "## Result",
        "",
        f"- Status: `{payload['decision']}`",
        f"- Artifact: `{payload['artifact_manifest']}`",
        f"- Shadow log: `{payload['shadow_log']}`",
        f"- Trades rehearsed: `{payload['trades_rehearsed']}`",
        f"- Shadow rows: `{parity['rows']}`",
        f"- Passed rows: `{parity['passed_rows']}`",
        f"- Failed rows: `{parity['failed_rows']}`",
        f"- Action counts: `{parity['action_counts']}`",
        f"- VIX coverage: `{payload['vix_coverage']:.3f}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in parity["checks"]:
        lines.append(f"| {check['name']} | {check['status']} | {check['detail']} |")
    lines.extend(
        [
            "",
            "## Remaining Gate",
            "",
            "This is still offline. The next gate is the same JSONL schema captured from an actual no-order live feed, with live stale-quote rejection and retained entry quote timestamps.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = load_protocol066_artifact(args.artifact_manifest)
    trades, steps = _read_sequence(args.sequence_dir)
    trade_uids = _select_trade_uids(trades, split=args.split, session_start=args.session_start, max_trades=args.max_trades)
    if not trade_uids:
        raise SystemExit("no trades selected for offline shadow rehearsal")
    selected_steps = steps[steps["trade_uid"].isin(trade_uids)].copy()
    selected_steps = selected_steps.sort_values(["session", "trade_uid", "step_idx"])
    vix_lookup = _load_vix_lookup(args.vix_dir, set(selected_steps["session"].unique()))

    shadow_rows: list[dict[str, Any]] = []
    fill_counts: dict[str, int] = {}
    vix_hits = 0
    vix_misses = 0
    for uid, group in selected_steps.groupby("trade_uid", sort=False):
        prepared, trade_fill_counts = _prepare_trade_steps(group, artifact)
        for column, count in trade_fill_counts.items():
            fill_counts[column] = fill_counts.get(column, 0) + int(count)
        value, recovery, decay = predict_protocol066_sequence(artifact, prepared)
        for local_idx, row in prepared.iterrows():
            if len(shadow_rows) >= args.max_shadow_rows:
                break
            vix_value, vix_ts, vix_source = _vix_at(vix_lookup, str(row["session"]), pd.Timestamp(row["quote_ts"]))
            if vix_value is None or vix_ts is None or vix_source is None:
                vix_misses += 1
                continue
            vix_hits += 1
            prediction = prediction_for_step(
                step_index=int(local_idx),
                value=value,
                recovery=recovery,
                decay=decay,
                override_threshold=artifact.selected_override_threshold,
                step_row=row,
            )
            shadow_rows.append(
                _shadow_observation(
                    artifact=artifact,
                    protocol_id=args.protocol_id,
                    row=row,
                    prediction=prediction,
                    vix_value=vix_value,
                    vix_ts=vix_ts,
                    vix_source=vix_source,
                )
            )
        if len(shadow_rows) >= args.max_shadow_rows:
            break

    config = ShadowParityConfig(
        protocol_id=args.protocol_id,
        max_quote_age_ms=args.max_quote_age_ms,
        max_context_age_ms=args.max_context_age_ms,
    )
    parity = summarize_shadow_parity(shadow_rows, config=config)
    shadow_log = args.out_dir / "offline_shadow_observations.jsonl"
    shadow_log.write_text("\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in shadow_rows) + "\n")
    decision = "pass" if parity["status"] == "pass" else "blocked"
    payload = {
        "decision": decision,
        "protocol_id": args.protocol_id,
        "protocol_label": args.protocol_label,
        "artifact_manifest": str(args.artifact_manifest),
        "shadow_log": str(shadow_log),
        "trades_rehearsed": len(trade_uids),
        "shadow_parity": parity,
        "feature_fill_counts": fill_counts,
        "vix_hits": vix_hits,
        "vix_misses": vix_misses,
        "vix_coverage": float(vix_hits / max(1, vix_hits + vix_misses)),
        "args": vars(args) | {"sequence_dir": str(args.sequence_dir), "artifact_manifest": str(args.artifact_manifest), "vix_dir": str(args.vix_dir), "out_dir": str(args.out_dir)},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(args.out_dir / "report.md")
    print(json.dumps({"decision": decision, "rows": parity["rows"], "failed_rows": parity["failed_rows"], "actions": parity["action_counts"]}, sort_keys=True))
    return 0 if parity["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
