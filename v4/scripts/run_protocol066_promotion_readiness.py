"""Protocol 068: promotion-readiness infrastructure for frozen Protocol 066.

This is not a model tweak and not paper/live approval. It freezes the Protocol
066 research candidate, replays its selected decisions through order-state
accounting, applies conservative extra-slippage stress, and reports live-data
parity blockers that must be closed before paper trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.sim.paper_replay import (
    DEFAULT_SLIPPAGE_SCENARIOS,
    PaperReplayConfig,
    SlippageScenario,
    apply_slippage,
    build_replay_frame,
    concentration_metrics,
    live_data_parity_checks,
    load_lifecycle_steps,
    load_selected_trades,
    order_state_summary,
    promotion_gate_status,
    summarize_by_split_seed,
)


_DEFAULT_PROTOCOL_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_066_protocol065_10seed_validation")
_DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_068_protocol066_promotion_readiness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", default="protocol066")
    parser.add_argument("--protocol-label", default="Protocol 066")
    parser.add_argument("--freeze-manifest", type=Path, default=Path("v4/promotion/PROTOCOL_066_FREEZE.json"))
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--selected-trades", type=Path, default=_DEFAULT_PROTOCOL_DIR / "selected_trades_sequence_exits.json")
    parser.add_argument("--protocol-report", type=Path, default=_DEFAULT_PROTOCOL_DIR / "report.json")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_060_lifecycle_sequence_dataset"),
    )
    parser.add_argument("--one-second-report", type=Path, default=_DEFAULT_PROTOCOL_DIR / "one_second_path_audit" / "report.json")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--order-sample-size", type=int, default=5000)
    parser.add_argument(
        "--scored-splits",
        nargs="+",
        default=["q3_2025", "q4_2025", "q1_2026", "march_2026"],
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_freeze(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _verify_freeze(freeze: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, spec in freeze.get("frozen_artifacts", {}).items():
        path = Path(spec["path"])
        exists = path.exists()
        actual = _sha256(path) if exists else None
        expected = spec.get("sha256")
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": exists,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "pass" if exists and actual == expected else "fail",
            }
        )
    return rows


def _load_one_second_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("summary")


def _scenario_rows(base_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        scenario.name: apply_slippage(base_frame, scenario)
        for scenario in DEFAULT_SLIPPAGE_SCENARIOS
    }


def _split_summaries_by_scenario(scenario_frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows = []
    for frame in scenario_frames.values():
        rows.extend(summarize_by_split_seed(frame))
    return rows


def _concentration_by_scenario(scenario_frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, frame in scenario_frames.items():
        out[name] = concentration_metrics(frame)
    return out


def _readiness_decision(
    *,
    freeze_checks: list[dict[str, Any]],
    parity_checks: list[dict[str, Any]],
    split_stress: list[dict[str, Any]],
    scored_splits: set[str],
) -> str:
    if any(check["status"] == "fail" for check in freeze_checks):
        return "blocked: freeze manifest hash check failed"
    gate = promotion_gate_status(parity_checks)
    if gate in {"fail", "blocked"}:
        return "blocked: live-data parity or deployment-artifact blockers remain"
    stressed = [
        row
        for row in split_stress
        if row["slippage_scenario"] == "extra_025_each_side"
        and row["eval_split"] in scored_splits
    ]
    if any(row["seed_total_pnl_median"] <= 0.0 for row in stressed):
        return "blocked: conservative slippage stress fails at least one scored split"
    return "ready_for_shadow_paper_harness_only"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 40) -> str:
    rows = rows[:max_rows]
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.0f}" if abs(value) >= 100 else f"{value:.3f}"
            cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body])


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    split_rows = [
        row
        for row in payload["slippage_split_summary"]
        if row["eval_split"] in set(payload["scored_splits"])
    ]
    protocol_label = payload["protocol_label"]
    lines = [
        f"# {protocol_label} Promotion-Readiness Replay",
        "",
        f"{protocol_label} is frozen as a research challenger. This report is promotion infrastructure, not paper/live approval.",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        f"Artifact manifest: `{payload.get('artifact_manifest') or 'not supplied'}`",
        "",
        "## Freeze Checks",
        "",
        _markdown_table(
            payload["freeze_checks"],
            ["artifact", "status", "exists", "expected_sha256", "actual_sha256"],
            max_rows=20,
        ),
        "",
        "## Slippage Stress",
        "",
        _markdown_table(
            split_rows,
            [
                "slippage_scenario",
                "eval_split",
                "seed_count",
                "seed_total_pnl_median",
                "seed_total_pnl_min",
                "positive_seed_fraction",
                "profit_factor_median",
                "trades_median",
            ],
            max_rows=80,
        ),
        "",
        "## Order-State Accounting",
        "",
        "The replay builds marketable one-contract order records from candidate_seen through exit_filled.",
        "",
        "```json",
        json.dumps(payload["order_state_summary"], indent=2),
        "```",
        "",
        "## Live-Data Parity Checks",
        "",
        _markdown_table(payload["live_data_parity_checks"], ["name", "status", "detail", "value"], max_rows=40),
        "",
        "## Blockers",
        "",
    ]
    blockers = [check for check in payload["live_data_parity_checks"] if check["status"] == "blocker"]
    if blockers:
        for check in blockers:
            lines.append(f"- `{check['name']}`: {check['detail']}")
    else:
        lines.append("- None")
    lines += [
        "",
        "## Next Step",
        "",
        f"Keep {protocol_label} frozen. Build the shadow paper harness around persisted model artifacts and live-feed parity before any broker-connected paper trading.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    freeze = _load_freeze(args.freeze_manifest)
    freeze_checks = _verify_freeze(freeze)
    selected = load_selected_trades(args.selected_trades)
    steps = load_lifecycle_steps(args.sequence_dir)
    base_frame = build_replay_frame(selected, steps)
    scenario_frames = _scenario_rows(base_frame)
    nbbo_frame = scenario_frames["nbbo_bid_ask"]
    one_second_summary = _load_one_second_summary(args.one_second_report)
    config = PaperReplayConfig(
        protocol_id=args.protocol_id,
        persisted_model_artifact_available=bool(args.artifact_manifest and args.artifact_manifest.exists()),
    )
    parity_checks = live_data_parity_checks(base_frame, one_second_summary, config=config)
    split_summary = _split_summaries_by_scenario(scenario_frames)
    order_summary = order_state_summary(
        nbbo_frame,
        config=config,
        scenario=SlippageScenario("nbbo_bid_ask"),
        sample_size=args.order_sample_size,
    )
    payload = {
        "protocol": "068_protocol066_promotion_readiness",
        "protocol_id": args.protocol_id,
        "protocol_label": args.protocol_label,
        "paid_data_downloaded": False,
        "freeze_manifest": str(args.freeze_manifest),
        "artifact_manifest": str(args.artifact_manifest) if args.artifact_manifest else None,
        "freeze": freeze,
        "freeze_checks": freeze_checks,
        "row_counts": {
            "selected_rows": int(len(selected)),
            "replay_rows": int(len(base_frame)),
            "matched_exit_step_fraction": float(base_frame["exit_bid"].notna().mean()),
        },
        "path_replay_quality": {
            "candidate_vs_path_abs_diff_p50": float(base_frame["candidate_pnl_diff_vs_path"].abs().quantile(0.50)),
            "candidate_vs_path_abs_diff_p95": float(base_frame["candidate_pnl_diff_vs_path"].abs().quantile(0.95)),
            "candidate_vs_path_abs_diff_p99": float(base_frame["candidate_pnl_diff_vs_path"].abs().quantile(0.99)),
        },
        "slippage_scenarios": [scenario.__dict__ | {"round_trip_cost": scenario.round_trip_cost} for scenario in DEFAULT_SLIPPAGE_SCENARIOS],
        "slippage_split_summary": split_summary,
        "concentration_by_scenario": _concentration_by_scenario(scenario_frames),
        "order_state_summary": order_summary,
        "one_second_summary_used": one_second_summary,
        "live_data_parity_checks": parity_checks,
        "scored_splits": list(args.scored_splits),
        "decision": _readiness_decision(
            freeze_checks=freeze_checks,
            parity_checks=parity_checks,
            split_stress=split_summary,
            scored_splits=set(args.scored_splits),
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_payload = _json_safe(payload)
    (args.out_dir / "report.json").write_text(json.dumps(json_payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(args.out_dir / "report.md", json_payload)
    print(json.dumps({"decision": payload["decision"], "row_counts": payload["row_counts"]}, indent=2))
    print(args.out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
