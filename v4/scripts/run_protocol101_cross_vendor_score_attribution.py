"""Attribute Protocol101 cross-vendor surface-score drift with frozen-model counterfactuals.

This is diagnostic-only. It does not train, tune, alter thresholds, or contact a
broker/data vendor. Each scenario changes one input family in the captured IBKR
trace to its paired historical value and rescores the frozen surface model.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np

from v4.live.protocol051_surface_edge import load_surface_edge_artifact, score_surface_decisions
from v4.model.hypothesis_protocol import SurfaceDecision


NY = ZoneInfo("America/New_York")
TOKEN_GROUPS = {
    "option_volume": (7, 21),
    "open_interest": (8, 20),
    "quote_spread": (0, 1, 2, 3, 4, 14, 19, 26, 52, 53, 55, 56, 57),
    "size_liquidity": (5, 6, 28),
    "greeks_decay": (9, 10, 11, 12, 22, 23, 24, 25, 27, 48, 49, 50, 51, 54, 55),
    "pattern_context": tuple(range(29, 48)),
    "omar_patterns": (37, 38),
    "discrete_break_pattern": (33, 34, 41),
}
SCALAR_GROUPS = {
    "market_window": tuple(range(0, 28)),
    "structure": tuple(range(28, 56)),
    "omar_structure": tuple(range(35, 42)),
    "discrete_break_pattern": (49,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-traces", type=Path, required=True)
    parser.add_argument("--historical-traces", type=Path, required=True)
    parser.add_argument("--surface-manifest", type=Path, required=True)
    parser.add_argument(
        "--session",
        default=None,
        help="Optional single-session filter. Omit to attribute all paired sessions.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-tolerance", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("payload")
    return nested if row.get("schema_version") == "Protocol101DecisionTraceV1" and isinstance(nested, dict) else row


def _key(row: dict[str, Any]) -> str:
    session = str(row.get("session") or "")
    timestamp = datetime.fromisoformat(str(row["decision_ts"]).replace("Z", "+00:00")).isoformat()
    return f"{session}|{timestamp}"


def _load(path: Path, session: str | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = _payload(json.loads(line))
            if session is None or str(row.get("session")) == session:
                rows[_key(row)] = row
    return rows


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _trace_arrays(row: dict[str, Any]) -> dict[str, Any]:
    scores = list((row.get("model_scores") or {}).get("surface_scores") or [])
    if len(scores) < 2:
        raise ValueError(f"trace has no surface scores at {row.get('decision_ts')}")
    width = len(scores) - 1
    universe = [item for item in row.get("candidate_universe") or [] if isinstance(item, dict)]
    if not universe:
        raise ValueError(f"trace has no candidate universe at {row.get('decision_ts')}")
    token_dim = max(len(item.get("token_features") or []) for item in universe)
    tokens = np.zeros((width, token_dim), dtype=np.float32)
    mask = np.zeros(width, dtype=bool)
    offsets = np.zeros(width, dtype=np.float32)
    rights = np.full(width, "C", dtype=object)
    contract_ids = np.asarray([f"invalid-{index}" for index in range(width)], dtype=object)
    for item in universe:
        token_index = int(item["token_idx"])
        features = np.asarray(item.get("token_features") or [], dtype=np.float32)
        if token_index < 0 or token_index >= width or len(features) != token_dim:
            continue
        tokens[token_index] = features
        mask[token_index] = True
        offsets[token_index] = float(item.get("offset_points") or 0.0)
        rights[token_index] = str(item.get("right") or "C")
        contract_ids[token_index] = str(item.get("contract_id") or contract_ids[token_index])
    features = row.get("features") or {}
    return {
        "scalar": np.asarray(features.get("scalar_features") or [], dtype=np.float32),
        "market_last": np.asarray(features.get("market_last") or [], dtype=np.float32),
        "tokens": tokens,
        "mask": mask,
        "offsets": offsets,
        "rights": rights,
        "contract_ids": contract_ids,
        "stored_scores": np.asarray(scores, dtype=np.float32),
    }


def _decision(row: dict[str, Any], arrays: dict[str, Any]) -> SurfaceDecision:
    return SurfaceDecision(
        session=str(row["session"]),
        decision_time=datetime.fromisoformat(str(row["decision_ts"]).replace("Z", "+00:00")),
        scalar_features=np.asarray(arrays["scalar"], dtype=np.float32),
        token_features=np.asarray(arrays["tokens"], dtype=np.float32),
        token_mask=np.asarray(arrays["mask"], dtype=bool),
        labels=np.zeros(len(arrays["mask"]), dtype=np.float32),
        offsets=np.asarray(arrays["offsets"], dtype=np.float32),
        rights=np.asarray(arrays["rights"], dtype=object),
        contract_ids=np.asarray(arrays["contract_ids"], dtype=object),
        market_last=np.asarray(arrays["market_last"], dtype=np.float32),
    )


def _replace_matching_tokens(
    live_arrays: dict[str, Any],
    historical_arrays: dict[str, Any],
    indices: Iterable[int],
) -> dict[str, Any]:
    out = {key: np.copy(value) if isinstance(value, np.ndarray) else value for key, value in live_arrays.items()}
    selected = tuple(int(index) for index in indices)
    historical_by_contract = {
        str(contract_id): index
        for index, contract_id in enumerate(historical_arrays["contract_ids"])
        if bool(historical_arrays["mask"][index])
    }
    for live_index, contract_id in enumerate(live_arrays["contract_ids"]):
        if not bool(live_arrays["mask"][live_index]):
            continue
        historical_index = historical_by_contract.get(str(contract_id))
        if historical_index is None:
            continue
        for feature_index in selected:
            if feature_index < out["tokens"].shape[1] and feature_index < historical_arrays["tokens"].shape[1]:
                out["tokens"][live_index, feature_index] = historical_arrays["tokens"][historical_index, feature_index]
    return out


def _replace_scalar(live_arrays: dict[str, Any], historical_arrays: dict[str, Any], indices: Iterable[int]) -> dict[str, Any]:
    out = {key: np.copy(value) if isinstance(value, np.ndarray) else value for key, value in live_arrays.items()}
    for index in indices:
        if int(index) < len(out["scalar"]) and int(index) < len(historical_arrays["scalar"]):
            out["scalar"][int(index)] = historical_arrays["scalar"][int(index)]
    return out


def _full_token_replacement(live_arrays: dict[str, Any], historical_arrays: dict[str, Any]) -> dict[str, Any]:
    out = {key: np.copy(value) if isinstance(value, np.ndarray) else value for key, value in live_arrays.items()}
    for key in ("tokens", "mask", "offsets", "rights", "contract_ids"):
        out[key] = np.copy(historical_arrays[key])
    return out


def _edge(scores: np.ndarray, decision: SurfaceDecision) -> tuple[float | None, str | None]:
    action_scores = np.asarray(scores[1:], dtype=float).copy()
    action_scores[~np.asarray(decision.token_mask, dtype=bool)] = -math.inf
    if not np.isfinite(action_scores).any():
        return None, None
    token_index = int(np.nanargmax(action_scores))
    return float(action_scores[token_index] - float(scores[0])), str(decision.contract_ids[token_index])


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(abs(value) for value in values)
    return {
        "n": len(values),
        "signed_mean": mean(values),
        "signed_median": median(values),
        "mean_abs": mean(ordered),
        "median_abs": median(ordered),
        "p95_abs": ordered[int(round((len(ordered) - 1) * 0.95))],
        "max_abs": ordered[-1],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    live = _load(args.live_traces, args.session)
    historical = _load(args.historical_traces, args.session)
    keys = sorted(set(live) & set(historical))
    if not keys:
        raise ValueError("no paired trace rows")

    artifact = load_surface_edge_artifact(args.surface_manifest)
    scenario_names = [
        "live_baseline",
        "historical_baseline",
        "live_with_historical_scalar",
        "live_with_historical_tokens",
    ]
    for group in TOKEN_GROUPS:
        scenario_names.append(f"live_with_historical_{group}")
    for group in SCALAR_GROUPS:
        if group != "discrete_break_pattern":
            scenario_names.append(f"live_with_historical_scalar_{group}")
    scenario_names.append("live_with_historical_discrete_break_pattern")
    scenario_names = list(dict.fromkeys(scenario_names))
    rows: list[dict[str, Any]] = []
    scenario_closures: dict[str, list[float]] = {
        name: [] for name in scenario_names if "with_historical" in name
    }
    scenario_effects: dict[str, list[float]] = {
        name: [] for name in scenario_names if "with_historical" in name
    }
    live_error = 0.0
    historical_error = 0.0
    batch_size = max(int(args.batch_size), 1)
    for batch_start in range(0, len(keys), batch_size):
        batch_keys = keys[batch_start : batch_start + batch_size]
        scenarios: dict[str, list[SurfaceDecision]] = {name: [] for name in scenario_names}
        stored_scores: dict[str, list[np.ndarray]] = {"live": [], "historical": []}
        for key in batch_keys:
            live_arrays = _trace_arrays(live[key])
            historical_arrays = _trace_arrays(historical[key])
            stored_scores["live"].append(live_arrays["stored_scores"])
            stored_scores["historical"].append(historical_arrays["stored_scores"])
            scenarios["live_baseline"].append(_decision(live[key], live_arrays))
            scenarios["historical_baseline"].append(_decision(historical[key], historical_arrays))
            full_scalar = _replace_scalar(
                live_arrays,
                historical_arrays,
                range(len(live_arrays["scalar"])),
            )
            scenarios["live_with_historical_scalar"].append(_decision(live[key], full_scalar))
            scenarios["live_with_historical_tokens"].append(
                _decision(live[key], _full_token_replacement(live_arrays, historical_arrays))
            )
            for group, indices in TOKEN_GROUPS.items():
                if group == "discrete_break_pattern":
                    continue
                replaced = _replace_matching_tokens(live_arrays, historical_arrays, indices)
                scenarios[f"live_with_historical_{group}"].append(_decision(live[key], replaced))
            for group, indices in SCALAR_GROUPS.items():
                if group == "discrete_break_pattern":
                    continue
                replaced = _replace_scalar(live_arrays, historical_arrays, indices)
                scenarios[f"live_with_historical_scalar_{group}"].append(
                    _decision(live[key], replaced)
                )
            discrete = _replace_scalar(
                live_arrays,
                historical_arrays,
                SCALAR_GROUPS["discrete_break_pattern"],
            )
            discrete = _replace_matching_tokens(
                discrete,
                historical_arrays,
                TOKEN_GROUPS["discrete_break_pattern"],
            )
            scenarios["live_with_historical_discrete_break_pattern"].append(
                _decision(live[key], discrete)
            )

        predictions = {
            name: score_surface_decisions(artifact, decisions)
            for name, decisions in scenarios.items()
        }
        live_error = max(
            live_error,
            max(
                float(
                    np.max(
                        np.abs(
                            predictions["live_baseline"][index]
                            - stored_scores["live"][index]
                        )
                    )
                )
                for index in range(len(batch_keys))
            ),
        )
        historical_error = max(
            historical_error,
            max(
                float(
                    np.max(
                        np.abs(
                            predictions["historical_baseline"][index]
                            - stored_scores["historical"][index]
                        )
                    )
                )
                for index in range(len(batch_keys))
            ),
        )
        for index, key in enumerate(batch_keys):
            live_edge, live_contract = _edge(
                predictions["live_baseline"][index],
                scenarios["live_baseline"][index],
            )
            historical_edge, historical_contract = _edge(
                predictions["historical_baseline"][index],
                scenarios["historical_baseline"][index],
            )
            if live_edge is None or historical_edge is None:
                continue
            ts = datetime.fromisoformat(key.split("|", 1)[-1]).astimezone(NY)
            row: dict[str, Any] = {
                "session": str(live[key].get("session") or historical[key].get("session") or ""),
                "decision_minute_et": ts.strftime("%H:%M"),
                "live_edge": live_edge,
                "historical_edge": historical_edge,
                "baseline_edge_delta": live_edge - historical_edge,
                "live_top_contract_id": live_contract,
                "historical_top_contract_id": historical_contract,
                "baseline_top_contract_match": live_contract == historical_contract,
            }
            baseline_gap = abs(live_edge - historical_edge)
            for name in scenario_closures:
                scenario_edge, scenario_contract = _edge(
                    predictions[name][index],
                    scenarios[name][index],
                )
                row[f"{name}_edge"] = scenario_edge
                row[f"{name}_top_contract_id"] = scenario_contract
                if scenario_edge is not None:
                    effect = scenario_edge - live_edge
                    closure = baseline_gap - abs(scenario_edge - historical_edge)
                    scenario_effects[name].append(effect)
                    scenario_closures[name].append(closure)
                    row[f"{name}_effect"] = effect
                    row[f"{name}_gap_closure"] = closure
            rows.append(row)
    if live_error > args.baseline_tolerance or historical_error > args.baseline_tolerance:
        raise ValueError(
            f"trace reconstruction exceeded tolerance: live={live_error} historical={historical_error} "
            f"tolerance={args.baseline_tolerance}"
        )

    summary = {
        "schema_version": "Protocol101CrossVendorScoreAttributionV1",
        "session": args.session or "ALL",
        "paired_minutes": len(rows),
        "frozen_model_baseline_reconstruction": {
            "tolerance": args.baseline_tolerance,
            "live_max_abs_score_error": live_error,
            "historical_max_abs_score_error": historical_error,
            "status": "pass",
        },
        "baseline_edge_delta": _summary([float(row["baseline_edge_delta"]) for row in rows]),
        "scenarios": {
            name: {
                "edge_effect_vs_live": _summary(scenario_effects[name]),
                "historical_gap_closure": _summary(scenario_closures[name]),
                "positive_gap_closure_minutes": sum(value > 0 for value in scenario_closures[name]),
                "negative_gap_closure_minutes": sum(value < 0 for value in scenario_closures[name]),
            }
            for name in scenario_closures
        },
        "largest_baseline_edge_drifts": sorted(
            rows,
            key=lambda row: abs(float(row["baseline_edge_delta"])),
            reverse=True,
        )[:20],
        "inputs": {
            "live_traces": str(args.live_traces),
            "historical_traces": str(args.historical_traces),
            "surface_manifest": str(args.surface_manifest),
        },
    }
    _write_csv(args.out_dir / "score_attribution_by_minute.csv", rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = [
        "# Protocol101 Cross-Vendor Score Attribution",
        "",
        f"- Session: `{args.session or 'ALL'}`",
        f"- Paired minutes: `{len(rows)}`",
        f"- Frozen live baseline reconstruction max error: `{live_error}`",
        f"- Frozen historical baseline reconstruction max error: `{historical_error}`",
        "- This is diagnostic counterfactual rescoring; no threshold or model parameter changed.",
        "",
        "## Scenario Summary",
        "",
    ]
    for name, payload in summary["scenarios"].items():
        effect = payload["edge_effect_vs_live"]
        closure = payload["historical_gap_closure"]
        report.append(
            f"- `{name}`: effect median/p95 abs=`{effect.get('median_abs')}` / `{effect.get('p95_abs')}`; "
            f"gap closure mean=`{closure.get('signed_mean')}`; improved/worsened minutes="
            f"`{payload['positive_gap_closure_minutes']}` / `{payload['negative_gap_closure_minutes']}`"
        )
    report.extend(["", "## Largest Baseline Edge Drifts", ""])
    for row in summary["largest_baseline_edge_drifts"][:10]:
        report.append(
            f"- `{row['decision_minute_et']} ET`: live/historical=`{row['live_edge']}` / `{row['historical_edge']}` "
            f"delta=`{row['baseline_edge_delta']}`"
        )
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
