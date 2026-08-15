"""Dry-run manifest-based ingestion for Protocol101 fair-contract training.

This verifies that a future owner-approved training run can load the canonical
protocol101-live-v1 manifest, apply the preregistered chronological split, and
construct candidate-level feature/label examples. It intentionally does not
train, tune thresholds, call vendors, call broker endpoints, or change runtime
defaults.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from v4.live.protocol101_synchronization import Protocol101FairContractTrainingDryRunV1
from v4.model.supervised_pilot import candidate_feature_vector


DEFAULT_DESIGN = Path("v4/audit/autoresearch/protocol101_fair_contract_training_design/summary.json")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_fair_contract_training_dry_run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--feature-sample-limit", type=int, default=20_000)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{path} expected list, got {type(rows).__name__}")
    return rows


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def split_lookup(split_policy: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split_name, key in (
        ("train", "train_sessions"),
        ("validation", "validation_sessions"),
        ("diagnostic_test", "diagnostic_test_sessions"),
    ):
        for session in split_policy.get(key) or []:
            lookup[str(session)] = split_name
    for row in split_policy.get("embargoed_sessions") or []:
        lookup[str(row.get("session"))] = "embargoed"
    return lookup


def update_numeric_summary(bucket: dict[str, Any], value: float) -> None:
    bucket["n"] += 1
    bucket["sum"] += value
    bucket["min"] = value if bucket["min"] is None else min(bucket["min"], value)
    bucket["max"] = value if bucket["max"] is None else max(bucket["max"], value)
    bucket["positive"] += int(value > 0.0)
    bucket["negative"] += int(value < 0.0)
    bucket["zero"] += int(value == 0.0)


def empty_numeric_summary() -> dict[str, Any]:
    return {"n": 0, "sum": 0.0, "min": None, "max": None, "positive": 0, "negative": 0, "zero": 0}


def mean_from_summary(summary: dict[str, Any]) -> float | None:
    return float(summary["sum"]) / int(summary["n"]) if int(summary["n"]) else None


def build_dry_run(design: dict[str, Any], *, feature_sample_limit: int) -> Protocol101FairContractTrainingDryRunV1:
    allowed = design.get("allowed_data") or {}
    manifest_path = Path(str(allowed.get("canonical_manifest") or ""))
    manifest = load_json(manifest_path)
    included = manifest.get("included_sessions") or []
    split_policy = design.get("split_policy") or {}
    lookup = split_lookup(split_policy)
    expected_contract = str(design.get("selected_feature_contract") or "protocol101-live-v1")

    split_summary: dict[str, Any] = {
        name: {
            "sessions": 0,
            "decision_rows": 0,
            "candidate_slots": 0,
            "tradable_candidates": 0,
            "embargoed": name == "embargoed",
        }
        for name in ("train", "validation", "diagnostic_test", "embargoed", "unassigned")
    }
    label_names: list[str] | None = None
    label_summaries: dict[str, dict[str, Any]] = {}
    feature_dim_counter: Counter[int] = Counter()
    row_shape_counter: Counter[str] = Counter()
    version_counter: Counter[str] = Counter()
    blockers: list[str] = []
    session_rows: list[dict[str, Any]] = []
    feature_samples = 0
    nonfinite_feature_values = 0
    total_feature_values = 0

    for item in included:
        session = str(item.get("session") or "")
        split = lookup.get(session, "unassigned")
        path = Path(str(item.get("processed_file") or ""))
        try:
            rows = load_rows(path)
        except Exception as exc:
            blockers.append(f"load_error:{session}:{type(exc).__name__}")
            continue
        split_summary[split]["sessions"] += 1
        session_tradable = 0
        session_slots = 0
        session_versions: Counter[str] = Counter()
        for row in rows:
            if not isinstance(row, dict):
                blockers.append(f"non_dict_row:{session}")
                continue
            version = str(row.get("feature_contract_version") or "UNKNOWN")
            version_counter[version] += 1
            session_versions[version] += 1
            if version != expected_contract:
                blockers.append(f"unexpected_feature_contract:{session}:{version}")
            option_ladder = np.asarray(row.get("option_ladder"))
            mask = np.asarray(row.get("candidate_mask"))
            labels = np.asarray(row.get("labels_net_pnl"))
            names = list(row.get("label_names") or [])
            if label_names is None:
                label_names = names
                label_summaries = {name: empty_numeric_summary() for name in label_names}
            elif names != label_names:
                blockers.append(f"label_names_mismatch:{session}")
            if option_ladder.ndim != 3 or mask.ndim != 2 or labels.ndim != 3:
                blockers.append(f"bad_tensor_rank:{session}")
                continue
            shape_key = f"option={tuple(option_ladder.shape)}|mask={tuple(mask.shape)}|labels={tuple(labels.shape)}"
            row_shape_counter[shape_key] += 1
            slots = int(mask.size)
            tradable = int(mask.sum())
            session_slots += slots
            session_tradable += tradable
            split_summary[split]["decision_rows"] += 1
            split_summary[split]["candidate_slots"] += slots
            split_summary[split]["tradable_candidates"] += tradable
            if split == "embargoed":
                continue
            true_indices = np.argwhere(mask)
            for strike_idx, right_idx in true_indices:
                if feature_samples < feature_sample_limit:
                    vector = candidate_feature_vector(row, int(strike_idx), int(right_idx))
                    feature_dim_counter[int(vector.shape[0])] += 1
                    total_feature_values += int(vector.size)
                    nonfinite_feature_values += int((~np.isfinite(vector)).sum())
                    feature_samples += 1
                for label_idx, label_name in enumerate(label_names or []):
                    value = finite_number(labels[int(strike_idx), int(right_idx), label_idx])
                    if value is not None:
                        update_numeric_summary(label_summaries[label_name], value)
        session_rows.append(
            {
                "session": session,
                "split": split,
                "processed_file": str(path),
                "decision_rows": len(rows),
                "candidate_slots": session_slots,
                "tradable_candidates": session_tradable,
                "feature_contract_versions": dict(sorted(session_versions.items())),
            }
        )

    if split_summary["unassigned"]["sessions"]:
        blockers.append("unassigned_sessions_present")
    if not label_names:
        blockers.append("no_label_names")
    if len(feature_dim_counter) != 1:
        blockers.append("inconsistent_feature_dimensions")
    label_blockers = []
    for label_name, summary in label_summaries.items():
        if int(summary["positive"]) == 0 and int(summary["negative"]) == 0:
            label_blockers.append(f"all_zero_label:{label_name}")
    blockers.extend(label_blockers)
    if design.get("model_training_authorized") is not False:
        blockers.append("design_training_authorization_not_false")
    if allowed.get("glob_loading_allowed") is not False:
        blockers.append("glob_loading_not_disabled")

    label_summary = {
        name: {
            **summary,
            "mean": mean_from_summary(summary),
            "positive_rate": (
                float(summary["positive"]) / int(summary["n"])
                if int(summary["n"])
                else None
            ),
        }
        for name, summary in label_summaries.items()
    }
    feature_summary = {
        "feature_samples": feature_samples,
        "feature_dimensions": {str(k): int(v) for k, v in sorted(feature_dim_counter.items())},
        "nonfinite_feature_values": nonfinite_feature_values,
        "total_feature_values_sampled": total_feature_values,
        "feature_imputation_required": bool(nonfinite_feature_values),
        "row_shapes": {k: int(v) for k, v in sorted(row_shape_counter.items())},
        "feature_contract_versions": {k: int(v) for k, v in sorted(version_counter.items())},
    }
    status = "pass" if not blockers else "blocked"
    decision = (
        "manifest_ingestion_ready_for_owner_approved_training_runner"
        if status == "pass"
        else "fix_manifest_ingestion_before_training_runner"
    )
    next_actions = [
        "Generate or attach causal ask-entry/bid-exit labels for the protocol101-live-v1 manifest before training.",
        "Implement owner-approved training runner using this manifest ingestion path after label readiness passes.",
        "Keep Q1 as development evidence; reserve a fresh chronological protected holdout before promotion.",
        "Do not enable paper-submit until a trained or selected candidate passes replay and shadow gates.",
    ]
    packet = Protocol101FairContractTrainingDryRunV1(
        status=status,
        decision=decision,
        selected_feature_contract=expected_contract,
        model_training_executed=False,
        threshold_tuning_executed=False,
        broker_endpoint_called=False,
        split_summary=split_summary,
        feature_summary=feature_summary,
        label_summary=label_summary,
        blockers=sorted(set(blockers)),
        next_actions=next_actions,
    )
    packet_session_rows = packet.to_dict()
    packet_session_rows["session_rows"] = session_rows
    return packet, packet_session_rows


def render_report(packet: Protocol101FairContractTrainingDryRunV1) -> str:
    split = packet.split_summary
    lines = [
        "# Protocol101 Fair Contract Training Dry Run",
        "",
        "## Decision",
        "",
        f"- Status: `{packet.status}`",
        f"- Decision: `{packet.decision}`",
        f"- Feature contract: `{packet.selected_feature_contract}`",
        f"- Model training executed: `{str(packet.model_training_executed).lower()}`",
        f"- Threshold tuning executed: `{str(packet.threshold_tuning_executed).lower()}`",
        f"- Broker endpoint called: `{str(packet.broker_endpoint_called).lower()}`",
        "",
        "## Split Summary",
        "",
    ]
    for name in ("train", "validation", "diagnostic_test", "embargoed", "unassigned"):
        row = split.get(name) or {}
        lines.append(
            f"- `{name}`: sessions=`{row.get('sessions')}`, rows=`{row.get('decision_rows')}`, "
            f"tradable=`{row.get('tradable_candidates')}`."
        )
    lines.extend(
        [
            "",
            "## Feature Summary",
            "",
            f"- Feature samples: `{packet.feature_summary['feature_samples']}`",
            f"- Feature dimensions: `{packet.feature_summary['feature_dimensions']}`",
            f"- Nonfinite feature values: `{packet.feature_summary['nonfinite_feature_values']}`",
            f"- Feature imputation required: `{str(packet.feature_summary['feature_imputation_required']).lower()}`",
            f"- Row shapes: `{packet.feature_summary['row_shapes']}`",
            "",
            "## Label Summary",
            "",
        ]
    )
    for name, summary in packet.label_summary.items():
        lines.append(
            f"- `{name}`: n=`{summary['n']}`, mean=`{summary['mean']}`, "
            f"positive_rate=`{summary['positive_rate']}`."
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{item}`" for item in packet.blockers) if packet.blockers else lines.append("- None.")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in packet.next_actions)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    design = load_json(args.design)
    packet, detailed = build_dry_run(design, feature_sample_limit=int(args.feature_sample_limit))
    (args.out_dir / "summary.json").write_text(
        json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "detailed_summary.json").write_text(
        json.dumps(detailed, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(packet))
    print(
        json.dumps(
            {
                "status": packet.status,
                "decision": packet.decision,
                "feature_dimensions": packet.feature_summary["feature_dimensions"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
