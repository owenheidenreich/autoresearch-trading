"""CLI for compiling and running the first autoresearch_v2 development suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .builtin import REFERENCE_EXIT, SHORT25_EXIT, write_builtin_hypotheses
from .cache import PredictionCache
from .compiler import ExperimentCompileError, canonical_json, load_and_compile
from .dataset import (
    DEFAULT_SESSION_MANIFEST,
    ROOT,
    SIGNED17,
    freeze_foundation,
    load_development_frame,
    rolling_folds,
)
from .models import fit_oof
from .registry import DuplicateSemanticHypothesis, read_registry, register
from .replay import (
    PairedReplayMismatch,
    paired_lockstep_component_replay,
    paired_lockstep_exit_replay,
    replay_absolute,
    serial_summary,
)
from .screens import (
    contract_screen,
    direction_screen,
    exit_screen,
    select_intents,
    target_screen,
    timing_screen,
)
from .statistics import paired_summary, session_blocked_max_t


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True) + "\n")


def engine_source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _status_from_screen(summary: Mapping[str, Any], spec: Any) -> str:
    if int(summary.get("n_sessions") or 0) < 20:
        return "UNDERPOWERED"
    if float(summary.get("mde80") or float("inf")) > spec.maximum_mde_dollars_per_session:
        return "UNDERPOWERED"
    if (
        float(summary.get("mean") or 0.0) < spec.practical_effect_dollars_per_session
        or float(summary.get("maxT_p_one_sided") or 1.0) > spec.alpha
    ):
        return "NO_INCREMENTAL_EDGE"
    return "PROVISIONAL_EDGE"


def _finalize_with_serial(
    preliminary: str,
    serial: Mapping[str, Any],
    spec: Any,
    *,
    exit_component: bool,
) -> str:
    if preliminary != "PROVISIONAL_EDGE" and not exit_component:
        return preliminary
    summary = serial["statistics"]
    if int(summary.get("n_sessions") or 0) < 20:
        return "UNDERPOWERED"
    if float(summary.get("mde80") or float("inf")) > spec.maximum_mde_dollars_per_session:
        return "UNDERPOWERED"
    if (
        float(summary.get("mean") or 0.0) >= spec.practical_effect_dollars_per_session
        and float(summary.get("p_one_sided") or 1.0) <= spec.alpha
    ):
        return "PROVISIONAL_EDGE"
    return "EXIT_ARTIFACT" if exit_component else "NO_INCREMENTAL_EDGE"


def _invalid_semantic_hash(payload: Mapping[str, Any]) -> str:
    mechanics = dict(payload)
    mechanics.pop("hypothesis_id", None)
    mechanics.pop("claim", None)
    return hashlib.sha256(canonical_json(mechanics).encode()).hexdigest()


def _record_invalid(
    registry_path: Path,
    payload: Mapping[str, Any],
    *,
    result_path: Path,
    engine_source_hash: str,
    allow_reexecution: bool,
) -> None:
    row = {
        "schema_version": "autoresearch_v2.semantic_registry.v1",
        "semantic_hash": _invalid_semantic_hash(payload),
        "hypothesis_id": payload.get("hypothesis_id", "unknown"),
        "component": payload.get("component", "unknown"),
        "epoch_id": payload.get("development_epoch", {}).get("epoch_id", "unknown"),
        "status": "INVALID_EXPERIMENT",
        "result_path": str(result_path),
        "engine_source_hash": engine_source_hash,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    existing = registry_path.read_text().splitlines() if registry_path.exists() else []
    duplicate = next(
        (
            json.loads(line)
            for line in existing
            if line and json.loads(line).get("semantic_hash") == row["semantic_hash"]
        ),
        None,
    )
    if duplicate is not None and not allow_reexecution:
        raise DuplicateSemanticHypothesis("invalid semantic hypothesis already registered")
    row["reexecution_of"] = duplicate.get("result_path") if duplicate else None
    row["previous_status"] = duplicate.get("status") if duplicate else None
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def initialize(*, foundation: Path, hypotheses: Path) -> dict[str, Any]:
    paths = write_builtin_hypotheses(hypotheses)
    frozen = freeze_foundation(foundation, session_manifest=DEFAULT_SESSION_MANIFEST)
    return {
        "foundation": str(foundation),
        "foundation_sha256": frozen["foundation_sha256"],
        "session_count": frozen["session_count"],
        "hypotheses": [str(path) for path in paths],
    }


def run_suite(
    *,
    foundation: Path,
    hypotheses: Path,
    output: Path,
    cache_root: Path | None = None,
    semantic_registry: Path | None = None,
    allow_semantic_rerun: bool = False,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    registry_path = semantic_registry or (
        ROOT / "v4/audit/autoresearch/autoresearch_v2_semantic_registry.jsonl"
    )
    source_hash = engine_source_hash()
    compiled: dict[str, Any] = {}
    invalid_results: dict[str, Any] = {}
    invalid_payloads: dict[str, Mapping[str, Any]] = {}
    for path in sorted(hypotheses.glob("*.json")):
        payload = json.loads(path.read_text())
        try:
            item = load_and_compile(path)
        except ExperimentCompileError as exc:
            name = str(payload.get("hypothesis_id", path.stem))
            result = {
                "hypothesis_id": name,
                "status": "INVALID_EXPERIMENT",
                "errors": list(exc.errors),
                "holdout_access_count": 0,
                "engine_source_hash": source_hash,
            }
            result_path = output / "results" / f"{name}.json"
            write_json(result_path, result)
            invalid_results[name] = result
            invalid_payloads[name] = payload
            continue
        compiled[item.spec.hypothesis_id] = item
        write_json(output / "compiled" / f"{item.spec.hypothesis_id}.json", item.payload())
    required = {
        "exit_short25_vs_frozen120_v1",
        "entry_timing_now_wait5_wait15_v1",
        "entry_direction_call_vs_put_v1",
        "entry_contract_ranking_full_epoch_v1",
        "entry_multi_policy_path_target_v1",
    }
    if not required.issubset(compiled):
        raise RuntimeError("one or more executable built-in hypotheses failed compilation")
    prior_hashes = {row["semantic_hash"] for row in read_registry(registry_path)}
    proposed_hashes = {item.semantic_hash for item in compiled.values()} | {
        _invalid_semantic_hash(payload) for payload in invalid_payloads.values()
    }
    duplicates = sorted(prior_hashes & proposed_hashes)
    if duplicates and not allow_semantic_rerun:
        raise DuplicateSemanticHypothesis(
            "semantic hypothesis already exists in the global registry: "
            + ",".join(duplicates)
        )

    frame, foundation_payload = load_development_frame(foundation)
    folds = rolling_folds(frame["session"].unique())
    if len(folds) != 5 or any(set(fold["model_fit"]) & set(fold["outer_test"]) for fold in folds):
        raise RuntimeError("paired five-fold construction failed")
    resolved_cache_root = cache_root or (output / "oof_cache")
    cache = PredictionCache(resolved_cache_root)
    reference_spec = compiled["exit_short25_vs_frozen120_v1"].spec
    reference_oof, reference_receipt = fit_oof(
        frame,
        spec=reference_spec,
        folds=folds,
        foundation_hash=foundation_payload["foundation_sha256"],
        cache=cache,
    )
    target_spec = compiled["entry_multi_policy_path_target_v1"].spec
    target_oof, target_receipt = fit_oof(
        frame,
        spec=target_spec,
        folds=folds,
        foundation_hash=foundation_payload["foundation_sha256"],
        cache=cache,
    )
    threshold = float(reference_spec.threshold.value)
    intents = select_intents(reference_oof, threshold=threshold, policy=REFERENCE_EXIT)

    contrasts: dict[str, dict[str, float]] = {}
    pairs: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    contrasts["exit_short25"], pairs["exit_short25"], metadata["exit_short25"] = exit_screen(
        intents, arm_policy=SHORT25_EXIT, baseline_policy=REFERENCE_EXIT
    )
    contrasts["direction"], pairs["direction"], metadata["direction"] = direction_screen(
        reference_oof, threshold=threshold, policy=REFERENCE_EXIT
    )
    contrasts["contract_ranking"], pairs["contract_ranking"], metadata["contract_ranking"] = contract_screen(
        reference_oof, threshold=threshold, policy=REFERENCE_EXIT
    )
    contrasts["wait_5m"], pairs["wait_5m"], metadata["wait_5m"] = timing_screen(
        reference_oof, intents, delay_minutes=5, policy=REFERENCE_EXIT
    )
    contrasts["wait_15m"], pairs["wait_15m"], metadata["wait_15m"] = timing_screen(
        reference_oof, intents, delay_minutes=15, policy=REFERENCE_EXIT
    )
    contrasts["multi_policy_target"], pairs["multi_policy_target"], metadata["multi_policy_target"] = target_screen(
        target_oof,
        reference_oof,
        threshold=threshold,
        policy=REFERENCE_EXIT,
    )
    max_permutations = min(
        compiled[name].spec.max_permutations for name in required
    )
    corrected = session_blocked_max_t(
        contrasts, permutations=max_permutations, seed=101
    )
    write_json(output / "contrasts" / "session_deltas.json", contrasts)

    results: dict[str, Any] = dict(invalid_results)
    mapping = {
        "exit_short25_vs_frozen120_v1": ["exit_short25"],
        "entry_direction_call_vs_put_v1": ["direction"],
        "entry_contract_ranking_full_epoch_v1": ["contract_ranking"],
        "entry_timing_now_wait5_wait15_v1": ["wait_5m", "wait_15m"],
        "entry_multi_policy_path_target_v1": ["multi_policy_target"],
    }
    route_key = {
        "exit_short25_vs_frozen120_v1": "exit_short25",
        "entry_direction_call_vs_put_v1": "direction",
        "entry_contract_ranking_full_epoch_v1": "contract_ranking",
        "entry_timing_now_wait5_wait15_v1": "wait_5m",
        "entry_multi_policy_path_target_v1": "multi_policy_target",
    }
    for hypothesis_id, contrast_names in mapping.items():
        item = compiled[hypothesis_id]
        primary = route_key[hypothesis_id]
        preliminary = _status_from_screen(corrected[primary], item.spec)
        result: dict[str, Any] = {
            "schema_version": "autoresearch_v2.result.v1",
            "hypothesis_id": hypothesis_id,
            "semantic_hash": item.semantic_hash,
            "component": item.spec.component,
            "status": preliminary,
            "development_epoch": item.spec.epoch_id,
            "holdout_access_count": 0,
            "protected_holdout_opened": False,
            "engine_source_hash": source_hash,
            "semantic_history_match": item.semantic_hash in prior_hashes,
            "screen": {
                name: {
                    "statistics": corrected[name],
                    "metadata": metadata[name],
                }
                for name in contrast_names
            },
            "family_correction": {
                "method": "session_blocked_sign_flip_maxT",
                "family": sorted(contrasts),
                "permutations": max_permutations,
            },
        }
        try:
            if hypothesis_id == "exit_short25_vs_frozen120_v1":
                paired_serial = paired_lockstep_exit_replay(
                    intents.to_dict("records"),
                    arm_policy=SHORT25_EXIT,
                    baseline_policy=REFERENCE_EXIT,
                )
                arm_trades, arm_state = replay_absolute(
                    intents.to_dict("records"), policy=SHORT25_EXIT, strategy="autoresearch_v2:absolute_short25"
                )
                base_trades, base_state = replay_absolute(
                    intents.to_dict("records"), policy=REFERENCE_EXIT, strategy="autoresearch_v2:absolute_reference120"
                )
                serial = {
                    "mode": "matched_lockstep_two_one_account_arms",
                    "statistics": paired_summary(paired_serial["session_deltas"]),
                    "paired": {key: value for key, value in paired_serial.items() if key != "trade_rows"},
                    "absolute_arm": serial_summary(arm_trades, arm_state),
                    "absolute_baseline": serial_summary(base_trades, base_state),
                }
                write_json(output / "trades" / "exit_short25_vs_frozen120_v1.json", paired_serial["trade_rows"])
                result["serial_replay"] = serial
                result["status"] = _finalize_with_serial(
                    preliminary, serial, item.spec, exit_component=True
                )
            elif preliminary == "PROVISIONAL_EDGE":
                replay = paired_lockstep_component_replay(
                    pairs[primary],
                    policy=REFERENCE_EXIT,
                    arm_name=item.spec.primary_contrast,
                    baseline_name=item.spec.baseline_name,
                )
                serial = {
                    "mode": "matched_lockstep_two_one_account_arms",
                    "statistics": paired_summary(replay["session_deltas"]),
                    "paired": {key: value for key, value in replay.items() if key != "session_deltas"},
                }
                result["serial_replay"] = serial
                result["status"] = _finalize_with_serial(
                    preliminary, serial, item.spec, exit_component=False
                )
            else:
                result["serial_replay"] = {
                    "routed": False,
                    "reason": "cheap paired component screen did not survive family correction and power/effect bars",
                }
        except (PairedReplayMismatch, ValueError, RuntimeError) as exc:
            result["status"] = "MECHANICAL_FAILURE"
            result["serial_replay"] = {"routed": True, "error": str(exc)}
        result_path = output / "results" / f"{hypothesis_id}.json"
        write_json(result_path, result)
        results[hypothesis_id] = result

    for hypothesis_id in mapping:
        result_path = output / "results" / f"{hypothesis_id}.json"
        register(
            registry_path,
            compiled[hypothesis_id],
            status=results[hypothesis_id]["status"],
            result_path=str(result_path),
            engine_source_hash=source_hash,
            allow_reexecution=allow_semantic_rerun,
        )
    for hypothesis_id, payload in invalid_payloads.items():
        _record_invalid(
            registry_path,
            payload,
            result_path=output / "results" / f"{hypothesis_id}.json",
            engine_source_hash=source_hash,
            allow_reexecution=allow_semantic_rerun,
        )

    suite = {
        "schema_version": "autoresearch_v2.suite_result.v1",
        "status": "complete",
        "evidence_grade": "development_only_non_promotable",
        "foundation_sha256": foundation_payload["foundation_sha256"],
        "foundation_session_count": foundation_payload["session_count"],
        "foundation_first_session": foundation_payload["first_session"],
        "foundation_last_session": foundation_payload["last_session"],
        "holdout_access_count": 0,
        "protected_holdout_opened": False,
        "engine_source_hash": source_hash,
        "semantic_registry": str(registry_path),
        "semantic_rerun_explicitly_allowed": bool(allow_semantic_rerun),
        "oof_cache_root": str(resolved_cache_root),
        "folds": folds,
        "reference_oof": reference_receipt,
        "multi_policy_oof": target_receipt,
        "terminal_status_counts": {
            status: sum(result["status"] == status for result in results.values())
            for status in (
                "INVALID_EXPERIMENT",
                "MECHANICAL_FAILURE",
                "UNDERPOWERED",
                "NO_INCREMENTAL_EDGE",
                "EXIT_ARTIFACT",
                "PROVISIONAL_EDGE",
                "CONFIRMED_EDGE",
            )
        },
        "results": {
            name: {
                "status": result["status"],
                "component": result.get("component"),
            }
            for name, result in sorted(results.items())
        },
    }
    write_json(output / "suite_result.json", suite)
    lines = [
        "# autoresearch_v2 development suite",
        "",
        f"- Evidence: `{suite['evidence_grade']}`",
        f"- Foundation: `{suite['foundation_sha256']}` ({suite['foundation_session_count']} sessions)",
        f"- Holdout opens: `{suite['holdout_access_count']}`",
        f"- OOF fits: `{reference_receipt['fit_count'] + target_receipt['fit_count']}` (cache hits `{reference_receipt['cache_hit_count'] + target_receipt['cache_hit_count']}`)",
        "",
        "| Hypothesis | Component | Status |",
        "|---|---|---|",
    ]
    for name, result in sorted(results.items()):
        lines.append(f"| `{name}` | `{result.get('component', 'compile')}` | `{result['status']}` |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("initialize")
    init.add_argument("--foundation", type=Path, required=True)
    init.add_argument("--hypotheses", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--foundation", type=Path, required=True)
    run.add_argument("--hypotheses", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--cache", type=Path)
    run.add_argument("--registry", type=Path)
    run.add_argument("--allow-semantic-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "initialize":
        print(json.dumps(initialize(foundation=args.foundation, hypotheses=args.hypotheses), indent=2))
    else:
        print(
            json.dumps(
                run_suite(
                    foundation=args.foundation,
                    hypotheses=args.hypotheses,
                    output=args.output,
                    cache_root=args.cache,
                    semantic_registry=args.registry,
                    allow_semantic_rerun=args.allow_semantic_rerun,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
