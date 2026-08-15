"""Execute the frozen Path-D Wave-1 exit-repair family offline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from v4.research.pathd_exit_runner import (
    PREREGISTRATION_PATH,
    PathDExitRunner,
    session_blocked_max_t,
    wave1_spec,
)
from v4.research.pathd_model_gate import acceptance_tier
from v4.research.pathd_research_loop import prior_art_check, run_wave
from v4.research.phase1_storage import ResearchRoots


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Path-D Wave 1 Exit Repair — Results",
        "",
        f"**Bottom line:** `{report['verdict']}`",
        "",
        "The protected holdout remained closed. No paid download, broker access, paper order, "
        "promotion, default change, runtime flag, or schedule mutation occurred.",
        "",
        "## Frozen registration and prior art",
        "",
        f"- Pre-registration: `{report['preregistration']['path']}`",
        f"- SHA-256: `{report['preregistration']['sha256']}`",
        "- Family size / budget: `8 / 8`",
        "- Canonical citations: `PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md` and "
        "`PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md`.",
        "",
        "| Hypothesis | Prior-art hits | Blocking hits | Status | Tier |",
        "|---|---:|---:|---|---|",
    ]
    loop_by_id = {row["hypothesis_id"]: row for row in report["loop"]["results"]}
    for identifier in report["declared_order"]:
        loop = loop_by_id.get(identifier, {})
        final = report["hypotheses"].get(identifier, {})
        lines.append(
            f"| {identifier} | {loop.get('prior_art_hits', 0)} | "
            f"{len(loop.get('prior_art_blocking', []))} | {loop.get('status', 'NOT_RUN')} | "
            f"{final.get('acceptance', {}).get('tier', 'NOT_RUN')} |"
        )
    lines.extend(("", "## Per-hypothesis gates", ""))
    for identifier in report["declared_order"]:
        row = report["hypotheses"].get(identifier)
        if row is None:
            lines.extend((f"### {identifier}", "", "Not run because the family stopped early.", ""))
            continue
        lines.extend(
            (
                f"### {identifier} — `{row['acceptance']['tier']}`",
                "",
                f"Status: `{row['diagnostics'].get('status')}`. Pooled policy "
                f"`${row['pooled_policy']:,.2f}` versus comparator "
                f"`${row['pooled_comparator']:,.2f}`; bootstrap delta LCB "
                f"`${row['bootstrap_lcb']:,.2f}`.",
                "",
                f"Fold deltas: `{json.dumps(row['fold_deltas'], sort_keys=True)}`",
                "",
                f"Negative controls accepted: `{json.dumps(row['negative_controls_accepted'], sort_keys=True)}`",
                "",
                "| Gate | Pass | Detail |",
                "|---|---:|---|",
            )
        )
        for test in row["rejection_tests"]:
            lines.append(
                f"| {test['name']} | {str(test['passed']).lower()} | "
                f"{str(test['detail']).replace('|', '/')} |"
            )
        components = row["acceptance"]["components"]
        lines.extend(
            (
                "",
                f"Acceptance components: `{json.dumps(components, sort_keys=True)}`",
                "",
                "Charter diagnostics:",
                "",
                "```json",
                json.dumps(row["diagnostics"]["charter_diagnostics"], indent=2, sort_keys=True),
                "```",
                "",
            )
        )
    lines.extend(
        (
            "## Family correction",
            "",
            "```json",
            json.dumps(report["max_t"], indent=2, sort_keys=True),
            "```",
            "",
            "## Claim boundary",
            "",
            "`TIER_A`, if present, means only worth a separately authorized forward live-paper test. "
            "It does not mean deployable. The protected historical firewall is spent.",
            "",
        )
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--entry-campaign",
        type=Path,
        default=Path("/Volumes/AR_TRADING_DATA/artifacts/entry_v2/campaign.json"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preregistration", type=Path, default=PREREGISTRATION_PATH)
    parser.add_argument("--volume-name", default="AR_TRADING_DATA")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preregistration = args.preregistration.resolve(strict=True)
    preregistration_text = preregistration.read_text()
    if "**Status:** `FROZEN_BEFORE_FIT`" not in preregistration_text:
        raise SystemExit("Wave-1 preregistration is not frozen")
    roots = ResearchRoots.resolve(
        data_root=args.data_root,
        scratch_root=args.scratch_root,
        artifact_root=args.artifact_root,
    )
    output_root = (
        args.output_root.resolve(strict=False)
        if args.output_root is not None
        else roots.artifact_root / "pathd_wave1_exit_repair_2026_08_03"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    freeze = {
        "schema_version": "pathd.wave1-preregistration-freeze.v1",
        "status": "FROZEN_BEFORE_FIT",
        "path": str(preregistration),
        "sha256": _sha256(preregistration),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "declared_family_size": 8,
        "budget": 8,
    }
    (output_root / "preregistration_freeze_receipt.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    )
    spec = wave1_spec()
    prior_art = {
        hypothesis.hypothesis_id: [asdict(hit) for hit in prior_art_check(hypothesis.mechanism)]
        for hypothesis in spec.hypotheses
    }
    runner = PathDExitRunner(
        roots=roots,
        entry_campaign_path=args.entry_campaign,
        output_root=output_root,
        expected_volume_name=args.volume_name,
    )
    loop = run_wave(spec, runner, registry_path=output_root / "hypothesis_registry.jsonl")
    invalid = loop["verdict"] == "INVALID"
    complete_family = loop["budget_spent"] == 8 and not invalid
    max_t = (
        session_blocked_max_t(runner.session_deltas)
        if complete_family
        else {
            "method": "session_blocked_max_t",
            "status": "NOT_EVALUATED_INVALID_OR_INCOMPLETE_FAMILY",
            "declared_family_size": 8,
            "trained_member_count": len(runner.session_deltas),
        }
    )
    hypotheses: dict[str, Any] = {}
    tiers: list[str] = []
    for hypothesis in spec.hypotheses:
        identifier = hypothesis.hypothesis_id
        if identifier not in runner.results:
            continue
        result = runner.results[identifier]
        survived = bool(max_t.get("survived", {}).get(identifier, False))
        acceptance = acceptance_tier(
            pooled_policy=result.pooled_policy,
            pooled_comparator=result.pooled_comparator,
            fold_deltas=result.fold_deltas,
            bootstrap_lcb=result.bootstrap_lcb,
            negative_controls_accepted=result.negative_controls_accepted,
            rejection_tests=result.rejection_tests,
            maxt_survived=survived,
            concentrated=result.concentrated,
        )
        tiers.append(str(acceptance["tier"]))
        hypotheses[identifier] = {
            "mechanism": hypothesis.mechanism,
            "params": dict(hypothesis.params),
            "prior_art": prior_art[identifier],
            "pooled_policy": result.pooled_policy,
            "pooled_comparator": result.pooled_comparator,
            "fold_deltas": dict(result.fold_deltas),
            "bootstrap_lcb": result.bootstrap_lcb,
            "negative_controls_accepted": dict(result.negative_controls_accepted),
            "rejection_tests": [test.as_dict() for test in result.rejection_tests],
            "diagnostics": dict(result.diagnostics),
            "max_t_p_one_sided": max_t.get("p_values", {}).get(identifier),
            "acceptance": acceptance,
        }
    if invalid or "INVALID" in tiers:
        verdict = "INVALID"
    elif "TIER_A" in tiers:
        verdict = "TIER_A"
    elif "TIER_B" in tiers:
        verdict = "TIER_B"
    else:
        verdict = "NO_EDGE"
    report = {
        "schema_version": "pathd.wave1-exit-repair-results.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "preregistration": freeze,
        "declared_order": [hypothesis.hypothesis_id for hypothesis in spec.hypotheses],
        "loop": loop,
        "max_t": max_t,
        "hypotheses": hypotheses,
        "protected_holdout_opened": False,
        "paid_download_executed": False,
        "broker_accessed": False,
        "paper_order_submitted": False,
        "promotion_or_default_changed": False,
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), default=_json).encode()
    ).hexdigest()
    (output_root / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json) + "\n"
    )
    (output_root / "report.md").write_text(_markdown(report))
    print(json.dumps({"verdict": verdict, "output_root": str(output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
