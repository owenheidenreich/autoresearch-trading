"""Validate a future Protocol101 fair-contract candidate before shadow/paper gates.

This gate does not train, tune, contact vendors, call broker endpoints, change
paper defaults, or promote a model. It only evaluates whether a completed
owner-approved fair-contract training result is strong enough to proceed to the
next strict serial/lifecycle replay harness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v4.live.protocol101_synchronization import (
    Protocol101FairContractCandidateValidationGateV1,
)


DEFAULT_RUNNER_PLAN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_runner/runner_plan.json"
)
DEFAULT_TRAINING_RESULT = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_runner/training_result.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_candidate_validation_gate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-plan", type=Path, default=DEFAULT_RUNNER_PLAN)
    parser.add_argument("--training-result", type=Path, default=DEFAULT_TRAINING_RESULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-validation-trades", type=int, default=20)
    parser.add_argument("--min-diagnostic-trades", type=int, default=20)
    parser.add_argument("--min-profit-factor", type=float, default=1.25)
    return parser.parse_args()


def load_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def render_report(packet: Protocol101FairContractCandidateValidationGateV1) -> str:
    lines = [
        "# Protocol101 Fair-Contract Candidate Validation Gate",
        "",
        "## Decision",
        "",
        f"- Status: `{packet.status}`",
        f"- Decision: `{packet.decision}`",
        f"- Feature contract: `{packet.selected_feature_contract}`",
        f"- Model training executed: `{str(packet.model_training_executed).lower()}`",
        f"- Threshold selection executed: `{str(packet.threshold_selection_executed).lower()}`",
        f"- Broker endpoint called: `{str(packet.broker_endpoint_called).lower()}`",
        f"- Paper-submit allowed: `{str(packet.paper_submit_allowed).lower()}`",
        "",
        "## Checks",
        "",
    ]
    for name, check in packet.checks.items():
        lines.append(
            f"- `{name}`: value=`{check.get('value')}`, required=`{check.get('required')}`, "
            f"pass=`{str(check.get('pass')).lower()}`."
        )
    if packet.metrics:
        lines.extend(["", "## Metrics", ""])
        for split_name in ("validation", "diagnostic_test"):
            metrics = packet.metrics.get(split_name) or {}
            lines.append(
                f"- `{split_name}`: trades=`{metrics.get('trades')}`, "
                f"total_pnl=`{metrics.get('total_pnl')}`, "
                f"profit_factor=`{metrics.get('profit_factor')}`, "
                f"max_drawdown=`{metrics.get('max_drawdown')}`."
            )
        lines.append(f"- Chosen threshold: `{packet.metrics.get('chosen_threshold')}`.")
        lines.append(f"- Model artifact: `{packet.metrics.get('model_out')}`.")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in packet.next_actions)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner_plan = load_json_optional(args.runner_plan)
    training_result = load_json_optional(args.training_result)
    packet = Protocol101FairContractCandidateValidationGateV1.evaluate(
        runner_plan=runner_plan,
        training_result=training_result,
        min_validation_trades=int(args.min_validation_trades),
        min_diagnostic_trades=int(args.min_diagnostic_trades),
        min_profit_factor=float(args.min_profit_factor),
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(packet))
    print(
        json.dumps(
            {
                "status": packet.status,
                "decision": packet.decision,
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
