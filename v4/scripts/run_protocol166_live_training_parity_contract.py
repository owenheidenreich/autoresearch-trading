"""Protocol166: write the live/training parity contract artifact.

This runner does not enable a new model in paper trading. It records the
contract that Protocol165 must satisfy before it can replace Protocol101 in the
live bot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v4.live.protocol166_parity_contract import default_protocol166_contract


LOOP_ID = "v4_aplus_hypothesis_166_live_training_parity_contract"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_PROTOCOL164_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_164_full_action_space_dataset/summary.json")
DEFAULT_PROTOCOL165_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_165_full_action_space_policy/summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol164-summary", type=Path, default=DEFAULT_PROTOCOL164_SUMMARY)
    parser.add_argument("--protocol165-summary", type=Path, default=DEFAULT_PROTOCOL165_SUMMARY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contract = default_protocol166_contract().to_dict()
    payload = {
        "protocol": "166_live_training_parity_contract",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "contract": contract,
        "protocol164_status": _status(args.protocol164_summary),
        "protocol165_status": _status(args.protocol165_summary),
        "operational_paper_default": "Protocol101 remains active until Protocol165 beats the frozen strict-serial promotion gate.",
        "replacement_rule": "A Protocol165 live runtime must validate candidates with v4.live.protocol166_parity_contract before paper-order enablement.",
        "decision": "contract_defined_protocol101_not_replaced",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def _status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"path": str(path), "exists": True, "status": "invalid_json"}
    return {
        "path": str(path),
        "exists": True,
        "protocol": payload.get("protocol"),
        "decision": payload.get("decision"),
        "rows": payload.get("rows"),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    contract = payload["contract"]
    lines = [
        "# Protocol166 Live/Training Parity Contract",
        "",
        "Protocol166 defines the candidate surface a future Protocol165 paper runtime must share with Protocol164 historical training.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Operational paper default: {payload['operational_paper_default']}",
        f"- Symbol/root: `{contract['symbol_root']}`",
        f"- Settlement: `{contract['settlement_style']}`",
        f"- Strike window: `ATM +/- ${contract['strike_window']:.0f}` in `${contract['strike_spacing']:.0f}` increments",
        f"- Entry/exit accounting: `{contract['entry_price']}` entry, `{contract['exit_price']}` exit, fees included `{contract['fees_included']}`",
        f"- Max contracts now: `{contract['max_contracts']}`",
        f"- Max concurrency: `{contract['max_concurrent_positions']}`",
        f"- No new entries after ET: `{contract['no_new_entries_after_et']}`",
        "",
        "## Disallowed Pre-Entry Gates",
        "",
    ]
    lines.extend(f"- `{name}`" for name in contract["disallowed_pre_entry_gates"])
    lines.extend(
        [
            "",
            "## Required Candidate Fields",
            "",
            ", ".join(f"`{field}`" for field in contract["required_candidate_fields"]),
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
