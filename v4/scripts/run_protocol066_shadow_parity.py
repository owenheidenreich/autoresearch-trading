from __future__ import annotations

import argparse
import json
from pathlib import Path

from v4.live.shadow_parity import (
    ShadowParityConfig,
    load_shadow_observations,
    summarize_shadow_parity,
    write_shadow_template,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_070_protocol066_shadow_parity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", default="protocol066")
    parser.add_argument("--protocol-label", default="Protocol 066")
    parser.add_argument("--shadow-log", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-quote-age-ms", type=int, default=1500)
    parser.add_argument("--max-context-age-ms", type=int, default=5000)
    return parser.parse_args()


def _write_report(
    out_dir: Path,
    summary: dict,
    *,
    shadow_log: Path | None,
    template_path: Path,
    protocol_label: str,
) -> Path:
    lines = [
        f"# {protocol_label} No-Order Shadow Parity",
        "",
        "No paid data was downloaded. No broker order endpoint is called by this harness.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Shadow log: `{shadow_log if shadow_log else 'not supplied'}`",
        f"- Rows: `{summary['rows']}`",
        f"- Passed rows: `{summary['passed_rows']}`",
        f"- Failed rows: `{summary['failed_rows']}`",
        f"- Warnings: `{summary['warning_count']}`",
        f"- Template: `{template_path}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in summary["checks"]:
        lines.append(f"| {check['name']} | {check['status']} | {check['detail']} |")
    if summary.get("failed_rows"):
        lines.extend(["", "## First Failures", ""])
        for result in summary["row_results"][:20]:
            if result["errors"]:
                lines.append(f"- Row {result['row_index']}: {result['errors']}")
    if not shadow_log:
        lines.extend(
            [
                "",
                "## Decision",
                "",
                "Blocked until a live no-order shadow JSONL is captured and passes parity. This is the correct state before paper trading.",
            ]
        )
    elif summary["status"] == "pass":
        lines.extend(
            [
                "",
                "## Decision",
                "",
                "Shadow parity passed for the supplied log. This still does not approve broker-connected paper trading by itself.",
            ]
        )
    else:
        lines.extend(["", "## Decision", "", "Shadow parity is not clean enough for the next promotion gate."])

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = ShadowParityConfig(
        protocol_id=args.protocol_id,
        max_quote_age_ms=args.max_quote_age_ms,
        max_context_age_ms=args.max_context_age_ms,
    )
    template_path = args.out_dir / "shadow_observation_template.jsonl"
    write_shadow_template(template_path, config)

    if args.shadow_log:
        rows = load_shadow_observations(args.shadow_log)
        summary = summarize_shadow_parity(rows, config=config)
    else:
        summary = {
            "status": "blocked",
            "protocol_id": config.protocol_id,
            "rows": 0,
            "passed_rows": 0,
            "failed_rows": 0,
            "warning_count": 0,
            "action_counts": {},
            "position_state_counts": {},
            "checks": [
                {
                    "name": "live_shadow_feed",
                    "status": "blocker",
                    "detail": "No no-order shadow JSONL supplied.",
                }
            ],
            "row_results": [],
        }

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report_path = _write_report(
        args.out_dir,
        summary,
        shadow_log=args.shadow_log,
        template_path=template_path,
        protocol_label=args.protocol_label,
    )
    print(report_path)


if __name__ == "__main__":
    main()
