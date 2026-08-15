"""Protocol 089: no-order shadow-paper ledger for frozen Protocol 081.

This is an offline/live-capture promotion-readiness harness. It reads router
JSONL, validates no-order shadow parity, and reconstructs one-contract lifecycle
accounting without calling broker order endpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v4.live.shadow_parity import ShadowParityConfig, load_shadow_observations, summarize_shadow_parity
from v4.sim.shadow_paper import ShadowPaperConfig, replay_shadow_paper


DEFAULT_SHADOW_LOG = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_088_protocol081_live_shadow_router/offline_router_shadow_observations.jsonl"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_089_protocol081_shadow_paper_ledger")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", default="protocol081")
    parser.add_argument("--protocol-label", default="Protocol 081")
    parser.add_argument("--shadow-log", type=Path, default=DEFAULT_SHADOW_LOG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-quote-age-ms", type=int, default=1500)
    parser.add_argument("--max-context-age-ms", type=int, default=5000)
    parser.add_argument("--require-all-closed", action="store_true")
    parser.add_argument("--require-terminal-final", action="store_true")
    parser.add_argument("--enforce-global-one-position", action="store_true")
    return parser.parse_args()


def _decision(parity: dict[str, Any], ledger: dict[str, Any]) -> str:
    if parity["status"] == "blocked":
        return "blocked: no shadow feed supplied"
    if parity["failed_rows"] > 0:
        return "blocked: shadow parity failed"
    if ledger["status"] == "fail":
        return "blocked: shadow-paper ledger failed safety checks"
    if ledger["open_trades"] > 0:
        return "usable_offline_rehearsal_with_open_window_warning"
    if any(check["status"] == "warn" for check in ledger["checks"]):
        return "usable_offline_rehearsal_with_warnings"
    return "pass_no_order_shadow_paper"


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.2f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    ledger = payload["ledger"]
    parity = payload["shadow_parity"]
    closed = [row for row in ledger["trade_ledgers"] if row["status"] == "closed"]
    open_rows = [row for row in ledger["trade_ledgers"] if row["status"] != "closed"]
    lines = [
        f"# {payload['protocol_label']} Shadow-Paper Ledger",
        "",
        "No paid data was downloaded. No broker order endpoint is called by this harness.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Shadow log: `{payload['shadow_log']}`",
        f"- Rows: `{ledger['rows']}`",
        f"- Trades: `{ledger['trades']}`",
        f"- Closed trades: `{ledger['closed_trades']}`",
        f"- Open at stream end: `{ledger['open_trades']}`",
        f"- Max concurrent positions observed: `{ledger['max_concurrent_positions']}`",
        "",
        "## Shadow Parity",
        "",
        f"- Status: `{parity['status']}`",
        f"- Failed rows: `{parity['failed_rows']}`",
        f"- Warnings: `{parity['warning_count']}`",
        "",
        "## Ledger Checks",
        "",
        _table(ledger["checks"], ["name", "status", "detail", "value"]),
        "",
        "## PnL Accounting",
        "",
        "```json",
        json.dumps(
            {
                "realized_pnl_closed": ledger["realized_pnl_closed"],
                "mark_pnl_all_trades": ledger["mark_pnl_all_trades"],
                "terminal_action_counts": ledger["terminal_action_counts"],
                "side_counts": ledger["side_counts"],
                "post_terminal_trades": sum(1 for row in ledger["trade_ledgers"] if row["post_terminal_row_count"] > 0),
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Closed Trade Sample",
        "",
        _table(
            closed[:20],
            [
                "trade_uid",
                "contract_id",
                "side",
                "terminal_action",
                "post_terminal_row_count",
                "entry_fill_price",
                "exit_fill_price",
                "realized_pnl",
            ],
        ),
    ]
    if open_rows:
        lines.extend(
            [
                "",
                "## Open Window Sample",
                "",
                "These trades are marked to bid at the end of the supplied shadow stream; they are not treated as realized exits.",
                "",
                _table(
                    open_rows[:20],
                    [
                        "trade_uid",
                        "contract_id",
                        "side",
                        "entry_fill_price",
                        "mark_price",
                        "mark_pnl",
                        "last_observed_time",
                    ],
                ),
            ]
        )
    if ledger["errors"] or any(row["errors"] for row in ledger["trade_ledgers"]):
        lines.extend(["", "## Errors", ""])
        for error in ledger["errors"][:20]:
            lines.append(f"- {error}")
        for row in ledger["trade_ledgers"]:
            for error in row["errors"]:
                lines.append(f"- {row['trade_uid']}: {error}")
    if ledger["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in ledger["warnings"]:
            lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "When IBKR market data is enabled, rerun this exact harness on a fresh live no-order JSONL with "
            "`--require-all-closed --enforce-global-one-position` before any broker-connected paper trading.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    rows = load_shadow_observations(args.shadow_log)
    parity_config = ShadowParityConfig(
        protocol_id=args.protocol_id,
        max_quote_age_ms=args.max_quote_age_ms,
        max_context_age_ms=args.max_context_age_ms,
    )
    parity = summarize_shadow_parity(rows, config=parity_config)
    ledger = replay_shadow_paper(
        rows,
        config=ShadowPaperConfig(
            protocol_id=args.protocol_id,
            require_all_closed=args.require_all_closed,
            require_terminal_final=args.require_terminal_final,
            enforce_global_one_position=args.enforce_global_one_position,
        ),
    )
    payload = {
        "protocol": "089_protocol081_shadow_paper_ledger",
        "protocol_id": args.protocol_id,
        "protocol_label": args.protocol_label,
        "paid_data_downloaded": False,
        "shadow_log": str(args.shadow_log),
        "shadow_parity": parity,
        "ledger": ledger,
        "decision": _decision(parity, ledger),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    (args.out_dir / "trade_ledgers.json").write_text(
        json.dumps(ledger["trade_ledgers"], indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"decision": payload["decision"], "rows": ledger["rows"], "trades": ledger["trades"]}, indent=2))
    print(args.out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
