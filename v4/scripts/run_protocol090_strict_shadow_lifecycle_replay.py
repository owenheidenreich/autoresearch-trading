"""Protocol 090: strict serial lifecycle replay for Protocol 081 shadow JSONL.

This is promotion-readiness infrastructure, not a model change. It takes the
offline router shadow stream, removes rows after terminal lifecycle decisions,
skips overlapping candidate trades, and then runs the strict no-order
shadow-paper gate that live data must pass later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v4.live.shadow_lifecycle import strict_serial_shadow_rows
from v4.live.shadow_parity import ShadowParityConfig, load_shadow_observations, summarize_shadow_parity
from v4.sim.shadow_paper import ShadowPaperConfig, replay_shadow_paper


DEFAULT_SHADOW_LOG = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_088_protocol081_live_shadow_router/offline_router_shadow_observations.jsonl"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_090_protocol081_strict_shadow_lifecycle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", default="protocol081")
    parser.add_argument("--protocol-label", default="Protocol 081")
    parser.add_argument("--shadow-log", type=Path, default=DEFAULT_SHADOW_LOG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-quote-age-ms", type=int, default=1500)
    parser.add_argument("--max-context-age-ms", type=int, default=5000)
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows) + ("\n" if rows else ""))


def _decision(parity: dict[str, Any], ledger: dict[str, Any]) -> str:
    if parity["failed_rows"] > 0 or parity["status"] == "blocked":
        return "blocked: strict shadow parity failed"
    if ledger["status"] != "pass":
        return "blocked: strict shadow-paper ledger failed"
    return "pass_strict_live_like_rehearsal"


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
    transform = payload["strict_transform"]
    lines = [
        f"# {payload['protocol_label']} Strict Shadow Lifecycle Replay",
        "",
        "No paid data was downloaded. No broker order endpoint is called by this harness.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Source shadow log: `{payload['source_shadow_log']}`",
        f"- Strict shadow log: `{payload['strict_shadow_log']}`",
        f"- Input rows/trades: `{transform['input_rows']}` / `{transform['input_trades']}`",
        f"- Output rows/trades: `{transform['output_rows']}` / `{transform['selected_trades']}`",
        f"- Overlap-skipped trades: `{transform['skipped_overlap_trades']}`",
        f"- Post-terminal rows removed: `{transform['post_terminal_rows_removed']}`",
        "",
        "## Strict Ledger Checks",
        "",
        _table(ledger["checks"], ["name", "status", "detail", "value"]),
        "",
        "## Accounting",
        "",
        "```json",
        json.dumps(
            {
                "terminal_action_counts": transform["terminal_action_counts"],
                "side_counts": ledger["side_counts"],
                "realized_pnl_closed": ledger["realized_pnl_closed"],
                "max_concurrent_positions": ledger["max_concurrent_positions"],
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Trade Sample",
        "",
        _table(
            ledger["trade_ledgers"][:20],
            ["trade_uid", "contract_id", "side", "terminal_action", "entry_fill_price", "exit_fill_price", "realized_pnl"],
        ),
        "",
        "## Interpretation",
        "",
        "This replay is a live-semantics infrastructure test over already-selected historical paths. It is not new model evidence and not paper/live approval.",
        "",
        "## Next Gate",
        "",
        "When live IBKR market data is enabled, the real no-order live JSONL must pass these same strict checks before broker-connected paper trading.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    source_rows = load_shadow_observations(args.shadow_log)
    strict = strict_serial_shadow_rows(source_rows, require_terminal=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strict_log = args.out_dir / "strict_serial_shadow_observations.jsonl"
    _write_jsonl(strict_log, strict.rows)
    parity = summarize_shadow_parity(
        strict.rows,
        config=ShadowParityConfig(
            protocol_id=args.protocol_id,
            max_quote_age_ms=args.max_quote_age_ms,
            max_context_age_ms=args.max_context_age_ms,
        ),
    )
    ledger = replay_shadow_paper(
        strict.rows,
        config=ShadowPaperConfig(
            protocol_id=args.protocol_id,
            require_all_closed=True,
            require_terminal_final=True,
            enforce_global_one_position=True,
        ),
    )
    payload = {
        "protocol": "090_protocol081_strict_shadow_lifecycle",
        "protocol_id": args.protocol_id,
        "protocol_label": args.protocol_label,
        "paid_data_downloaded": False,
        "source_shadow_log": str(args.shadow_log),
        "strict_shadow_log": str(strict_log),
        "strict_transform": strict.summary,
        "shadow_parity": parity,
        "ledger": ledger,
        "decision": _decision(parity, ledger),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "trade_ledgers.json").write_text(json.dumps(ledger["trade_ledgers"], indent=2, sort_keys=True) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "input_trades": strict.summary["input_trades"],
                "selected_trades": strict.summary["selected_trades"],
                "output_rows": strict.summary["output_rows"],
            },
            indent=2,
        )
    )
    print(args.out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
