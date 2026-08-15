"""Export a human-readable causal replay from one simulator result.

The exporter is policy-agnostic.  It can document known-answer fixtures now
and, after the gates open, the exact same format can document an out-of-fold
fitted policy.  It refuses to call an artifact out-of-fold without a bound
artifact hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.ops.causal_day_simulator import SimulationResult


class ReplayExportError(RuntimeError):
    """A replay would be incomplete, overwritten, or misrepresented."""


def _display(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return ["_None._"]
    present = [column for column in columns if column in frame]
    lines = [
        "| " + " | ".join(present) + " |",
        "|" + "|".join("---" for _ in present) + "|",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_display(row[column]) for column in present) + " |")
    return lines


def export_replay(
    result: SimulationResult,
    out_dir: Path,
    *,
    policy_source: str,
    policy_artifact_sha256: str | None = None,
    out_of_fold: bool = False,
) -> dict[str, Any]:
    if out_dir.exists():
        raise ReplayExportError(f"refusing to overwrite replay: {out_dir}")
    if not policy_source.strip():
        raise ReplayExportError("policy_source is required")
    if out_of_fold and not policy_artifact_sha256:
        raise ReplayExportError("an out-of-fold replay requires a policy artifact SHA-256")
    if policy_artifact_sha256 is not None and (
        len(policy_artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in policy_artifact_sha256)
    ):
        raise ReplayExportError("policy artifact SHA-256 is malformed")

    out_dir.mkdir(parents=True, exist_ok=False)
    event_path = out_dir / "events.csv"
    trade_path = out_dir / "trades.csv"
    ladder_path = out_dir / "considered_ladder.csv"
    report_path = out_dir / "replay.md"
    result.events.to_csv(event_path, index=False)
    result.trades.to_csv(trade_path, index=False)
    result.considered_ladder.to_csv(ladder_path, index=False)

    lines = [
        f"# Causal day replay — {result.session}",
        "",
        f"- Policy source: `{policy_source}`",
        f"- Out-of-fold fitted policy: `{out_of_fold}`",
        f"- Policy artifact SHA-256: `{policy_artifact_sha256 or 'NONE'}`",
        f"- Risk reading: `{result.risk_mode}`; trade cap: `{result.trade_cap}`",
        f"- Starting equity: `${result.starting_equity_usd:,.2f}`",
        f"- Ending cash: `{_display(result.ending_cash_usd)}`",
        f"- Realised P&L: `${result.realised_pnl_usd:,.2f}`",
        f"- Blocked terminal position: `{result.blocked_terminal_position}`",
        "",
    ]
    if not out_of_fold:
        lines.extend(
            [
                "> This is an implementation/known-answer replay, not evidence of model skill or profitability.",
                "",
            ]
        )

    lines.extend(["## Trades", ""])
    lines.extend(
        _markdown_table(
            result.trades,
            [
                "entry_minute",
                "exit_minute",
                "origin_regime",
                "raw_symbol",
                "right",
                "strike",
                "entry_ask",
                "exit_bid",
                "net_pnl_usd",
                "minutes_held",
                "entry_otm_depth_points",
                "maximum_itm_depth_points",
                "final_itm_depth_points",
                "otm_to_itm_conversion",
                "time_to_cross_minutes",
                "underlying_mfe_points",
                "underlying_mae_points",
                "maximum_unrealised_usd",
                "minimum_unrealised_usd",
                "exit_reason",
                "exit_type",
            ],
        )
    )
    lines.extend(["", "## Minute-by-minute decisions", ""])
    lines.extend(
        _markdown_table(
            result.events,
            [
                "minute",
                "role",
                "action",
                "status",
                "requested_contract_id",
                "action_probability",
                "reason",
                "eligible_contracts",
                "cash_usd",
                "realised_pnl_usd",
                "policy_diagnostics_json",
            ],
        )
    )

    buys = result.events[result.events["status"].eq("filled_ask")]
    lines.extend(["", "## Ladder considered at filled entries", ""])
    if buys.empty or result.considered_ladder.empty:
        lines.append("_No filled entry with captured ladder state._")
    else:
        for _, buy in buys.iterrows():
            minute = str(buy["minute"])
            lines.extend(["", f"### {minute}", ""])
            considered = result.considered_ladder[
                result.considered_ladder["minute"].astype(str).eq(minute)
            ].copy()
            if "policy_probability" in considered:
                considered = considered.sort_values(
                    ["selected", "policy_probability"], ascending=[False, False], na_position="last"
                )
            lines.extend(
                _markdown_table(
                    considered,
                    [
                        "raw_symbol",
                        "right",
                        "strike",
                        "moneyness_itm_points",
                        "bid",
                        "ask",
                        "spread_usd",
                        "self_delta",
                        "self_gamma_dollars",
                        "self_theta_per_minute",
                        "policy_probability",
                        "selected",
                    ],
                )
            )
    report_path.write_text("\n".join(lines) + "\n")

    artifacts = {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for name, path in (
            ("events", event_path),
            ("trades", trade_path),
            ("considered_ladder", ladder_path),
            ("report", report_path),
        )
    }
    receipt: dict[str, Any] = {
        "schema_version": "v5.causal-day-replay.v2",
        "created_on": "2026-08-14",
        "session": result.session,
        "policy_source": policy_source,
        "policy_artifact_sha256": policy_artifact_sha256,
        "out_of_fold": out_of_fold,
        "risk_mode": result.risk_mode,
        "trade_cap": result.trade_cap,
        "trades": int(len(result.trades)),
        "events": int(len(result.events)),
        "considered_ladder_rows": int(len(result.considered_ladder)),
        "model_skill_claim": False if not out_of_fold else None,
        "artifacts": artifacts,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Library-backed exporter; construct SimulationResult in Python and call export_replay."
    )
    parser.parse_args()
    raise ReplayExportError("CLI import of an in-memory SimulationResult is intentionally unsupported")


if __name__ == "__main__":
    raise SystemExit(main())
