"""Issue a model-free known-answer/smoke receipt for the day simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.ops.causal_day_simulator import Action, DecisionState, simulate_session
from v5.ops.measure_fill_quality import QUOTE_CORPUS


def abstain(state: DecisionState) -> Action:
    return Action("HOLD" if state.position is not None else "ABSTAIN", reason="smoke_no_trade")


def run(
    quote_root: Path,
    coverage_csv: Path,
    minute_features_path: Path,
    simulator_path: Path,
    declaration_path: Path,
    settlement_receipt_path: Path,
    evidence_dir: Path,
) -> dict:
    if evidence_dir.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {evidence_dir}")
    coverage = pd.read_csv(coverage_csv)
    settlement_receipt = json.loads(settlement_receipt_path.read_text())
    settlement_validation = settlement_receipt.get("validation", {})
    if (
        settlement_receipt.get("schema_version")
        != "v5.causal-day-terminal-settlement-audit.v1"
        or settlement_validation.get("status") != "VALIDATED_FOR_TERMINAL_ACCOUNTING"
        or settlement_validation.get("mismatches") != 0
        or settlement_receipt.get("sessions", {}).get("all_exact_identity") is not True
    ):
        raise RuntimeError("terminal settlement receipt is not validated for accounting")
    sessions = coverage.loc[coverage["included_for_episode_build"], "session"].astype(str).tolist()
    chosen = [sessions[0], sessions[len(sessions) // 2], sessions[-1]]
    features = pd.read_parquet(minute_features_path)
    rows = []
    for session in chosen:
        quotes = pd.read_parquet(
            quote_root / f"databento_spxw_0dte_{session}.parquet"
        )
        for risk_mode in ("ticket_only", "ticket_and_breaker"):
            for cap in (1, 2, 3):
                result = simulate_session(
                    quotes,
                    session,
                    abstain,
                    trade_cap=cap,
                    risk_mode=risk_mode,
                    minute_features=features,
                )
                assert result.trades.empty
                assert result.ending_cash_usd == 10_000.0
                assert result.realised_pnl_usd == 0.0
                assert not result.blocked_terminal_position
                assert len(result.events) == 386
                assert result.events["role"].isin(["morning_entry", "afternoon_entry"]).all()
                rows.append(
                    {
                        "session": session,
                        "risk_mode": risk_mode,
                        "trade_cap": cap,
                        "events": len(result.events),
                        "trades": len(result.trades),
                        "ending_cash_usd": result.ending_cash_usd,
                    }
                )

    evidence_dir.mkdir(parents=True, exist_ok=False)
    smoke = pd.DataFrame(rows)
    smoke_path = evidence_dir / "no_trade_smoke.csv"
    smoke.to_csv(smoke_path, index=False)
    receipt = {
        "schema_version": "v5.causal-day-simulator.v3",
        "created_on": "2026-08-14",
        "purpose": "deterministic no-trade real-data smoke plus known-answer test identity",
        "simulator_path": str(simulator_path),
        "simulator_sha256": file_sha256(simulator_path),
        "declaration_sha256": file_sha256(declaration_path),
        "terminal_settlement_receipt": {
            "path": str(settlement_receipt_path),
            "sha256": file_sha256(settlement_receipt_path),
            "receipt_sha256": settlement_receipt["receipt_sha256"],
            "status": settlement_validation["status"],
        },
        "sessions": chosen,
        "smoke_cells": len(smoke),
        "risk_modes": ["ticket_only", "ticket_and_breaker"],
        "trade_caps": [1, 2, 3],
        "assertions": {
            "events_per_session": 386,
            "no_trade_ending_cash_usd": 10000.0,
            "trades": 0,
            "blocked_terminal_positions": 0,
            "only_flat_entry_roles_seen": True,
        },
        "known_answer_tests": [
            "origin regime owns exit across 12:46",
            "first later bid, never later-best bid",
            "unvalidated terminal settlement blocks score",
            "validated settlement is labelled and uses intrinsic",
            "zero recovery is a labelled sensitivity, not a bid",
            "ask-in/bid-out fee charged once",
            "trade ledger tracks OTM-to-ITM conversion and underlying excursions",
            "overlapping buy rejected",
            "trade cap enforced",
            "120-minute maximum hold enforced",
            "policy sees the whole live chain while entry actions stay in the declared OTM band",
            "held contract remains visible separately when its quote becomes one-sided",
            "replay captures every considered contract and declared probability",
            "ineligible contract probabilities are rejected",
            "stateful policy resets at every session boundary",
            "stateful policy without an episode-reset contract is refused",
        ],
        "known_answer_test_count": 16,
        "terminal_primary_status": "VALIDATED_CASH_SETTLEMENT_WHEN_NO_EXECUTABLE_BID_REMAINS",
        "model_fit": False,
        "artifacts": {
            "no_trade_smoke": {
                "path": str(smoke_path),
                "sha256": file_sha256(smoke_path),
                "size_bytes": smoke_path.stat().st_size,
            }
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    (evidence_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(evidence_dir / "receipt.json")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", type=Path, default=QUOTE_CORPUS)
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/causal_day_trader_coverage_2026_08_14_attempt002/"
            "session_coverage.csv"
        ),
    )
    parser.add_argument(
        "--minute-features",
        type=Path,
        default=Path("/Volumes/AR_TRADING_DATA/derived/causal_day_trader_v1/minutes.parquet"),
    )
    parser.add_argument(
        "--simulator-path", type=Path, default=Path("v5/ops/causal_day_simulator.py")
    )
    parser.add_argument(
        "--declaration", type=Path, default=Path("v5/work/entry-exit-attribution/DECLARATION_V2.json")
    )
    parser.add_argument(
        "--settlement-receipt",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/causal_day_terminal_settlement_2026_08_14/receipt.json"
        ),
    )
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.quotes,
        args.coverage_csv,
        args.minute_features,
        args.simulator_path,
        args.declaration,
        args.settlement_receipt,
        args.evidence_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
