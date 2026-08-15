"""Produce structurally selected, non-model replay examples.

These examples prove the replay plumbing on owned sessions.  Sessions are
selected only from coverage position (first/middle/last with a 09:35 candidate,
plus the first 09:35 no-contract day), never from profit or path outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256, minute_number
from v5.ops.causal_day_simulator import Action, DecisionState, simulate_session
from v5.ops.export_causal_day_replay import export_replay
from v5.research.causal_day_reporting import summarize_results


class Fixed0935ClockPolicy:
    """Known-answer plumbing policy: nearest OTM at 09:35, sell after 60m."""

    def __call__(self, state: DecisionState) -> Action:
        if state.position is not None:
            held = minute_number(state.minute) - minute_number(state.position.entry_minute)
            if held >= 60:
                return Action(
                    "SELL",
                    reason="known_answer_60m_clock",
                    probability=1.0,
                    diagnostics={"policy_kind": "fixed_clock_fixture", "minutes_held": held},
                )
            return Action(
                "HOLD",
                reason="known_answer_wait_for_60m",
                probability=1.0,
                diagnostics={"policy_kind": "fixed_clock_fixture", "minutes_held": held},
            )
        if state.minute != "09:35" or state.entry_candidates.empty:
            return Action(
                "ABSTAIN",
                reason="known_answer_not_entry_boundary_or_no_contract",
                probability=1.0,
                diagnostics={"policy_kind": "fixed_clock_fixture"},
            )
        ordered = state.entry_candidates.sort_values(
            ["moneyness_itm_points", "contract_id"], ascending=[False, True]
        )
        contract_id = str(ordered.iloc[0]["contract_id"])
        probabilities = {
            str(value): float(str(value) == contract_id)
            for value in state.entry_candidates["contract_id"]
        }
        return Action(
            "BUY",
            contract_id,
            reason="known_answer_nearest_otm_at_0935",
            probability=1.0,
            candidate_probabilities=probabilities,
            diagnostics={"policy_kind": "fixed_clock_fixture", "selection": "nearest_otm"},
        )


def structurally_selected_sessions(coverage: pd.DataFrame) -> list[str]:
    included = coverage[coverage["included_for_episode_build"].astype(bool)].copy()
    with_contract = included[included["first_decision_eligible_contracts"].gt(0)][
        "session"
    ].astype(str).tolist()
    without_contract = included[included["first_decision_eligible_contracts"].eq(0)][
        "session"
    ].astype(str).tolist()
    if len(with_contract) < 3 or not without_contract:
        raise RuntimeError("coverage cannot supply the declared structural replay examples")
    selected = [
        with_contract[0],
        with_contract[len(with_contract) // 2],
        with_contract[-1],
        without_contract[0],
    ]
    if len(set(selected)) != 4:
        raise RuntimeError("structural replay selection produced duplicate sessions")
    return selected


def run(quote_root: Path, coverage_csv: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {out_dir}")
    coverage = pd.read_csv(coverage_csv)
    sessions = structurally_selected_sessions(coverage)
    out_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    results = []
    for session in sessions:
        quote_path = quote_root / f"databento_spxw_0dte_{session}.parquet"
        quotes = pd.read_parquet(quote_path)
        result = simulate_session(
            quotes,
            session,
            Fixed0935ClockPolicy(),
            trade_cap=1,
            capture_replay_state=True,
        )
        results.append(result)
        replay_dir = out_dir / session
        replay = export_replay(
            result,
            replay_dir,
            policy_source="known_answer_nearest_otm_0935_plus_60m_clock",
            out_of_fold=False,
        )
        rows.append(
            {
                "session": session,
                "selection_basis": (
                    "first_0935_no_contract_session"
                    if int(
                        coverage.loc[
                            coverage["session"].astype(str).eq(session),
                            "first_decision_eligible_contracts",
                        ].iloc[0]
                    )
                    == 0
                    else "coverage_position_among_0935_eligible_sessions"
                ),
                "trades": int(len(result.trades)),
                "realised_pnl_usd": float(result.realised_pnl_usd),
                "blocked_terminal": bool(result.blocked_terminal_position),
                "replay_receipt": str(replay_dir / "receipt.json"),
                "replay_receipt_sha256": file_sha256(replay_dir / "receipt.json"),
                "considered_ladder_rows": replay["considered_ladder_rows"],
            }
        )
        sources.append(
            {
                "session": session,
                "path": str(quote_path),
                "size_bytes": quote_path.stat().st_size,
                "sha256": file_sha256(quote_path),
            }
        )

    receipt: dict[str, Any] = {
        "schema_version": "v5.causal-day-replay-plumbing.v2",
        "created_on": "2026-08-14",
        "purpose": "prove full minute/ladder replay export without fitting or outcome-selected storytelling",
        "selection_law": {
            "traded_examples": "first, middle and last included session with a causal 09:35 candidate",
            "abstention_example": "first included session without a causal 09:35 candidate",
            "outcome_used_for_selection": False,
        },
        "policy": "nearest OTM at 09:35 when available; fixed 60-minute exit",
        "examples": rows,
        "accounting_metric_smoke": summarize_results(results),
        "implementation_hashes": {
            str(path): file_sha256(path)
            for path in (
                Path("v5/work/entry-exit-attribution/DECLARATION_V2.json"),
                Path("v5/ops/causal_day_simulator.py"),
                Path("v5/ops/export_causal_day_replay.py"),
                Path("v5/research/causal_day_reporting.py"),
            )
        },
        "sources": sources,
        "fit_performed": False,
        "model_skill_claim": False,
        "completion_limit": (
            "These are implementation examples, not the required win/loss/abstention replays from a "
            "chronological fitted policy."
        ),
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    path = out_dir / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quotes",
        type=Path,
        default=Path(
            "/Users/och/.autoresearch-trading/pathd_2025-08-01_2026-07-31/"
            "aligned/normalized"
        ),
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/causal_day_trader_coverage_2026_08_14_attempt002/"
            "session_coverage.csv"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.quotes, args.coverage_csv, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
