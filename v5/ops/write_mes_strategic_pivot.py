"""Write the self-hashed, no-outcome MES strategic pivot receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.mes_strategic_pivot import (
    scaled_data_cost,
    target_stop_economics,
    translate_es_spread_to_mes,
)


SPREAD_RECEIPT = Path(
    "v4/audit/autoresearch/pathd_es_spread_measurement_2026_08_04/"
    "spread_measurement_result.json"
)
OHLCV_COST_RECEIPT = Path("v4/audit/databento_es_vwap_downloads.jsonl")
POWER_RECEIPT = Path("v5/history/jobs/measurement-review/analysis_receipt.json")
ACTION_VALUE_RECEIPT = Path(
    "v4/audit/autoresearch/causal_day_action_value_economics_2026_08_14_attempt003/"
    "receipt.json"
)
DEFAULT_OUT = Path(
    "v4/audit/autoresearch/mes_strategic_pivot_2026_08_14/receipt.json"
)
EXPECTED_MES_SESSIONS_HIGH = 1830
OWNER_COST_CAP_USD = 200.0


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def build_receipt() -> dict:
    spread = _load(SPREAD_RECEIPT)
    power = _load(POWER_RECEIPT)
    action = _load(ACTION_VALUE_RECEIPT)
    ohlcv_rows = [json.loads(line) for line in OHLCV_COST_RECEIPT.read_text().splitlines() if line]
    ohlcv_cost = sum(float(row["cost_estimate_usd"]) for row in ohlcv_rows)
    bbo_cost = float(spread["acquisition_receipt"]["estimated_cost_usd"])
    bbo_sessions = int(spread["acquisition_receipt"]["session_count"])
    translated = translate_es_spread_to_mes(float(spread["elevated_rv15"]["mean_ticks"]))
    cost_scale = scaled_data_cost(
        EXPECTED_MES_SESSIONS_HIGH,
        ohlcv_cost_usd=ohlcv_cost,
        ohlcv_sessions=len(ohlcv_rows),
        bbo_cost_usd=bbo_cost,
        bbo_sessions=bbo_sessions,
    )
    if cost_scale["scaled_total_usd"] >= OWNER_COST_CAP_USD:
        raise RuntimeError("empirical scaled estimate no longer fits the proposed owner cap")

    payload = {
        "schema_version": "v5.mes-strategic-pivot.v1",
        "created_on": "2026-08-14",
        "status": "DECISION_GRADE_DIFFERENT_GAME_PACKET_NOT_A_PROFITABLE_POLICY",
        "closed_game": {
            "instrument": "SPXW 0DTE long call/put",
            "selected_trades": action["primary_midpoint_firewall"]["trades"],
            "mean_midpoint_usd_per_trade": action["primary_midpoint_firewall"][
                "mean_usd_per_trade"
            ],
            "why_closed": "negative before spread; bid economics cannot improve on midpoint",
        },
        "selected_different_game": {
            "instrument": "one Micro E-mini S&P 500 future (MES)",
            "mechanism": [
                "linear five-dollar-per-index-point exposure",
                "no option premium, theta, implied-volatility premium or strike selection",
                "one position at a time with explicit WAIT, ENTER_LONG, ENTER_SHORT and EXIT",
                "clock is a causal feature; 10:00 and 13:30 are diagnostics, never forced trades",
            ],
            "why_preferred": (
                "it preserves the 10-30 point directional thesis while removing every option-layer "
                "mechanism that bound the completed long-SPXW experiments"
            ),
        },
        "instrument_terms": {
            "multiplier_usd_per_point": 5.0,
            "minimum_tick_points": 0.25,
            "minimum_tick_usd": 1.25,
            "launch_date": "2019-05-06",
            "sources": {
                "cme_contract": "https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.margins.html",
                "cme_launch": "https://www.cmegroup.com/notices/ser/2019/04/SER-8360.html",
                "ibkr_commission": "https://www.interactivebrokers.com/en/pricing/commissions-futures.php",
                "ibkr_exchange_fee": "https://www.interactivebrokers.com/en/accounts/fees/CME.php",
            },
            "public_terms_checked_on": "2026-08-14",
        },
        "cost_translation": {
            **translated.to_dict(),
            "spread_source": "measured ES.FUT bbo-1s elevated-volatility proxy",
            "proxy_limitation": (
                "MES spread has not been measured; exact MES BBO replaces this proxy before any fit "
                "or economic claim"
            ),
            "fee_limitation": (
                "public commission and exchange fees are not a paper-fill receipt; the 0.55-point "
                "ceiling is a research assumption and exact fees must be reverified"
            ),
        },
        "defined_move_economics": [
            target_stop_economics(target, 5.0) for target in (10.0, 20.0, 30.0)
        ],
        "risk_law_for_a_future_test": {
            "starting_equity_usd": 10000,
            "contracts": 1,
            "initial_stop_points": 5.0,
            "maximum_trades_per_session": 2,
            "one_position_at_a_time": True,
            "no_overnight_position": True,
            "historical_worst_trade_gate_usd": -500,
            "warning": "a stop is not a guaranteed maximum loss; gap/slippage stress remains mandatory",
        },
        "evidence_requirement": {
            "current_owned_es_sessions": power["non_empty_sessions"],
            "sessions_for_one_trade_per_day": power["forward_confirmation_sessions"],
            "interpretation": (
                "at 60 minutes a one-point net edge needs 3,807 sessions, two points needs 952, "
                "and three points needs 423 under the simple one-hypothesis calculation; controls, "
                "multiplicity and 4/5 chronology can only require more"
            ),
            "minimum_acquisition_scope": (
                "all available MES RTH sessions from 2019-05-06 through 2026-07-31, with the final "
                "252 sessions reserved before policy construction"
            ),
            "estimated_available_session_range": [1800, EXPECTED_MES_SESSIONS_HIGH],
            "model_size_ceiling": 50,
            "why_50": (
                "retains the completed conservative design budget and prevents the new data from "
                "becoming permission for a broad architecture search"
            ),
        },
        "data_budget": {
            **cost_scale,
            "estimate_basis": (
                "scaled from exact on-disk Databento estimates: ES OHLCV-1m $2.592221274972/311 "
                "sessions and ES bbo-1s $1.478327661753/20 sessions"
            ),
            "owner_cap_usd": OWNER_COST_CAP_USD,
            "prepurchase_rule": (
                "a read-only exact MES.FUT metadata estimate must fit the cap; otherwise abort and "
                "return to the owner without downloading"
            ),
        },
        "future_test_minimums": {
            "single_shared_time_aware_model": True,
            "separate_morning_afternoon_networks": False,
            "entry_objective": "ENTER_LONG or ENTER_SHORT now versus WAIT",
            "exit_ownership": "the entry owns its position until close across any clock boundary",
            "execution": "first eligible MES BBO after the causal decision plus measured fees",
            "controls": [
                "outcome-blind time/direction-matched control",
                "identically fitted shuffled-label null",
            ],
            "required_kills": [
                "positive executable net with corrected lower bound above zero",
                "positive versus both controls",
                "positive absolutely and versus controls in at least four of five folds",
                "per-feature timestamp and future-mutation audits",
                "historical worst trade no worse than -$500 on the $10,000 account",
            ],
        },
        "owner_controlled_change": {
            "needed": True,
            "exact_decision": (
                "replace the long-SPXW-only research charter with a MES research charter and authorize "
                "up to $200 for exact MES.FUT OHLCV-1m plus bbo-1s history from launch through "
                "2026-07-31; research only, no broker, paper or live orders"
            ),
            "not_authorized_by_this_packet": [
                "data purchase",
                "broker contact",
                "paper or live order",
                "promotion",
            ],
        },
        "source_hashes": {
            str(path): file_sha256(path)
            for path in (SPREAD_RECEIPT, OHLCV_COST_RECEIPT, POWER_RECEIPT, ACTION_VALUE_RECEIPT)
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    receipt = build_receipt()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

