"""Seal the one-member defined-risk strategic comparison before outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256


COVERAGE = Path(
    "v4/audit/autoresearch/causal_day_trader_coverage_2026_08_14_attempt002/session_coverage.csv"
)
SETTLEMENT = Path(
    "v4/audit/autoresearch/causal_day_terminal_settlement_2026_08_14/receipt.json"
)
OUT_ROOT = Path("/Volumes/AR_TRADING_DATA/derived/defined_risk_iron_fly_v1")
EVIDENCE_DIR = Path(
    "v4/audit/autoresearch/defined_risk_iron_fly_2026_08_14_attempt001"
)
IMPLEMENTATION = (
    Path("v5/research/defined_risk_iron_fly.py"),
    Path("v5/ops/measure_defined_risk_iron_fly.py"),
    Path("v5/ops/write_defined_risk_iron_fly_declaration.py"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or OUT_ROOT.exists() or EVIDENCE_DIR.exists():
        raise RuntimeError("refusing to overwrite declaration, output, or evidence")
    payload = {
        "schema_version": "v5.defined-risk-iron-fly-declaration.v1",
        "declared_on": "2026-08-14",
        "purpose": "one model-free test of the strongest $10,000-compatible direction-neutral short-premium structure",
        "family": {
            "primary_members": 1,
            "project_multiplicity_family": 33,
            "reason": "retain the 32 previously inspected two-sided cells plus this selected defined-risk translation",
            "nearby_width_time_or_horizon_retry_forbidden": True,
        },
        "structure": {
            "name": "five_point_atm_iron_fly",
            "entry_minute": "15:00",
            "target_exit_minute": "15:15",
            "center": "nearest entry-visible strike with ATM call/put and exact +/-5 point wings; lower strike wins distance tie",
            "legs": ["short ATM call at bid", "short ATM put at bid", "long K+5 call at ask", "long K-5 put at ask"],
            "exit": "first complete four-leg touch at/after 15:15 with combo debit <=5 points; otherwise validated PM settlement",
            "fees": "four measured $3.08 option round trips",
            "risk": "entry-defined maximum loss including fees must be <=$500, exactly 5% of $10,000; otherwise abstain",
            "trade_cap": 1,
        },
        "selection_basis_before_outcomes": {
            "time": "prior evidence locates the measured variance premium in the final two hours and names 15:00; multiplicity is retained",
            "width": "5 points is the widest expiry-defined risk compatible with the existing 5% daily breaker",
            "direction_neutral": "the measured effect is variance premium and no stable directional selector exists",
        },
        "primary_metric": "mean aggressive-touch net P&L per all 243 sessions, counting abstentions as zero",
        "controls": ["same-minute naked short straddle", "same-minute long straddle", "iron-fly mid-to-mid gross"],
        "inference_standard": {
            "one_sided_bootstrap_draws": 20000,
            "alpha": 0.05,
            "multiplicity_family": 33,
            "chronological_blocks": 5,
            "positive_blocks_required": 4,
            "strategic_breakthrough": "positive mean, corrected lower bound >0, >=4/5 positive blocks, and max loss <=$500",
        },
        "failure_consequence": "this exact tight-risk direction-neutral structure closes without a width/time retry; broader spread classes remain owner-gated and unproven",
        "inputs": {
            "coverage_path": str(COVERAGE),
            "coverage_sha256": file_sha256(COVERAGE),
            "settlement_receipt_path": str(SETTLEMENT),
            "settlement_receipt_sha256": file_sha256(SETTLEMENT),
        },
        "implementation_hashes": {str(path): file_sha256(path) for path in IMPLEMENTATION},
        "outputs": {"root": str(OUT_ROOT), "evidence_dir": str(EVIDENCE_DIR)},
        "forbidden": [
            "choose a different width, time, horizon, side, or fill after outcomes",
            "drop sessions lacking an entry structure or later executable combo",
            "market-maker or midpoint fill claim",
            "model fit, paper order, live order, or reserved session",
        ],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(payload["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
