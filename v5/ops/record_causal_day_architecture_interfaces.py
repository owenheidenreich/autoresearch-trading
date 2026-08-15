"""Record the implemented-but-unfitted causal policy architecture family."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research import causal_day_architectures as architectures
from v5.research import causal_day_policy_gate as gate
from v5.research import causal_day_tensorizer as tensorizer


DIMENSIONS = architectures.ArchitectureDimensions(
    candle_features=len(tensorizer.CANDLE_FEATURES),
    ladder_features=len(tensorizer.LADDER_FEATURES),
    account_features=5,
    position_features=10,
    clock_features=5,
    hidden_size=8,
)


def run(out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {out_dir}")
    family: dict[str, Any] = {}
    for name in gate.ARCHITECTURES:
        model = architectures.build_architecture(name, DIMENSIONS)
        parameters = architectures.trainable_parameter_count(model)
        family[name] = {
            "parameter_count": parameters,
            "sequence": gate.ARCHITECTURES[name].sequence,
            "shared_encoder": gate.ARCHITECTURES[name].shared_encoder,
            "heads": list(gate.ARCHITECTURES[name].heads),
            "conditional_followup": gate.ARCHITECTURES[name].conditional_followup,
            "current_fit_blockers": list(
                gate.fit_blockers(
                    name,
                    sessions=243,
                    trainable_parameters=parameters,
                    g1_passed=False,
                    do_not_retest_reopened=False,
                    shared_specialization_justified=False,
                )
            ),
        }
    source_path = Path(inspect.getsourcefile(architectures) or "")
    gate_path = Path(inspect.getsourcefile(gate) or "")
    tensorizer_path = Path(inspect.getsourcefile(tensorizer) or "")
    declaration_path = Path("v5/work/entry-exit-attribution/DECLARATION_V2.json")
    receipt: dict[str, Any] = {
        "schema_version": "v5.causal-day-architecture-interfaces.v3",
        "created_on": "2026-08-14",
        "purpose": "bind the common unfitted forward interface for the declared architecture comparison",
        "declaration": {
            "path": str(declaration_path),
            "sha256": file_sha256(declaration_path),
        },
        "dimensions": dict(DIMENSIONS.__dict__),
        "input_contract": {
            "candles": "right-padded completed session prefix; at least one visible row",
            "candle_features": list(tensorizer.CANDLE_FEATURES),
            "maximum_candle_prefix": tensorizer.MAX_CANDLE_PREFIX,
            "ladder": "current whole live two-sided contract set with an explicit visibility mask",
            "ladder_features": list(tensorizer.LADDER_FEATURES),
            "entry_action_mask": (
                "visible, declared near-ATM OTM contracts affordable from current cash; "
                "deep strikes remain context but cannot become actions"
            ),
            "account": "causal cash, realised P&L and occupancy state",
            "account_features": [
                "cash_fraction_of_start",
                "realised_pnl_fraction_of_start",
                "trades_opened_fraction_of_three",
                "remaining_trade_cap_fraction",
                "breaker_triggered",
            ],
            "position": "causal held-contract and path state",
            "position_features": [
                "is_holding",
                "signed_right",
                "strike_distance_scaled",
                "entry_ask_fraction_of_start",
                "unrealised_pnl_fraction_of_start",
                "maximum_favourable_fraction_of_start",
                "maximum_adverse_fraction_of_start",
                "minutes_held_fraction_of_120",
                "entry_moneyness_scaled",
                "signed_origin_regime",
            ],
            "clock": "causal time encoding including declared regime context",
            "clock_features": [
                "session_elapsed_fraction",
                "session_remaining_fraction",
                "is_morning_entry_regime",
                "time_sine",
                "time_cosine",
            ],
            "roles": list(gate.ROLES),
        },
        "output_contract": {
            "entry": "abstain logit plus one logit per currently visible ladder contract",
            "exit": "hold/sell logits",
            "routing": "opening regime owns its matching exit role until close",
        },
        "mask_mutation_invariant_tested": True,
        "family": family,
        "source_hashes": {
            "architectures": {"path": str(source_path), "sha256": file_sha256(source_path)},
            "fit_gate": {"path": str(gate_path), "sha256": file_sha256(gate_path)},
            "tensorizer": {
                "path": str(tensorizer_path),
                "sha256": file_sha256(tensorizer_path),
            },
        },
        "fit_performed": False,
        "weights_saved": False,
        "economic_result": None,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=False)
    path = out_dir / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
