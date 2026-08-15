"""Record why the activated goal cannot honestly fit its proposed models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research import causal_day_architectures as architecture_code
from v5.research import causal_day_tensorizer as tensorizer
from v5.research.causal_day_policy_gate import ARCHITECTURES, fit_blockers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=243)
    parser.add_argument(
        "--declaration",
        type=Path,
        default=Path("v5/work/entry-exit-attribution/DECLARATION_V2.json"),
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {args.out_dir}")
    sources = {
        "status": Path("v5/STATUS.md"),
        "do_not_retest": Path("v5/research/history/DO_NOT_RETEST.md"),
        "knobs": Path("v5/research/knobs.py"),
        "declaration": args.declaration,
        "gate": Path("v5/research/causal_day_policy_gate.py"),
    }
    architectures = {}
    dimensions = architecture_code.ArchitectureDimensions(
        candle_features=len(tensorizer.CANDLE_FEATURES),
        ladder_features=len(tensorizer.LADDER_FEATURES),
        account_features=5,
        position_features=10,
        clock_features=5,
        hidden_size=8,
    )
    for name, spec in ARCHITECTURES.items():
        parameters = architecture_code.trainable_parameter_count(
            architecture_code.build_architecture(name, dimensions)
        )
        architectures[name] = {
            "sequence": spec.sequence,
            "shared_encoder": spec.shared_encoder,
            "heads": list(spec.heads),
            "conditional_followup": spec.conditional_followup,
            "trainable_parameters": parameters,
            "blockers": list(
                fit_blockers(
                    name,
                    sessions=args.sessions,
                    trainable_parameters=parameters,
                    g1_passed=False,
                    do_not_retest_reopened=False,
                    shared_specialization_justified=False,
                )
            ),
        }
    receipt = {
        "schema_version": "v5.causal-day-model-block.v2",
        "created_on": "2026-08-14",
        "owner_fit_authorization_present": True,
        "meaning": (
            "owner authorization satisfies the action-tier requirement but does not rewrite the "
            "G1, do-not-retest, or frozen neural sample-size evidence rules"
        ),
        "sessions": args.sessions,
        "fit_performed": False,
        "architectures": architectures,
        "source_hashes": {name: file_sha256(path) for name, path in sources.items()},
        "legitimate_reopening_requirements": [
            "an explicit STATUS/G1 training-precondition change with its required evidence",
            "an explicit new-evidence reopening of the later quote-priced selective-entry closure",
            "for neural comparison, at least 1140 independent sessions or the registry's named pre-registered published argument and governance change",
            "for four independent specialists, prior chronological evidence that shared heads specialize materially"
        ],
        "promotion_authorized": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(args.out_dir / "receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
