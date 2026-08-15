"""Freeze the outcome-blind V5 instability attribution before it runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v5.ops.analyze_causal_day_v5_instability import (
    CANDIDATE_COLUMNS,
    PROXIES,
)
from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_architectures import computed_parameter_counts
from v5.research.causal_day_attribution import TIME_BANDS, clock_baseline


DEFAULT_V5 = Path("v5/work/entry-exit-attribution/DECLARATION_V5.json")
DEFAULT_FIT = Path(
    "v4/audit/autoresearch/causal_day_magnitude_fit_2026_08_14_attempt001/receipt.json"
)
DEFAULT_CALIBRATION = Path(
    "v4/audit/autoresearch/causal_day_rank_selector_calibration_2026_08_14_attempt001/receipt.json"
)
DEFAULT_CACHE = Path("/Volumes/AR_TRADING_DATA/derived/causal_day_magnitude_cache_v3")
DEFAULT_CANDIDATES = Path(
    "/Volumes/AR_TRADING_DATA/derived/causal_day_trader_v2/candidates.parquet"
)
DEFAULT_V1 = Path(
    "v5/work/entry-exit-attribution/ATTRIBUTION_DECLARATION_V1.json"
)
IMPLEMENTATIONS = (
    Path("v5/research/causal_day_attribution.py"),
    Path("v5/ops/analyze_causal_day_v5_instability.py"),
    Path("v5/ops/write_causal_day_v5_attribution_declaration.py"),
)


def _verified(path: Path, schema: str) -> dict:
    value = json.loads(path.read_text())
    expected = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"self-hash mismatch: {path}")
    if value.get("schema_version") != schema:
        raise RuntimeError(f"unexpected schema: {path}")
    return value


def build_declaration(
    *,
    v5_path: Path,
    fit_path: Path,
    calibration_path: Path,
    cache_root: Path,
    candidates_path: Path,
    output_root: Path,
    evidence_dir: Path,
    v1_path: Path,
) -> dict:
    v5 = _verified(v5_path, "v5.causal-day-trader-fit-declaration.v5")
    fit = _verified(fit_path, "v5.causal-day-magnitude-fit.v1")
    calibration = _verified(
        calibration_path, "v5.causal-day-rank-selector-calibration.v1"
    )
    if calibration.get("economics_read") is not False:
        raise RuntimeError("attribution must bind the pre-economics calibration")
    if v5["fit_reuse"]["fit_receipt"]["sha256"] != file_sha256(fit_path):
        raise RuntimeError("V5 and fit receipt binding changed")
    if calibration["candidate_table"]["sha256"] != file_sha256(candidates_path):
        raise RuntimeError("candidate table changed after V5 calibration")
    v1 = _verified(
        v1_path, "v5.causal-day-v5-instability-attribution-declaration.v1"
    )

    value = {
        "schema_version": "v5.causal-day-v5-instability-attribution-declaration.v2",
        "declared_on": "2026-08-14",
        "supersedes_declaration_sha256": file_sha256(v1_path),
        "v1_pre_output_defect": {
            "status": "FAILED_SAFE_BEFORE_ATTRIBUTION_OUTPUT",
            "cause": (
                "The V1 analyzer reconstructed normalized model logits but compared "
                "them with stored predictions expressed in SPX points"
            ),
            "repair": (
                "Multiply full score, state offset, contract-node contribution and "
                "all neutralized offsets by the frozen 30-point target scale"
            ),
            "outcome_or_attribution_metric_read": False,
            "v1_declaration_receipt_sha256": v1["receipt_sha256"],
        },
        "purpose": (
            "Explain V5 fold concentration and opening-minute activation without reading "
            "P&L, fitting a new model, changing a selector, or choosing an economic subgroup"
        ),
        "closed_policy": {
            "declaration": {"path": str(v5_path), "sha256": file_sha256(v5_path)},
            "architecture": "neural_four_head",
            "horizon_minutes": 120,
            "computed_parameters": computed_parameter_counts()["neural_four_head"],
            "new_fit": False,
            "new_operating_point": False,
        },
        "immutable_inputs": {
            "fit_receipt": {"path": str(fit_path), "sha256": file_sha256(fit_path)},
            "rank_calibration": {
                "path": str(calibration_path),
                "sha256": file_sha256(calibration_path),
            },
            "cache_receipt": {
                "path": str(cache_root / "receipt.json"),
                "sha256": file_sha256(cache_root / "receipt.json"),
            },
            "candidate_table": {
                "path": str(candidates_path),
                "sha256": file_sha256(candidates_path),
            },
        },
        "population": {
            "law": "The unchanged 150 OOF scored sessions and 423,053 real neural-four-head 120m contract predictions",
            "folds": 5,
            "reserved_sessions_forbidden_on_or_after": "2026-08-06",
        },
        "architectural_hypothesis": {
            "statement": (
                "The linear ContractHead is additive in [state,node], so every candle, "
                "clock, account and global-ladder state contributes one common offset to "
                "all contracts in a routed minute. Contract score differences and the "
                "call/put/strike argmax depend only on the contract node and role head."
            ),
            "numerical_proof": [
                "reconstruct every stored OOF score as state_offset + contract_node_contribution",
                "require maximum absolute reconstruction error <= 1e-5",
                "require the state offset to be constant within every session-minute",
                "require full-score and contract-node argmax identity on every minute",
            ],
            "failure_consequence": (
                "If verified, no existing Job-39 architecture can answer the owner's "
                "state-conditioned direction/contract-choice question; a next policy must "
                "contain an explicit state-by-contract interaction"
            ),
        },
        "declared_diagnostics": {
            "activation_drift": {
                "cells": "all five frozen fold cutoffs applied only to their original OOF score blocks",
                "metrics": [
                    "cutoff percentile and scored maximum",
                    "crossing minutes and sessions",
                    "first crossing minute",
                    "share of first crossings at 09:35--09:39",
                ],
                "severe_rule": "cutoff exceeds every scored minute in at least 2 of 5 folds",
            },
            "time_surface": {
                "bands": list(TIME_BANDS),
                "named_minutes": ["10:00", "13:30"],
                "outcome_comparison": False,
            },
            "exact_score_decomposition": [
                "state offset",
                "contract node contribution",
                "within-minute and between-minute variance fractions",
                "state/node covariance",
            ],
            "state_channel_neutralizations": {
                "candle_state": "replace the GRU terminal state by zero at the fitted fusion input",
                "global_ladder_summary": "replace the permutation-invariant ladder summary by zero while preserving contract nodes",
                "explicit_clock": "replace the five clock channels by their frozen all-entry-minute mean",
                "clock_baseline": clock_baseline().tolist(),
                "interpretation": "descriptive fitted-model sensitivity, not causal economic importance",
            },
            "contract_proxies": {
                "candidate_columns_read": list(CANDIDATE_COLUMNS),
                "reported_proxies": list(PROXIES),
                "metrics": ["pooled Spearman", "within-minute Pearson after demeaning"],
                "proxy_dominated_rule": "largest absolute within-minute correlation >= 0.75",
            },
        },
        "classification_rules": {
            "architectural_separability": "all four numerical proof conditions pass",
            "early_entry_collapse": "at least 75% of activated sessions first cross during 09:35--09:39",
            "next_policy_if_separable": "COMPACT_STATE_CONTRACT_INTERACTION_REQUIRED",
            "next_policy_if_not_separable_but_proxy_dominated": "PROXY_DOMINATED_DO_NOT_FIT_SEQUENCE_POLICY",
            "otherwise": "ATTRIBUTION_INCONCLUSIVE_REQUIRE_MORE_DIAGNOSIS",
        },
        "forbidden": [
            "read P&L, bid, exit value, MFE, MAE or any forward target column",
            "evaluate a new threshold, rank target, selector, seed, architecture or horizon",
            "report a profitable time, side, premium, delta or fold subgroup",
            "use target correlation to choose an attribution result",
            "fit or update any model",
            "open reserved sessions",
        ],
        "outputs": {
            "root": str(output_root),
            "evidence_dir": str(evidence_dir),
        },
        "implementation_hashes": {
            str(path): file_sha256(path) for path in IMPLEMENTATIONS
        },
    }
    value["receipt_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
    return value


def run(
    *,
    out_path: Path,
    v5_path: Path,
    fit_path: Path,
    calibration_path: Path,
    cache_root: Path,
    candidates_path: Path,
    output_root: Path,
    evidence_dir: Path,
    v1_path: Path,
) -> dict:
    if out_path.exists():
        raise RuntimeError(f"refusing to overwrite declaration: {out_path}")
    value = build_declaration(
        v5_path=v5_path,
        fit_path=fit_path,
        calibration_path=calibration_path,
        cache_root=cache_root,
        candidates_path=candidates_path,
        output_root=output_root,
        evidence_dir=evidence_dir,
        v1_path=v1_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(out_path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--v5", type=Path, default=DEFAULT_V5)
    parser.add_argument("--fit-receipt", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1)
    args = parser.parse_args()
    run(
        out_path=args.out,
        v5_path=args.v5,
        fit_path=args.fit_receipt,
        calibration_path=args.calibration,
        cache_root=args.cache_root,
        candidates_path=args.candidates,
        output_root=args.output_root,
        evidence_dir=args.evidence_dir,
        v1_path=args.v1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
