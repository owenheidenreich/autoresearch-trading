"""Run Phase 4b and apply Fable's three-way verdict rule exactly as written.

Preconditions, both hard: the declaration's implementation hashes must match the
files on disk, and the Phase-4a headline must reproduce bit-for-bit at
+8.649074789891253pp. The run is deterministic, so anything else means the code
or the corpus has moved and the phase stops rather than reporting a number
against a changed baseline.

Label-side only: the four first-touch columns and no dollar column.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json
from v5.ops.run_feature_information_preflight import (
    SEED,
    build_probe_frame,
    verify_declaration,
)
from v5.research.autoresearch.budget import AlphaLedger
from v5.research.causal_day_chain_state_lifecycle import STATE_CANDLE_FEATURES
from v5.research.chain_internal_features import (
    CHAIN_STATE_FEATURES,
    CONTRACT_CHAIN_FEATURES,
)
from v5.research.feature_information_decomposition import (
    GAIN_MINUTE_COLUMN,
    LOSS_MINUTE_COLUMN,
    NULL_DRAWS,
    PHASE_4A_HEADLINE_LIFT_PP,
    PRIMARY_LABEL,
    TIE_BREAK_DRAWS,
    DecompositionError,
    apply_verdict,
    decompose,
    matched_precision,
    out_of_fold_scores,
    permute_label_side,
    select_top,
)
from v5.research.feature_information_preflight import (
    CLOCK_FEATURES_NAMED,
    CONTRACT_BASE_FEATURES,
    INNER_FOLDS,
    ORDERING_FEATURES,
    PROBE_FEATURES,
    run_probe,
)
from v5.research.lifecycle_episode_adapter import CorpusIndex
from v5.research.lifecycle_trainer import ChronologySplit

SCHEMA_VERSION = "v5.phase-4b-decomposition-run.v1"
EXPERIMENT_ID = "job46.phase4b.feature_information_decomposition.v1"

EXTRA_COLUMNS = (GAIN_MINUTE_COLUMN, LOSS_MINUTE_COLUMN)

GROUPS = {
    "chain_internal": tuple(CHAIN_STATE_FEATURES),
    "tape": tuple(STATE_CANDLE_FEATURES),
    "clock": CLOCK_FEATURES_NAMED,
    "contract_price": CONTRACT_BASE_FEATURES,
    "contract_chain": tuple(CONTRACT_CHAIN_FEATURES),
    "ordering": ORDERING_FEATURES,
}


def _labelled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[np.isfinite(frame[PRIMARY_LABEL].to_numpy(float))].reset_index(drop=True)


def _augment_label_side(frame: pd.DataFrame, index: CorpusIndex) -> pd.DataFrame:
    """Attach the two touch-minute columns without disturbing a feature or a row."""

    parts = []
    for session in sorted(frame["session"].unique()):
        row = index[str(session)]
        parts.append(
            pd.read_parquet(
                row.table("candidates"),
                columns=["session", "entry_minute", "contract_id", *EXTRA_COLUMNS],
            )
        )
    extra = pd.concat(parts, ignore_index=True)
    extra["entry_minute"] = extra["entry_minute"].astype(str)
    before = len(frame)
    merged = frame.merge(extra, on=["session", "entry_minute", "contract_id"], how="left")
    if len(merged) != before:
        raise DecompositionError(
            f"augmenting the probe frame changed its row count ({before} -> {len(merged)}); "
            "the candidate key is not unique"
        )
    return merged


def reproduce_phase_4a(frame: pd.DataFrame, folds) -> dict[str, Any]:
    """Refuse to continue unless the published headline comes back exactly."""

    result = run_probe(frame, folds, label_column=PRIMARY_LABEL, seed=SEED, name="phase_4a_replay")
    observed = result.lift * 100.0
    if observed != PHASE_4A_HEADLINE_LIFT_PP:
        raise DecompositionError(
            f"Phase 4a headline did not reproduce: {observed!r} against the published "
            f"{PHASE_4A_HEADLINE_LIFT_PP!r}. The run is deterministic, so the code or the "
            "corpus has moved; Phase 4b stops."
        )
    return {"reproduced_lift_pp": observed, "matches_published": True}


def run(
    *,
    index: CorpusIndex,
    declaration_path: Path,
    receipt_path: Path,
    ledger_path: Path,
    cache_path: Path,
) -> dict[str, Any]:
    declaration = verify_declaration(declaration_path)
    split = ChronologySplit.build(index.sessions)
    prefix = list(split.training_prefix)
    folds = split.inner_folds(INNER_FOLDS)

    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
    else:
        frame = build_probe_frame(index, prefix)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)

    if set(EXTRA_COLUMNS) - set(frame.columns):
        # Augment rather than rebuild. `run_feature_information_preflight` is pinned
        # by the Phase-4a declaration and its column list cannot gain the touch
        # minutes, and rebuilding would re-derive the chain features -- which must
        # stay bit-identical for the headline precondition to mean anything. A left
        # merge adds label-side columns and touches no feature.
        frame = _augment_label_side(frame, index)
    labelled = _labelled(frame)
    precondition = reproduce_phase_4a(labelled, folds)

    scores = out_of_fold_scores(labelled, folds, label_column=PRIMARY_LABEL, seed=SEED)
    scored = np.isfinite(scores)
    population = labelled[scored].reset_index(drop=True)
    selected = select_top(population["session"].to_numpy(), scores[scored])

    # --- D1 ---------------------------------------------------------------
    observed = decompose(population, selected, label_column=PRIMARY_LABEL)
    null_ordering: list[float] = []
    null_resolution: list[float] = []
    for draw in range(NULL_DRAWS):
        permuted = permute_label_side(labelled, seed=SEED + 50_000 + draw)
        null_scores = out_of_fold_scores(
            permuted, folds, label_column=PRIMARY_LABEL, seed=SEED
        )
        mask = np.isfinite(null_scores)
        null_population = permuted[mask].reset_index(drop=True)
        null_selected = select_top(
            null_population["session"].to_numpy(), null_scores[mask]
        )
        null_result = decompose(null_population, null_selected, label_column=PRIMARY_LABEL)
        null_ordering.append(null_result.ordering_component * 100.0)
        null_resolution.append(null_result.resolution_component * 100.0)
        if (draw + 1) % 20 == 0:
            print(f"  null {draw + 1}/{NULL_DRAWS}", flush=True)
    verdict = apply_verdict(observed, null_ordering)

    # --- per fold and per year -------------------------------------------
    by_fold = []
    for position, (_, holdout) in enumerate(folds):
        mask = np.isin(population["session"].to_numpy(), holdout)
        if not mask.any() or not selected[mask].any():
            continue
        part = decompose(
            population[mask].reset_index(drop=True), selected[mask], label_column=PRIMARY_LABEL
        )
        by_fold.append({"fold": position, **part.payload()})
    by_year = []
    years = population["session"].str.slice(0, 4).to_numpy()
    for year in sorted(set(years)):
        mask = years == year
        if not selected[mask].any():
            continue
        part = decompose(
            population[mask].reset_index(drop=True), selected[mask], label_column=PRIMARY_LABEL
        )
        by_year.append({"year": year, **part.payload()})

    # --- D2 ---------------------------------------------------------------
    matched = matched_precision(population, selected, label_column=PRIMARY_LABEL)
    agrees = bool(
        np.sign(matched["difference_pp"]) == np.sign(observed.ordering_component * 100.0)
        or abs(observed.ordering_component * 100.0) < 1e-12
    )

    # --- D3 ---------------------------------------------------------------
    ablation: dict[str, Any] = {}
    for name, fields in GROUPS.items():
        alone = _tie_break_average(labelled, folds, fields)
        rest = tuple(f for f in PROBE_FEATURES if f not in set(fields))
        without = _tie_break_average(labelled, folds, rest)
        ablation[name] = {
            "n_fields": len(fields),
            "alone_lift_pp_mean": alone,
            "without_lift_pp_mean": without,
        }
    ablation["realised_vol_15m_alone"] = {
        "n_fields": 1,
        "alone_lift_pp_mean": _tie_break_average(labelled, folds, ("realised_vol_15m",)),
    }
    ablation["full_minus_realised_vol_15m"] = {
        "n_fields": len(PROBE_FEATURES) - 1,
        "alone_lift_pp_mean": _tie_break_average(
            labelled, folds, tuple(f for f in PROBE_FEATURES if f != "realised_vol_15m")
        ),
    }

    ledger = AlphaLedger(
        ledger_path,
        option_sessions=len(prefix),
        universe=declaration["alpha_ledger"]["universe"],
    )
    entry = ledger.record(
        experiment_id=EXPERIMENT_ID,
        declaration_sha256=declaration["declaration_sha256"],
        declared_on=declaration["declared_on"],
        outcome="PASS" if verdict["verdict"] == "A_ORDERING_INFORMATION" else "FAIL",
        observed_accuracy=None,
        accuracy_lower_bound=None,
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "declaration": str(declaration_path),
        "declaration_sha256": declaration["declaration_sha256"],
        "precondition": precondition,
        "population": {
            "training_prefix_sessions": len(prefix),
            "scored_actions": int(scored.sum()),
            "sessions_scored": int(population["session"].nunique()),
            "selected": int(selected.sum()),
        },
        "d1_decomposition": observed.payload(),
        "d1_null": {
            "draws": NULL_DRAWS,
            "permutation": "label-side record (label, gain minute, loss minute) permuted as a block within session",
            "ordering_pp": null_ordering,
            "resolution_pp": null_resolution,
        },
        "d1_by_fold": by_fold,
        "d1_by_year": by_year,
        "d2_matched_control": {**matched, "agrees_with_d1_direction": agrees},
        "d3_ablation_randomised_tie_break": {
            "tie_break_draws": TIE_BREAK_DRAWS,
            "note": "replaces the Phase 4a 'alone' numbers; no verdict weight",
            "groups": ablation,
        },
        "verdict": verdict,
        "alpha_ledger": {
            "path": str(ledger_path),
            "experiments_run": ledger.experiments_run,
            "entry_index": entry.index,
            "entry_hash": entry.entry_hash,
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _tie_break_average(frame: pd.DataFrame, folds, features) -> float:
    """Mean lift over seeded tie-break draws, so minute-common groups are readable."""

    scores = out_of_fold_scores(
        frame, folds, label_column=PRIMARY_LABEL, seed=SEED, features=features
    )
    mask = np.isfinite(scores)
    population = frame[mask].reset_index(drop=True)
    sessions = population["session"].to_numpy()
    label = population[PRIMARY_LABEL].to_numpy(float)
    base = float(label.mean())
    lifts = []
    for draw in range(TIE_BREAK_DRAWS):
        selected = select_top(sessions, scores[mask], tie_break_seed=SEED + 90_000 + draw)
        lifts.append(float(label[selected].mean() - base) * 100.0)
    return float(np.mean(lifts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--backfill-quotes", type=Path, required=True)
    parser.add_argument("--owned-quotes", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()

    index = CorpusIndex.from_receipt(
        args.build_receipt,
        corpus_root=args.corpus_root,
        quote_roots={"owned": args.owned_quotes, "backfill": args.backfill_quotes},
    )
    payload = run(
        index=index,
        declaration_path=args.declaration,
        receipt_path=args.receipt,
        ledger_path=args.ledger,
        cache_path=args.cache,
    )
    d1, verdict = payload["d1_decomposition"], payload["verdict"]
    print(f"\nPHASE 4B — {verdict['verdict']}")
    print(f"  {verdict['meaning']}")
    print(
        f"  lift {d1['lift_pp']:+.2f}pp = resolution {d1['resolution_component_pp']:+.2f}pp "
        f"+ ordering {d1['ordering_component_pp']:+.2f}pp"
    )
    print(
        f"  P(resolved) {d1['p_resolved_population']:.4f} -> {d1['p_resolved_selected']:.4f} | "
        f"P(gain first | resolved) {d1['p_gain_first_given_resolved_population']:.4f} -> "
        f"{d1['p_gain_first_given_resolved_selected']:.4f}"
    )
    print(
        f"  ordering null: p97.5 {verdict['null_p975_pp']:+.2f}pp over {verdict['null_draws']} draws, "
        f"{verdict['draws_at_or_above_observed']} at or above observed"
    )
    print(
        f"  D2 matched: {payload['d2_matched_control']['difference_pp']:+.2f}pp, "
        f"agrees with D1: {payload['d2_matched_control']['agrees_with_d1_direction']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
