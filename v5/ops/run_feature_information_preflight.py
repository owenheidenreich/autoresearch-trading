"""Run Phase 4a and apply its pre-declared verdict rule exactly as written.

The declaration is written and hashed before this runs, and this refuses to run
against a declaration whose implementation hashes no longer match the code -- so
the thresholds cannot move once a number is on the screen.

Reads member P's binary label and nothing economic. Touches the chronological
training prefix and no score block. Fits nothing that is promoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.autoresearch.budget import AlphaLedger
from v5.research.causal_day_tensorizer import CANDLE_FEATURES, _candle_feature_frame, _clock_vector
from v5.research.chain_internal_features import (
    CHAIN_STATE_FEATURES,
    CONTRACT_CHAIN_FEATURES,
    chain_state,
    contract_chain_features,
)
from v5.research.causal_day_chain_state_lifecycle import STATE_CANDLE_FEATURES
from v5.research.feature_information_preflight import (
    CLOCK_FEATURES_NAMED,
    INNER_FOLDS,
    PROBE_FEATURES,
    PROBE_PARAMETERS,
    PreflightError,
    apply_verdict,
    plant_known_answer,
    run_probe,
)
from v5.research.lifecycle_episode_adapter import (
    PRIMARY_LABEL,
    CorpusIndex,
    CorpusSession,
    assert_declared_tape,
)
from v5.research.lifecycle_trainer import ChronologySplit

SCHEMA_VERSION = "v5.phase-4a-preflight-run.v1"
EXPERIMENT_ID = "job46.phase4a.feature_information_preflight.v1"
SEED = 20260819

CANDIDATE_COLUMNS = (
    "session",
    "entry_minute",
    "contract_id",
    "ask",
    "spread",
    "self_iv",
    "self_theta_per_minute",
    "is_call",
    "moneyness_itm_points",
    PRIMARY_LABEL,
)
LADDER_COLUMNS = (
    "session",
    "minute",
    "contract_id",
    "is_call",
    "strike",
    "mid",
    "self_iv",
    "moneyness_itm_points",
    "bid_size",
    "ask_size",
    "spread",
)


def session_frame(row: CorpusSession) -> pd.DataFrame:
    """One session's candidate actions with the declared feature set attached."""

    assert_declared_tape(row)
    candidates = pd.read_parquet(row.table("candidates"), columns=list(CANDIDATE_COLUMNS))
    if candidates.empty:
        # A legitimate no-trade day under the signed ticket cap. It contributes no
        # action to a probe over actions, and is reported rather than dropped.
        return candidates.assign(**{name: pd.Series(dtype=float) for name in PROBE_FEATURES})

    ladder = pd.read_parquet(row.table("ladder"), columns=list(LADDER_COLUMNS))
    ladder = ladder.sort_values(["minute", "contract_id"]).reset_index(drop=True)

    state = chain_state(ladder).rename(columns={"minute": "entry_minute"})
    contract = contract_chain_features(ladder)
    contract = pd.concat(
        [ladder.loc[:, ["minute", "contract_id"]].reset_index(drop=True), contract], axis=1
    ).rename(columns={"minute": "entry_minute"})

    candles = pd.read_parquet(row.table("candles"))
    tape = _candle_feature_frame(candles).loc[:, list(CANDLE_FEATURES)]
    tape["entry_minute"] = candles["knowable_at"].astype(str).to_numpy()
    tape = tape.loc[:, ["entry_minute", *STATE_CANDLE_FEATURES]]

    frame = candidates.copy()
    frame["entry_minute"] = frame["entry_minute"].astype(str)
    frame = frame.merge(
        state.loc[:, ["entry_minute", *CHAIN_STATE_FEATURES]], on="entry_minute", how="left"
    )
    frame = frame.merge(
        contract.loc[:, ["entry_minute", "contract_id", *CONTRACT_CHAIN_FEATURES]],
        on=["entry_minute", "contract_id"],
        how="left",
    )
    frame = frame.merge(tape, on="entry_minute", how="left")

    clock = np.vstack([_clock_vector(minute) for minute in frame["entry_minute"]])
    for index, name in enumerate(CLOCK_FEATURES_NAMED):
        frame[name] = clock[:, index]
    frame["is_call"] = frame["is_call"].astype(float)
    return frame


def build_probe_frame(index: CorpusIndex, sessions: Sequence[str]) -> pd.DataFrame:
    frames = []
    for position, session in enumerate(sessions):
        frames.append(session_frame(index[session]))
        if (position + 1) % 50 == 0:
            print(f"  probe frame {position + 1}/{len(sessions)}", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    return frame


def verify_declaration(path: Path) -> dict[str, Any]:
    """Refuse a declaration whose code has moved since it was signed."""

    declaration = json.loads(path.read_text())
    semantic = {k: v for k, v in declaration.items() if k != "declaration_sha256"}
    actual = hashlib.sha256(canonical_json(semantic)).hexdigest()
    if declaration.get("declaration_sha256") != actual:
        raise PreflightError(f"{path.name}: declaration hash does not match its own content")
    for source, expected in declaration["implementation_hashes"].items():
        got = file_sha256(Path(source))
        if got != expected:
            raise PreflightError(
                f"{path.name}: {source} has changed since the declaration was written "
                f"({got[:12]} != {expected[:12]}); the thresholds may not move after a run"
            )
    return declaration


def run(
    *,
    index: CorpusIndex,
    declaration_path: Path,
    receipt_path: Path,
    ledger_path: Path,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    declaration = verify_declaration(declaration_path)
    split = ChronologySplit.build(index.sessions)
    prefix = list(split.training_prefix)

    if cache_path is not None and cache_path.exists():
        frame = pd.read_parquet(cache_path)
        print(f"reloaded probe frame: {len(frame)} rows", flush=True)
    else:
        frame = build_probe_frame(index, prefix)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache_path, index=False)

    labelled = frame[np.isfinite(frame[PRIMARY_LABEL].to_numpy(float))].reset_index(drop=True)
    folds = split.inner_folds(INNER_FOLDS)

    real = run_probe(labelled, folds, label_column=PRIMARY_LABEL, seed=SEED, name="real_features")
    planted_frame = plant_known_answer(labelled, label_column=PRIMARY_LABEL, seed=SEED + 101)
    planted = run_probe(
        planted_frame, folds, label_column=PRIMARY_LABEL, seed=SEED, name="planted_known_answer"
    )
    verdict = apply_verdict(real, planted)

    ledger = AlphaLedger(
        ledger_path,
        option_sessions=len(prefix),
        universe=declaration["alpha_ledger"]["universe"],
    )
    entry = ledger.record(
        experiment_id=EXPERIMENT_ID,
        declaration_sha256=declaration["declaration_sha256"],
        declared_on=declaration["declared_on"],
        outcome="PASS" if verdict["verdict"] == "PROCEED" else "FAIL",
        # Deliberately null. This preflight measures a precision LIFT at a fixed
        # operating rate, not the directional accuracy the ledger's bar is
        # denominated in. Writing the precision into that column would put a
        # number under a heading that means something else, and the ledger's
        # `bar_at_time_of_run` would then read as a bar this experiment was
        # judged against. It was not: the verdict is the declaration's +4.0pp
        # rule. The ledger is charged here for its actual purpose -- the
        # hash-chained count that stops an experiment being quietly unrun.
        observed_accuracy=None,
        accuracy_lower_bound=None,
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "declaration": str(declaration_path),
        "declaration_sha256": declaration["declaration_sha256"],
        "seed": SEED,
        "population": {
            "corpus_sessions": len(index),
            "training_prefix_sessions": len(prefix),
            "prefix_first": prefix[0],
            "prefix_last": prefix[-1],
            "candidate_actions": int(len(frame)),
            "labelled_actions": int(len(labelled)),
            "unlabelled_actions": int(len(frame) - len(labelled)),
            "score_blocks_touched": 0,
        },
        "probe": {
            "parameters": PROBE_PARAMETERS,
            "features": list(PROBE_FEATURES),
            "inner_folds": INNER_FOLDS,
            "folds_used": real.folds,
        },
        "real_features": real.payload(),
        "planted_known_answer": planted.payload(),
        "verdict": verdict,
        "alpha_ledger": {
            "path": str(ledger_path),
            "experiments_run": ledger.experiments_run,
            "entry_index": entry.index,
            "entry_hash": entry.entry_hash,
            "accuracy_columns_null_because": (
                "Phase 4a measures a precision lift at a fixed operating rate, not a "
                "directional accuracy; the ledger is charged for the count, and the "
                "verdict is the declaration's rule"
            ),
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--backfill-quotes", type=Path, required=True)
    parser.add_argument("--owned-quotes", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
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
    verdict = payload["verdict"]
    real = payload["real_features"]
    planted = payload["planted_known_answer"]
    print(f"\nPHASE 4A — {verdict['verdict']}")
    print(f"  {verdict['meaning']}")
    print(
        f"  plant: lift {planted['lift_pp']:+.2f}pp "
        f"(bar {verdict['plant_recovery_bar_pp']:+.1f}pp) -> "
        f"{'recovered' if verdict['plant_recovered'] else 'NOT recovered'}"
    )
    print(
        f"  real:  base {real['base_rate']:.4f}, precision {real['precision_at_operating_rate']:.4f}, "
        f"lift {real['lift_pp']:+.2f}pp, upper {real['verdict_lift_upper_pp']:+.2f}pp "
        f"(bar {verdict['lift_bar_pp']:+.1f}pp)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
