"""Small fixed-budget OOF model runner."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .cache import PredictionCache, prediction_cache_key, stable_hash
from .causality import assert_mutate_future_invariant
from .dataset import POLICY_INDEX, target_column
from .schema import HypothesisSpec
from v4.research.pathd_feature_admission_ledger import admitted_feature_matrix


def compose_target(frame: pd.DataFrame, spec: HypothesisSpec) -> np.ndarray:
    columns = [target_column(policy) for policy in spec.target.fields]
    values = frame.loc[:, columns].to_numpy(float)
    weights = np.asarray(spec.target.weights, dtype=float)
    weights = weights / weights.sum()
    return np.sum(values * weights.reshape(1, -1), axis=1)


def make_model(spec: HypothesisSpec, *, seed: int) -> HistGradientBoostingRegressor:
    parameters = dict(spec.model.hyperparameters)
    parameters["random_state"] = int(seed)
    return HistGradientBoostingRegressor(**parameters)


def fit_oof(
    frame: pd.DataFrame,
    *,
    spec: HypothesisSpec,
    folds: Sequence[Mapping[str, Any]],
    foundation_hash: str,
    cache: PredictionCache,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = tuple(feature.name for feature in spec.features)
    # Phase-0 admission is checked before any estimator is constructed or any
    # fit matrix can be materialized.  Missing/tampered ledgers fail closed.
    admitted_feature_matrix(frame, features)
    future_columns = [
        column
        for column in frame.columns
        if column.startswith(
            ("pnl_", "mid_pnl_", "exit_ns_", "source_exit_ns_", "quote_age_ms_", "reason_", "exit_bid_", "deadline_ns_", "invalid_")
        )
    ]
    seed = int(spec.model.seeds[0])
    outputs = []
    receipts = []
    for fold in folds:
        train = frame[frame["session"].isin(fold["model_fit"])].copy()
        test = frame[frame["session"].isin(fold["outer_test"])].copy()
        train["_target"] = compose_target(train, spec)
        test["_target"] = compose_target(test, spec)
        train = train[np.isfinite(train["_target"].to_numpy(float))].copy()
        test = test[np.isfinite(test["_target"].to_numpy(float))].copy()
        test = test.sort_values("candidate_uid").reset_index(drop=True)
        row_hash = stable_hash(test["candidate_uid"].tolist())
        target_payload = {
            "kind": spec.target.kind,
            "fields": list(spec.target.fields),
            "weights": list(spec.target.weights),
        }
        model_payload = {
            "family": spec.model.family,
            "hyperparameters": dict(spec.model.hyperparameters),
        }
        key = prediction_cache_key(
            features=features,
            target=target_payload,
            model=model_payload,
            fold=dict(fold),
            seed=seed,
            foundation_hash=foundation_hash,
        )
        cached = cache.load(key, row_hash=row_hash)
        cache_hit = cached is not None
        if cached is None:
            model = make_model(spec, seed=seed)
            model.fit(train.loc[:, features].to_numpy(float), train["_target"].to_numpy(float))
            predictions = model.predict(test.loc[:, features].to_numpy(float))
        else:
            predictions, model = cached
        invariance = assert_mutate_future_invariant(
            test,
            feature_columns=features,
            future_columns=future_columns,
            scorer=lambda matrix, fitted=model: fitted.predict(matrix.to_numpy(float)),
        )
        if not cache_hit:
            cache.store(
                key,
                predictions,
                row_hash=row_hash,
                model=model,
                mutate_future_invariance_status=str(invariance["status"]),
            )
        test["_pred"] = np.asarray(predictions, dtype=float)
        test["fold"] = int(fold["fold"])
        outputs.append(test)
        receipts.append(
            {
                "fold": int(fold["fold"]),
                "train_sessions": len(fold["model_fit"]),
                "test_sessions": len(fold["outer_test"]),
                "train_rows": len(train),
                "test_rows": len(test),
                "cache_key": key,
                "cache_hit": cache_hit,
                "mutate_future_invariance": invariance,
            }
        )
    if len(receipts) > spec.model.max_fits:
        raise RuntimeError("OOF fold count exceeded preregistered model fit budget")
    return pd.concat(outputs, ignore_index=True), {
        "features": list(features),
        "target": {
            "kind": spec.target.kind,
            "fields": list(spec.target.fields),
            "weights": list(spec.target.weights),
        },
        "seed": seed,
        "folds": receipts,
        "fit_count": sum(not item["cache_hit"] for item in receipts),
        "cache_hit_count": sum(item["cache_hit"] for item in receipts),
    }
