"""Executable causality guards."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


class FutureMutationFailure(RuntimeError):
    pass


def assert_mutate_future_invariant(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    future_columns: Sequence[str],
    scorer: Callable[[pd.DataFrame], np.ndarray],
) -> dict[str, object]:
    baseline = np.asarray(scorer(frame.loc[:, list(feature_columns)]), dtype=float)
    mutated = frame.copy()
    for index, column in enumerate(future_columns, start=1):
        if column in mutated:
            values = pd.to_numeric(mutated[column], errors="coerce").to_numpy(float)
            mutated[column] = np.where(np.isfinite(values), values * -(index + 1) + 1_000_003.0, values)
    after = np.asarray(scorer(mutated.loc[:, list(feature_columns)]), dtype=float)
    if baseline.shape != after.shape or not np.array_equal(baseline, after, equal_nan=True):
        raise FutureMutationFailure("mutating future outcomes changed decision-time scores")
    return {
        "status": "pass",
        "row_count": int(len(frame)),
        "feature_count": len(feature_columns),
        "mutated_future_columns": [column for column in future_columns if column in frame],
    }
