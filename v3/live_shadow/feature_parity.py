from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureParityResult:
    n_features: int
    max_abs_diff: float
    mean_abs_diff: float
    missing_features: list[str]
    mismatched_features: list[str]

    @property
    def passed(self) -> bool:
        return not self.missing_features and not self.mismatched_features


def compare_feature_rows(
    *,
    feature_names: list[str],
    historical: dict[str, float],
    live_style: dict[str, float],
    atol: float = 1e-6,
) -> FeatureParityResult:
    missing = [name for name in feature_names if name not in live_style]
    diffs: list[float] = []
    mismatched: list[str] = []
    for name in feature_names:
        if name in missing:
            continue
        left = float(historical.get(name, np.nan))
        right = float(live_style.get(name, np.nan))
        if not np.isfinite(left) and not np.isfinite(right):
            diff = 0.0
        else:
            diff = abs(left - right)
        diffs.append(float(diff))
        if diff > atol:
            mismatched.append(name)
    return FeatureParityResult(
        n_features=len(feature_names),
        max_abs_diff=float(max(diffs) if diffs else 0.0),
        mean_abs_diff=float(np.mean(diffs) if diffs else 0.0),
        missing_features=missing,
        mismatched_features=mismatched,
    )

