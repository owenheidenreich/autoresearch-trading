"""Paired session statistics, power checks, and family maxT correction."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from scipy.stats import t as student_t


def paired_summary(values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, float | int | None]:
    sample = np.asarray([float(value) for _, value in sorted(values.items())], dtype=float)
    sample = sample[np.isfinite(sample)]
    n = int(len(sample))
    if n == 0:
        return {"n_sessions": 0, "mean": None, "std": None, "se": None, "ci_low": None, "ci_high": None, "t": None, "p_one_sided": None, "mde80": None}
    mean = float(np.mean(sample))
    std = float(np.std(sample, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n else float("inf")
    critical = float(student_t.ppf(1.0 - alpha / 2.0, n - 1)) if n > 1 else float("inf")
    statistic = mean / se if se > 0 else (float("inf") if mean > 0 else 0.0)
    p_one = float(student_t.sf(statistic, n - 1)) if n > 1 else 1.0
    mde80 = 2.80 * se
    return {
        "n_sessions": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci_low": mean - critical * se if math.isfinite(critical) else None,
        "ci_high": mean + critical * se if math.isfinite(critical) else None,
        "t": statistic,
        "p_one_sided": p_one,
        "mde80": mde80,
    }


def session_blocked_max_t(
    contrasts: Mapping[str, Mapping[str, float]],
    *,
    permutations: int,
    seed: int = 101,
) -> dict[str, dict[str, float | int | None]]:
    sessions = sorted({session for values in contrasts.values() for session in values})
    names = sorted(contrasts)
    matrix = np.full((len(sessions), len(names)), np.nan, dtype=float)
    for column, name in enumerate(names):
        values = contrasts[name]
        for row, session in enumerate(sessions):
            if session in values:
                matrix[row, column] = float(values[session])

    def t_values(values: np.ndarray) -> np.ndarray:
        count = np.sum(np.isfinite(values), axis=0)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0, ddof=1)
        se = std / np.sqrt(count)
        return np.divide(mean, se, out=np.zeros_like(mean), where=(count > 1) & (se > 0))

    observed = t_values(matrix)
    exceed = np.zeros(len(names), dtype=int)
    rng = np.random.default_rng(seed)
    for _ in range(int(permutations)):
        signs = rng.choice((-1.0, 1.0), size=(len(sessions), 1))
        null_max = float(np.nanmax(t_values(matrix * signs)))
        exceed += null_max >= observed
    output = {}
    for index, name in enumerate(names):
        summary = paired_summary(contrasts[name])
        summary["maxT_p_one_sided"] = float((exceed[index] + 1) / (permutations + 1))
        output[name] = summary
    return output
