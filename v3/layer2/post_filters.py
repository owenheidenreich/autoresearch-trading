"""L2 post-selection veto layer (Phase 2b).

Loads the frozen filter spec from `post_filters_v0` and exposes:

  - `apply_avoid_filters_mask(pred_df)` -> boolean mask, True = veto
  - `report_favor_diagnostics(chosen_df)` -> dict of per-rule metrics
  - `apply_avoid_filters(chosen_df)` -> filtered chosen_df (drop-only mode,
    used in offline analyses where the per-bar candidate frame is not
    available)

Runtime semantics: the live pipeline composes this mask into
`_select_daily_trades`'s eligible_mask, which is rescan-after-veto at
the bar→day level (see v3/reference/l2_post_filter_semantics_2026_04_26.md).
This module does NOT modify pred_df in-place; callers AND the result
into their existing eligibility expression.

Favor rules are diagnostic-only — `report_favor_diagnostics` computes
their per-rule kept-distribution metrics for monitoring but never
modifies the trade set. There is no boost / sizing / priority logic in
this Phase 2 implementation.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from v3.layer2.post_filters_v0 import (
    ALL_RULES,
    AVOID_RULES,
    FAVOR_RULES,
    FILTER_SET_ID,
    FilterRule,
    attach_buckets,
)

logger = logging.getLogger(__name__)


def _ensure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure sigma_b / iv_b / cell columns exist using the frozen tertiles.

    Returns a frame containing the bucket columns; original is unchanged
    unless they're already present.
    """
    if {"sigma_b", "iv_b", "cell"}.issubset(df.columns):
        return df
    if "sigma_pos" not in df.columns or "iv_percentile" not in df.columns:
        raise KeyError(
            "post_filters: pred_df must contain sigma_pos and iv_percentile "
            "to apply cell-conditional rules."
        )
    return attach_buckets(df)


def _rule_mask(df: pd.DataFrame, rule: FilterRule) -> np.ndarray:
    return df.apply(rule.predicate, axis=1).astype(bool).values


def apply_avoid_filters_mask(
    pred_df: pd.DataFrame, rules: Iterable[FilterRule] | None = None
) -> np.ndarray:
    """Return a boolean array (length len(pred_df)) — True = veto.

    Caller composes this with the existing eligibility mask:
      eligible_mask = base_eligibility & ~apply_avoid_filters_mask(pred_df)
    """
    rules = list(rules) if rules is not None else list(AVOID_RULES)
    if not rules:
        return np.zeros(len(pred_df), dtype=bool)
    df = _ensure_buckets(pred_df)
    # Map the unified policy's column conventions to the predicate's
    # expected names. Predicates read `side` (call/put); pred_df uses
    # `chosen_side`. Predicates read `bar_index` and `cell`/`sigma_b`/`iv_b`
    # which are present.
    if "side" not in df.columns and "chosen_side" in df.columns:
        df = df.assign(side=df["chosen_side"])
    mask = np.zeros(len(df), dtype=bool)
    veto_counts: dict[str, int] = {}
    for rule in rules:
        m = _rule_mask(df, rule)
        veto_counts[rule.rule_id] = int(m.sum())
        mask |= m
    if veto_counts:
        logger.info(
            "post_filters[%s] veto counts: %s (union=%d / %d rows)",
            FILTER_SET_ID,
            veto_counts,
            int(mask.sum()),
            len(df),
        )
    return mask


def apply_avoid_filters(
    chosen_df: pd.DataFrame, rules: Iterable[FilterRule] | None = None
) -> pd.DataFrame:
    """Drop-only convenience: returns chosen_df with vetoed rows removed.

    Used in offline analyses (Phase 1) where we operate on the post-
    selection chosen_trades frame and rescan is not possible.
    """
    mask = apply_avoid_filters_mask(chosen_df, rules)
    return chosen_df[~mask].copy()


def report_favor_diagnostics(
    chosen_df: pd.DataFrame, hl_col: str = "hl"
) -> dict:
    """Compute per-favor-rule kept-distribution metrics.

    Does NOT modify chosen_df. Returns a dict keyed by rule_id with:
      - n_match (count of trades matching the rule)
      - mean_hl (over matching rows)
      - sum_hl
      - profit_factor (over matching rows)

    Used for monitoring / log output post-FW. Favor rules are explicitly
    NOT applied to admission, sizing, or priority in Phase 2.
    """
    df = _ensure_buckets(chosen_df)
    if "side" not in df.columns and "chosen_side" in df.columns:
        df = df.assign(side=df["chosen_side"])
    out: dict[str, dict] = {"_filter_set_id": FILTER_SET_ID}
    for rule in FAVOR_RULES:
        m = _rule_mask(df, rule)
        if not m.any() or hl_col not in df.columns:
            out[rule.rule_id] = {"n_match": int(m.sum()), "mean_hl": float("nan"),
                                 "sum_hl": float("nan"), "pf": float("nan")}
            continue
        hls = df.loc[m, hl_col].dropna().values
        pos = float(hls[hls > 0].sum())
        neg = float(hls[hls < 0].sum())
        pf = float("inf") if neg == 0 and pos > 0 else (
            float(pos / abs(neg)) if neg < 0 else float("nan")
        )
        out[rule.rule_id] = {
            "n_match": int(m.sum()),
            "mean_hl": float(np.mean(hls)) if len(hls) else float("nan"),
            "sum_hl": float(np.sum(hls)) if len(hls) else float("nan"),
            "pf": pf,
            "tags": list(rule.tags),
        }
    return out
