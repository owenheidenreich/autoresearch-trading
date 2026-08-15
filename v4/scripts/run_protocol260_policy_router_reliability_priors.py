"""EXP_POLICY_ROUTER_RELIABILITY_PRIORS_V1.

Historically Protocol260. Protocol258 showed the complete-stream router beats
PAPER_DEFAULT_PROTOCOL101 on every required block, but loses to the
premium-blend stream on recent 2026 because some Protocol101 substitutions
block better premium-blend trades. This experiment keeps the same router action
space and adds causal reliability priors computed only from each fold's training
history.

The priors are inputs, not gates: the neural router may still choose either
stream, but it can see whether similar proposals had positive expectancy in the
allowed past. No paid data is downloaded and no broker endpoint is called.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import v4.scripts.run_protocol254_policy_router_union_slot_aware as p254
import v4.scripts.run_protocol255_policy_router_history_features as p255
import v4.scripts.run_protocol258_policy_router_complete_stream_history as p258


PRIOR_FEATURES = [
    "prior_reward_mean",
    "prior_positive_rate",
    "prior_reward_per_premium",
    "prior_observation_log1p",
    "prior_backoff_level",
]
SPLIT_ORDER = {
    "q4_2024": 0,
    "q1_2025": 1,
    "q2_2025": 2,
    "q3_2025": 3,
    "q4_2025": 4,
    "q1_2026": 5,
    "march_2026": 6,
    "recent_2026": 7,
}


def main() -> int:
    original_loader = p254.load_router_proposals
    base_feature_columns = list(p254.FEATURE_COLUMNS)
    p254.ROLE_LABEL = "EXP_POLICY_ROUTER_RELIABILITY_PRIORS_V1"
    p254.HISTORICAL_ID = "Protocol260"
    p254.CANDIDATE_LABEL = "CHALLENGER_POLICY_ROUTER_RELIABILITY_PRIORS_V1"
    p254.DEFAULT_INPUT = p254.Path(
        "v4/audit/autoresearch/v4_aplus_hypothesis_257_complete_router_proposal_stream/complete_router_proposal_stream.csv"
    )
    p254.DEFAULT_OUT_DIR = p254.Path("v4/audit/autoresearch/v4_aplus_hypothesis_260_policy_router_reliability_priors")
    p254.FEATURE_COLUMNS = list(dict.fromkeys([*p254.FEATURE_COLUMNS, *p255.HISTORY_FEATURES, *PRIOR_FEATURES]))

    def load_with_priors(path: p254.Path) -> pd.DataFrame:
        active_columns = list(p254.FEATURE_COLUMNS)
        p254.FEATURE_COLUMNS = base_feature_columns
        try:
            base = original_loader(path)
        finally:
            p254.FEATURE_COLUMNS = active_columns
        history = p258.add_fold_aware_history_features(base)
        return add_causal_reliability_priors(history)

    p254.load_router_proposals = load_with_priors
    return p254.main()


def add_causal_reliability_priors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["moneyness_bucket"] = [moneyness(right, offset) for right, offset in zip(out["right"], out["offset"])]
    out["premium_bucket"] = pd.cut(
        pd.to_numeric(out["entry_premium"], errors="coerce").fillna(-1.0),
        bins=[-1.0, 0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 1_000_000.0],
        labels=["missing", "0-500", "500-1000", "1000-1500", "1500-2000", "2000-3000", "3000-4000", "4000+"],
    ).astype(str)
    out["time_bucket"] = pd.cut(
        pd.to_numeric(out["minute_from_open"], errors="coerce").fillna(-1.0),
        bins=[-1.0, 30.0, 90.0, 180.0, 300.0, 390.0],
        labels=["open_30", "morning", "midday", "afternoon", "late"],
    ).astype(str)
    for column in PRIOR_FEATURES:
        out[column] = np.nan

    fold_lookup = {str(fold["name"]): fold for fold in p254.FOLDS}
    if "router_fold" not in out.columns:
        out["router_fold"] = ""
    for router_fold, fold_frame in out.groupby("router_fold", sort=False):
        fold = fold_lookup.get(str(router_fold))
        if not fold:
            continue
        train_splits = [str(split) for split in fold["train_splits"]]
        for split in sorted(fold_frame["reported_split"].astype(str).unique(), key=lambda item: SPLIT_ORDER.get(item, 999)):
            target_index = fold_frame[fold_frame["reported_split"].astype(str).eq(split)].index
            if split in train_splits:
                split_rank = SPLIT_ORDER.get(split, 999)
                source_splits = [item for item in train_splits if SPLIT_ORDER.get(item, 999) < split_rank]
            else:
                source_splits = train_splits
            source = fold_frame[fold_frame["reported_split"].astype(str).isin(source_splits)].copy()
            if source.empty or len(target_index) == 0:
                continue
            fill_priors(out, source, target_index)

    for column in PRIOR_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def fill_priors(out: pd.DataFrame, source: pd.DataFrame, target_index: pd.Index) -> None:
    key_sets = [
        ["policy", "right", "moneyness_bucket", "premium_bucket", "time_bucket"],
        ["policy", "right", "moneyness_bucket", "time_bucket"],
        ["policy", "right", "time_bucket"],
        ["policy", "right"],
        ["policy"],
    ]
    remaining = out.loc[target_index, PRIOR_FEATURES[0]].isna()
    for level, keys in enumerate(key_sets, start=1):
        if not bool(remaining.any()):
            break
        stats = source_stats(source, keys)
        if stats.empty:
            continue
        remaining_index = remaining[remaining].index
        lookup_target = out.loc[remaining_index, keys].reset_index().merge(stats, on=keys, how="left").set_index("index")
        matched = lookup_target["prior_observation_log1p"].notna()
        if not bool(matched.any()):
            continue
        matched_index = lookup_target.index[matched]
        for column in ["prior_reward_mean", "prior_positive_rate", "prior_reward_per_premium", "prior_observation_log1p"]:
            out.loc[matched_index, column] = pd.to_numeric(lookup_target.loc[matched_index, column], errors="coerce")
        out.loc[matched_index, "prior_backoff_level"] = float(level)
        remaining = out.loc[target_index, PRIOR_FEATURES[0]].isna()


def source_stats(source: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    tmp = source.copy()
    tmp["reward"] = pd.to_numeric(tmp["candidate_pnl"], errors="coerce").fillna(0.0)
    tmp["positive"] = (tmp["reward"] > 0.0).astype(float)
    tmp["premium"] = pd.to_numeric(tmp["entry_premium"], errors="coerce").clip(lower=0.0).fillna(0.0)
    grouped = tmp.groupby(keys, dropna=False)
    stats = grouped.agg(
        reward_sum=("reward", "sum"),
        reward_mean=("reward", "mean"),
        positive_rate=("positive", "mean"),
        premium_sum=("premium", "sum"),
        observation_count=("reward", "size"),
    ).reset_index()
    stats["prior_reward_mean"] = stats["reward_mean"].astype(float)
    stats["prior_positive_rate"] = stats["positive_rate"].astype(float)
    stats["prior_reward_per_premium"] = np.divide(
        stats["reward_sum"],
        stats["premium_sum"].replace(0.0, np.nan),
    ).replace([np.inf, -np.inf], np.nan)
    stats["prior_observation_log1p"] = np.log1p(stats["observation_count"].astype(float))
    return stats[[*keys, "prior_reward_mean", "prior_positive_rate", "prior_reward_per_premium", "prior_observation_log1p"]]


def moneyness(right: Any, offset: Any) -> str:
    try:
        value = float(offset)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(value):
        return "unknown"
    if abs(value) <= 2.5:
        return "ATM"
    side = str(right)
    if side == "C":
        return "ITM" if value < 0.0 else "OTM"
    if side == "P":
        return "ITM" if value > 0.0 else "OTM"
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
