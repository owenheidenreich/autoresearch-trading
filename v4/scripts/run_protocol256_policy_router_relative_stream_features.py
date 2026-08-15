"""EXP_POLICY_ROUTER_RELATIVE_STREAM_FEATURES_V1.

Historically Protocol256. Protocol255 showed that short history helps, but the
router still over-substitutes Protocol101 proposals in recent 2026. This adds
causal same-decision comparative features so the model can compare the two
proposal streams directly:

* premium ratio versus the other stream,
* score advantage versus the event max,
* same-side / same-contract stream agreement,
* relative offset and spread.

No paid data is downloaded. No broker endpoint is called. This does not change
the paper default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import v4.scripts.run_protocol254_policy_router_union_slot_aware as p254
import v4.scripts.run_protocol255_policy_router_history_features as p255


RELATIVE_FEATURES = [
    "event_min_premium",
    "event_median_premium",
    "candidate_premium_minus_event_min",
    "candidate_premium_ratio_to_event_min",
    "event_max_score",
    "candidate_score_minus_event_max",
    "event_min_spread_pct",
    "candidate_spread_pct_minus_event_min",
    "other_stream_available",
    "streams_same_side",
    "streams_same_contract",
    "other_stream_premium",
    "premium_minus_other_stream",
    "premium_ratio_to_other_stream",
    "offset_abs_minus_other_stream",
    "score_margin_minus_other_stream",
]


def main() -> int:
    original_loader = p254.load_router_proposals
    base_feature_columns = list(p254.FEATURE_COLUMNS)
    p254.ROLE_LABEL = "EXP_POLICY_ROUTER_RELATIVE_STREAM_FEATURES_V1"
    p254.HISTORICAL_ID = "Protocol256"
    p254.CANDIDATE_LABEL = "CHALLENGER_POLICY_ROUTER_RELATIVE_STREAM_FEATURES_V1"
    p254.DEFAULT_OUT_DIR = p254.Path("v4/audit/autoresearch/v4_aplus_hypothesis_256_policy_router_relative_stream_features")
    p254.FEATURE_COLUMNS = list(dict.fromkeys([*p254.FEATURE_COLUMNS, *p255.HISTORY_FEATURES, *RELATIVE_FEATURES]))

    def load_with_relative_history(path: p254.Path) -> pd.DataFrame:
        active_columns = list(p254.FEATURE_COLUMNS)
        p254.FEATURE_COLUMNS = base_feature_columns
        try:
            base = original_loader(path)
        finally:
            p254.FEATURE_COLUMNS = active_columns
        return add_relative_features(p255.add_history_features(base))

    p254.load_router_proposals = load_with_relative_history
    return p254.main()


def add_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    event_key = ["reported_split", "seed", "session", "decision_time"]
    out["event_min_premium"] = out.groupby(event_key)["entry_premium"].transform("min")
    out["event_median_premium"] = out.groupby(event_key)["entry_premium"].transform("median")
    out["candidate_premium_minus_event_min"] = out["entry_premium"] - out["event_min_premium"]
    out["candidate_premium_ratio_to_event_min"] = out["entry_premium"] / out["event_min_premium"].replace(0.0, np.nan)
    out["event_max_score"] = out.groupby(event_key)["score"].transform("max")
    out["candidate_score_minus_event_max"] = out["score"] - out["event_max_score"]
    out["event_min_spread_pct"] = out.groupby(event_key)["entry_spread_pct"].transform("min")
    out["candidate_spread_pct_minus_event_min"] = out["entry_spread_pct"] - out["event_min_spread_pct"]
    for column in [
        "other_stream_available",
        "streams_same_side",
        "streams_same_contract",
        "other_stream_premium",
        "premium_minus_other_stream",
        "premium_ratio_to_other_stream",
        "offset_abs_minus_other_stream",
        "score_margin_minus_other_stream",
    ]:
        out[column] = 0.0

    for _, group in out.groupby(event_key, sort=False):
        if group["policy"].nunique() < 2:
            continue
        for row_index, row in group.iterrows():
            other = group[group["policy"] != row["policy"]]
            if other.empty:
                continue
            # If both streams somehow offer multiple contracts at the same timestamp,
            # compare against the highest-score proposal from the other stream.
            other_row = other.sort_values("score", ascending=False).iloc[0]
            out.at[row_index, "other_stream_available"] = 1.0
            out.at[row_index, "streams_same_side"] = 1.0 if str(row["right"]) == str(other_row["right"]) else 0.0
            out.at[row_index, "streams_same_contract"] = 1.0 if str(row["contract_id"]) == str(other_row["contract_id"]) else 0.0
            other_premium = float(other_row["entry_premium"])
            out.at[row_index, "other_stream_premium"] = other_premium
            out.at[row_index, "premium_minus_other_stream"] = float(row["entry_premium"]) - other_premium
            out.at[row_index, "premium_ratio_to_other_stream"] = (
                float(row["entry_premium"]) / other_premium if other_premium > 0.0 else 0.0
            )
            out.at[row_index, "offset_abs_minus_other_stream"] = abs(float(row["offset"])) - abs(float(other_row["offset"]))
            out.at[row_index, "score_margin_minus_other_stream"] = float(row["score_margin"]) - float(other_row["score_margin"])

    for column in RELATIVE_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
