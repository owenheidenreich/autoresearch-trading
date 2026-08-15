"""EXP_POLICY_ROUTER_COMPLETE_STREAM_HISTORY_V1.

Historically Protocol258. This is the complete-stream version of the
history-feature policy router. It uses the fold-aware proposal stream created by
Protocol257, so Q3/Q4/Q1/recent checks are evaluated against proposals produced
by the correct chronological training artifact for each fold.

This is an experiment, not a paper-default change. It does not download data or
call a broker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import v4.scripts.run_protocol254_policy_router_union_slot_aware as p254
import v4.scripts.run_protocol255_policy_router_history_features as p255


def main() -> int:
    original_loader = p254.load_router_proposals
    base_feature_columns = list(p254.FEATURE_COLUMNS)
    p254.ROLE_LABEL = "EXP_POLICY_ROUTER_COMPLETE_STREAM_HISTORY_V1"
    p254.HISTORICAL_ID = "Protocol258"
    p254.CANDIDATE_LABEL = "CHALLENGER_POLICY_ROUTER_COMPLETE_STREAM_HISTORY_V1"
    p254.DEFAULT_INPUT = p254.Path(
        "v4/audit/autoresearch/v4_aplus_hypothesis_257_complete_router_proposal_stream/complete_router_proposal_stream.csv"
    )
    p254.DEFAULT_OUT_DIR = p254.Path(
        "v4/audit/autoresearch/v4_aplus_hypothesis_258_policy_router_complete_stream_history"
    )
    p254.FEATURE_COLUMNS = list(dict.fromkeys([*p254.FEATURE_COLUMNS, *p255.HISTORY_FEATURES]))

    def load_with_fold_aware_history(path: p254.Path) -> pd.DataFrame:
        active_columns = list(p254.FEATURE_COLUMNS)
        p254.FEATURE_COLUMNS = base_feature_columns
        try:
            base = original_loader(path)
        finally:
            p254.FEATURE_COLUMNS = active_columns
        return add_fold_aware_history_features(base)

    p254.load_router_proposals = load_with_fold_aware_history
    return p254.main()


def add_fold_aware_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    fold_prefix = ["router_fold"] if "router_fold" in out.columns else []
    event_key = [*fold_prefix, "reported_split", "seed", "session", "decision_time"]
    event_count = out.groupby(event_key)["policy"].transform("size")
    stream_count = out.groupby(event_key)["policy"].transform("nunique")
    out["event_candidate_count"] = event_count.astype(float)
    out["event_has_both_streams"] = (stream_count >= 2).astype(float)
    out["score_rank_in_event"] = out.groupby(event_key)["score"].rank(method="first", ascending=False).astype(float)
    out["premium_rank_in_event"] = out.groupby(event_key)["entry_premium"].rank(method="first", ascending=True).astype(float)

    for column in p255.HISTORY_FEATURES:
        if column not in out.columns:
            out[column] = 0.0

    session_group = [*fold_prefix, "reported_split", "seed", "session"]
    for _, group in out.groupby(session_group, sort=False):
        group = group.sort_values(["decision_dt", "policy", "contract_id"])
        indices = list(group.index)
        times = group["decision_dt"].to_numpy()
        times_ns = np.asarray([pd.Timestamp(value).value for value in times], dtype=np.int64)
        underlying = pd.to_numeric(group["entry_underlying"], errors="coerce").to_numpy(dtype=float)
        right = group["right"].astype(str).to_numpy()
        policy = group["policy"].astype(str).to_numpy()
        for pos, row_index in enumerate(indices):
            t = times_ns[pos]
            prev_end = int(np.searchsorted(times_ns, t, side="left"))
            prev_15_start = int(np.searchsorted(times_ns, t - 15 * 60 * 1_000_000_000, side="left"))
            prev_30_start = int(np.searchsorted(times_ns, t - 30 * 60 * 1_000_000_000, side="left"))
            recent_15 = slice(prev_15_start, prev_end)
            recent_30 = slice(prev_30_start, prev_end)
            out.at[row_index, "proposal_count_prev_15m"] = float(prev_end - prev_15_start)
            out.at[row_index, "proposal_count_prev_30m"] = float(prev_end - prev_30_start)
            out.at[row_index, "same_side_count_prev_30m"] = float(np.sum(right[recent_30] == right[pos]))
            out.at[row_index, "same_policy_count_prev_30m"] = float(np.sum(policy[recent_30] == policy[pos]))
            out.at[row_index, "minutes_since_same_side"] = p255.minutes_since_last(times_ns, right == right[pos], pos)
            out.at[row_index, "minutes_since_same_policy"] = p255.minutes_since_last(times_ns, policy == policy[pos], pos)
            out.at[row_index, "underlying_change_prev_5m"] = p255.underlying_change(times_ns, underlying, pos, minutes=5)
            out.at[row_index, "underlying_change_prev_15m"] = p255.underlying_change(times_ns, underlying, pos, minutes=15)
            out.at[row_index, "underlying_change_prev_30m"] = p255.underlying_change(times_ns, underlying, pos, minutes=30)
    for column in p255.HISTORY_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
