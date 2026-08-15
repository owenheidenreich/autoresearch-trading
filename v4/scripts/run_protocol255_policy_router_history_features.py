"""EXP_POLICY_ROUTER_HISTORY_FEATURES_V1.

Historically Protocol255. This is a narrow follow-up to Protocol254. The union
router had useful Q1/March signal but was too memoryless and underperformed the
premium-blend stream on recent 2026. This runner keeps the same proposal union
and action space, but adds causal short-history features:

* recent proposal density,
* recent same-side and same-policy clustering,
* minutes since the same side/policy appeared,
* event-level stream agreement,
* causal underlying drift over prior windows.

It does not download data, call a broker, or change the paper default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import v4.scripts.run_protocol254_policy_router_union_slot_aware as p254


HISTORY_FEATURES = [
    "event_candidate_count",
    "event_has_both_streams",
    "score_rank_in_event",
    "premium_rank_in_event",
    "proposal_count_prev_15m",
    "proposal_count_prev_30m",
    "same_side_count_prev_30m",
    "same_policy_count_prev_30m",
    "minutes_since_same_side",
    "minutes_since_same_policy",
    "underlying_change_prev_5m",
    "underlying_change_prev_15m",
    "underlying_change_prev_30m",
]


def main() -> int:
    original_loader = p254.load_router_proposals
    base_feature_columns = list(p254.FEATURE_COLUMNS)
    p254.ROLE_LABEL = "EXP_POLICY_ROUTER_HISTORY_FEATURES_V1"
    p254.HISTORICAL_ID = "Protocol255"
    p254.CANDIDATE_LABEL = "CHALLENGER_POLICY_ROUTER_HISTORY_FEATURES_V1"
    p254.DEFAULT_OUT_DIR = p254.Path("v4/audit/autoresearch/v4_aplus_hypothesis_255_policy_router_history_features")
    p254.FEATURE_COLUMNS = list(dict.fromkeys([*p254.FEATURE_COLUMNS, *HISTORY_FEATURES]))

    def load_with_history(path: p254.Path) -> pd.DataFrame:
        active_columns = list(p254.FEATURE_COLUMNS)
        p254.FEATURE_COLUMNS = base_feature_columns
        try:
            base = original_loader(path)
        finally:
            p254.FEATURE_COLUMNS = active_columns
        return add_history_features(base)

    p254.load_router_proposals = load_with_history
    return p254.main()


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    event_key = ["reported_split", "seed", "session", "decision_time"]
    event_count = out.groupby(event_key)["policy"].transform("size")
    stream_count = out.groupby(event_key)["policy"].transform("nunique")
    out["event_candidate_count"] = event_count.astype(float)
    out["event_has_both_streams"] = (stream_count >= 2).astype(float)
    out["score_rank_in_event"] = out.groupby(event_key)["score"].rank(method="first", ascending=False).astype(float)
    out["premium_rank_in_event"] = out.groupby(event_key)["entry_premium"].rank(method="first", ascending=True).astype(float)

    for column in HISTORY_FEATURES:
        if column not in out.columns:
            out[column] = 0.0

    for _, group in out.groupby(["reported_split", "seed", "session"], sort=False):
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
            out.at[row_index, "minutes_since_same_side"] = minutes_since_last(times_ns, right == right[pos], pos)
            out.at[row_index, "minutes_since_same_policy"] = minutes_since_last(times_ns, policy == policy[pos], pos)
            out.at[row_index, "underlying_change_prev_5m"] = underlying_change(times_ns, underlying, pos, minutes=5)
            out.at[row_index, "underlying_change_prev_15m"] = underlying_change(times_ns, underlying, pos, minutes=15)
            out.at[row_index, "underlying_change_prev_30m"] = underlying_change(times_ns, underlying, pos, minutes=30)
    for column in HISTORY_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def minutes_since_last(times_ns: np.ndarray, mask: np.ndarray, pos: int) -> float:
    prior = np.flatnonzero(mask[:pos])
    if len(prior) == 0:
        return 999.0
    return float((times_ns[pos] - times_ns[int(prior[-1])]) / 60_000_000_000.0)


def underlying_change(times_ns: np.ndarray, underlying: np.ndarray, pos: int, *, minutes: int) -> float:
    t = times_ns[pos]
    target = t - int(minutes) * 60 * 1_000_000_000
    prior_idx = int(np.searchsorted(times_ns, target, side="right")) - 1
    if prior_idx < 0 or not np.isfinite(underlying[pos]) or not np.isfinite(underlying[prior_idx]):
        return 0.0
    return float(underlying[pos] - underlying[prior_idx])


if __name__ == "__main__":
    raise SystemExit(main())
