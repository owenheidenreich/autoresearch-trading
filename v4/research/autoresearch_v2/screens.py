"""Cheap exact-pair component screens run before serial replay."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .dataset import target_column


Pair = tuple[str, dict[str, Any], dict[str, Any]]


def _session_deltas(pairs: Iterable[Pair], *, target: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for _, arm, baseline in pairs:
        grouped.setdefault(str(arm["session"]), []).append(
            float(arm[target]) - float(baseline[target])
        )
    return {session: float(np.mean(values)) for session, values in sorted(grouped.items())}


def select_intents(frame: pd.DataFrame, *, threshold: float, policy: str) -> pd.DataFrame:
    target = target_column(policy)
    eligible = frame[
        np.isfinite(frame["_pred"].to_numpy(float))
        & np.isfinite(frame[target].to_numpy(float))
        & (frame["_pred"].to_numpy(float) > float(threshold))
    ].copy()
    eligible = eligible.sort_values(
        ["session", "decision_time_ns", "_pred", "candidate_uid"],
        ascending=[True, True, False, True],
    )
    return eligible.groupby(["session", "decision_time_ns"], sort=True).head(1).reset_index(drop=True)


def exit_screen(
    intents: pd.DataFrame, *, arm_policy: str, baseline_policy: str
) -> tuple[dict[str, float], list[Pair], dict[str, Any]]:
    arm_target = target_column(arm_policy)
    baseline_target = target_column(baseline_policy)
    pairs: list[Pair] = []
    grouped: dict[str, list[float]] = {}
    for row in intents.to_dict("records"):
        if not np.isfinite([row[arm_target], row[baseline_target]]).all():
            continue
        signal = f"{row['session']}|{row['decision_time_ns']}"
        pairs.append((signal, row, row))
        grouped.setdefault(str(row["session"]), []).append(
            float(row[arm_target]) - float(row[baseline_target])
        )
    deltas = {session: float(np.mean(values)) for session, values in sorted(grouped.items())}
    return deltas, pairs, {"pair_count": len(pairs), "session_count": len(deltas)}


def direction_screen(
    frame: pd.DataFrame, *, threshold: float, policy: str, premium_caliper: float = 1.5
) -> tuple[dict[str, float], list[Pair], dict[str, Any]]:
    target = target_column(policy)
    pairs: list[Pair] = []
    eligible_decisions = 0
    for (session, decision), group in frame.groupby(["session", "decision_time_ns"], sort=True):
        calls = {
            int(row["offset"]): row
            for row in group[group["right"] == "C"].to_dict("records")
        }
        puts = {
            int(row["offset"]): row
            for row in group[group["right"] == "P"].to_dict("records")
        }
        candidates = []
        for offset, call in calls.items():
            put = puts.get(-offset)
            if put is None or abs(float(call["entry_ask"]) - float(put["entry_ask"])) > premium_caliper:
                continue
            candidates.append((abs(float(call["entry_ask"]) - float(put["entry_ask"])), abs(offset), call, put))
        if not candidates:
            continue
        eligible_decisions += 1
        _, _, call, put = min(candidates, key=lambda item: (item[0], item[1], -max(item[2]["_pred"], item[3]["_pred"])))
        if max(float(call["_pred"]), float(put["_pred"])) <= threshold:
            continue
        arm_row, baseline_row = (call, put) if float(call["_pred"]) >= float(put["_pred"]) else (put, call)
        if np.isfinite([arm_row[target], baseline_row[target]]).all():
            pairs.append((f"{session}|{decision}", arm_row, baseline_row))
    return _session_deltas(pairs, target=target), pairs, {
        "pair_count": len(pairs),
        "eligible_decisions": eligible_decisions,
        "premium_caliper_dollars": premium_caliper,
    }


def contract_screen(
    frame: pd.DataFrame, *, threshold: float, policy: str, premium_caliper: float = 1.5
) -> tuple[dict[str, float], list[Pair], dict[str, Any]]:
    target = target_column(policy)
    pairs: list[Pair] = []
    for (session, decision, right), group in frame.groupby(
        ["session", "decision_time_ns", "right"], sort=True
    ):
        ranked = group.sort_values(["_pred", "candidate_uid"], ascending=[False, True])
        arm = ranked.iloc[0]
        if float(arm["_pred"]) <= threshold:
            continue
        baseline_pool = group[
            (group["entry_ask"] - float(arm["entry_ask"])).abs() <= premium_caliper
        ].sort_values(["nearest_atm_rank", "candidate_uid"])
        if baseline_pool.empty:
            continue
        baseline = baseline_pool.iloc[0]
        if np.isfinite([arm[target], baseline[target]]).all():
            pairs.append(
                (
                    f"{session}|{decision}|{right}",
                    arm.to_dict(),
                    baseline.to_dict(),
                )
            )
    return _session_deltas(pairs, target=target), pairs, {
        "pair_count": len(pairs),
        "premium_caliper_dollars": premium_caliper,
    }


def timing_screen(
    frame: pd.DataFrame,
    intents: pd.DataFrame,
    *,
    delay_minutes: int,
    policy: str,
) -> tuple[dict[str, float], list[Pair], dict[str, Any]]:
    target = target_column(policy)
    lookup = {
        (str(row["session"]), int(row["decision_time_ns"]), str(row["contract_id"])): row
        for row in frame.to_dict("records")
    }
    pairs: list[Pair] = []
    for now in intents.to_dict("records"):
        later = lookup.get(
            (
                str(now["session"]),
                int(now["decision_time_ns"]) + int(delay_minutes * 60 * 1_000_000_000),
                str(now["contract_id"]),
            )
        )
        if later is None or not np.isfinite([now[target], later[target]]).all():
            continue
        pairs.append((f"{now['session']}|{now['decision_time_ns']}", later, now))
    return _session_deltas(pairs, target=target), pairs, {
        "pair_count": len(pairs),
        "delay_minutes": int(delay_minutes),
        "source_intent_count": len(intents),
        "paired_coverage": float(len(pairs) / len(intents)) if len(intents) else 0.0,
    }


def target_screen(
    candidate_oof: pd.DataFrame,
    baseline_oof: pd.DataFrame,
    *,
    threshold: float,
    policy: str,
    premium_caliper: float = 1.5,
    moneyness_caliper: int = 15,
) -> tuple[dict[str, float], list[Pair], dict[str, Any]]:
    target = target_column(policy)
    candidate = select_intents(candidate_oof, threshold=threshold, policy=policy)
    baseline = select_intents(baseline_oof, threshold=threshold, policy=policy)
    baseline_lookup = {
        (str(row["session"]), int(row["decision_time_ns"])): row
        for row in baseline.to_dict("records")
    }
    pairs: list[Pair] = []
    for arm in candidate.to_dict("records"):
        other = baseline_lookup.get((str(arm["session"]), int(arm["decision_time_ns"])))
        if other is None:
            continue
        if arm["right"] != other["right"]:
            continue
        if abs(float(arm["entry_ask"]) - float(other["entry_ask"])) > premium_caliper:
            continue
        if abs(int(arm["abs_moneyness"]) - int(other["abs_moneyness"])) > moneyness_caliper:
            continue
        pairs.append((f"{arm['session']}|{arm['decision_time_ns']}", arm, other))
    return _session_deltas(pairs, target=target), pairs, {
        "pair_count": len(pairs),
        "candidate_intents": len(candidate),
        "baseline_intents": len(baseline),
        "premium_caliper_dollars": premium_caliper,
        "moneyness_caliper_points": moneyness_caliper,
    }
