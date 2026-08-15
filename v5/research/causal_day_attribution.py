"""Outcome-blind attribution primitives for the closed V5 magnitude policy."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from v5.research.causal_day_architectures import CausalPolicyBatch, ENTRY_ROLES


class AttributionError(RuntimeError):
    """The frozen attribution contract was violated."""


TIME_BANDS = (
    "opening_0935_0949",
    "magic_0950_1010",
    "morning_rest_1011_1245",
    "afternoon_pre_algo_1246_1319",
    "algo_1320_1340",
    "afternoon_rest_1341_1500",
)


def clock_baseline() -> np.ndarray:
    """Mean raw five-channel clock vector over all 09:35--15:00 decisions."""

    values = []
    for minute_number in range(9 * 60 + 35, 15 * 60 + 1):
        elapsed = (minute_number - 570) / 390.0
        angle = 2.0 * pi * elapsed
        values.append(
            [
                elapsed,
                (960 - minute_number) / 390.0,
                float(minute_number < 766),
                sin(angle),
                cos(angle),
            ]
        )
    return np.asarray(values, dtype=np.float32).mean(axis=0)


def time_band(minute: str) -> str:
    value = str(minute)
    if "09:35" <= value <= "09:49":
        return TIME_BANDS[0]
    if "09:50" <= value <= "10:10":
        return TIME_BANDS[1]
    if "10:11" <= value <= "12:45":
        return TIME_BANDS[2]
    if "12:46" <= value <= "13:19":
        return TIME_BANDS[3]
    if "13:20" <= value <= "13:40":
        return TIME_BANDS[4]
    if "13:41" <= value <= "15:00":
        return TIME_BANDS[5]
    raise AttributionError(f"minute is outside the frozen entry clock: {minute}")


@dataclass(frozen=True)
class DecomposedScores:
    full: Tensor
    state_offset: Tensor
    contract_node: Tensor
    no_candle_state_offset: Tensor
    no_ladder_summary_offset: Tensor
    neutral_clock_offset: Tensor
    reconstruction_max_abs_error: float


def _masked_ladder_summary(nodes: Tensor, mask: Tensor) -> Tensor:
    visible = torch.where(mask.unsqueeze(-1), nodes, torch.zeros_like(nodes))
    count = mask.sum(dim=1, keepdim=True).clamp(min=1).to(nodes.dtype)
    mean = visible.sum(dim=1) / count
    negative = torch.finfo(nodes.dtype).min
    maximum = torch.where(mask.unsqueeze(-1), nodes, negative).max(dim=1).values
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


def _encoder_components(model: torch.nn.Module, batch: CausalPolicyBatch) -> tuple[Tensor, Tensor, Tensor]:
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "candle_gru"):
        raise AttributionError("V5 attribution requires the declared neural shared encoder")
    batch.validate(encoder.dimensions)
    candle_values = torch.where(
        batch.candle_mask.unsqueeze(-1), batch.candles, torch.zeros_like(batch.candles)
    )
    projected = torch.tanh(encoder.candle_projection(candle_values))
    sequence, _ = encoder.candle_gru(projected)
    last = batch.candle_mask.sum(dim=1) - 1
    candle_state = sequence[
        torch.arange(batch.batch_size, device=last.device), last
    ]

    ladder_values = torch.where(
        batch.ladder_mask.unsqueeze(-1), batch.ladder, torch.zeros_like(batch.ladder)
    )
    nodes = encoder.ladder_node(ladder_values)
    ladder_summary = _masked_ladder_summary(nodes, batch.ladder_mask)
    return candle_state, ladder_summary, nodes


def _fused_state(
    model: torch.nn.Module,
    batch: CausalPolicyBatch,
    *,
    candle_state: Tensor,
    ladder_summary: Tensor,
    clock: Tensor,
) -> Tensor:
    return model.encoder.fusion(
        torch.cat(
            (
                candle_state,
                ladder_summary,
                batch.account,
                batch.position,
                clock,
            ),
            dim=-1,
        )
    )


@torch.no_grad()
def decompose_contract_scores(
    model: torch.nn.Module,
    batch: CausalPolicyBatch,
) -> DecomposedScores:
    """Split scores into the common state offset and contract-only term.

    ``ContractHead`` is a linear map over ``[state, node]``. Consequently
    ``score_i - score_j`` cancels the complete candle/clock/global-ladder
    state for any two contracts in the same routed minute. This function
    reproduces that identity numerically from each fitted checkpoint.
    """

    if not hasattr(model, "entry_heads"):
        raise AttributionError("V5 attribution requires shared entry heads")
    if any(role not in ENTRY_ROLES for role in batch.roles):
        raise AttributionError("attribution batch may contain only entry roles")
    candle_state, ladder_summary, nodes = _encoder_components(model, batch)
    full_state = _fused_state(
        model,
        batch,
        candle_state=candle_state,
        ladder_summary=ladder_summary,
        clock=batch.clock,
    )
    no_candle_state = _fused_state(
        model,
        batch,
        candle_state=torch.zeros_like(candle_state),
        ladder_summary=ladder_summary,
        clock=batch.clock,
    )
    no_ladder_state = _fused_state(
        model,
        batch,
        candle_state=candle_state,
        ladder_summary=torch.zeros_like(ladder_summary),
        clock=batch.clock,
    )
    baseline = torch.as_tensor(
        clock_baseline(), dtype=batch.clock.dtype, device=batch.clock.device
    ).unsqueeze(0).expand_as(batch.clock)
    neutral_clock_state = _fused_state(
        model,
        batch,
        candle_state=candle_state,
        ladder_summary=ladder_summary,
        clock=baseline,
    )

    b, n = batch.batch_size, batch.ladder.shape[1]
    full = batch.candles.new_full((b, n), -torch.inf)
    node_term = batch.candles.new_full((b, n), -torch.inf)
    offsets = batch.candles.new_full((b,), torch.nan)
    no_candle_offsets = offsets.clone()
    no_ladder_offsets = offsets.clone()
    neutral_clock_offsets = offsets.clone()
    hidden = int(nodes.shape[-1])

    for role, head in model.entry_heads.items():
        selected = torch.tensor(
            [value == role for value in batch.roles],
            dtype=torch.bool,
            device=batch.candles.device,
        )
        if not selected.any():
            continue
        weight = head.contract.weight.squeeze(0)
        if int(weight.numel()) != hidden * 2:
            raise AttributionError("entry head is not the declared additive state/node map")
        state_weight, node_weight = weight[:hidden], weight[hidden:]
        bias = head.contract.bias.squeeze(0)
        role_offset = full_state @ state_weight + bias
        role_node = nodes @ node_weight
        offsets[selected] = role_offset[selected]
        no_candle_offsets[selected] = (no_candle_state @ state_weight + bias)[selected]
        no_ladder_offsets[selected] = (no_ladder_state @ state_weight + bias)[selected]
        neutral_clock_offsets[selected] = (neutral_clock_state @ state_weight + bias)[selected]
        node_term[selected] = role_node[selected]
        full[selected] = role_offset[selected].unsqueeze(1) + role_node[selected]

    full = full.masked_fill(~batch.entry_action_mask, -torch.inf)
    node_term = node_term.masked_fill(~batch.entry_action_mask, -torch.inf)
    expected = model(batch).contract_logits
    usable = batch.entry_action_mask
    error = float(torch.max(torch.abs(full[usable] - expected[usable])).cpu())
    if not torch.isfinite(offsets).all() or error > 1e-5:
        raise AttributionError(f"additive score reconstruction failed: {error}")
    return DecomposedScores(
        full=full,
        state_offset=offsets,
        contract_node=node_term,
        no_candle_state_offset=no_candle_offsets,
        no_ladder_summary_offset=no_ladder_offsets,
        neutral_clock_offset=neutral_clock_offsets,
        reconstruction_max_abs_error=error,
    )


def activation_drift(
    scores: pd.DataFrame,
    *,
    fold_cutoffs: dict[int, float],
) -> dict[str, Any]:
    required = {"session", "entry_minute", "fold", "full_score", "state_offset"}
    missing = sorted(required - set(scores))
    if missing:
        raise AttributionError(f"activation frame missing columns: {missing}")
    minute = (
        scores.groupby(["session", "entry_minute", "fold"], as_index=False, sort=True)
        .agg(minute_max_score=("full_score", "max"), state_offset=("state_offset", "first"))
    )
    minute["time_band"] = minute["entry_minute"].map(time_band)
    folds = []
    first_crossings = []
    for fold in range(1, 6):
        part = minute[minute["fold"].astype(int).eq(fold)].copy()
        if part.empty or fold not in fold_cutoffs:
            raise AttributionError(f"fold {fold} is absent from activation attribution")
        cutoff = float(fold_cutoffs[fold])
        part["crossed"] = part["minute_max_score"].ge(cutoff)
        crossing = part[part["crossed"]]
        first = (
            crossing.sort_values(["session", "entry_minute"], kind="mergesort")
            .drop_duplicates("session", keep="first")
        )
        first_crossings.extend(
            {
                "fold": fold,
                "session": str(row.session),
                "entry_minute": str(row.entry_minute),
            }
            for row in first.itertuples(index=False)
        )
        maximum = float(part["minute_max_score"].max())
        folds.append(
            {
                "fold": fold,
                "cutoff_points": cutoff,
                "scored_sessions": int(part["session"].nunique()),
                "scored_minutes": len(part),
                "scored_max_points": maximum,
                "cutoff_above_every_scored_minute": bool(cutoff > maximum),
                "cutoff_percentile_in_scored_minutes": float(
                    part["minute_max_score"].lt(cutoff).mean()
                ),
                "crossing_minutes": len(crossing),
                "crossing_sessions": int(crossing["session"].nunique()),
                "first_crossing_at_0935_0939": int(
                    first["entry_minute"].between("09:35", "09:39").sum()
                ),
            }
        )
    first_frame = pd.DataFrame(first_crossings)
    activated = len(first_frame)
    opening = (
        int(first_frame["entry_minute"].between("09:35", "09:39").sum())
        if activated
        else 0
    )
    band_rows = []
    for (fold, band), part in minute.groupby(["fold", "time_band"], sort=True):
        band_rows.append(
            {
                "fold": int(fold),
                "time_band": str(band),
                "minutes": len(part),
                "mean_minute_max_score": float(part["minute_max_score"].mean()),
                "median_minute_max_score": float(part["minute_max_score"].median()),
                "mean_state_offset": float(part["state_offset"].mean()),
            }
        )
    exact_times = []
    for named in ("10:00", "13:30"):
        for fold, part in minute[minute["entry_minute"].eq(named)].groupby("fold", sort=True):
            exact_times.append(
                {
                    "fold": int(fold),
                    "minute": named,
                    "sessions": len(part),
                    "mean_minute_max_score": float(part["minute_max_score"].mean()),
                    "mean_state_offset": float(part["state_offset"].mean()),
                }
            )
    return {
        "folds": folds,
        "first_crossings": first_crossings,
        "activated_sessions": activated,
        "first_crossing_at_0935_0939": opening,
        "early_first_crossing_share": float(opening / activated) if activated else None,
        "time_band_surface": band_rows,
        "named_minute_surface": exact_times,
        "severe_calibration_drift": sum(
            row["cutoff_above_every_scored_minute"] for row in folds
        )
        >= 2,
    }


def score_decomposition_metrics(scores: pd.DataFrame) -> dict[str, Any]:
    required = {
        "session",
        "entry_minute",
        "contract_id",
        "full_score",
        "state_offset",
        "contract_node_contribution",
    }
    missing = sorted(required - set(scores))
    if missing:
        raise AttributionError(f"decomposition frame missing columns: {missing}")
    full = scores["full_score"].to_numpy(float)
    state = scores["state_offset"].to_numpy(float)
    node = scores["contract_node_contribution"].to_numpy(float)
    reconstruction = np.abs(full - state - node)
    group = scores.groupby(["session", "entry_minute"], sort=False)
    minute_mean = group["full_score"].transform("mean").to_numpy(float)
    grand = float(np.mean(full))
    total_ss = float(np.square(full - grand).sum())
    within_ss = float(np.square(full - minute_mean).sum())
    full_choice = (
        scores.sort_values(
            ["session", "entry_minute", "full_score", "contract_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["session", "entry_minute"])
        [["session", "entry_minute", "contract_id"]]
    )
    node_choice = (
        scores.sort_values(
            [
                "session",
                "entry_minute",
                "contract_node_contribution",
                "contract_id",
            ],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["session", "entry_minute"])
        [["session", "entry_minute", "contract_id"]]
    )
    choice = full_choice.merge(
        node_choice,
        on=["session", "entry_minute"],
        suffixes=("_full", "_node"),
        validate="one_to_one",
    )
    state_unique = group["state_offset"].nunique(dropna=False)
    return {
        "rows": len(scores),
        "minutes": len(state_unique),
        "maximum_abs_full_minus_state_minus_node": float(reconstruction.max()),
        "state_offset_constant_within_every_minute": bool(state_unique.eq(1).all()),
        "full_argmax_equals_contract_node_argmax_rate": float(
            choice["contract_id_full"].eq(choice["contract_id_node"]).mean()
        ),
        "total_score_variance": float(np.var(full)),
        "state_offset_variance": float(np.var(state)),
        "contract_node_variance": float(np.var(node)),
        "state_node_covariance": float(np.cov(state, node, ddof=0)[0, 1]),
        "within_minute_score_variance_fraction": (
            float(within_ss / total_ss) if total_ss > 0.0 else None
        ),
        "between_minute_score_variance_fraction": (
            float(1.0 - within_ss / total_ss) if total_ss > 0.0 else None
        ),
    }


def state_channel_metrics(scores: pd.DataFrame) -> dict[str, Any]:
    columns = {
        "candle_state": "no_candle_state_offset",
        "global_ladder_summary": "no_ladder_summary_offset",
        "explicit_clock": "neutral_clock_offset",
    }
    minute = scores.drop_duplicates(["session", "entry_minute"])
    full = minute["state_offset"].to_numpy(float)
    results = {}
    for name, column in columns.items():
        ablated = minute[column].to_numpy(float)
        change = full - ablated
        correlation = (
            float(np.corrcoef(full, ablated)[0, 1])
            if np.std(full) > 0 and np.std(ablated) > 0
            else None
        )
        results[name] = {
            "minutes": len(minute),
            "mean_absolute_offset_change": float(np.mean(np.abs(change))),
            "root_mean_square_offset_change": float(np.sqrt(np.mean(np.square(change)))),
            "change_standard_deviation": float(np.std(change)),
            "full_vs_ablated_correlation": correlation,
        }
    largest = max(
        results,
        key=lambda value: results[value]["mean_absolute_offset_change"],
    )
    return {"channels": results, "largest_mean_absolute_channel": largest}


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def contract_proxy_metrics(
    scores: pd.DataFrame,
    proxies: tuple[str, ...],
) -> dict[str, Any]:
    required = {"session", "entry_minute", "contract_node_contribution", *proxies}
    missing = sorted(required - set(scores))
    if missing:
        raise AttributionError(f"proxy frame missing columns: {missing}")
    results = {}
    group_keys = ["session", "entry_minute"]
    for proxy in proxies:
        subset = scores[[*group_keys, "contract_node_contribution", proxy]].copy()
        subset[proxy] = pd.to_numeric(subset[proxy], errors="coerce")
        subset = subset.dropna()
        node = subset["contract_node_contribution"].to_numpy(float)
        value = subset[proxy].to_numpy(float)
        pooled_rank_node = pd.Series(node).rank(method="average").to_numpy(float)
        pooled_rank_value = pd.Series(value).rank(method="average").to_numpy(float)
        node_within = node - subset.groupby(group_keys)[
            "contract_node_contribution"
        ].transform("mean").to_numpy(float)
        value_within = value - subset.groupby(group_keys)[proxy].transform("mean").to_numpy(float)
        results[proxy] = {
            "rows": len(subset),
            "pooled_spearman": _correlation(pooled_rank_node, pooled_rank_value),
            "within_minute_pearson": _correlation(node_within, value_within),
        }
    strongest = max(
        results,
        key=lambda name: abs(results[name]["within_minute_pearson"] or 0.0),
    )
    return {
        "proxies": results,
        "strongest_within_minute_proxy": strongest,
        "strongest_abs_within_minute_correlation": abs(
            results[strongest]["within_minute_pearson"] or 0.0
        ),
    }


def classify_attribution(
    *,
    activation: dict[str, Any],
    decomposition: dict[str, Any],
    channels: dict[str, Any],
    proxies: dict[str, Any],
) -> dict[str, Any]:
    separable = bool(
        decomposition["state_offset_constant_within_every_minute"]
        and decomposition["full_argmax_equals_contract_node_argmax_rate"] >= 1.0 - 1e-12
        and decomposition["maximum_abs_full_minus_state_minus_node"] <= 1e-5
    )
    proxy_dominated = bool(
        proxies["strongest_abs_within_minute_correlation"] >= 0.75
    )
    early_collapse = bool(
        activation["activated_sessions"] > 0
        and activation["early_first_crossing_share"] is not None
        and activation["early_first_crossing_share"] >= 0.75
    )
    if separable:
        next_class = "COMPACT_STATE_CONTRACT_INTERACTION_REQUIRED"
        reason = (
            "Every current contract head is additive, so day state cannot change call/put/strike "
            "ranking; the next policy must include an explicit state-by-contract interaction"
        )
    elif proxy_dominated:
        next_class = "PROXY_DOMINATED_DO_NOT_FIT_SEQUENCE_POLICY"
        reason = "Contract ranking is dominated by a static causal contract proxy"
    else:
        next_class = "ATTRIBUTION_INCONCLUSIVE_REQUIRE_MORE_DIAGNOSIS"
        reason = "The frozen attribution did not isolate a sufficiently specific mechanism"
    return {
        "architectural_separability_verified": separable,
        "severe_calibration_drift": bool(activation["severe_calibration_drift"]),
        "early_entry_collapse": early_collapse,
        "contract_proxy_dominated": proxy_dominated,
        "largest_state_offset_channel": channels["largest_mean_absolute_channel"],
        "next_policy_class": next_class,
        "reason": reason,
    }
