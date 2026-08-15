"""Frozen causal selector and outcome-blind control for compact action values.

The learned WAIT estimate is not allowed to make abstention worse than doing
nothing.  At inference, WAIT therefore has the structural zero-dollar floor
present in its own target.  This is a feasibility law, not a tuned prediction
threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v5.ops.audit_causal_day_coverage import ENTRY_MINUTES
from v5.ops.build_causal_day_dataset import minute_number


ENTER_SCORE = "predicted_q_enter_bid_120m_usd"
WAIT_SCORE = "predicted_q_wait_bid_120m_usd"
KEYS = ("session", "entry_minute", "contract_id")


class ActionValueSelectionError(RuntimeError):
    """Predictions cannot support the frozen one-trade inference law."""


def _finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ActionValueSelectionError(f"{column} contains a non-finite value")


def select_first_positive_advantage(
    candidate_predictions: pd.DataFrame,
    minute_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Take the first minute where an ENTER prediction beats feasible WAIT.

    The walk is causal within each scored session: only the current minute's
    predictions enter the decision.  The one-trade cap ends the walk after the
    first entry.  Exact score ties are broken by contract identity, never by an
    outcome or a later-day rank.
    """

    # Refuse the Job-39 defect class at the executable selector boundary.  The
    # proof is outcome-free and constructs a canonical model whose unbounded
    # outputs strictly satisfy this exact ENTER > max($0, WAIT) comparison.
    from v5.research.causal_day_action_value_attainability import (
        action_value_selector_attainability,
    )

    action_value_selector_attainability()

    candidate_required = {*KEYS, "fold", "shuffled_label_null", ENTER_SCORE}
    minute_required = {
        "session",
        "entry_minute",
        "fold",
        "shuffled_label_null",
        WAIT_SCORE,
    }
    missing_candidate = sorted(candidate_required - set(candidate_predictions.columns))
    missing_minute = sorted(minute_required - set(minute_predictions.columns))
    if missing_candidate or missing_minute:
        raise ActionValueSelectionError(
            f"prediction columns missing: candidates={missing_candidate}, minutes={missing_minute}"
        )
    candidates = candidate_predictions.copy()
    minutes = minute_predictions.copy()
    for frame in (candidates, minutes):
        frame["session"] = frame["session"].astype(str)
        frame["entry_minute"] = frame["entry_minute"].astype(str)
    candidates["contract_id"] = candidates["contract_id"].astype(str)
    if candidates.duplicated(list(KEYS)).any():
        raise ActionValueSelectionError("candidate prediction key is not unique")
    if minutes.duplicated(["session", "entry_minute"]).any():
        raise ActionValueSelectionError("minute prediction key is not unique")
    _finite(candidates, (ENTER_SCORE, "fold"))
    _finite(minutes, (WAIT_SCORE, "fold"))
    if set(minutes["entry_minute"].unique()) - set(ENTRY_MINUTES):
        raise ActionValueSelectionError("minute prediction lies outside the decision clock")
    for session, frame in minutes.groupby("session", sort=True):
        if tuple(frame.sort_values("entry_minute")["entry_minute"]) != tuple(ENTRY_MINUTES):
            raise ActionValueSelectionError(f"{session}: incomplete 326-minute WAIT clock")
        if frame["fold"].astype(int).nunique() != 1:
            raise ActionValueSelectionError(f"{session}: WAIT rows span score folds")
        if frame["shuffled_label_null"].astype(bool).nunique() != 1:
            raise ActionValueSelectionError(f"{session}: WAIT rows mix real and null fits")
    if set(candidates["session"]) - set(minutes["session"]):
        raise ActionValueSelectionError("candidate session lacks a WAIT clock")
    null_values = set(candidates["shuffled_label_null"].astype(bool)) | set(
        minutes["shuffled_label_null"].astype(bool)
    )
    if len(null_values) != 1:
        raise ActionValueSelectionError("one selection call cannot mix real and null predictions")

    wait_columns = ["session", "entry_minute", "fold", "shuffled_label_null", WAIT_SCORE]
    scored = candidates.merge(
        minutes[wait_columns],
        on=["session", "entry_minute"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_wait"),
    )
    if scored[WAIT_SCORE].isna().any():
        raise ActionValueSelectionError("candidate minute lacks a WAIT prediction")
    if not scored["fold"].astype(int).eq(scored["fold_wait"].astype(int)).all():
        raise ActionValueSelectionError("candidate and WAIT fold assignments differ")
    if not scored["shuffled_label_null"].astype(bool).eq(
        scored["shuffled_label_null_wait"].astype(bool)
    ).all():
        raise ActionValueSelectionError("candidate and WAIT null identities differ")

    selected: list[pd.Series] = []
    for session, minute_clock in minutes.groupby("session", sort=True):
        session_candidates = scored[scored["session"].eq(session)]
        for minute in ENTRY_MINUTES:
            block = session_candidates[session_candidates["entry_minute"].eq(minute)]
            if block.empty:
                continue
            wait_raw = float(
                minute_clock.loc[minute_clock["entry_minute"].eq(minute), WAIT_SCORE].iloc[0]
            )
            feasible_wait = max(0.0, wait_raw)
            chosen = block.sort_values(
                [ENTER_SCORE, "contract_id"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0].copy()
            enter = float(chosen[ENTER_SCORE])
            if enter <= feasible_wait:
                continue
            chosen["predicted_q_wait_raw_usd"] = wait_raw
            chosen["effective_q_wait_usd"] = feasible_wait
            chosen["predicted_action_advantage_usd"] = enter - feasible_wait
            chosen["selector"] = "first_enter_above_structural_wait_floor"
            selected.append(chosen)
            break
    if not selected:
        return pd.DataFrame(
            columns=[*candidate_predictions.columns, "predicted_q_wait_raw_usd", "effective_q_wait_usd", "predicted_action_advantage_usd", "selector"]
        )
    result = pd.DataFrame(selected).drop(
        columns=["fold_wait", "shuffled_label_null_wait"], errors="ignore"
    )
    if result["session"].duplicated().any():
        raise ActionValueSelectionError("one-trade selector emitted two trades in a session")
    return result.sort_values("session", kind="mergesort").reset_index(drop=True)


def select_outcome_blind_matched_control(
    selected_causal: pd.DataFrame,
    causal_population: pd.DataFrame,
) -> pd.DataFrame:
    """Match session, regime and side, then time, delta and premium.

    A same-minute alternate always wins when one exists.  If that side has only
    one eligible contract at the selected minute, the nearest minute inside the
    same opening regime is used.  No payoff column is accepted or inspected.
    """

    required = {
        *KEYS,
        "entry_regime",
        "right",
        "self_delta",
        "entry_ask_usd",
    }
    missing_selected = sorted(required - set(selected_causal.columns))
    missing_population = sorted(required - set(causal_population.columns))
    if missing_selected or missing_population:
        raise ActionValueSelectionError(
            f"matched-control columns missing: selected={missing_selected}, population={missing_population}"
        )
    selected = selected_causal[list(required)].copy()
    population = causal_population[list(required)].copy()
    for frame in (selected, population):
        frame["session"] = frame["session"].astype(str)
        frame["entry_minute"] = frame["entry_minute"].astype(str)
        frame["contract_id"] = frame["contract_id"].astype(str)
        frame["right"] = frame["right"].astype(str)
        frame["entry_regime"] = frame["entry_regime"].astype(str)
        _finite(frame, ("self_delta", "entry_ask_usd"))
    if selected["session"].duplicated().any():
        raise ActionValueSelectionError("matched control expects at most one target per session")
    if population.duplicated(list(KEYS)).any():
        raise ActionValueSelectionError("causal control population key is not unique")
    if (pd.to_numeric(population["entry_ask_usd"], errors="coerce") <= 0.0).any():
        raise ActionValueSelectionError("matched-control premium must be positive")

    rows: list[pd.Series] = []
    for target in selected.itertuples(index=False):
        pool = population[
            population["session"].eq(target.session)
            & population["entry_regime"].eq(target.entry_regime)
            & population["right"].eq(target.right)
            & ~(
                population["entry_minute"].eq(target.entry_minute)
                & population["contract_id"].eq(target.contract_id)
            )
        ].copy()
        if pool.empty:
            raise ActionValueSelectionError(
                f"{target.session}: no outcome-blind same-regime same-side control"
            )
        pool["match_minute_distance"] = (
            pool["entry_minute"].map(minute_number) - minute_number(target.entry_minute)
        ).abs()
        pool["match_abs_delta_distance"] = (
            pd.to_numeric(pool["self_delta"], errors="raise").abs() - abs(float(target.self_delta))
        ).abs()
        pool["match_log_premium_distance"] = np.abs(
            np.log(pd.to_numeric(pool["entry_ask_usd"], errors="raise") / float(target.entry_ask_usd))
        )
        chosen = pool.sort_values(
            [
                "match_minute_distance",
                "match_abs_delta_distance",
                "match_log_premium_distance",
                "contract_id",
            ],
            kind="mergesort",
        ).iloc[0].copy()
        chosen["matched_for_entry_minute"] = target.entry_minute
        chosen["matched_for_contract_id"] = target.contract_id
        rows.append(chosen)
    if not rows:
        return pd.DataFrame(columns=[*required, "matched_for_entry_minute", "matched_for_contract_id"])
    result = pd.DataFrame(rows)
    if result["session"].duplicated().any() or len(result) != len(selected):
        raise ActionValueSelectionError("matched control did not preserve trade count")
    return result.sort_values("session", kind="mergesort").reset_index(drop=True)
