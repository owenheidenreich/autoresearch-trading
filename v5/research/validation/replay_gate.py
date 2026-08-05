"""The corrected economic validation gate.

This replaces ``v4/research/pathd_phase1_replay.py``, which contains four
confirmed defects that would make a future pass untrustworthy:

1. **The fee sensitivity cancelled.**  Every cell recomputed policy and
   comparator with the *same* fee and tested only their difference, so the fee
   dropped out algebraically and eight cells were really four latency tests.
   Fixed by requiring an **absolute** lower bound above zero in every fee and
   latency cell, and by stating the paired-delta identity out loud instead of
   letting it hide.
2. **A negative control imported the future.**  ``time_shifted`` was built with
   ``shift(-1)``, which is the next row.  Fixed by using a causal lag,
   ``shift(+1)``.
3. **One positive fold counted as skill.**  ``positive_target_skill_folds > 0``
   passed on one fold of five.  Fixed by requiring 4 of 5, session-clustered.
4. **"Powered" counted rows.**  It checked ``sessions >= 30`` and five distinct
   fold labels.  Fixed by requiring a pre-run power receipt frozen before the
   evaluation rows are touched, whose power is computed rather than asserted,
   and whose declared session count must equal the evaluated frame — a dropped
   no-trade day is refused, not silently averaged away.

Two further weaknesses are also repaired: the comparator was selected as the
best of fifteen on the same rows it was judged against, so it must now arrive
pre-frozen with a hash; and negative controls were compared on pooled totals
only, so they must now fail the **same full gate**.

The module is source-neutral and model-free.  It consumes one row per calendar
session — including zero on a no-trade day — and computes nothing about the
market itself.  It cannot fit, tune, promote, or contact anything.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from v5.research.knobs import frozen_value


GATE_SCHEMA_VERSION = "v5.replay-gate.v1"
POWER_SCHEMA_VERSION = "v5.power-receipt.v1"
COMPARATOR_SCHEMA_VERSION = "v5.comparator-lock.v1"
DEFAULT_BLOCK_SESSIONS = 5
DEFAULT_BOOTSTRAP_SAMPLES = 2000
MINIMUM_POWER = 0.80
CONTROL_NAMES = ("constant", "sign_reversed", "causal_lag", "session_shuffled")


class ReplayGateError(RuntimeError):
    """The gate cannot be evaluated as specified."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash_unsigned(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


@dataclass(frozen=True)
class PowerReceipt:
    """Evidence, frozen before evaluation, that the index can resolve the effect.

    Defect 4's replacement.  Counting sessions and fold labels is not power; the
    declared effect, the session standard deviation and the resulting required
    session count all have to be written down before any outcome is seen.
    """

    schema_version: str
    declared_effect: float
    session_sd: float
    required_sessions: int
    achieved_power: float
    index_sessions: int
    frozen_on: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_powered(self) -> "PowerReceipt":
        if self.schema_version != POWER_SCHEMA_VERSION:
            raise ReplayGateError(f"unknown power receipt schema: {self.schema_version}")
        if _hash_unsigned(self.to_dict()) != self.receipt_sha256:
            raise ReplayGateError("power receipt self-hash mismatch; the bytes were edited")
        if self.declared_effect <= 0.0 or self.session_sd <= 0.0:
            raise ReplayGateError("a power receipt needs a positive effect and dispersion")
        if self.achieved_power < MINIMUM_POWER:
            raise ReplayGateError(
                f"index has {self.achieved_power:.3f} power for the declared effect; "
                f"{MINIMUM_POWER:.2f} is required before the outcome may be read"
            )
        if self.index_sessions < self.required_sessions:
            raise ReplayGateError(
                f"index has {self.index_sessions} sessions; the receipt requires "
                f"{self.required_sessions}"
            )
        return self


def analytic_power(
    *, declared_effect: float, session_sd: float, index_sessions: int
) -> float:
    """One-sided 5% normal power to detect ``declared_effect`` on this index."""

    if declared_effect <= 0.0 or session_sd <= 0.0 or index_sessions < 1:
        raise ReplayGateError("power needs a positive effect, dispersion, and index")
    z = declared_effect * math.sqrt(index_sessions) / session_sd - 1.645
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def make_power_receipt(
    *,
    declared_effect: float,
    session_sd: float,
    index_sessions: int,
    frozen_on: str,
    simulated_power: float | None = None,
) -> PowerReceipt:
    """Build a receipt, deriving both power and the required session count.

    ``N = ceil(((1.645 + 0.842) * sd / effect) ** 2)`` is the one-sided 5%,
    80%-power normal lower bound used throughout the gate-chain audit.
    ``achieved_power`` is computed from the same normal model, never accepted
    from the caller.  A session-block simulation may only *lower* it via
    ``simulated_power`` — serial dependence reduces power, it never adds any.
    """

    if declared_effect <= 0.0 or session_sd <= 0.0:
        raise ReplayGateError("a power receipt needs a positive effect and dispersion")
    required = int(np.ceil(((1.645 + 0.842) * session_sd / declared_effect) ** 2))
    achieved = analytic_power(
        declared_effect=declared_effect,
        session_sd=session_sd,
        index_sessions=index_sessions,
    )
    if simulated_power is not None:
        if not 0.0 <= float(simulated_power) <= 1.0:
            raise ReplayGateError("simulated power must be a probability")
        if float(simulated_power) > achieved:
            raise ReplayGateError(
                "a simulation may not claim more power than the independent-session "
                f"analytic bound ({achieved:.3f})"
            )
        achieved = float(simulated_power)
    unsigned = {
        "schema_version": POWER_SCHEMA_VERSION,
        "declared_effect": float(declared_effect),
        "session_sd": float(session_sd),
        "required_sessions": required,
        "achieved_power": achieved,
        "index_sessions": int(index_sessions),
        "frozen_on": frozen_on,
    }
    return PowerReceipt(**unsigned, receipt_sha256=_hash_unsigned(unsigned))


@dataclass(frozen=True)
class ComparatorLock:
    """The comparator's identity, fixed before the evaluation rows are read.

    Selecting the best of fifteen comparators on the same rows it is judged
    against makes the paired inference invalid.  The lock records which one was
    chosen, when, and on what basis.
    """

    schema_version: str
    comparator_name: str
    selected_on: str
    selection_basis: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_frozen_before(self, evaluation_date: str) -> "ComparatorLock":
        if self.schema_version != COMPARATOR_SCHEMA_VERSION:
            raise ReplayGateError(f"unknown comparator lock schema: {self.schema_version}")
        if _hash_unsigned(self.to_dict()) != self.receipt_sha256:
            raise ReplayGateError("comparator lock self-hash mismatch; the bytes were edited")
        if self.selected_on >= evaluation_date:
            raise ReplayGateError(
                f"comparator was selected on {self.selected_on}, not before the "
                f"{evaluation_date} evaluation; it may not be reselected on these rows"
            )
        return self


def make_comparator_lock(
    *, comparator_name: str, selected_on: str, selection_basis: str
) -> ComparatorLock:
    unsigned = {
        "schema_version": COMPARATOR_SCHEMA_VERSION,
        "comparator_name": comparator_name,
        "selected_on": selected_on,
        "selection_basis": selection_basis,
    }
    return ComparatorLock(**unsigned, receipt_sha256=_hash_unsigned(unsigned))


@dataclass(frozen=True)
class Cell:
    """One declared fee and latency combination, and its result columns."""

    name: str
    fee_per_side: float
    latency_seconds: int
    policy_column: str
    comparator_column: str


@dataclass
class GateResult:
    schema_version: str = GATE_SCHEMA_VERSION
    verdict: str = "NOT_SUPPORTED"
    failures: list[str] = field(default_factory=list)
    cells: dict[str, Any] = field(default_factory=dict)
    folds: dict[str, Any] = field(default_factory=dict)
    controls: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["receipt_sha256"] = _hash_unsigned(payload)
        return payload


def block_bootstrap_lower_bound(
    values: Sequence[float],
    *,
    confidence: float | None = None,
    block_sessions: int = DEFAULT_BLOCK_SESSIONS,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> float:
    """One-sided lower confidence bound on the mean, moving-block by session.

    Sessions are resampled in contiguous blocks rather than independently,
    because consecutive trading days are not independent draws.  Treating them
    as independent is exactly the optimism the measurement work warned about.
    """

    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ReplayGateError("cannot bootstrap an empty session index")
    if not np.isfinite(array).all():
        raise ReplayGateError("session values contain non-finite entries")
    level = float(frozen_value("confidence_level") if confidence is None else confidence)
    block = max(1, min(int(block_sessions), array.size))
    starts_available = array.size - block + 1
    blocks_needed = int(np.ceil(array.size / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, starts_available, size=(samples, blocks_needed))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(samples, -1)
    means = array[indices[:, : array.size]].mean(axis=1)
    return float(np.quantile(means, 1.0 - level))


def causal_lag_control(values: Sequence[float]) -> np.ndarray:
    """Defect 2's replacement: shift(+1), the previous row, never the next one."""

    return pd.Series(list(values), dtype=float).shift(1).fillna(0.0).to_numpy()


def _fold_signs(frame: pd.DataFrame, column: str, fold_column: str) -> dict[str, float]:
    return {
        str(fold): float(rows[column].mean())
        for fold, rows in frame.groupby(fold_column, sort=True)
    }


def _passes_fold_rule(fold_means: Mapping[str, float]) -> bool:
    rule = str(frozen_value("fold_pass_rule"))
    needed, _, total = rule.partition("_of_")
    if len(fold_means) != int(total):
        return False
    return sum(value > 0.0 for value in fold_means.values()) >= int(needed)


def evaluate(
    sessions: pd.DataFrame,
    *,
    cells: Sequence[Cell],
    controls: Mapping[str, str],
    power_receipt: PowerReceipt,
    comparator_lock: ComparatorLock,
    evaluation_date: str,
    skill_folds_positive: int,
    skill_null_passed_known_answer_gate: bool,
    session_column: str = "session",
    fold_column: str = "fold",
    seed: int = 0,
    _is_control_pass: bool = False,
) -> GateResult:
    """Evaluate the full corrected gate on one row per calendar session."""

    result = GateResult()
    for column in (session_column, fold_column):
        if column not in sessions.columns:
            raise ReplayGateError(f"session frame missing column: {column}")
    if sessions[session_column].duplicated().any():
        raise ReplayGateError("one row per calendar session is required; found duplicates")
    if not cells:
        raise ReplayGateError("at least one fee/latency cell must be declared")

    frame = sessions.sort_values(session_column).reset_index(drop=True)
    fold_count = int(frozen_value("fold_count"))
    if frame[fold_column].nunique() != fold_count:
        result.failures.append(
            f"index has {frame[fold_column].nunique()} folds; {fold_count} are required"
        )

    power_receipt.assert_powered()
    if len(frame) != power_receipt.index_sessions:
        raise ReplayGateError(
            f"session frame has {len(frame)} rows but the power receipt declared "
            f"{power_receipt.index_sessions} eligible sessions; a no-trade day must "
            "appear as a zero row, never be dropped"
        )
    comparator_lock.assert_frozen_before(evaluation_date)

    # --- defect 1: absolute economics at every fee and latency cell ----------
    paired_by_latency: dict[int, list[float]] = {}
    for cell in cells:
        for column in (cell.policy_column, cell.comparator_column):
            if column not in frame.columns:
                raise ReplayGateError(f"session frame missing column: {column}")
        absolute = frame[cell.policy_column].to_numpy(float)
        paired = absolute - frame[cell.comparator_column].to_numpy(float)
        absolute_lcb = block_bootstrap_lower_bound(absolute, seed=seed)
        paired_lcb = block_bootstrap_lower_bound(paired, seed=seed + 1)
        paired_by_latency.setdefault(cell.latency_seconds, []).append(float(paired.mean()))

        cell_frame = frame.assign(_absolute=absolute, _paired=paired)
        absolute_folds = _fold_signs(cell_frame, "_absolute", fold_column)
        paired_folds = _fold_signs(cell_frame, "_paired", fold_column)
        result.cells[cell.name] = {
            "fee_per_side": cell.fee_per_side,
            "latency_seconds": cell.latency_seconds,
            "absolute_mean": float(absolute.mean()),
            "absolute_lower_bound": absolute_lcb,
            "paired_mean": float(paired.mean()),
            "paired_lower_bound": paired_lcb,
            "absolute_positive_folds": absolute_folds,
            "paired_positive_folds": paired_folds,
        }
        if absolute_lcb <= 0.0:
            result.failures.append(
                f"{cell.name}: absolute net lower bound {absolute_lcb:.4f} is not above zero"
            )
        if paired_lcb <= 0.0:
            result.failures.append(
                f"{cell.name}: paired lower bound {paired_lcb:.4f} is not above zero"
            )
        # --- defect 3: four of five folds, on both quantities ----------------
        if not _passes_fold_rule(absolute_folds):
            result.failures.append(f"{cell.name}: absolute net fails the 4-of-5 fold rule")
        if not _passes_fold_rule(paired_folds):
            result.failures.append(f"{cell.name}: paired delta fails the 4-of-5 fold rule")

    # State the fee identity rather than letting it silently pass as evidence.
    for latency_seconds, deltas in sorted(paired_by_latency.items()):
        if len(deltas) > 1 and np.allclose(deltas, deltas[0], atol=1e-9):
            result.notes.append(
                f"latency {latency_seconds}s: the paired delta is identical across fee levels, "
                "as it must be algebraically; fee cells are evidence only for absolute economics"
            )

    # --- defect 3 again: exit-target skill needs 4 of 5, not 1 of 5 ----------
    needed_skill_folds = int(str(frozen_value("fold_pass_rule")).partition("_of_")[0])
    result.folds = {
        "skill_folds_positive": int(skill_folds_positive),
        "skill_folds_required": needed_skill_folds,
        "skill_null_passed_known_answer_gate": bool(skill_null_passed_known_answer_gate),
    }
    if not skill_null_passed_known_answer_gate:
        result.failures.append("the skill null has not passed a known-answer gate")
    if int(skill_folds_positive) < needed_skill_folds:
        result.failures.append(
            f"exit-target skill is positive in {skill_folds_positive} folds; "
            f"{needed_skill_folds} are required"
        )

    # --- negative controls must fail the SAME full gate ----------------------
    if not _is_control_pass:
        missing = set(CONTROL_NAMES) - set(controls)
        if missing:
            result.failures.append(f"missing required negative controls: {sorted(missing)}")
        for name, column in sorted(controls.items()):
            if column not in frame.columns:
                raise ReplayGateError(f"session frame missing control column: {column}")
            control_cells = [
                Cell(
                    name=cell.name,
                    fee_per_side=cell.fee_per_side,
                    latency_seconds=cell.latency_seconds,
                    policy_column=column,
                    comparator_column=cell.comparator_column,
                )
                for cell in cells
            ]
            control_result = evaluate(
                frame,
                cells=control_cells,
                controls={},
                power_receipt=power_receipt,
                comparator_lock=comparator_lock,
                evaluation_date=evaluation_date,
                skill_folds_positive=skill_folds_positive,
                skill_null_passed_known_answer_gate=skill_null_passed_known_answer_gate,
                session_column=session_column,
                fold_column=fold_column,
                seed=seed,
                _is_control_pass=True,
            )
            passed = not control_result.failures
            result.controls[name] = {"passed_full_gate": passed}
            if passed:
                result.failures.append(
                    f"negative control '{name}' passed the full gate; the result is invalid"
                )

    if not _is_control_pass:
        if any(entry["passed_full_gate"] for entry in result.controls.values()):
            result.verdict = "INVALID"
        elif result.failures:
            result.verdict = "NOT_SUPPORTED"
        else:
            result.verdict = "SUPPORTED"
    return result
