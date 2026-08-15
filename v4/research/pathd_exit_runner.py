"""Bounded Wave-1 runner for the Phase-1 Path-D exit repair.

The runner deliberately reuses the frozen Phase-1 feature, label, fold, model,
fill, and OOF machinery.  It changes only the pre-registered training weights,
deterministic class sampling, policy utility contribution, and fallback arm.
It has no broker, download, protected-holdout, promotion, or runtime capability.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from v4.research.pathd_model_gate import (
    TestResult,
    decile_monotonicity,
    exit_first_step_rate,
    four_bucket_distribution,
    harvest_ratio,
    label_balance,
    pnl_concentration,
    tail_preservation,
    underwater_duration,
)
from v4.research.pathd_research_loop import ExperimentResult, Hypothesis, WaveSpec
from v4.research.phase1_exit_model import (
    EXIT_FEATURE_NAMES,
    MAX_CALIBRATION_ROWS,
    MAX_FIT_ROWS,
    _deterministic_sha_sample,
    _partition_pairs,
    fit_hgb_baseline,
    stable_hash,
)
from v4.research.phase1_storage import ResearchRoots, preflight_roots


WAVE_ID = "pathd-wave1-exit-repair-2026-08-03"
PREREGISTRATION_PATH = Path(
    "v4/docs/protocol101/training/research/"
    "PATHD_WAVE1_EXIT_REPAIR_PREREGISTRATION_2026_08_03.md"
)
SEEDS = (301, 302, 303)
MAXT_SEED = 1065
MAXT_DRAWS = 9_999
BOOTSTRAP_SEED = 901
BOOTSTRAP_DRAWS = 10_000


class ExitRunnerError(RuntimeError):
    """A frozen Wave-1 contract or input was violated."""


def wave1_spec() -> WaveSpec:
    """Return the exact eight-member family frozen in the preregistration."""

    rows = (
        ("W1-H01", "recovery penalty", True, False, False, 1.0),
        ("W1-H02", "recovery penalty", True, False, True, 1.0),
        ("W1-H03", "rebalanced exit label", False, True, False, 1.0),
        ("W1-H04", "risk lower bound calibration", False, True, False, 0.5),
        ("W1-H05", "recovery penalty", True, True, False, 1.0),
        ("W1-H06", "rebalanced exit label", False, True, True, 1.0),
        ("W1-H07", "recovery penalty", True, True, True, 1.0),
        ("W1-H08", "recovery penalty", True, True, True, 0.5),
    )
    hypotheses = tuple(
        Hypothesis(
            hypothesis_id=identifier,
            mechanism=mechanism,
            params={
                "recovery_penalty": recovery,
                "class_balance": balance,
                "fallback": fallback,
                "lcb_weight": lcb_weight,
            },
            rationale="Frozen Path-D Wave-1 exit repair family.",
        )
        for identifier, mechanism, recovery, balance, fallback, lcb_weight in rows
    )
    return WaveSpec(
        wave_id=WAVE_ID,
        objective="Repair the Phase-1 exit while preserving loss defense and the convex tail.",
        hypotheses=hypotheses,
        budget=8,
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def recovery_weight_multiplier(frame: pd.DataFrame) -> np.ndarray:
    """Exact Protocol065 four-factor multiplier frozen for Wave 1."""

    required = {"a_ref_dollars", "recovery_300_dollars", "seconds_held"}
    if required - set(frame):
        raise ExitRunnerError("recovery-weight input columns are missing")
    target = frame["a_ref_dollars"].to_numpy(float)
    recovery = _sigmoid((frame["recovery_300_dollars"].to_numpy(float) - 250.0) / 200.0)
    regret = _sigmoid((target - 100.0) / 200.0)
    early = _sigmoid((5.0 - frame["seconds_held"].to_numpy(float) / 60.0) / 2.0)
    penalty = (target > 0.0).astype(float) * recovery * regret * early
    return 1.0 + 4.0 * penalty


def _identity_rank(frame: pd.DataFrame) -> np.ndarray:
    required = {"session", "trajectory_id", "decision_time_ns"}
    if required - set(frame):
        raise ExitRunnerError("class-balance identity columns are missing")
    keys = (
        frame["session"].astype(str)
        + "|"
        + frame["trajectory_id"].astype(str)
        + "|"
        + frame["decision_time_ns"].astype(str)
    )
    return np.fromiter(
        (int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) for value in keys),
        dtype=np.uint64,
        count=len(frame),
    )


def deterministic_class_balance(frame: pd.DataFrame, *, maximum_rows: int) -> pd.DataFrame:
    """Preserve both target signs and select an exact 50/50 SHA-ranked sample."""

    finite = frame[np.isfinite(frame["a_ref_dollars"].to_numpy(float))].copy()
    positive = finite[finite["a_ref_dollars"] > 0.0].copy()
    negative = finite[finite["a_ref_dollars"] <= 0.0].copy()
    per_class = min(len(positive), len(negative), maximum_rows // 2)
    if per_class <= 0:
        raise ExitRunnerError("class balance requires both positive and non-positive labels")

    def take(source: pd.DataFrame) -> pd.DataFrame:
        if len(source) <= per_class:
            return source
        ranks = _identity_rank(source)
        selected = np.argpartition(ranks, per_class - 1)[:per_class]
        return source.iloc[selected]

    result = pd.concat((take(positive), take(negative)), ignore_index=True)
    return result.sort_values(
        ["session", "trajectory_id", "decision_time_ns"], kind="mergesort"
    ).reset_index(drop=True)


def _load_scope_sample(
    scratch_root: Path, sessions: Sequence[str], *, maximum_rows: int
) -> pd.DataFrame:
    """Phase-1 balanced trajectory sample with the recovery column retained."""

    pairs = _partition_pairs(scratch_root, sessions)
    if not pairs:
        raise ExitRunnerError("no exit partitions for Wave-1 scope")
    per_session = max(1, int(math.ceil(maximum_rows / len(pairs))))
    pieces: list[pd.DataFrame] = []
    for _session, session_pairs in pairs.items():
        per_trajectory = max(1, int(math.ceil(per_session / len(session_pairs))))
        for feature_path, label_path in session_pairs:
            features = pd.read_parquet(feature_path)
            labels = pd.read_parquet(label_path)
            identity = ["session", "trajectory_id", "decision_time_ns"]
            if not features[identity].equals(labels[identity]):
                raise ExitRunnerError(f"feature/label identity drift: {feature_path}")
            joined = features.merge(
                labels[identity + ["a_ref_dollars", "recovery_300_dollars"]],
                on=identity,
                validate="one_to_one",
            )
            pieces.append(_deterministic_sha_sample(joined, per_trajectory))
    sample = _deterministic_sha_sample(pd.concat(pieces, ignore_index=True), maximum_rows)
    sample = sample[np.isfinite(sample["a_ref_dollars"].to_numpy(float))].reset_index(drop=True)
    if sample.empty:
        raise ExitRunnerError("Wave-1 scope has no finite targets")
    trajectory_count = sample.groupby(["session", "trajectory_id"])[
        "decision_time_ns"
    ].transform("size").to_numpy(float)
    trajectories_per_session = sample.groupby("session")["trajectory_id"].transform(
        "nunique"
    ).to_numpy(float)
    weights = 1.0 / (trajectory_count * trajectories_per_session)
    weights *= len(weights) / float(weights.sum())
    sample["_base_weight"] = weights
    return sample


def _bootstrap_delta_lcb(session_delta: pd.Series) -> float:
    values = session_delta.to_numpy(float)
    if not len(values):
        return 0.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for start in range(0, BOOTSTRAP_DRAWS, 1_000):
        count = min(1_000, BOOTSTRAP_DRAWS - start)
        indexes = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indexes].mean(axis=1)
    return float(np.quantile(means, 0.05))


def _hold_utility(prediction: pd.DataFrame, lcb_weight: float) -> np.ndarray:
    return (
        prediction["mean"].to_numpy(float)
        + float(lcb_weight)
        * (prediction["mean_lcb90"].to_numpy(float) - prediction["mean"].to_numpy(float))
        + 0.25 * np.minimum(prediction["q10"].to_numpy(float), 0.0)
    )


def _exit_index(
    features: pd.DataFrame, utility: np.ndarray, *, fallback: bool
) -> tuple[int, str]:
    premium = float(features["entry_fill_option_price"].iloc[0]) * 100.0
    floor = features["current_net_pnl_dollars"].to_numpy(float) <= -0.50 * premium
    request = np.asarray(utility, dtype=float) <= 0.0
    reason_arrays: list[tuple[str, np.ndarray]] = [
        ("CATASTROPHIC_FLOOR", floor),
        ("LEARNED_EXIT", request),
    ]
    if fallback:
        returns = features["current_return_on_entry_premium"].to_numpy(float)
        seconds = features["seconds_held"].to_numpy(float)
        reason_arrays.extend(
            (("TARGET_100", returns >= 1.0), ("MAX_HOLD_25M", seconds >= 1_500.0))
        )
    best: tuple[int, int, str] | None = None
    for priority, (reason, flags) in enumerate(reason_arrays):
        indexes = np.flatnonzero(flags)
        if len(indexes):
            candidate = (int(indexes[0]), priority, reason)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    return (len(features) - 1, "FORCED_FLAT") if best is None else (best[0], best[2])


def _value_at(
    features: pd.DataFrame, labels: pd.DataFrame, utility: np.ndarray, *, fallback: bool
) -> tuple[float, int, str]:
    index, reason = _exit_index(features, utility, fallback=fallback)
    if reason == "FORCED_FLAT":
        return float(labels["hold_to_1555_value_dollars"].iloc[0]), index, reason
    return float(labels["exit_until_filled_value_dollars"].iloc[index]), index, reason


def _horizon_value(labels: pd.DataFrame, features: pd.DataFrame, seconds: int) -> float:
    indexes = np.flatnonzero(features["seconds_held"].to_numpy(float) >= seconds)
    if not len(indexes):
        return float(labels["hold_to_1555_value_dollars"].iloc[0])
    return float(labels["exit_until_filled_value_dollars"].iloc[int(indexes[0])])


def _stop_target_value(
    labels: pd.DataFrame, features: pd.DataFrame, *, stop: float, target: float
) -> float:
    returns = features["current_return_on_entry_premium"].to_numpy(float)
    indexes = np.flatnonzero((returns <= stop) | (returns >= target))
    if not len(indexes):
        return float(labels["hold_to_1555_value_dollars"].iloc[0])
    return float(labels["exit_until_filled_value_dollars"].iloc[int(indexes[0])])


def _interpolate_utility(source: np.ndarray, target_length: int) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    if target_length <= 0 or not len(source):
        raise ExitRunnerError("session-shuffle utility cannot be empty")
    if len(source) == 1:
        return np.full(target_length, source[0], dtype=float)
    return np.interp(
        np.linspace(0.0, 1.0, target_length),
        np.linspace(0.0, 1.0, len(source)),
        source,
    )


def _session_shuffle_sources(
    records: Sequence[Mapping[str, Any]], utilities: Mapping[str, np.ndarray]
) -> dict[str, str]:
    """Frozen SHA permutation within fold/role, preferring a different session."""

    assignments: dict[str, str] = {}
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault((int(row["outer_fold"]), str(row["entry_policy_role"])), []).append(row)
    for group, rows in grouped.items():
        targets = sorted(rows, key=lambda row: str(row["trajectory_id"]))
        sources = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{MAXT_SEED}|session-shuffle|{group}|{row['trajectory_id']}".encode()
            ).hexdigest(),
        )
        best_shift = 0
        best_mismatch = -1
        for shift in range(1, len(sources) + 1):
            mismatch = sum(
                str(target["session"]) != str(sources[(index + shift) % len(sources)]["session"])
                for index, target in enumerate(targets)
            )
            if mismatch > best_mismatch:
                best_shift, best_mismatch = shift, mismatch
            if mismatch == len(targets):
                break
        for index, target in enumerate(targets):
            source = sources[(index + best_shift) % len(sources)]
            target_id = str(target["trajectory_id"])
            source_id = str(source["trajectory_id"])
            if target_id not in utilities or source_id not in utilities:
                raise ExitRunnerError("session-shuffle utility identity is missing")
            assignments[target_id] = source_id
    return assignments


def _not_evaluated(name: str, detail: str) -> TestResult:
    return TestResult(name=name, passed=False, detail=detail, metrics={"evaluated": 0.0})


class PathDExitRunner:
    """Callable ``Hypothesis -> ExperimentResult`` for the frozen Wave-1 family."""

    def __init__(
        self,
        *,
        roots: ResearchRoots,
        entry_campaign_path: Path,
        output_root: Path,
        expected_volume_name: str = "AR_TRADING_DATA",
    ) -> None:
        self.roots = roots
        self.scratch_root = Path(roots.scratch_root)
        self.entry_campaign_path = Path(entry_campaign_path)
        self.output_root = Path(output_root)
        self.expected_volume_name = expected_volume_name
        self.results: dict[str, ExperimentResult] = {}
        self.session_deltas: dict[str, pd.Series] = {}
        self.raw_packets: dict[str, dict[str, Any]] = {}
        self._campaign = self._load_entry_campaign()

    def _load_entry_campaign(self) -> dict[str, Any]:
        payload = json.loads(self.entry_campaign_path.read_text())
        semantic = dict(payload)
        observed = semantic.pop("campaign_sha256", None)
        if observed != stable_hash(semantic):
            raise ExitRunnerError("entry campaign receipt drift")
        if payload.get("protected_holdout_opened") is not False:
            raise ExitRunnerError("protected holdout must remain closed")
        return payload

    def _fold_specs(self) -> list[tuple[int, tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
        specs = []
        for fold_key, manifest_path in sorted(
            self._campaign["fold_manifests"].items(), key=lambda item: int(item[0])
        ):
            manifest = json.loads(Path(manifest_path).read_text())
            semantic = dict(manifest)
            observed = semantic.pop("manifest_sha256", None)
            if observed != stable_hash(semantic):
                raise ExitRunnerError("entry fold manifest drift")
            train = tuple(map(str, manifest["fold_spec"]["train"]))
            test = tuple(map(str, manifest["fold_spec"]["test"]))
            calibration_count = max(1, int(math.ceil(0.20 * len(train))))
            specs.append((int(fold_key), train[:-calibration_count], train[-calibration_count:], test))
        if len(specs) != 5:
            raise ExitRunnerError("Wave 1 requires exactly five OOF folds")
        return specs

    def _sample_path(self, fold: int, role: str) -> Path:
        return self.output_root / "sample_cache" / f"fold={fold}" / f"{role}.parquet"

    def _sample(self, fold: int, role: str, sessions: Sequence[str], cap: int) -> pd.DataFrame:
        path = self._sample_path(fold, role)
        if path.is_file():
            return pd.read_parquet(path)
        sample = _load_scope_sample(self.scratch_root, sessions, maximum_rows=cap)
        path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_parquet(path, index=False, compression="zstd")
        return sample

    def _preflight(self, hypothesis: Hypothesis) -> tuple[bool, list[dict[str, Any]], TestResult]:
        balance = bool(hypothesis.params["class_balance"])
        reports: list[dict[str, Any]] = []
        effective_labels: list[np.ndarray] = []
        passed = True
        for fold, fit_sessions, calibration_sessions, _test in self._fold_specs():
            for role, sessions, cap in (
                ("fit", fit_sessions, MAX_FIT_ROWS),
                ("calibration", calibration_sessions, MAX_CALIBRATION_ROWS),
            ):
                raw = self._sample(fold, role, sessions, cap)
                original = label_balance(raw["a_ref_dollars"].to_numpy(float))
                effective = deterministic_class_balance(raw, maximum_rows=cap) if balance else raw
                checked = label_balance(effective["a_ref_dollars"].to_numpy(float))
                reports.append(
                    {
                        "fold": fold,
                        "role": role,
                        "original": original.as_dict(),
                        "effective": checked.as_dict(),
                        "original_rows": len(raw),
                        "effective_rows": len(effective),
                    }
                )
                effective_labels.append(effective["a_ref_dollars"].to_numpy(float))
                passed = passed and checked.passed
        aggregate = label_balance(np.concatenate(effective_labels))
        return passed and aggregate.passed, reports, aggregate

    def _fit_and_score(self, hypothesis: Hypothesis, directory: Path) -> dict[int, Path]:
        prediction_roots: dict[int, Path] = {}
        for fold, fit_sessions, calibration_sessions, test_sessions in self._fold_specs():
            fit = self._sample(fold, "fit", fit_sessions, MAX_FIT_ROWS)
            calibration = self._sample(
                fold, "calibration", calibration_sessions, MAX_CALIBRATION_ROWS
            )
            if bool(hypothesis.params["class_balance"]):
                fit = deterministic_class_balance(fit, maximum_rows=MAX_FIT_ROWS)
                calibration = deterministic_class_balance(
                    calibration, maximum_rows=MAX_CALIBRATION_ROWS
                )
            # Arrow-backed cached Parquet columns may expose a read-only NumPy
            # view.  Variant normalization is intentionally local and must not
            # mutate the frozen cached sample.
            weights = fit["_base_weight"].to_numpy(dtype=float, copy=True)
            if bool(hypothesis.params["recovery_penalty"]):
                weights *= recovery_weight_multiplier(fit)
            weights *= len(weights) / float(weights.sum())
            artifact = fit_hgb_baseline(
                fit.loc[:, EXIT_FEATURE_NAMES],
                fit["a_ref_dollars"],
                calibration.loc[:, EXIT_FEATURE_NAMES],
                calibration["a_ref_dollars"],
                sample_weight=weights,
                seeds=SEEDS,
            )
            model_dir = directory / "models" / f"fold={fold}"
            model_dir.mkdir(parents=True, exist_ok=False)
            model_path = model_dir / "model.joblib"
            joblib.dump(artifact, model_path)
            manifest = {
                "schema_version": "pathd.wave1-exit-variant.v1",
                "hypothesis_id": hypothesis.hypothesis_id,
                "params": dict(hypothesis.params),
                "fold": fold,
                "seeds": list(SEEDS),
                "feature_names": list(EXIT_FEATURE_NAMES),
                "model_path": str(model_path),
                "model_sha256": _sha256_path(model_path),
                "fit_sessions": list(fit_sessions),
                "calibration_sessions": list(calibration_sessions),
                "test_sessions": list(test_sessions),
                "fill_law_modified": False,
                "causal_clock_modified": False,
                "oof_firewall_modified": False,
                "protected_holdout_opened": False,
            }
            manifest["manifest_sha256"] = stable_hash(manifest)
            (model_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            prediction_root = directory / "oof_predictions" / f"fold={fold}"
            prediction_roots[fold] = prediction_root
            for session, session_pairs in _partition_pairs(self.scratch_root, test_sessions).items():
                session_root = prediction_root / f"session={session}"
                session_root.mkdir(parents=True, exist_ok=True)
                for feature_path, _label_path in session_pairs:
                    features = pd.read_parquet(feature_path)
                    prediction = artifact.predict(features.loc[:, EXIT_FEATURE_NAMES])
                    prediction["utility_hold"] = _hold_utility(
                        prediction, float(hypothesis.params["lcb_weight"])
                    )
                    identity = features[["session", "trajectory_id", "decision_time_ns"]]
                    pd.concat(
                        [identity.reset_index(drop=True), prediction.reset_index(drop=True)], axis=1
                    ).to_parquet(session_root / feature_path.name, index=False, compression="zstd")
        return prediction_roots

    def _prediction_path(self, directory: Path, item: Mapping[str, Any]) -> Path:
        return (
            directory
            / "oof_predictions"
            / f"fold={int(item['outer_fold'])}"
            / f"session={item['session']}"
            / f"{item['trajectory_id']}.parquet"
        )

    def _replay(self, hypothesis: Hypothesis, directory: Path, balance_test: TestResult) -> dict[str, Any]:
        index = pd.read_parquet(
            self._campaign.get(
                "evaluation_trajectory_index_path", self._campaign["trajectory_index_path"]
            )
        ).sort_values(["session", "trajectory_id"], kind="mergesort")
        records = index.to_dict("records")
        utilities: dict[str, np.ndarray] = {}
        for item in records:
            prediction = pd.read_parquet(self._prediction_path(directory, item))
            utilities[str(item["trajectory_id"])] = prediction["utility_hold"].to_numpy(
                dtype=np.float32
            )
        shuffle_sources = _session_shuffle_sources(records, utilities)
        rows: list[dict[str, Any]] = []
        fallback = bool(hypothesis.params["fallback"])
        for item in records:
            session = str(item["session"])
            trajectory_id = str(item["trajectory_id"])
            feature_path = (
                self.scratch_root / "exit_features" / f"session={session}" / f"{trajectory_id}.parquet"
            )
            label_path = (
                self.scratch_root / "exit_labels" / f"session={session}" / f"{trajectory_id}.parquet"
            )
            features = pd.read_parquet(feature_path)
            labels = pd.read_parquet(label_path)
            utility = utilities[trajectory_id].astype(float)
            learned_value, exit_index, exit_reason = _value_at(
                features, labels, utility, fallback=fallback
            )
            source = shuffle_sources[trajectory_id]
            shuffled = _interpolate_utility(utilities[source], len(utility))
            negative_values = {
                "sign_reversed": _value_at(features, labels, -utility, fallback=fallback)[0],
                "session_shuffled": _value_at(features, labels, shuffled, fallback=fallback)[0],
                "constant": _value_at(features, labels, np.zeros(len(utility)), fallback=fallback)[0],
            }
            hold = float(labels["hold_to_1555_value_dollars"].iloc[0])
            premium = float(features["entry_fill_option_price"].iloc[0]) * 100.0
            row = {
                **item,
                "control_exit_value": hold,
                "learned_exit_value": learned_value,
                "learned_exit_index": exit_index,
                "learned_exit_reason": exit_reason,
                "trajectory_rows": len(labels),
                "learned_exit_fraction": exit_index / max(1, len(labels) - 1),
                "entry_premium_dollars": premium,
                "peak_available_value": float(
                    np.nanmax(labels["exit_until_filled_value_dollars"].to_numpy(float))
                ),
                "initial_exit_score": float(utility[0]),
                "initial_a_ref": float(labels["a_ref_dollars"].iloc[0]),
                "exit_immediate": float(labels["exit_until_filled_value_dollars"].iloc[0]),
                "exit_60s": _horizon_value(labels, features, 60),
                "exit_300s": _horizon_value(labels, features, 300),
                "exit_900s": _horizon_value(labels, features, 900),
                "stop50_target100": _stop_target_value(
                    labels, features, stop=-0.50, target=1.00
                ),
                "stop25_target50": _stop_target_value(
                    labels, features, stop=-0.25, target=0.50
                ),
                **{f"negative_{key}": value for key, value in negative_values.items()},
            }
            rows.append(row)
        frame = pd.DataFrame(rows)
        for _role, role_frame in frame.groupby("entry_policy_role", sort=True):
            fractions = role_frame["learned_exit_fraction"].to_numpy(float)
            target_indexes = role_frame.index.to_numpy()
            for seed in range(8):
                rng = np.random.default_rng(
                    int(
                        hashlib.sha256(
                            f"{_role}|matched-rate|{seed}".encode()
                        ).hexdigest()[:16],
                        16,
                    )
                )
                assigned = fractions.copy()
                rng.shuffle(assigned)
                for frame_index, fraction in zip(target_indexes, assigned, strict=True):
                    item = frame.loc[frame_index]
                    labels = pd.read_parquet(
                        self.scratch_root
                        / "exit_labels"
                        / f"session={item['session']}"
                        / f"{item['trajectory_id']}.parquet"
                    )
                    curve = labels["exit_until_filled_value_dollars"].to_numpy(float)
                    curve_index = min(
                        len(curve) - 1,
                        int(round(fraction * max(0, len(curve) - 1))),
                    )
                    frame.at[frame_index, f"matched_random_{seed}"] = float(curve[curve_index])

        # Wave 1 repairs the exit head, so its gate population is every OOF exit
        # trajectory (the 1,031-row population used by the handoff diagnostics),
        # not the legacy four-box headline's 153 learned-entry rows.  The latter
        # spans only three outer folds and therefore cannot satisfy the frozen
        # five-fold acceptance contract.  Preserve the four-box cells below as
        # diagnostics, but score the exit candidate on all OOF trajectories.
        evaluation = frame.copy()
        four_boxes = {
            f"{role}/{exit_kind}": float(part[value_column].sum())
            for role, part in frame.groupby("entry_policy_role", sort=True)
            for exit_kind, value_column in (
                ("control_exit", "control_exit_value"),
                ("learned_exit", "learned_exit_value"),
            )
        }
        comparator_columns = [
            "exit_immediate",
            "control_exit_value",
            "exit_60s",
            "exit_300s",
            "exit_900s",
            "stop50_target100",
            "stop25_target50",
            *[f"matched_random_{seed}" for seed in range(8)],
        ]
        comparator_pnl = {name: float(evaluation[name].sum()) for name in comparator_columns}
        best_name = max(comparator_columns, key=comparator_pnl.__getitem__)
        evaluation["paired_delta"] = evaluation["learned_exit_value"] - evaluation[best_name]
        session_delta = evaluation.groupby("session", sort=True)["paired_delta"].sum()
        fold_delta = {
            str(int(fold)): float(part["paired_delta"].sum())
            for fold, part in evaluation.groupby("outer_fold", sort=True)
        }
        negative_pnl = {
            name: float(evaluation[f"negative_{name}"].sum())
            for name in ("sign_reversed", "session_shuffled", "constant")
        }
        negative_accepted = {
            name: value > comparator_pnl[best_name] for name, value in negative_pnl.items()
        }
        rejection_tests = [
            exit_first_step_rate(evaluation["learned_exit_index"].to_numpy(int)),
            tail_preservation(
                evaluation["learned_exit_value"].to_numpy(float),
                evaluation["control_exit_value"].to_numpy(float),
            ),
            decile_monotonicity(
                evaluation["initial_exit_score"].to_numpy(float),
                evaluation["initial_a_ref"].to_numpy(float),
                evaluation["outer_fold"].to_numpy(int),
            ),
            balance_test,
        ]
        returns = evaluation["learned_exit_value"].to_numpy(float) / evaluation[
            "entry_premium_dollars"
        ].to_numpy(float)
        bucket = four_bucket_distribution(returns)
        rejection_tests.append(bucket)
        ordered = evaluation.sort_values(["session", "trajectory_id"], kind="mergesort")
        charter = {
            "four_bucket_distribution": bucket.as_dict(),
            "harvest_ratio": harvest_ratio(
                evaluation["learned_exit_value"].to_numpy(float),
                evaluation["peak_available_value"].to_numpy(float),
            ),
            "underwater_duration": underwater_duration(
                ordered["learned_exit_value"].cumsum().to_numpy(float)
            ),
            "pnl_concentration": pnl_concentration(
                evaluation["learned_exit_value"].to_numpy(float)
            ),
        }
        positive_total = float(
            evaluation.loc[evaluation["paired_delta"] > 0, "paired_delta"].sum()
        )
        session_positive = evaluation.groupby("session")["paired_delta"].sum().clip(lower=0.0)
        weekdays = evaluation["session"].map(
            lambda value: date.fromisoformat(str(value)).strftime("%A")
        )
        weekday_positive = evaluation.assign(weekday=weekdays).groupby("weekday")[
            "paired_delta"
        ].sum().clip(lower=0.0)
        session_share = float(session_positive.max() / positive_total) if positive_total > 0 else 1.0
        weekday_share = float(weekday_positive.max() / positive_total) if positive_total > 0 else 1.0
        concentration = {
            "positive_paired_delta_dollars": positive_total,
            "max_session_positive_share": session_share,
            "max_weekday_positive_share": weekday_share,
            "limit": 0.50,
            "passed": bool(session_share <= 0.50 and weekday_share <= 0.50),
        }
        trajectory_path = directory / "trajectory_outcomes.parquet"
        frame.to_parquet(trajectory_path, index=False, compression="zstd")
        return {
            "pooled_policy": float(evaluation["learned_exit_value"].sum()),
            "pooled_comparator": comparator_pnl[best_name],
            "best_comparator": best_name,
            "comparators": comparator_pnl,
            "four_boxes": four_boxes,
            "gate_population": {
                "name": "all_oof_exit_trajectories",
                "trajectories": len(evaluation),
                "folds": int(evaluation["outer_fold"].nunique()),
            },
            "fold_deltas": fold_delta,
            "bootstrap_lcb": _bootstrap_delta_lcb(session_delta),
            "negative_control_pnl": negative_pnl,
            "negative_controls_accepted": negative_accepted,
            "rejection_tests": rejection_tests,
            "charter_diagnostics": charter,
            "concentration": concentration,
            "session_delta": session_delta,
            "trajectory_outcomes_path": str(trajectory_path),
            "trajectory_outcomes_sha256": _sha256_path(trajectory_path),
        }

    def __call__(self, hypothesis: Hypothesis) -> ExperimentResult:
        if hypothesis not in wave1_spec().hypotheses:
            raise ExitRunnerError("hypothesis is not a member of the frozen Wave-1 family")
        directory = self.output_root / hypothesis.hypothesis_id
        directory.mkdir(parents=True, exist_ok=False)
        storage_before = preflight_roots(
            self.roots, expected_volume_name=self.expected_volume_name
        )
        preflight_passed, balance_reports, aggregate_balance = self._preflight(hypothesis)
        if not preflight_passed:
            detail = "Not evaluated because label_balance failed before fitting."
            rejection_tests = (
                _not_evaluated("exit_first_step_rate", detail),
                _not_evaluated("tail_preservation", detail),
                _not_evaluated("decile_monotonicity", detail),
                aggregate_balance,
                _not_evaluated("four_bucket_distribution", detail),
            )
            diagnostics = {
                "status": "PREFLIGHT_REJECTED_LABEL_BALANCE",
                "params": dict(hypothesis.params),
                "label_balance_by_fold": balance_reports,
                "charter_diagnostics": {
                    "four_bucket_distribution": rejection_tests[-1].as_dict(),
                    "harvest_ratio": {"status": "NOT_EVALUATED"},
                    "underwater_duration": {"status": "NOT_EVALUATED"},
                    "pnl_concentration": {"status": "NOT_EVALUATED"},
                },
                "storage_before": storage_before,
                "storage_after": preflight_roots(
                    self.roots, expected_volume_name=self.expected_volume_name
                ),
                "protected_holdout_opened": False,
            }
            result = ExperimentResult(
                pooled_policy=0.0,
                pooled_comparator=0.0,
                fold_deltas={str(index): 0.0 for index in range(5)},
                bootstrap_lcb=0.0,
                negative_controls_accepted={
                    "sign_reversed": False,
                    "session_shuffled": False,
                    "constant": False,
                },
                rejection_tests=rejection_tests,
                maxt_survived=False,
                concentrated=True,
                diagnostics=diagnostics,
            )
            self.results[hypothesis.hypothesis_id] = result
            self.raw_packets[hypothesis.hypothesis_id] = diagnostics
            (directory / "result.json").write_text(
                json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
            )
            return result

        self._fit_and_score(hypothesis, directory)
        replay = self._replay(hypothesis, directory, aggregate_balance)
        storage_after = preflight_roots(
            self.roots, expected_volume_name=self.expected_volume_name
        )
        diagnostics = {
            "status": "TRAINED_AND_REPLAYED",
            "params": dict(hypothesis.params),
            "label_balance_by_fold": balance_reports,
            "best_comparator": replay["best_comparator"],
            "comparators": replay["comparators"],
            "four_boxes": replay["four_boxes"],
            "gate_population": replay["gate_population"],
            "negative_control_pnl": replay["negative_control_pnl"],
            "charter_diagnostics": replay["charter_diagnostics"],
            "concentration": replay["concentration"],
            "trajectory_outcomes_path": replay["trajectory_outcomes_path"],
            "trajectory_outcomes_sha256": replay["trajectory_outcomes_sha256"],
            "storage_before": storage_before,
            "storage_after": storage_after,
            "protected_holdout_opened": False,
            "fill_law_modified": False,
            "causal_clock_modified": False,
            "oof_firewall_modified": False,
        }
        result = ExperimentResult(
            pooled_policy=replay["pooled_policy"],
            pooled_comparator=replay["pooled_comparator"],
            fold_deltas=replay["fold_deltas"],
            bootstrap_lcb=replay["bootstrap_lcb"],
            negative_controls_accepted=replay["negative_controls_accepted"],
            rejection_tests=replay["rejection_tests"],
            maxt_survived=False,
            concentrated=not bool(replay["concentration"]["passed"]),
            diagnostics=diagnostics,
        )
        self.results[hypothesis.hypothesis_id] = result
        self.session_deltas[hypothesis.hypothesis_id] = replay["session_delta"]
        self.raw_packets[hypothesis.hypothesis_id] = diagnostics
        packet = {
            **diagnostics,
            "pooled_policy": result.pooled_policy,
            "pooled_comparator": result.pooled_comparator,
            "fold_deltas": dict(result.fold_deltas),
            "bootstrap_lcb": result.bootstrap_lcb,
            "negative_controls_accepted": dict(result.negative_controls_accepted),
            "rejection_tests": [test.as_dict() for test in result.rejection_tests],
        }
        (directory / "result.json").write_text(
            json.dumps(packet, indent=2, sort_keys=True) + "\n"
        )
        return result


def session_blocked_max_t(
    session_deltas: Mapping[str, pd.Series], *, draws: int = MAXT_DRAWS, seed: int = MAXT_SEED
) -> dict[str, Any]:
    """One-sided family maxT via shared session sign flips."""

    if not session_deltas:
        return {"method": "session_blocked_max_t", "draws": draws, "p_values": {}}
    sessions = sorted(set().union(*(set(series.index.astype(str)) for series in session_deltas.values())))
    names = sorted(session_deltas)
    matrix = np.column_stack(
        [session_deltas[name].reindex(sessions, fill_value=0.0).to_numpy(float) for name in names]
    )
    means = matrix.mean(axis=0)
    standard_error = matrix.std(axis=0, ddof=1) / math.sqrt(len(matrix))
    observed = np.divide(
        means,
        standard_error,
        out=np.where(means > 0.0, np.inf, np.where(means < 0.0, -np.inf, 0.0)),
        where=standard_error > 0,
    )
    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(names), dtype=int)
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        signs = rng.choice((-1.0, 1.0), size=(count, len(sessions)))
        permuted_values = signs[:, :, None] * matrix[None, :, :]
        permuted_means = permuted_values.mean(axis=1)
        permuted_se = permuted_values.std(axis=1, ddof=1) / math.sqrt(len(sessions))
        permuted_t = np.divide(
            permuted_means,
            permuted_se,
            out=np.where(
                permuted_means > 0.0,
                np.inf,
                np.where(permuted_means < 0.0, -np.inf, 0.0),
            ),
            where=permuted_se > 0,
        )
        maxima = permuted_t.max(axis=1)
        exceed += (maxima[:, None] >= observed[None, :]).sum(axis=0)
    p_values = {
        name: float((count + 1) / (draws + 1))
        for name, count in zip(names, exceed, strict=True)
    }
    return {
        "method": "session_blocked_max_t",
        "draws": draws,
        "seed": seed,
        "declared_family_size": 8,
        "trained_member_count": len(names),
        "session_count": len(sessions),
        "observed_t": dict(zip(names, map(float, observed), strict=True)),
        "p_values": p_values,
        "survived": {name: value <= 0.05 for name, value in p_values.items()},
    }


__all__ = [
    "ExitRunnerError",
    "PathDExitRunner",
    "deterministic_class_balance",
    "recovery_weight_multiplier",
    "session_blocked_max_t",
    "wave1_spec",
]
