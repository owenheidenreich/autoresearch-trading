"""Attribute Stage-1 failures to entries, fixed exits, or other gates.

This is a read-only diagnostic. It does not fit a model, select a threshold,
change a gate, or authorize Stage-2 execution. Its frozen decision law is
preregistered independently of the H1-H3 results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.model.protocol101_scoped_stage1_hgb import POLICY_HOLD_MINUTES
from v4.model.protocol101_serial_simulator import (
    SerialCandidate,
    SerialSimulatorConfig,
    simulate_serial_candidates,
)
from v4.scripts.protocol101_training_scope import load_training_scope


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_DESIGN_DIR = (
    BASE_AUDIT / "protocol101_stage1_fixed_exit_binding_attribution_design"
)
DEFAULT_SELECTION_DIR = (
    BASE_AUDIT
    / "protocol101_scoped_canonical_stage1_cross_hypothesis_selection"
)
DEFAULT_OUT_DIR = (
    BASE_AUDIT / "protocol101_stage1_fixed_exit_binding_attribution_attempt001"
)
HYPOTHESES = ("H0", "H1", "H2", "H3")
POLICIES = tuple(range(7))
SEEDS = (42, 43, 44)
FOLDS = tuple(range(5))
FEE = 3.0
MIN_PATH_COVERAGE = 0.95
MIN_LABEL_COVERAGE = 0.99
MIN_LOSERS = 30
MIN_SALVAGEABLE_LOSER_SHARE = 0.30
MIN_CAUSAL_STATE_AUC = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preregister", "run"), required=True)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def preregistration() -> dict[str, Any]:
    payload = {
        "schema_version": "Protocol101Stage1FixedExitBindingAttributionDesignV1",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "attribution_source_sha256": sha256_path(Path(__file__)),
        "scope": "read_only_existing_artifact_failure_attribution",
        "input_route_required": "real_signal_attribution_required",
        "signal_definition": "accepted_G2_and_G5_for_the_same_hypothesis_policy",
        "candidate_population": (
            "all H0-H3 hypothesis-policy rows accepted as real signal; "
            "all three seeds and all five folds"
        ),
        "counterfactuals": {
            "same_entry_intents_each_single_fixed_policy": list(POLICIES),
            "loss_only_best_of_seven_oracle": (
                "preserve source-policy PnL for winners; for source-policy "
                "losers choose the best of seven label policies"
            ),
            "minimum_all_seven_label_lookup_coverage": MIN_LABEL_COVERAGE,
            "best_of_seven_oracle": "diagnostic upper bound only",
            "serial_replay": (
                "one account, original entry-intent timestamps, counterfactual "
                "policy cooldown, $3 fee reserve"
            ),
        },
        "path_evidence": {
            "source": "existing governed processed session rows only",
            "executed_trade_deserialization": (
                "executed replay rows omit cooldown by schema; reconstruct it "
                "from the frozen source policy solely for path diagnostics"
            ),
            "features": [
                "current bid-to-entry PnL at 5m and 10m",
                "causal MFE at 5m and 10m",
                "causal giveback at 5m and 10m",
                "full observed MFE and MAE",
            ],
            "minimum_coverage": MIN_PATH_COVERAGE,
            "minimum_source_policy_losers": MIN_LOSERS,
            "minimum_loser_share_with_positive_MFE": MIN_SALVAGEABLE_LOSER_SHARE,
            "minimum_univariate_causal_state_AUC": MIN_CAUSAL_STATE_AUC,
        },
        "routing_law": {
            "fixed_exit_composition_requires_design": (
                "at least one single fixed policy repairs G4 for every seed "
                "while retaining G1 and G7"
            ),
            "stage2_design_may_be_drafted": (
                "no single fixed policy repairs the candidate; loss-only "
                "oracle preserves G1 and repairs G4 and G7 for every seed; "
                "path coverage and loser sample pass; salvageable-MFE share "
                "passes; at least one 10-minute causal state statistic reaches "
                "the frozen AUC bar"
            ),
            "otherwise": "fixed_exits_not_shown_to_be_the_binding_problem",
        },
        "interpretation_limits": [
            "oracle counterfactuals are not deployable",
            "passing attribution authorizes a Stage-2 design draft only",
            "no learned exit may be trained by this audit",
            "G8 calibration failure is reported separately and never attributed to exits",
        ],
        "forbidden": [
            "model fitting",
            "threshold selection or tuning",
            "feature changes",
            "gate changes",
            "protected holdout or recorder data",
            "broker or paper submit",
            "promotion, runtime, launchd, or real-money changes",
        ],
    }
    payload["preregistration_hash"] = stable_hash(payload)
    return payload


def write_preregistration(design_dir: Path) -> dict[str, Any]:
    path = design_dir / "preregistration.json"
    proposed = preregistration()
    if path.exists():
        existing = load_json(path)
        expected = existing.get("preregistration_hash")
        without_hash = dict(existing)
        without_hash.pop("preregistration_hash", None)
        if not expected or expected != stable_hash(without_hash):
            raise RuntimeError("existing attribution preregistration hash mismatch")
        return existing
    design_dir.mkdir(parents=True, exist_ok=False)
    write_json(path, proposed)
    return proposed


def deserialize_intent(
    payload: dict[str, Any],
    *,
    default_cooldown_minutes: float | None = None,
) -> SerialCandidate:
    cooldown = payload.get("cooldown_minutes", default_cooldown_minutes)
    if cooldown is None:
        raise KeyError("cooldown_minutes")
    max_hold = payload.get("max_hold_minutes", default_cooldown_minutes)
    return SerialCandidate(
        split=str(payload["split"]),
        session=str(payload["session"]),
        decision_time=datetime.fromisoformat(str(payload["decision_time"])),
        contract_id=str(payload["contract_id"]),
        right=str(payload["right"]),
        offset=float(payload["offset"]),
        entry_ask=float(payload["entry_ask"]),
        score=float(payload["score"]),
        raw_label_pnl=float(payload["raw_label_pnl"]),
        cooldown_minutes=float(cooldown),
        max_hold_minutes=(
            None
            if max_hold is None
            else float(max_hold)
        ),
        feature_hash=str(payload.get("feature_hash") or ""),
        source_quote_time=str(payload.get("source_quote_time") or ""),
        source_context_time=str(payload.get("source_context_time") or ""),
        strategy=str(payload.get("strategy") or ""),
        metadata=dict(payload.get("metadata") or {}),
    )


def simulator_metrics(intents: list[SerialCandidate]) -> dict[str, Any]:
    trades, state = simulate_serial_candidates(
        sorted(intents, key=lambda row: (row.session, row.decision_time)),
        config=SerialSimulatorConfig(
            starting_cash=10_000.0,
            max_daily_loss_fraction_of_session_start_equity=0.05,
            affordability_reserve_per_trade=FEE,
        ),
    )
    pnl = np.asarray([float(row.raw_label_pnl) for row in trades], dtype=float)
    equity = np.asarray(
        state.equity_by_account.get("validation", [10_000.0]), dtype=float
    )
    peak = np.maximum.accumulate(equity)
    max_drawdown = float(np.max(peak - equity))
    net_pnl = float(pnl.sum()) if len(pnl) else 0.0
    calmar = (
        net_pnl / max_drawdown
        if max_drawdown > 0.0
        else (math.inf if net_pnl > 0.0 else -math.inf)
    )
    return {
        "intents": len(intents),
        "trades": len(trades),
        "net_pnl": net_pnl,
        "minimum_equity": float(np.min(equity)),
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "trade_rows": [asdict(row) for row in trades],
    }


class SessionRows:
    def __init__(self) -> None:
        scope = load_training_scope()
        self.paths = {session: path for session, path in scope.sessions}
        self.cache: dict[str, tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]] = {}

    def load(
        self, session: str
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        if session not in self.cache:
            with self.paths[session].open("rb") as handle:
                rows = pickle.load(handle)
            by_time = {
                pd.Timestamp(row["decision_time"]).isoformat(): row
                for row in rows
                if isinstance(row, dict) and row.get("decision_time")
            }
            self.cache[session] = (rows, by_time)
        return self.cache[session]

    def label_vector(self, intent: SerialCandidate) -> np.ndarray:
        _rows, by_time = self.load(intent.session)
        key = pd.Timestamp(intent.decision_time).isoformat()
        row = by_time[key]
        ids = np.asarray(row["contract_ids"], dtype=object)
        matches = np.argwhere(ids == intent.contract_id)
        if len(matches) != 1:
            raise KeyError(
                f"contract lookup {intent.session} {key} {intent.contract_id}: "
                f"{len(matches)} matches"
            )
        strike_idx, right_idx = (int(item) for item in matches[0])
        return np.asarray(
            row["labels_net_pnl"], dtype=float
        )[strike_idx, right_idx, :]

    def bid_path(self, intent: SerialCandidate) -> list[tuple[float, float]]:
        rows, _by_time = self.load(intent.session)
        entry = pd.Timestamp(intent.decision_time)
        path: list[tuple[float, float]] = []
        for row in rows:
            timestamp = pd.Timestamp(row["decision_time"])
            elapsed = (timestamp - entry).total_seconds() / 60.0
            if elapsed < 0.0 or elapsed > 384.0:
                continue
            metadata = (row.get("contract_quote_metadata") or {}).get(
                intent.contract_id
            ) or {}
            try:
                bid = float(metadata.get("bid"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(bid) and bid >= 0.0:
                path.append(
                    (
                        elapsed,
                        (bid - float(intent.entry_ask)) * 100.0 - FEE,
                    )
                )
        return path


def counterfactual_intent(
    intent: SerialCandidate,
    *,
    label_vector: np.ndarray,
    policy: int,
) -> SerialCandidate:
    return replace(
        intent,
        raw_label_pnl=float(label_vector[policy]) - FEE,
        cooldown_minutes=float(POLICY_HOLD_MINUTES[policy]),
        max_hold_minutes=float(POLICY_HOLD_MINUTES[policy]),
        metadata={**intent.metadata, "attribution_policy": int(policy)},
    )


def rank_auc(values: list[float], positive: list[bool]) -> float | None:
    frame = pd.DataFrame({"value": values, "positive": positive}).dropna()
    positives = int(frame["positive"].sum())
    negatives = len(frame) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = frame["value"].rank(method="average")
    rank_sum = float(ranks[frame["positive"]].sum())
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    return max(float(auc), float(1.0 - auc))


def path_summary(
    trades: list[SerialCandidate],
    *,
    sessions: SessionRows,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        path = sessions.bid_path(trade)
        if not path:
            rows.append({"covered": False, "source_loss": trade.raw_label_pnl < 0.0})
            continue
        values = np.asarray([value for _minute, value in path], dtype=float)
        row: dict[str, Any] = {
            "covered": True,
            "source_loss": trade.raw_label_pnl < 0.0,
            "mfe": float(np.max(values)),
            "mae": float(np.min(values)),
        }
        for horizon in (5.0, 10.0):
            eligible = [(minute, value) for minute, value in path if minute <= horizon]
            if eligible:
                current = eligible[-1][1]
                mfe = max(value for _minute, value in eligible)
                row[f"pnl_{int(horizon)}m"] = float(current)
                row[f"mfe_{int(horizon)}m"] = float(mfe)
                row[f"giveback_{int(horizon)}m"] = float(mfe - current)
        rows.append(row)
    covered = [row for row in rows if row.get("covered")]
    losers = [row for row in covered if row["source_loss"]]
    aucs: dict[str, float | None] = {}
    for name in (
        "pnl_5m",
        "mfe_5m",
        "giveback_5m",
        "pnl_10m",
        "mfe_10m",
        "giveback_10m",
    ):
        selected = [row for row in covered if name in row]
        aucs[name] = rank_auc(
            [float(row[name]) for row in selected],
            [bool(row["source_loss"]) for row in selected],
        )
    return {
        "trades": len(trades),
        "covered_trades": len(covered),
        "coverage": len(covered) / len(trades) if trades else 0.0,
        "source_policy_losers": len(losers),
        "salvageable_losers_positive_mfe": sum(
            float(row["mfe"]) > 0.0 for row in losers
        ),
        "salvageable_loser_share": (
            sum(float(row["mfe"]) > 0.0 for row in losers) / len(losers)
            if losers
            else 0.0
        ),
        "causal_state_univariate_auc": aucs,
        "max_10m_causal_state_auc": max(
            (
                value
                for name, value in aucs.items()
                if name.endswith("10m") and value is not None
            ),
            default=None,
        ),
    }


def batch_dir(hypothesis: str) -> Path:
    return (
        BASE_AUDIT
        / f"protocol101_scoped_canonical_stage1_{hypothesis.lower()}_attempt001"
    )


def unit_payloads(hypothesis: str, policy: int) -> list[dict[str, Any]]:
    summary = load_json(batch_dir(hypothesis) / "summary.json")
    payloads: list[dict[str, Any]] = []
    for reference in summary.get("unit_artifacts") or []:
        payload = load_json(Path(reference["path"]))
        unit = payload["unit"]
        if int(unit["policy_index"]) == int(policy):
            payloads.append(payload)
    return payloads


def candidate_attribution(
    *,
    hypothesis: str,
    source_policy: int,
    sessions: SessionRows,
) -> dict[str, Any]:
    units = unit_payloads(hypothesis, source_policy)
    by_seed_fold: dict[tuple[int, int], dict[str, Any]] = {}
    all_executed: list[SerialCandidate] = []
    lookup_total = 0
    lookup_finite = 0
    for payload in units:
        unit = payload["unit"]
        seed = int(unit["seed"])
        fold = int(payload["fold"])
        intents = [
            deserialize_intent(row)
            for row in unit["validation"]["entry_intents"]
        ]
        vectors: list[np.ndarray] = []
        for intent in intents:
            lookup_total += 1
            vector = sessions.label_vector(intent)
            vectors.append(vector)
            if len(vector) == len(POLICIES) and np.isfinite(vector).all():
                lookup_finite += 1
        source_actual = simulator_metrics(intents)
        reported = unit["validation"]["metrics"]
        if not math.isclose(
            float(source_actual["net_pnl"]),
            float(reported["net_pnl"]),
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                f"source replay mismatch {hypothesis} p{source_policy} "
                f"seed{seed} fold{fold}"
            )
        fixed: dict[str, Any] = {}
        for policy in POLICIES:
            alternate = [
                counterfactual_intent(
                    intent, label_vector=vector, policy=policy
                )
                for intent, vector in zip(intents, vectors)
                if np.isfinite(vector[policy])
            ]
            fixed[str(policy)] = simulator_metrics(alternate)
        best_all: list[SerialCandidate] = []
        loss_only: list[SerialCandidate] = []
        for intent, vector in zip(intents, vectors):
            if not np.isfinite(vector).all():
                continue
            best_policy = int(np.argmax(vector))
            best_all.append(
                counterfactual_intent(
                    intent, label_vector=vector, policy=best_policy
                )
            )
            chosen = (
                best_policy if float(intent.raw_label_pnl) < 0.0 else source_policy
            )
            loss_only.append(
                counterfactual_intent(
                    intent, label_vector=vector, policy=chosen
                )
            )
        by_seed_fold[(seed, fold)] = {
            "sessions": len(payload["validation_sessions"]),
            "source": source_actual,
            "fixed": fixed,
            "_fixed_intents": {
                str(policy): [
                    counterfactual_intent(
                        intent, label_vector=vector, policy=policy
                    )
                    for intent, vector in zip(intents, vectors)
                    if np.isfinite(vector[policy])
                ]
                for policy in POLICIES
            },
            "best_of_seven_oracle": simulator_metrics(best_all),
            "_best_of_seven_intents": best_all,
            "loss_only_oracle": simulator_metrics(loss_only),
            "_loss_only_intents": loss_only,
        }
        all_executed.extend(
            deserialize_intent(
                row,
                default_cooldown_minutes=float(
                    POLICY_HOLD_MINUTES[source_policy]
                ),
            )
            for row in unit["validation"]["trades"]
        )
    seed_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        fold_rows = [by_seed_fold[(seed, fold)] for fold in FOLDS]
        fixed_results: dict[str, Any] = {}
        for policy in POLICIES:
            rows = [row["fixed"][str(policy)] for row in fold_rows]
            pooled = simulator_metrics(
                [
                    intent
                    for fold_row in fold_rows
                    for intent in fold_row["_fixed_intents"][str(policy)]
                ]
            )
            fixed_results[str(policy)] = {
                "G1": sum(row["net_pnl"] > 0.0 for row in rows) >= 4
                and sum(row["net_pnl"] for row in rows) > 0.0,
                "G4": pooled["calmar"] >= 1.0
                and all(row["minimum_equity"] >= 5_000.0 for row in rows),
                "G7": all(
                    0.3
                    <= row["trades"] / max(1, fold_row["sessions"])
                    <= 6.0
                    for row, fold_row in zip(rows, fold_rows)
                ),
                "folds": rows,
                "continuous_oof_replay": pooled,
            }
        loss_rows = [row["loss_only_oracle"] for row in fold_rows]
        pooled_loss_only = simulator_metrics(
            [
                intent
                for fold_row in fold_rows
                for intent in fold_row["_loss_only_intents"]
            ]
        )
        seed_results.append(
            {
                "seed": seed,
                "fixed_policies": fixed_results,
                "loss_only_oracle_G1": sum(
                    row["net_pnl"] > 0.0 for row in loss_rows
                )
                >= 4
                and sum(row["net_pnl"] for row in loss_rows) > 0.0,
                "loss_only_oracle_G4": all(
                    row["minimum_equity"] >= 5_000.0 for row in loss_rows
                )
                and pooled_loss_only["calmar"] >= 1.0,
                "loss_only_oracle_G7": all(
                    0.3
                    <= row["trades"] / max(1, fold_row["sessions"])
                    <= 6.0
                    for row, fold_row in zip(loss_rows, fold_rows)
                ),
                "loss_only_oracle_folds": loss_rows,
                "loss_only_oracle_continuous_oof_replay": pooled_loss_only,
            }
        )
    single_fixed_repairs = [
        policy
        for policy in POLICIES
        if all(
            row["fixed_policies"][str(policy)]["G1"]
            and row["fixed_policies"][str(policy)]["G4"]
            and row["fixed_policies"][str(policy)]["G7"]
            for row in seed_results
        )
    ] if (lookup_finite / lookup_total if lookup_total else 0.0) >= MIN_LABEL_COVERAGE else []
    paths = path_summary(all_executed, sessions=sessions)
    learned_exit_evidence = bool(
        not single_fixed_repairs
        and (lookup_finite / lookup_total if lookup_total else 0.0)
        >= MIN_LABEL_COVERAGE
        and all(
            row["loss_only_oracle_G1"]
            and row["loss_only_oracle_G4"]
            and row["loss_only_oracle_G7"]
            for row in seed_results
        )
        and paths["coverage"] >= MIN_PATH_COVERAGE
        and paths["source_policy_losers"] >= MIN_LOSERS
        and paths["salvageable_loser_share"] >= MIN_SALVAGEABLE_LOSER_SHARE
        and (paths["max_10m_causal_state_auc"] or 0.0) >= MIN_CAUSAL_STATE_AUC
    )
    return {
        "hypothesis": hypothesis,
        "source_policy": source_policy,
        "label_lookup": {
            "total": lookup_total,
            "finite_all_seven": lookup_finite,
            "coverage": lookup_finite / lookup_total if lookup_total else 0.0,
        },
        "single_fixed_policies_repairing_G1_G4_G7_all_seeds": (
            single_fixed_repairs
        ),
        "seed_results": seed_results,
        "path_evidence": paths,
        "learned_exit_design_evidence_pass": learned_exit_evidence,
    }


def run_attribution(
    *,
    design: dict[str, Any],
    selection_dir: Path,
) -> dict[str, Any]:
    selection = load_json(selection_dir / "comparison.json")
    if selection.get("status") != "pass":
        raise RuntimeError("cross-hypothesis selection is not accepted")
    if selection.get("routing") != "real_signal_attribution_required":
        return {
            "status": "not_applicable",
            "routing": selection.get("routing"),
            "candidates": [],
            "decision": "attribution_not_run_for_selection_route",
        }
    signal_rows = [
        row
        for row in selection.get("policy_comparison") or []
        if bool(row.get("real_signal_G2_G5"))
    ]
    sessions = SessionRows()
    candidates = [
        candidate_attribution(
            hypothesis=str(row["hypothesis"]),
            source_policy=int(row["policy_index"]),
            sessions=sessions,
        )
        for row in signal_rows
    ]
    fixed = [
        {
            "hypothesis": row["hypothesis"],
            "source_policy": row["source_policy"],
            "repair_policies": row[
                "single_fixed_policies_repairing_G1_G4_G7_all_seeds"
            ],
        }
        for row in candidates
        if row["single_fixed_policies_repairing_G1_G4_G7_all_seeds"]
    ]
    learned = [
        {
            "hypothesis": row["hypothesis"],
            "source_policy": row["source_policy"],
        }
        for row in candidates
        if row["learned_exit_design_evidence_pass"]
    ]
    if fixed:
        decision = "fixed_exit_composition_requires_separate_design"
    elif learned:
        decision = "stage2_design_may_be_drafted"
    else:
        decision = "fixed_exits_not_shown_to_be_the_binding_problem"
    return {
        "status": "complete",
        "selection_routing": selection["routing"],
        "preregistration_hash": design["preregistration_hash"],
        "selection_comparison_sha256": sha256_path(
            selection_dir / "comparison.json"
        ),
        "candidates": candidates,
        "fixed_exit_composition_candidates": fixed,
        "learned_exit_design_candidates": learned,
        "decision": decision,
        "G8_interpretation": (
            "calibration remains a separate binding gate and is not repaired "
            "or attributed by this exit diagnostic"
        ),
        "side_effects": {
            "model_training_executed": False,
            "threshold_selection_executed": False,
            "gate_or_feature_changed": False,
            "protected_holdout_or_recorder_read": False,
            "broker_or_paper_called": False,
            "promotion_runtime_launchd_or_real_money_changed": False,
        },
    }


def report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Stage-1 Fixed-Exit Binding Attribution",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        "- This packet fitted no model and selected no threshold.",
        "",
    ]
    for row in payload.get("candidates") or []:
        paths = row["path_evidence"]
        lines.extend(
            [
                f"## {row['hypothesis']} / policy {row['source_policy']}",
                "",
                "- Single fixed-policy repairs: "
                f"`{row['single_fixed_policies_repairing_G1_G4_G7_all_seeds']}`",
                f"- Path coverage: `{paths['coverage']:.4f}`",
                f"- Source-policy losers: `{paths['source_policy_losers']}`",
                "- Salvageable loser share: "
                f"`{paths['salvageable_loser_share']:.4f}`",
                "- Maximum 10-minute causal-state AUC: "
                f"`{paths['max_10m_causal_state_auc']}`",
                "- Learned-exit design evidence: "
                f"`{row['learned_exit_design_evidence_pass']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    design = write_preregistration(args.design_dir)
    if args.mode == "preregister":
        print(
            json.dumps(
                {
                    "status": "preregistered",
                    "preregistration_hash": design["preregistration_hash"],
                    "path": str(args.design_dir / "preregistration.json"),
                },
                indent=2,
            )
        )
        return 0
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = run_attribution(design=design, selection_dir=args.selection_dir)
    payload["schema_version"] = "Protocol101Stage1FixedExitBindingAttributionV1"
    payload["generated_at_utc"] = datetime.now(UTC).isoformat()
    payload["summary_hash"] = stable_hash(payload)
    write_json(args.out_dir / "summary.json", payload)
    (args.out_dir / "report.md").write_text(report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "candidate_count": len(payload.get("candidates") or []),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
