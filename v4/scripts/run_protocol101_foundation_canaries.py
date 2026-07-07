"""Protocol101 foundation canaries: prove the pipeline rejects fake edge.

Diagnostics-tier only. This harness deliberately feeds the governed pipeline
broken inputs and requires the correct reaction to each:

1. shuffled_label_null_silent  - a model trained on permuted labels must find
   no edge on clean evaluation data (silence proves the metric stack cannot
   hallucinate signal from structure alone);
2. planted_edge_detected       - the same model/metric stack fed a feature
   that secretly encodes the label must scream (power proves silence above is
   meaningful, not blindness);
3. tampered_pickle_blocked     - a byte-modified processed session must be
   refused by the governed loader via processed_sha256 mismatch;
4. lag_zero_build_rejected     - a session rebuilt with zero index-context lag
   (one minute of lookahead) must fail acceptance;
5. random_selection_null_band  - the no-skill selection PnL distribution is
   recorded as the reference band for all future stage-1 experiments.

No model is persisted, no threshold is tuned, nothing is promoted. Labels are
consumed for diagnostics-tier null calibration only.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.model.protocol101_governed_loader import (
    load_governed_loader_artifacts,
    validate_governance_artifacts,
    validate_session_for_role,
)
from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import (
    AcceptanceThresholds,
    stable_hash,
    verify_session,
)


SCHEMA_VERSION = "Protocol101FoundationCanariesV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_foundation_canaries")
DEFAULT_ERA_MANIFEST = Path("v4/audit/autoresearch/protocol101_session_era_manifest/summary.json")
DEFAULT_ROLE_POLICY = Path("v4/audit/autoresearch/protocol101_era_role_policy/summary.json")
REGISTRY_DIR_TEMPLATE = "v4/audit/autoresearch/protocol101_owned_raw_acceptance_{tag}_v33_full/summary.json"
TRAIN_MONTH_TAGS = ("2024_10", "2024_11", "2024_12", "2025_01")
EVAL_MONTH_TAGS = ("2025_02", "2025_03")
ROLE = "diagnostics_only"
ROW_STRIDE = 3
MAX_CANDIDATES_PER_ROW = 8
RANDOM_NULL_DRAWS = 200
TOP_SHARE = 0.10
SHUFFLED_MAX_ABS_Z = 3.0
# Rho-metric commissioning history (kept deliberately, 2026-07-06/07):
# v1 used pooled rho(score, pnl) with an a-priori |rho|<0.02 bound — failed at
# rho=-0.025 with clean selection z; bound was arbitrary. v2 replaced the
# bound with a measured within-session permutation band — the band came back
# entirely negative ([-0.0142,-0.0021]), exposing that the METRIC, not the
# bound, was wrong: scores predict P(win) while pnl magnitudes are
# structurally anti-correlated with win probability across the option ladder
# (far-OTM: rare wins, huge payoffs), so pooled rho-vs-pnl measures payoff
# geometry, not information. v3 (current): mean per-session rho(score,
# win_indicator) — like-for-like, judged against its own within-session
# permutation null, which is zero-centered by construction. The economic
# criterion (selection z vs random-selection null) was clean in all versions.
RHO_NULL_DRAWS = 300
RHO_NULL_LOW_QUANTILE = 0.005
RHO_NULL_HIGH_QUANTILE = 0.995
MIN_SESSION_EXAMPLES_FOR_RHO = 50
PLANTED_MIN_Z = 5.0
PLANTED_MIN_RHO = 0.50
SEED = 20260706


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp/protocol101_foundation_canaries_scratch"))
    parser.add_argument("--skip-lag-zero-build", action="store_true", help="Skip the builder-invoking canary (for fast reruns).")
    return parser.parse_args()


def governed_sessions(month_tags: tuple[str, ...]) -> list[tuple[str, Path]]:
    """Return (session, processed_path) for every placeable diagnostics session."""
    out: list[tuple[str, Path]] = []
    for tag in month_tags:
        registry_path = Path(REGISTRY_DIR_TEMPLATE.format(tag=tag))
        artifacts = load_governed_loader_artifacts(
            acceptance_registry_path=registry_path,
            era_manifest_path=DEFAULT_ERA_MANIFEST,
            role_policy_path=DEFAULT_ROLE_POLICY,
        )
        governance_blockers = validate_governance_artifacts(artifacts)
        if governance_blockers:
            raise SystemExit(f"governance blockers for {tag}: {governance_blockers}")
        for record in artifacts.acceptance_registry.get("sessions", []):
            if record.get("status") != "pass":
                continue
            processed_path = Path(record["processed"]["processed_path"])
            validation = validate_session_for_role(
                session=record["session"],
                role=ROLE,
                processed_path=processed_path,
                artifacts=artifacts,
                governance_blockers=governance_blockers,
            )
            if validation["placeable"]:
                out.append((str(record["session"]), processed_path))
    return sorted(out)


def extract_matrix(sessions: list[tuple[str, Path]]) -> dict[str, np.ndarray]:
    """Flatten (row, masked candidate, policy) tuples into a feature matrix."""
    features: list[np.ndarray] = []
    pnl: list[float] = []
    session_ids: list[int] = []
    row_keys: list[int] = []
    row_counter = 0
    for session_idx, (_session, path) in enumerate(sessions):
        rows = pickle.load(path.open("rb"))
        for row_idx in range(0, len(rows), ROW_STRIDE):
            row = rows[row_idx]
            mask = np.asarray(row.get("candidate_mask"), dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue
            ladder = np.asarray(row.get("option_ladder"), dtype=float)
            labels = np.asarray(row.get("labels_net_pnl"), dtype=float)
            offsets = np.asarray(row.get("strike_offsets"), dtype=float)
            window = np.asarray(row.get("market_window"), dtype=float)
            market = window[-1]
            decision = pd.Timestamp(row.get("decision_time"))
            minute_of_day = decision.hour * 60 + decision.minute
            candidates = np.argwhere(mask)
            if len(candidates) > MAX_CANDIDATES_PER_ROW:
                keep = np.linspace(0, len(candidates) - 1, MAX_CANDIDATES_PER_ROW).astype(int)
                candidates = candidates[keep]
            row_counter += 1
            for strike_idx, right_idx in candidates:
                for policy_idx in range(labels.shape[2]):
                    value = float(labels[strike_idx, right_idx, policy_idx])
                    if not np.isfinite(value):
                        continue
                    features.append(
                        np.concatenate(
                            [
                                market,
                                ladder[strike_idx, right_idx],
                                [
                                    float(offsets[strike_idx]),
                                    float(right_idx),
                                    float(policy_idx),
                                    float(minute_of_day),
                                ],
                            ]
                        )
                    )
                    pnl.append(value)
                    session_ids.append(session_idx)
                    row_keys.append(row_counter)
    return {
        "X": np.asarray(features, dtype=float),
        "pnl": np.asarray(pnl, dtype=float),
        "session": np.asarray(session_ids, dtype=int),
        "row": np.asarray(row_keys, dtype=int),
    }


def mean_within_session_rho(
    scores: np.ndarray,
    outcome: np.ndarray,
    session: np.ndarray,
) -> float:
    """Mean per-session Spearman rho between score and binary win outcome.

    Judged day by day, exactly as a day-trading chooser is judged; immune to
    the cross-ladder payoff-geometry confound that poisons pooled rho-vs-pnl.
    """
    from scipy.stats import spearmanr

    rhos = []
    for session_idx in np.unique(session):
        mask = session == session_idx
        if mask.sum() < MIN_SESSION_EXAMPLES_FOR_RHO:
            continue
        wins = outcome[mask]
        if wins.min() == wins.max():
            continue
        value = spearmanr(scores[mask], wins).statistic
        if np.isfinite(value):
            rhos.append(float(value))
    return float(np.mean(rhos)) if rhos else 0.0


def rho_permutation_null(
    scores: np.ndarray,
    evaluation: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Measured null band for the within-session win-rho metric.

    Permutes win outcomes within session against fixed scores; the resulting
    band is zero-centered by construction and reflects the true sampling
    variability of the metric on this clustered data.
    """
    outcome = (evaluation["pnl"] > 0).astype(float)
    rhos = []
    for _ in range(RHO_NULL_DRAWS):
        permuted = shuffle_within_session(outcome, evaluation["session"], rng)
        rhos.append(mean_within_session_rho(scores, permuted, evaluation["session"]))
    values = np.asarray(rhos)
    return {
        "draws": RHO_NULL_DRAWS,
        "low": float(np.quantile(values, RHO_NULL_LOW_QUANTILE)),
        "high": float(np.quantile(values, RHO_NULL_HIGH_QUANTILE)),
        "low_quantile": RHO_NULL_LOW_QUANTILE,
        "high_quantile": RHO_NULL_HIGH_QUANTILE,
    }


def train_and_score(
    train: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    *,
    train_labels: np.ndarray,
    rng: np.random.Generator,
    with_rho_null: bool = False,
) -> dict[str, Any]:
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_iter=150, random_state=SEED)
    model.fit(np.nan_to_num(train["X"], nan=0.0), (train_labels > 0).astype(int))
    scores = model.predict_proba(np.nan_to_num(evaluation["X"], nan=0.0))[:, 1]
    pooled_rho_vs_pnl = float(spearmanr(scores, evaluation["pnl"]).statistic)
    win_rho = mean_within_session_rho(
        scores, (evaluation["pnl"] > 0).astype(float), evaluation["session"]
    )
    top_count = max(int(len(scores) * TOP_SHARE), 1)
    top_idx = np.argsort(scores)[-top_count:]
    top_mean_pnl = float(evaluation["pnl"][top_idx].mean())
    null_means = np.asarray(
        [
            float(evaluation["pnl"][rng.choice(len(scores), size=top_count, replace=False)].mean())
            for _ in range(RANDOM_NULL_DRAWS)
        ]
    )
    null_mu = float(null_means.mean())
    null_sigma = float(null_means.std(ddof=1)) or 1e-9
    result = {
        "within_session_win_rho": win_rho,
        "pooled_rho_vs_pnl_informational": pooled_rho_vs_pnl,
        "top_share": TOP_SHARE,
        "top_count": int(top_count),
        "top_mean_pnl": top_mean_pnl,
        "random_null_mean": null_mu,
        "random_null_sigma": null_sigma,
        "selection_z": float((top_mean_pnl - null_mu) / null_sigma),
        "random_null_p05": float(np.quantile(null_means, 0.05)),
        "random_null_p95": float(np.quantile(null_means, 0.95)),
    }
    if with_rho_null:
        result["rho_null_band"] = rho_permutation_null(scores, evaluation, rng)
    return result


def shuffle_within_session(pnl: np.ndarray, session: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = pnl.copy()
    for session_idx in np.unique(session):
        mask = session == session_idx
        shuffled[mask] = rng.permutation(shuffled[mask])
    return shuffled


def tampered_pickle_canary(scratch_dir: Path, sessions: list[tuple[str, Path]]) -> dict[str, Any]:
    session, source = sessions[0]
    month_tag = session[:7].replace("-", "_")
    registry_path = Path(REGISTRY_DIR_TEMPLATE.format(tag=month_tag))
    scratch_dir.mkdir(parents=True, exist_ok=True)
    tampered_path = scratch_dir / f"{session}.pkl"
    rows = pickle.load(source.open("rb"))
    rows[0]["market_window"] = np.asarray(rows[0]["market_window"], dtype=float) + 0.0001
    with tampered_path.open("wb") as handle:
        pickle.dump(rows, handle)
    artifacts = load_governed_loader_artifacts(
        acceptance_registry_path=registry_path,
        era_manifest_path=DEFAULT_ERA_MANIFEST,
        role_policy_path=DEFAULT_ROLE_POLICY,
    )
    validation = validate_session_for_role(
        session=session,
        role=ROLE,
        processed_path=tampered_path,
        artifacts=artifacts,
    )
    return {
        "session": session,
        "tampered_path": str(tampered_path),
        "placeable": bool(validation["placeable"]),
        "blockers": validation["blockers"],
        "passed": not validation["placeable"]
        and any("hash" in blocker or "path_differs" in blocker for blocker in validation["blockers"]),
    }


def lag_zero_canary(scratch_dir: Path) -> dict[str, Any]:
    session = "2024-10-01"
    normalized_dir = scratch_dir / "normalized_lag0"
    processed_dir = scratch_dir / "processed_lag0"
    command = [
        sys.executable,
        "-m",
        "v4.scripts.build_databento_neural_dataset",
        "--start-date", session,
        "--end-date", session,
        "--raw-root", "data/raw",
        "--normalized-dir", str(normalized_dir),
        "--processed-dir", str(processed_dir),
        "--context-mode", "official",
        "--official-spx-dir", "data/vendor/thetadata/index/spx_1m",
        "--official-vix-dir", "data/vendor/thetadata/index/vix_1m",
        "--feature-contract", "protocol101-live-v1",
        "--compute-live-policy-labels",
        "--diagnostic-index-context-lag-minutes", "0",
        "--summary-out", str(scratch_dir / "lag0_build_summary.json"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    if completed.returncode != 0:
        return {
            "session": session,
            "build_returncode": completed.returncode,
            "passed": False,
            "note": "lag-0 build itself failed; canary inconclusive",
            "stderr_tail": (completed.stderr or "")[-500:],
        }
    record = verify_session(
        session=session,
        raw_root=Path("data/raw"),
        spx_dir=Path("data/vendor/thetadata/index/spx_1m"),
        vix_dir=Path("data/vendor/thetadata/index/vix_1m"),
        processed_dir=processed_dir,
        thresholds=AcceptanceThresholds(),
    )
    failed_checks = sorted(name for name, ok in record["checks"].items() if not ok)
    return {
        "session": session,
        "build_returncode": 0,
        "acceptance_status": record["status"],
        "failed_checks": failed_checks,
        "passed": record["status"] == "fail"
        and bool({"context_lag_exact_one_minute", "no_future_context"} & set(failed_checks)),
    }


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(SEED)
    if args.scratch_dir.exists():
        shutil.rmtree(args.scratch_dir)
    args.scratch_dir.mkdir(parents=True)

    train_sessions = governed_sessions(TRAIN_MONTH_TAGS)
    eval_sessions = governed_sessions(EVAL_MONTH_TAGS)
    train = extract_matrix(train_sessions)
    evaluation = extract_matrix(eval_sessions)

    true_run = train_and_score(train, evaluation, train_labels=train["pnl"], rng=rng)
    shuffled_labels = shuffle_within_session(train["pnl"], train["session"], rng)
    shuffled_run = train_and_score(
        train, evaluation, train_labels=shuffled_labels, rng=rng, with_rho_null=True
    )

    planted_train = {**train, "X": np.column_stack([train["X"], train["pnl"] + rng.normal(0, 1.0, len(train["pnl"]))])}
    planted_eval = {
        **evaluation,
        "X": np.column_stack([evaluation["X"], evaluation["pnl"] + rng.normal(0, 1.0, len(evaluation["pnl"]))]),
    }
    planted_run = train_and_score(planted_train, planted_eval, train_labels=train["pnl"], rng=rng)

    tampered = tampered_pickle_canary(args.scratch_dir, train_sessions)
    lag_zero = (
        {"passed": None, "note": "skipped via --skip-lag-zero-build"}
        if args.skip_lag_zero_build
        else lag_zero_canary(args.scratch_dir)
    )

    rho_band = shuffled_run["rho_null_band"]
    canaries = {
        "shuffled_label_null_silent": bool(
            abs(shuffled_run["selection_z"]) < SHUFFLED_MAX_ABS_Z
            and rho_band["low"] <= shuffled_run["within_session_win_rho"] <= rho_band["high"]
        ),
        "planted_edge_detected": bool(
            planted_run["selection_z"] > PLANTED_MIN_Z
            and planted_run["within_session_win_rho"] > PLANTED_MIN_RHO
        ),
        "tampered_pickle_blocked": bool(tampered["passed"]),
        "lag_zero_build_rejected": bool(lag_zero["passed"]) if lag_zero["passed"] is not None else None,
    }
    hard_verdicts = [value for value in canaries.values() if value is not None]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if all(hard_verdicts) else "fail",
        "role": ROLE,
        "evidence_grade": "diagnostics_tier_null_calibration",
        "labels_used_for_strategy_selection": False,
        "model_persisted": False,
        "thresholds": {
            "shuffled_max_abs_z": SHUFFLED_MAX_ABS_Z,
            "rho_criterion": "measured permutation null band (within-session label permutation)",
            "rho_null_draws": RHO_NULL_DRAWS,
            "rho_null_quantiles": [RHO_NULL_LOW_QUANTILE, RHO_NULL_HIGH_QUANTILE],
            "planted_min_z": PLANTED_MIN_Z,
            "planted_min_rho": PLANTED_MIN_RHO,
            "top_share": TOP_SHARE,
            "random_null_draws": RANDOM_NULL_DRAWS,
            "row_stride": ROW_STRIDE,
            "max_candidates_per_row": MAX_CANDIDATES_PER_ROW,
            "seed": SEED,
        },
        "train_months": list(TRAIN_MONTH_TAGS),
        "eval_months": list(EVAL_MONTH_TAGS),
        "train_session_count": len(train_sessions),
        "eval_session_count": len(eval_sessions),
        "train_example_count": int(len(train["pnl"])),
        "eval_example_count": int(len(evaluation["pnl"])),
        "true_data_run_informational": true_run,
        "shuffled_label_run": shuffled_run,
        "planted_edge_run": planted_run,
        "tampered_pickle": tampered,
        "lag_zero_build": lag_zero,
        "canaries": canaries,
    }
    payload["canaries_hash"] = stable_hash(payload)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Protocol101 Foundation Canaries",
        "",
        f"- Status: `{payload['status']}`",
        f"- Train: {len(train_sessions)} sessions ({', '.join(TRAIN_MONTH_TAGS)}), {payload['train_example_count']} examples",
        f"- Eval: {len(eval_sessions)} sessions ({', '.join(EVAL_MONTH_TAGS)}), {payload['eval_example_count']} examples",
        "",
        "## Canary Verdicts",
        "",
    ]
    for name, verdict in canaries.items():
        rendered = "PASS" if verdict else "SKIPPED" if verdict is None else "FAIL"
        lines.append(f"- `{name}`: **{rendered}**")
    lines.extend(
        [
            "",
            "## Key Numbers",
            "",
            f"- True-data run (informational): win-rho={true_run['within_session_win_rho']:.4f}, selection z={true_run['selection_z']:.2f}",
            f"- Shuffled-label run: win-rho={shuffled_run['within_session_win_rho']:.4f} (measured null band [{rho_band['low']:.4f}, {rho_band['high']:.4f}]), selection z={shuffled_run['selection_z']:.2f}",
            f"- Planted-edge run: win-rho={planted_run['within_session_win_rho']:.4f}, selection z={planted_run['selection_z']:.2f}",
            f"- Random-selection null band (top-decile mean PnL): p05={true_run['random_null_p05']:.2f}, p95={true_run['random_null_p95']:.2f}",
            "",
            "## Guardrails",
            "",
            "- Diagnostics tier only; no model persisted; no thresholds tuned; nothing promoted.",
            "- The true-data run is informational and is NOT a strategy-performance claim.",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "canaries": canaries,
                "true_win_rho": round(true_run["within_session_win_rho"], 4),
                "shuffled_win_rho": round(shuffled_run["within_session_win_rho"], 4),
                "shuffled_z": round(shuffled_run["selection_z"], 2),
                "planted_win_rho": round(planted_run["within_session_win_rho"], 4),
                "planted_z": round(planted_run["selection_z"], 2),
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
