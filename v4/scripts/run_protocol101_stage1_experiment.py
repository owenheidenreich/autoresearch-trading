"""Protocol101 stage-1 experiment runner: the governed autoresearch loop.

One invocation = one preregistered experiment judged against the owner-approved
gates (v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md, approved
2026-07-07). The preregistration artifact is written BEFORE any training so a
result can never rewrite its own hypothesis.

Founding constraints (measured, not stylistic):
- Models train on PAYOFF (net PnL), never win probability — P(win)-ranked
  selection was measured losing more than random (canary, 2026-07-07).
- Selection nulls are stratified (session x policy x near/far) with a refit
  permutation envelope; uniform-random nulls are confounded and inadmissible.
- All data flows through the governed loader; holdout sessions are untouchable.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.model.protocol101_governed_loader import (
    load_governed_loader_artifacts,
    validate_governance_artifacts,
    validate_session_for_role,
)
from v4.model.protocol101_serial_simulator import (
    SerialCandidate,
    SerialSimulatorConfig,
    simulate_serial_candidates,
)
from v4.scripts.run_protocol101_foundation_canaries import (
    mean_within_session_rho,
    shuffle_within_groups,
    shuffle_within_session,
)
from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import (
    PINNED_LABEL_POLICIES,
    stable_hash,
)


SCHEMA_VERSION = "Protocol101Stage1ExperimentV1"
GATES_DOC = "v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md"
DEFAULT_ERA_MANIFEST = Path("v4/audit/autoresearch/protocol101_session_era_manifest/summary.json")
DEFAULT_ROLE_POLICY = Path("v4/audit/autoresearch/protocol101_era_role_policy/summary.json")
REGISTRY_DIR_TEMPLATE = "v4/audit/autoresearch/protocol101_owned_raw_acceptance_{tag}_v35_full/summary.json"
ALL_MONTH_TAGS = (
    "2024_10", "2024_11", "2024_12", "2025_01", "2025_02", "2025_03",
    "2025_04", "2025_05", "2025_06", "2025_07", "2025_08", "2025_09",
    "2025_10", "2025_11", "2025_12",
)

FEE_PER_TRADE = 3.00
FEE_SENSITIVITY = (2.00, 5.00)
FOLD_COUNT = 5
EMBARGO_SESSIONS = 1
TEST_REGION_SHARE = 0.50
SELECTION_SEEDS = (42, 43, 44)
CONFIRMATION_SEED = 99
REFIT_NULL_MODELS = 5
STRAT_NULL_DRAWS = 200
ROW_STRIDE = 2
MAX_CANDIDATES_PER_ROW = 10
G2_POOLED_MIN_Z = 3.0
G5_WORST_SEED_MIN_Z = 2.0
G4_MAX_DRAWDOWN = 1_500.0
G7_TRADES_PER_DAY = (0.3, 6.0)
G8_MAX_ECE = 0.10
PRE_PROGRAM_LAST_SESSION = "2025-06-30"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("v4/audit/autoresearch"))
    parser.add_argument("--max-iter", type=int, default=200)
    return parser.parse_args()


def governed_sessions(role: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for tag in ALL_MONTH_TAGS:
        registry_path = Path(REGISTRY_DIR_TEMPLATE.format(tag=tag))
        artifacts = load_governed_loader_artifacts(
            acceptance_registry_path=registry_path,
            era_manifest_path=DEFAULT_ERA_MANIFEST,
            role_policy_path=DEFAULT_ROLE_POLICY,
        )
        blockers = validate_governance_artifacts(artifacts)
        if blockers:
            raise SystemExit(f"governance blockers for {tag}: {blockers}")
        for record in artifacts.acceptance_registry.get("sessions", []):
            if record.get("status") != "pass":
                continue
            processed_path = Path(record["processed"]["processed_path"])
            validation = validate_session_for_role(
                session=record["session"],
                role=role,
                processed_path=processed_path,
                artifacts=artifacts,
                governance_blockers=blockers,
            )
            if validation["placeable"]:
                out.append((str(record["session"]), processed_path))
    return sorted(out)


@dataclass
class CandidateTable:
    """Flat candidate/policy table with everything selection and replay need."""

    X: np.ndarray
    pnl: np.ndarray
    session_idx: np.ndarray
    stratum: np.ndarray
    session_name: list[str]
    decision_time: list[str]
    contract_id: list[str]
    right: list[str]
    offset: np.ndarray
    entry_ask: np.ndarray
    policy_idx: np.ndarray


def extract_table(sessions: list[tuple[str, Path]]) -> CandidateTable:
    feats: list[np.ndarray] = []
    pnl: list[float] = []
    session_idx: list[int] = []
    stratum: list[int] = []
    session_name: list[str] = []
    decision_time: list[str] = []
    contract_id: list[str] = []
    right: list[str] = []
    offset: list[float] = []
    entry_ask: list[float] = []
    policy_col: list[int] = []
    for s_idx, (name, path) in enumerate(sessions):
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
            metadata = row.get("contract_quote_metadata") or {}
            ids = np.asarray(row.get("contract_ids"), dtype=object)
            decision = pd.Timestamp(row.get("decision_time"))
            minute_of_day = decision.hour * 60 + decision.minute
            candidates = np.argwhere(mask)
            if len(candidates) > MAX_CANDIDATES_PER_ROW:
                keep = np.linspace(0, len(candidates) - 1, MAX_CANDIDATES_PER_ROW).astype(int)
                candidates = candidates[keep]
            for k_idx, r_idx in candidates:
                cid = str(ids[k_idx, r_idx])
                ask = (metadata.get(cid) or {}).get("ask")
                if ask is None or not np.isfinite(float(ask)):
                    continue
                for p_idx in range(labels.shape[2]):
                    value = float(labels[k_idx, r_idx, p_idx])
                    if not np.isfinite(value):
                        continue
                    feats.append(
                        np.concatenate(
                            [
                                market,
                                ladder[k_idx, r_idx],
                                [float(offsets[k_idx]), float(r_idx), float(p_idx), float(minute_of_day)],
                            ]
                        )
                    )
                    pnl.append(value)
                    session_idx.append(s_idx)
                    near = int(abs(float(offsets[k_idx])) <= 20.0)
                    stratum.append(s_idx * 1000 + p_idx * 10 + near)
                    session_name.append(name)
                    decision_time.append(decision.isoformat())
                    contract_id.append(cid)
                    right.append("C" if int(r_idx) == 0 else "P")
                    offset.append(float(offsets[k_idx]))
                    entry_ask.append(float(ask))
                    policy_col.append(int(p_idx))
    return CandidateTable(
        X=np.asarray(feats, dtype=float),
        pnl=np.asarray(pnl, dtype=float),
        session_idx=np.asarray(session_idx, dtype=int),
        stratum=np.asarray(stratum, dtype=int),
        session_name=session_name,
        decision_time=decision_time,
        contract_id=contract_id,
        right=right,
        offset=np.asarray(offset, dtype=float),
        entry_ask=np.asarray(entry_ask, dtype=float),
        policy_idx=np.asarray(policy_col, dtype=int),
    )


def subset(table: CandidateTable, mask: np.ndarray) -> CandidateTable:
    idx = np.flatnonzero(mask)
    return CandidateTable(
        X=table.X[idx],
        pnl=table.pnl[idx],
        session_idx=table.session_idx[idx],
        stratum=table.stratum[idx],
        session_name=[table.session_name[i] for i in idx],
        decision_time=[table.decision_time[i] for i in idx],
        contract_id=[table.contract_id[i] for i in idx],
        right=[table.right[i] for i in idx],
        offset=table.offset[idx],
        entry_ask=table.entry_ask[idx],
        policy_idx=table.policy_idx[idx],
    )


def fold_boundaries(session_names: list[str]) -> list[dict[str, Any]]:
    """Expanding-window folds over the chronological session list."""
    unique = sorted(set(session_names))
    test_region = unique[int(len(unique) * (1.0 - TEST_REGION_SHARE)):]
    windows = np.array_split(np.asarray(test_region), FOLD_COUNT)
    folds = []
    for k, window in enumerate(windows):
        window = list(window)
        first_test = window[0]
        train_pool = [s for s in unique if s < first_test]
        train = train_pool[:-EMBARGO_SESSIONS] if EMBARGO_SESSIONS else train_pool
        folds.append(
            {
                "fold": k,
                "train_sessions": train,
                "embargo_sessions": train_pool[-EMBARGO_SESSIONS:] if EMBARGO_SESSIONS else [],
                "test_sessions": window,
            }
        )
    return folds


def train_payoff_model(table: CandidateTable, seed: int, max_iter: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed)
    model.fit(np.nan_to_num(table.X, nan=0.0), table.pnl)
    return model


def select_trades(table: CandidateTable, scores: np.ndarray, fee: float) -> np.ndarray:
    """One candidate per decision minute: argmax predicted payoff if > fee."""
    order = {}
    for i, (sess, dt) in enumerate(zip(table.session_name, table.decision_time)):
        key = (sess, dt)
        if scores[i] > fee and (key not in order or scores[i] > scores[order[key]]):
            order[key] = i
    return np.asarray(sorted(order.values()), dtype=int)


def replay(table: CandidateTable, selected: np.ndarray, fee: float, split: str) -> dict[str, Any]:
    policies = {int(p["policy_idx"]): p for p in PINNED_LABEL_POLICIES}
    candidates = []
    for i in selected:
        hold = float(policies[int(table.policy_idx[i])]["max_hold_minutes"])
        candidates.append(
            SerialCandidate(
                split=split,
                session=table.session_name[i],
                decision_time=pd.Timestamp(table.decision_time[i]).to_pydatetime(),
                contract_id=table.contract_id[i],
                right=table.right[i],
                offset=float(table.offset[i]),
                entry_ask=float(table.entry_ask[i]),
                score=0.0,
                raw_label_pnl=float(table.pnl[i]) - fee,
                cooldown_minutes=hold,
                max_hold_minutes=hold,
            )
        )
    trades, state = simulate_serial_candidates(candidates, config=SerialSimulatorConfig())
    equity = state.equity_by_account.get(split, [10_000.0])
    peaks = np.maximum.accumulate(np.asarray(equity))
    drawdown = float((peaks - np.asarray(equity)).max()) if len(equity) else 0.0
    session_days = len(set(table.session_name))
    return {
        "fee_per_trade": fee,
        "candidates_selected": int(len(selected)),
        "trades_executed": len(trades),
        "skipped": state.skipped,
        "net_pnl": float(sum(t.raw_label_pnl for t in trades)),
        "final_cash": float(equity[-1]) if equity else 10_000.0,
        "max_drawdown": drawdown,
        "trades_per_day": float(len(trades) / session_days) if session_days else 0.0,
        "win_rate_diagnostic": float(np.mean([t.raw_label_pnl > 0 for t in trades])) if trades else 0.0,
    }


def stratified_selection_z(
    table: CandidateTable, scores: np.ndarray, selected: np.ndarray, rng: np.random.Generator
) -> float:
    if len(selected) == 0:
        return 0.0
    observed = float(table.pnl[selected].mean())
    null_means = [
        float(shuffle_within_groups(table.pnl, table.stratum, rng)[selected].mean())
        for _ in range(STRAT_NULL_DRAWS)
    ]
    mu, sigma = float(np.mean(null_means)), float(np.std(null_means, ddof=1)) or 1e-9
    return float((observed - mu) / sigma)


def expected_calibration_error(scores: np.ndarray, wins: np.ndarray, bins: int = 10) -> float:
    """ECE of an isotonic P(win) read-out fitted on the payoff scores."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    half = len(scores) // 2
    iso.fit(scores[:half], wins[:half])
    probs = iso.predict(scores[half:])
    outcomes = wins[half:]
    edges = np.quantile(probs, np.linspace(0, 1, bins + 1))
    ece, total = 0.0, len(probs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (probs >= lo) & (probs <= hi)
        if in_bin.sum() == 0:
            continue
        ece += in_bin.sum() / total * abs(probs[in_bin].mean() - outcomes[in_bin].mean())
    return float(ece)


def main() -> int:
    args = parse_args()
    out_dir = args.out_root / f"protocol101_stage1_{args.experiment_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "hypothesis": args.hypothesis,
        "gates_doc": GATES_DOC,
        "model": f"HistGradientBoostingRegressor(max_iter={args.max_iter})",
        "target": "labels_net_pnl (gross payoff); selection threshold = fee",
        "fee_per_trade": FEE_PER_TRADE,
        "fee_sensitivity": FEE_SENSITIVITY,
        "folds": FOLD_COUNT,
        "embargo_sessions": EMBARGO_SESSIONS,
        "test_region_share": TEST_REGION_SHARE,
        "selection_seeds": SELECTION_SEEDS,
        "confirmation_seed": CONFIRMATION_SEED,
        "refit_null_models": REFIT_NULL_MODELS,
        "row_stride": ROW_STRIDE,
        "max_candidates_per_row": MAX_CANDIDATES_PER_ROW,
        "registry_template": REGISTRY_DIR_TEMPLATE,
        "month_tags": ALL_MONTH_TAGS,
    }
    config["config_hash"] = stable_hash(config)
    prereg_path = out_dir / "preregistration.json"
    prereg_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"preregistered": str(prereg_path), "config_hash": config["config_hash"]}))

    train_ok = {s for s, _ in governed_sessions("train")}
    test_ok = {s for s, _ in governed_sessions("test")}
    sessions = governed_sessions("train")
    table = extract_table(sessions)
    folds = fold_boundaries(table.session_name)
    for fold in folds:
        assert all(s in train_ok for s in fold["train_sessions"])
        assert all(s in test_ok for s in fold["test_sessions"])

    session_arr = np.asarray(table.session_name)
    seed_results: dict[int, Any] = {}
    for seed in SELECTION_SEEDS:
        fold_rows = []
        pooled_scores, pooled_pnl, pooled_stratum, pooled_session = [], [], [], []
        pooled_selected_mask = []
        for fold in folds:
            train_mask = np.isin(session_arr, fold["train_sessions"])
            test_mask = np.isin(session_arr, fold["test_sessions"])
            train_tbl, test_tbl = subset(table, train_mask), subset(table, test_mask)
            model = train_payoff_model(train_tbl, seed, args.max_iter)
            scores = model.predict(np.nan_to_num(test_tbl.X, nan=0.0))
            rng = np.random.default_rng(seed * 100 + fold["fold"])
            fold_row = {"fold": fold["fold"], "test_sessions": len(fold["test_sessions"])}
            for fee in (FEE_PER_TRADE, *FEE_SENSITIVITY):
                selected = select_trades(test_tbl, scores, fee)
                sim = replay(test_tbl, selected, fee, f"seed{seed}_fold{fold['fold']}")
                key = "primary" if fee == FEE_PER_TRADE else f"fee_{fee:.2f}"
                fold_row[key] = sim
                if fee == FEE_PER_TRADE:
                    fold_row["selection_z"] = stratified_selection_z(test_tbl, scores, selected, rng)
                    sel_mask = np.zeros(len(test_tbl.pnl), dtype=bool)
                    sel_mask[selected] = True
                    pooled_selected_mask.append(sel_mask)
            fold_rows.append(fold_row)
            pooled_scores.append(scores)
            pooled_pnl.append(test_tbl.pnl)
            pooled_stratum.append(test_tbl.stratum)
            pooled_session.append(test_tbl.session_idx)
        scores_all = np.concatenate(pooled_scores)
        pnl_all = np.concatenate(pooled_pnl)
        stratum_all = np.concatenate(pooled_stratum)
        session_all = np.concatenate(pooled_session)
        win_rho = mean_within_session_rho(scores_all, (pnl_all > 0).astype(float), session_all)
        ece = expected_calibration_error(scores_all, (pnl_all > 0).astype(float))
        seed_results[seed] = {
            "folds": fold_rows,
            "pooled_win_rho": win_rho,
            "pooled_ece": ece,
            "pooled_net_pnl": float(sum(f["primary"]["net_pnl"] for f in fold_rows)),
            "profitable_folds": int(sum(f["primary"]["net_pnl"] > 0 for f in fold_rows)),
            "worst_fold_drawdown": float(max(f["primary"]["max_drawdown"] for f in fold_rows)),
            "mean_selection_z": float(np.mean([f["selection_z"] for f in fold_rows])),
            "trades_per_day": float(np.mean([f["primary"]["trades_per_day"] for f in fold_rows])),
        }

    # Refit permutation envelope on the last fold's split (seed 42, pooled test region)
    rng = np.random.default_rng(4242)
    refit_z = []
    last_fold = folds[-1]
    train_mask = np.isin(session_arr, last_fold["train_sessions"])
    test_mask = np.isin(session_arr, last_fold["test_sessions"])
    train_tbl, test_tbl = subset(table, train_mask), subset(table, test_mask)
    for k in range(REFIT_NULL_MODELS):
        shuffled = shuffle_within_session(train_tbl.pnl, train_tbl.session_idx, np.random.default_rng(9000 + k))
        refit_model = train_payoff_model(
            CandidateTable(**{**train_tbl.__dict__, "pnl": shuffled}), seed=SELECTION_SEEDS[0], max_iter=args.max_iter
        )
        refit_scores = refit_model.predict(np.nan_to_num(test_tbl.X, nan=0.0))
        refit_selected = select_trades(test_tbl, refit_scores, FEE_PER_TRADE)
        refit_z.append(stratified_selection_z(test_tbl, refit_scores, refit_selected, rng))

    # Gates (primary fee, per approved doc)
    per_seed = list(seed_results.values())
    worst = min(per_seed, key=lambda r: r["pooled_net_pnl"])
    era_split = {"pre_program": [], "owned_2025h2": []}
    for r in per_seed:
        for f in r["folds"]:
            fold_sessions = folds[f["fold"]]["test_sessions"]
            era = "pre_program" if max(fold_sessions) <= PRE_PROGRAM_LAST_SESSION else "owned_2025h2"
            era_split[era].append(f["primary"]["net_pnl"])
    gates = {
        "G1_profitability": all(r["profitable_folds"] >= 4 and r["pooled_net_pnl"] > 0 for r in per_seed[:1])
        and per_seed[0]["pooled_net_pnl"] > 0,
        "G2_beats_no_skill": per_seed[0]["mean_selection_z"] >= G2_POOLED_MIN_Z
        and per_seed[0]["mean_selection_z"] > max(refit_z) + 2.0,
        "G3_beats_heuristics": None,
        "G4_drawdown": all(r["worst_fold_drawdown"] <= G4_MAX_DRAWDOWN for r in per_seed),
        "G5_seed_robustness": worst["pooled_net_pnl"] > 0 and worst["mean_selection_z"] >= G5_WORST_SEED_MIN_Z,
        "G6_era_guard": all(np.median(v) > 0 for v in era_split.values() if v) or "regime_bound_requires_owner_review",
        "G7_frequency_band": all(
            G7_TRADES_PER_DAY[0] <= r["trades_per_day"] <= G7_TRADES_PER_DAY[1] for r in per_seed
        ),
        "G8_calibration": all(r["pooled_ece"] <= G8_MAX_ECE for r in per_seed),
        "G9_confirmation": None,
    }
    payload = {
        **config,
        "labels_used_for_strategy_selection": True,
        "model_tier": True,
        "promotion": False,
        "paper_submit_allowed": False,
        "seed_results": seed_results,
        "refit_null_selection_z": refit_z,
        "era_fold_pnl": era_split,
        "gates": {k: (bool(v) if isinstance(v, bool) else v) for k, v in gates.items()},
    }
    payload["result_hash"] = stable_hash(payload)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    lines = [
        f"# Stage-1 Experiment `{args.experiment_id}`",
        "",
        f"- Hypothesis: {args.hypothesis}",
        f"- Config hash: `{config['config_hash']}`",
        f"- Sessions: {len(set(table.session_name))} | examples: {len(table.pnl)}",
        "",
        "## Per-seed (primary fee $%.2f)" % FEE_PER_TRADE,
        "",
    ]
    for seed, r in seed_results.items():
        lines.append(
            f"- seed {seed}: pooled net ${r['pooled_net_pnl']:.0f}, {r['profitable_folds']}/5 folds "
            f"profitable, mean sel z {r['mean_selection_z']:.2f}, win-rho {r['pooled_win_rho']:.3f}, "
            f"max DD ${r['worst_fold_drawdown']:.0f}, {r['trades_per_day']:.2f} tr/day, ECE {r['pooled_ece']:.3f}"
        )
    lines += ["", f"- Refit-null selection z envelope: {[round(z, 2) for z in refit_z]}", "", "## Gates", ""]
    for k, v in payload["gates"].items():
        lines.append(f"- {k}: **{v}**")
    lines += ["", "## Guardrails", "", "- Model-tier experiment; no promotion, no paper-submit, holdout untouched."]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(out_dir / "report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
