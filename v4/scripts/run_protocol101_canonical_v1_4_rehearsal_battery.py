"""Canonical v1.4 battery runner for rehearsal / sealed-confirmation days.

Runs the frozen attempt005 battery semantics on a parameterized session set:
  - L1 fixed probes + L3 transfer probes under the frozen v1.4 selection
    contract (thresholds/epsilons loaded from the contract, never re-derived)
  - L3 diagnostic models trained on the burned-day historical rows ONLY and
    scored on the evaluation days (per the confirmation preregistration)
  - compact L0 field-divergence/bias and L2 source-discriminator checks
  - absolute pass criteria (no reduction-vs-baseline gates)

Modes:
  rehearsal (default): eval = validation+development days, train = burned.
  burned_validation:   eval = train = burned days; must reproduce attempt005
                       (runner self-test; run this before trusting rehearsal).

This script modifies no frozen artifact. It patches module-level SESSIONS on
the imported audit modules in-process only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v4.scripts import run_protocol101_canonical_v1_l0_l2_design_audit as l0l2
from v4.scripts import run_protocol101_canonical_v1_l1_l3_probe_audit as l1l3
from v4.scripts import run_protocol101_canonical_v1_4_near_atm_band_restriction_probe as v14

AUDIT = Path("v4/audit/autoresearch")
FROZEN = {
    "contract": (
        AUDIT / "protocol101_canonical_v1_4_near_atm_band_restriction_attempt005/selection_contract_v1_4.json",
        "602fd8eff564a059ad114dd051b6793cb50bcc50269c83b81bdfe25aa119ef57",
    ),
    "feature_definition": (
        AUDIT / "protocol101_canonical_v1_l0_l2_design_audit_attempt001/canonical_feature_definition.json",
        "628b162b66fcce50c0bd08e7bd34606ecf9a4a95afc0cd566f12bb79429d5bb8",
    ),
    "sealing_rule": (
        AUDIT / "protocol101_sealed_day_assignment/sealing_rule.json",
        "a9fb2c25e0b31c42b6846925f198930f44594cf7567a0970c767169ed468a89e",
    ),
    "confirmation_prereg": (
        AUDIT / "protocol101_canonical_v1_sealed_confirmation_preregistration/preregistration.md",
        "13176551ed26f4a42f4a9fce204f3059591a55d24d2833caf1c29b8401ed06b9",
    ),
}
ATTEMPT001_DIR = AUDIT / "protocol101_canonical_v1_l1_l3_probe_audit_attempt001"
L0_PREREG = AUDIT / "protocol101_canonical_v1_l0_l2_design_audit_attempt001/preregistration.json"
BURNED = ("2026-06-30", "2026-07-01", "2026-07-02")
BURNED_PREFIX = "protocol101_live_v2_candidate_universe_parity_source_aligned"
REPAIRED_PROBES = ("option_mid_momentum", "internal_iv_expansion_compression")
EXPECTED_TRACE_ROWS_PER_SESSION = 360
EXPECTED_SLOT_ROWS_PER_SESSION = 360 * 42


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=v14.json_default) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["rehearsal", "burned_validation"], default="rehearsal")
    p.add_argument("--eval-sessions", nargs="+", default=["2026-07-10", "2026-07-13", "2026-07-14"])
    p.add_argument("--eval-prefix", default="protocol101_canonical_rehearsal")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--l0-l2-dir", type=Path, default=l1l3.DEFAULT_L0_L2_DIR)
    p.add_argument("--group1-dir", type=Path, default=l1l3.DEFAULT_GROUP1_DIR)
    p.add_argument("--group1-definition", type=Path, default=l1l3.DEFAULT_GROUP1_DEFINITION)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def build_frame(args: argparse.Namespace, prefix: str, sessions: tuple[str, ...]):
    l0l2.SESSIONS = tuple(sessions)
    l1l3.SESSIONS = tuple(sessions)
    ns = SimpleNamespace(
        l0_l2_dir=args.l0_l2_dir,
        group1_dir=args.group1_dir,
        group1_definition=args.group1_definition,
        trace_prefix=prefix,
    )
    return v14.build_paired_frame(ns)


def full_session_trace_blockers(
    readiness: dict[str, Any], sessions: tuple[str, ...]
) -> list[str]:
    """Reject partial captures before they can be scored as parity failures."""
    trace_sessions = (readiness.get("trace_readiness") or {}).get("sessions") or {}
    blockers: list[str] = []
    for session in sessions:
        evidence = trace_sessions.get(session) or {}
        for plane in ("historical", "ibkr"):
            rows = evidence.get(f"{plane}_rows")
            slot_rows = evidence.get(f"{plane}_slot_rows")
            if rows != EXPECTED_TRACE_ROWS_PER_SESSION:
                blockers.append(
                    f"incomplete_{plane}_trace_rows:{session}:"
                    f"{rows}!={EXPECTED_TRACE_ROWS_PER_SESSION}"
                )
            if slot_rows != EXPECTED_SLOT_ROWS_PER_SESSION:
                blockers.append(
                    f"incomplete_{plane}_slot_rows:{session}:"
                    f"{slot_rows}!={EXPECTED_SLOT_ROWS_PER_SESSION}"
                )
    return blockers


def count_material_true_reorderings(merged: pd.DataFrame) -> int:
    """Count material reorderings under either supported merged-frame schema."""
    material_column = next(
        (
            column
            for column in (
                "selected_slot_swap_economically_material",
                "economically_material",
            )
            if column in merged.columns
        ),
        None,
    )
    if material_column is None or "true_score_reordering" not in merged.columns:
        raise ValueError("merged decisions lack material-reordering columns")
    return int(
        (
            (merged["true_score_reordering"] == True)  # noqa: E712
            & (merged[material_column] == True)  # noqa: E712
        ).sum()
    )


def frozen_l1_thresholds(contract: dict[str, Any], l1_summary: dict[str, Any]) -> dict[str, Any]:
    """Attempt001 dict-shaped thresholds with the two v1.4 recalibrated values
    injected from the frozen contract. Nothing recomputed."""
    thresholds = json.loads(json.dumps(l1_summary["thresholds"]))
    contract_l1 = contract["action_thresholds"]["l1"]
    for probe, value in contract_l1.items():
        entry = thresholds.setdefault(probe, {})
        if probe not in REPAIRED_PROBES:
            prior = entry.get("threshold")
            if prior is not None and not np.isclose(float(prior), float(value)):
                raise SystemExit(f"contract/attempt001 threshold mismatch for {probe}: {prior} vs {value}")
        entry["threshold"] = float(value)
    return thresholds


def build_l3_reference(contract: dict[str, Any], l3_csv: Path) -> pd.DataFrame:
    ref = pd.read_csv(l3_csv)[["probe", "threshold"]].copy()
    contract_l3 = contract["action_thresholds"]["l3"]
    for probe, value in contract_l3.items():
        mask = ref["probe"] == probe
        if mask.any() and not np.isclose(float(ref.loc[mask, "threshold"].iloc[0]), float(value)):
            raise SystemExit(f"contract/attempt001 L3 threshold mismatch for {probe}")
    return ref


def build_populations_cross_day(
    *,
    eval_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    admitted: list[str],
    l1_thresholds: dict[str, Any],
    previous_l3: pd.DataFrame,
    epsilon: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    populations: list[pd.DataFrame] = []
    for probe, family in v14.L1_PROBE_FAMILIES.items():
        populations.append(
            v14.score_population_rows(
                paired=eval_frame,
                kind="L1",
                probe=probe,
                feature_family=family,
                threshold=float(l1_thresholds[probe]["threshold"]),
                epsilon=float(epsilon[probe]["epsilon"]),
                historical_scores=v14.l1_v14_score(eval_frame, probe, "historical", l1_thresholds),
                ibkr_scores=v14.l1_v14_score(eval_frame, probe, "ibkr", l1_thresholds),
            )
        )
    train_mask = train_frame["y_15m_conservative_return_ibkr"].notna()
    y = (train_frame.loc[train_mask, "y_15m_conservative_return_ibkr"].astype(float) > 0).astype(int).to_numpy()
    for subset_name in v14.L3_SUBSETS:
        cols = v14.l3_feature_columns(subset_name, admitted)
        for model_name in ("logistic", "bounded_hgb"):
            probe = f"{subset_name}_{model_name}"
            row = previous_l3[previous_l3["probe"] == probe]
            if row.empty:
                continue
            model = v14.train_l3_model(model_name)
            model.fit(v14.matrix(train_frame.loc[train_mask], cols, "_historical"), y)
            populations.append(
                v14.score_population_rows(
                    paired=eval_frame,
                    kind="L3",
                    probe=probe,
                    feature_family="+".join(v14.L3_SUBSETS[subset_name]) + "+G1",
                    threshold=float(row.iloc[0]["threshold"]),
                    epsilon=float(epsilon[probe]["epsilon"]),
                    historical_scores=model.predict_proba(v14.matrix(eval_frame, cols, "_historical"))[:, 1],
                    ibkr_scores=model.predict_proba(v14.matrix(eval_frame, cols, "_ibkr"))[:, 1],
                    subset=subset_name,
                    model=model_name,
                )
            )
    return pd.concat(populations, ignore_index=True)


def compact_l0(eval_frame: pd.DataFrame, admitted: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    prereg = json.loads(L0_PREREG.read_text())
    rows = []
    vol = pd.to_numeric(eval_frame.get("realized_5m_spx_vol_historical",
                        eval_frame.get("realized_5m_spx_vol", pd.Series(np.nan, index=eval_frame.index))), errors="coerce")
    y = pd.to_numeric(eval_frame["y_15m_conservative_return_ibkr"], errors="coerce")
    for feature in admitted:
        h = pd.to_numeric(eval_frame.get(f"{feature}_historical"), errors="coerce")
        i = pd.to_numeric(eval_frame.get(f"{feature}_ibkr"), errors="coerce")
        if h is None or i is None:
            continue
        both = h.notna() & i.notna()
        applicable = h.notna() | i.notna()
        coverage = float(both.sum() / applicable.sum()) if applicable.any() else 0.0
        drift = (h - i).abs()
        # frozen policy historical_reference_stddev_with_floor_v1: scale_floor = 1.0
        ref_std = max(float(h.std() or 0.0), 1.0)
        std_p95 = float(drift[both].quantile(0.95)) / ref_std if both.any() else np.nan
        d = (h - i)[both]
        def rho(target):
            t = target[both]
            ok = d.notna() & t.notna()
            if ok.sum() < 50:
                return np.nan
            return float(spearmanr(d[ok], t[ok]).statistic)
        rows.append({
            "feature": feature, "coverage": coverage, "standardized_abs_p95": std_p95,
            "rho_y": rho(y), "rho_abs_y": rho(y.abs()), "rho_vol": rho(vol),
            "pass": bool(coverage >= 0.80 and std_p95 <= 0.10
                         and all(abs(x) <= 0.05 for x in (rho(y) or 0, rho(y.abs()) or 0, rho(vol) or 0)
                                 if not np.isnan(x))),
        })
    table = pd.DataFrame(rows)
    summary = {
        "features": len(table),
        "failing": table.loc[~table["pass"], "feature"].tolist(),
        "max_standardized_p95": float(table["standardized_abs_p95"].max()),
        "gates_source": str(L0_PREREG),
        "inference_grade": "design_grade_point_estimates",
    }
    return table, summary


def compact_l2(eval_frame: pd.DataFrame, admitted: list[str], eval_prefix: str,
               eval_sessions: tuple[str, ...]) -> dict[str, Any]:
    def stack(cols_suffix_pairs):
        frames, labels, groups = [], [], []
        for suffix, label in (("_historical", 0), ("_ibkr", 1)):
            cols = [c for c in cols_suffix_pairs if c.endswith(suffix)]
            X = eval_frame[cols].apply(pd.to_numeric, errors="coerce")
            X.columns = [c.removesuffix(suffix) for c in cols]
            frames.append(X)
            labels.append(np.full(len(X), label))
            groups.append(eval_frame["session_date"].to_numpy())
        X = pd.concat(frames, ignore_index=True).fillna(0.0).to_numpy()
        return X, np.concatenate(labels), np.concatenate(groups)

    feat_cols = [f"{f}{s}" for f in admitted for s in ("_historical", "_ibkr")
                 if f"{f}{s}" in eval_frame.columns]
    X, yl, groups = stack(feat_cols)
    hgb = HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=42)
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lodo = LeaveOneGroupOut()
    out: dict[str, Any] = {}
    for name, clf in (("hgb", hgb), ("logistic", logit)):
        scores = cross_val_score(clf, X, yl, cv=lodo, groups=groups, scoring="roc_auc")
        out[f"slot_{name}_auc"] = float(np.mean(scores))
        out[f"slot_{name}_fold_aucs"] = [float(s) for s in scores]
    # null control: odd/even minutes within historical plane, grouped by
    # minute so per-minute memorization is impossible (a minute's rows never
    # straddle train/test).
    from sklearn.model_selection import GroupKFold
    minute_idx = pd.factorize(eval_frame["decision_minute_et"].astype(str))[0]
    Xh = eval_frame[[c for c in feat_cols if c.endswith("_historical")]].apply(
        pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    out["null_control_auc"] = float(np.mean(cross_val_score(
        HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=42),
        Xh, minute_idx % 2, cv=LeaveOneGroupOut(),
        groups=eval_frame["session_date"].to_numpy(), scoring="roc_auc")))
    # positive control: raw fields from the pre-pairing slot frame (the
    # paired frame drops raw_quote_age_ms, the strongest source fingerprint).
    l0l2.SESSIONS = tuple(eval_sessions)
    l1l3.SESSIONS = tuple(eval_sessions)
    slot_frame, _ = l0l2.build_slot_frame(eval_prefix)
    raw_fields = [c for c in ("raw_bid", "raw_ask", "raw_spread", "raw_quote_age_ms")
                  if c in slot_frame.columns]
    if raw_fields and "source" in slot_frame.columns:
        Xr = slot_frame[raw_fields].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        yr = (slot_frame["source"].astype(str) == "ibkr").astype(int).to_numpy()
        gr = slot_frame["session_date"].to_numpy()
        out["positive_control_auc"] = float(np.mean(cross_val_score(
            HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=42),
            Xr, yr, cv=LeaveOneGroupOut(), groups=gr, scoring="roc_auc")))
        out["positive_control_fields"] = raw_fields
    else:
        out["positive_control_auc"] = None
        out["positive_control_fields"] = []
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} exists; pass --force")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = {"status": "started", "mode": args.mode,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "broker_endpoint_called": False, "paid_data_download": False,
                "paper_submit_allowed": False, "threshold_optimization_executed": False}
    write_json(out_dir / "progress.json", progress)

    hash_report, mismatches = {}, []
    for name, (path, expected) in FROZEN.items():
        actual = sha256_path(path) if path.exists() else "MISSING"
        hash_report[name] = {"path": str(path), "expected": expected, "actual": actual}
        if actual != expected:
            mismatches.append(name)
    write_json(out_dir / "frozen_input_hashes.json", hash_report)
    if mismatches:
        write_json(out_dir / "routing_decision.json", {
            "routing_decision": "rehearsal_insufficient_artifacts",
            "blockers": [f"hash_mismatch_{m}" for m in mismatches]})
        return

    contract = json.loads(FROZEN["contract"][0].read_text())
    l1_summary = json.loads((ATTEMPT001_DIR / "l1_summary.json").read_text())
    l1_thresholds = frozen_l1_thresholds(contract, l1_summary)
    previous_l3 = build_l3_reference(contract, ATTEMPT001_DIR / "l3_transfer_probe_results.csv")
    epsilon = contract["frozen_epsilons"]

    if args.mode == "burned_validation":
        eval_sessions, eval_prefix = BURNED, BURNED_PREFIX
    else:
        eval_sessions, eval_prefix = tuple(args.eval_sessions), args.eval_prefix

    eval_frame, admitted, readiness = build_frame(args, eval_prefix, eval_sessions)
    readiness_blockers = list(readiness["blockers"])
    readiness_blockers.extend(full_session_trace_blockers(readiness, eval_sessions))
    if readiness_blockers:
        write_json(out_dir / "routing_decision.json", {
            "routing_decision": "rehearsal_insufficient_artifacts",
            "blockers": readiness_blockers})
        return
    if args.mode == "burned_validation":
        train_frame = eval_frame
    else:
        train_frame, _, train_readiness = build_frame(args, BURNED_PREFIX, BURNED)
        if train_readiness["blockers"]:
            write_json(out_dir / "routing_decision.json", {
                "routing_decision": "rehearsal_insufficient_artifacts",
                "blockers": ["train:" + b for b in train_readiness["blockers"]]})
            return

    progress["status"] = "building_populations"
    write_json(out_dir / "progress.json", progress)
    population = build_populations_cross_day(
        eval_frame=eval_frame, train_frame=train_frame, admitted=admitted,
        l1_thresholds=l1_thresholds, previous_l3=previous_l3, epsilon=epsilon)
    decisions = v14.select_decisions(population, k=2.0, fallback_rule="score_independent_nearest_atm")
    merged = v14.enrich_disagreement_classes(v14.merge_plane_decisions(decisions))
    summary_table = v14.summarize_all(merged)
    summary_table.to_csv(out_dir / "l1_l3_results.csv", index=False)

    per_day = (merged.assign(agree=merged["action_historical"] == merged["action_ibkr"])
               .groupby(["probe", "session_date"])["agree"].mean().reset_index())
    per_day.to_csv(out_dir / "per_day_action_agreement.csv", index=False)

    progress["status"] = "l0_l2"
    write_json(out_dir / "progress.json", progress)
    l0_table, l0_summary = compact_l0(eval_frame, admitted)
    l0_table.to_csv(out_dir / "l0_results.csv", index=False)
    l2_summary = compact_l2(eval_frame, admitted, eval_prefix, eval_sessions)
    write_json(out_dir / "l2_results.json", l2_summary)

    non_random = summary_table[summary_table["probe"] != "random_with_guards_null"]
    action_fail = non_random[non_random["action_agreement"].astype(float) < 0.98]
    slot_col = "selected_slot_agreement_mutually_confident_enter"
    n_col = "mutually_confident_enter_n"
    slot_rows = non_random[pd.to_numeric(non_random[n_col], errors="coerce") >= 30]
    slot_fail = slot_rows[pd.to_numeric(slot_rows[slot_col], errors="coerce") < 0.99]
    material_reorder = count_material_true_reorderings(merged)
    allowance = int(2 * len(eval_sessions) / 10)
    per_day_fail = per_day[per_day["agree"] < 0.98]

    failures = []
    if len(action_fail):
        failures.append("action_agreement:" + ",".join(action_fail["probe"]))
    if len(slot_fail):
        failures.append("slot_agreement:" + ",".join(slot_fail["probe"]))
    if material_reorder > allowance:
        failures.append(f"material_true_reorderings:{material_reorder}>allowance:{allowance}")
    if l0_summary["failing"]:
        failures.append("l0:" + ",".join(l0_summary["failing"][:5]))
    if l2_summary["slot_hgb_auc"] > 0.55 or l2_summary["slot_logistic_auc"] > 0.55:
        failures.append(f"l2_auc:{l2_summary['slot_hgb_auc']:.4f}")
    if l2_summary["positive_control_auc"] is None:
        failures.append("l2_positive_control_unavailable")
    elif l2_summary["positive_control_auc"] < 0.80:
        failures.append(f"l2_positive_control:{l2_summary['positive_control_auc']:.4f}<0.80")
    if not (0.45 <= l2_summary["null_control_auc"] <= 0.55):
        failures.append(f"l2_null_control:{l2_summary['null_control_auc']:.4f}_outside_[0.45,0.55]")

    routing = {
        "schema_version": "Protocol101CanonicalV14RehearsalBatteryRoutingV1",
        "attempt_id": out_dir.name,
        "mode": args.mode,
        "evidence_grade": ("burned_validation_self_test" if args.mode == "burned_validation"
                           else "rehearsal_design_grade"),
        "eval_sessions": list(eval_sessions),
        "trained_on": "eval_frame_self" if args.mode == "burned_validation" else list(BURNED),
        "routing_decision": ("rehearsal_pass" if not failures else
                             "rehearsal_failed_" + failures[0].split(":")[0]),
        "failures": failures,
        "material_true_reordering_count": material_reorder,
        "material_reordering_allowance": allowance,
        "per_day_action_agreement_below_gate": per_day_fail.to_dict("records"),
        "l0_summary": l0_summary,
        "l2_summary": {k: v for k, v in l2_summary.items() if "fold" not in k},
        "frozen_contract_sha256": FROZEN["contract"][1],
        "no_confirmation_claim": True,
    }
    write_json(out_dir / "routing_decision.json", routing)

    lines = [
        f"# Canonical v1.4 Battery — {args.mode} ({', '.join(eval_sessions)})",
        "",
        f"- Routing: `{routing['routing_decision']}`",
        f"- Evidence grade: `{routing['evidence_grade']}` (no confirmation claim)",
        f"- Material true reorderings: `{material_reorder}` (allowance `{allowance}`)",
        f"- L2 slot AUC hgb/logistic: `{l2_summary['slot_hgb_auc']:.4f}` / `{l2_summary['slot_logistic_auc']:.4f}`"
        f" | null `{l2_summary['null_control_auc']:.4f}` | positive `{l2_summary['positive_control_auc']}`",
        f"- L0 max standardized p95: `{l0_summary['max_standardized_p95']:.4f}`; failing: `{l0_summary['failing']}`",
        "",
        "## Probe action agreement",
        "",
    ]
    for _, r in summary_table.iterrows():
        lines.append(f"- `{r['probe']}`: action `{float(r['action_agreement']):.4f}`"
                     f" slot `{float(r['selected_slot_agreement']):.4f}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    progress["status"] = "complete"
    progress["routing_decision"] = routing["routing_decision"]
    progress["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(out_dir / "progress.json", progress)
    print(json.dumps({k: routing[k] for k in ("routing_decision", "failures", "mode")}, indent=1))


if __name__ == "__main__":
    main()
