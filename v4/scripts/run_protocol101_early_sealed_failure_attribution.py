"""Attribute the July 15/22 canonical-v1.4 failures to recorder health.

This is a post-open diagnostic. It keeps the frozen v1.4 thresholds, models,
features, epsilons, and selection rules unchanged, then reports probe behavior
inside objective recorder-health strata. It makes no confirmation claim.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts import run_protocol101_canonical_v1_4_near_atm_band_restriction_probe as v14
from v4.scripts import run_protocol101_canonical_v1_4_rehearsal_battery as battery


SESSIONS = ("2026-07-15", "2026-07-22")
PREFIX = "protocol101_canonical_early_confirmation"
CAPTURE_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture_sealed"
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/"
    "protocol101_canonical_v1_4_early_sealed_failure_attribution_2026_07_25"
)
SPX_TOLERANCES = (0.10, 0.50, 0.75, 1.00, 2.00)
HISTORY_WINDOWS_MINUTES = (5, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=v14.json_default) + "\n")


def quality_path(session: str) -> Path:
    capture_id = f"protocol101-recorder-{session}"
    return CAPTURE_ROOT / session / capture_id / "ibkr_capture_quality.json"


def trace_path(session: str) -> Path:
    slug = session.replace("-", "_")
    return (
        Path("v4/audit/autoresearch")
        / f"{PREFIX}_ibkr_capture_replay_{slug}"
        / "decision_traces.jsonl"
    )


def replay_opening_context_ready(session: str) -> bool:
    rows = [json.loads(line) for line in trace_path(session).read_text().splitlines() if line.strip()]
    flags = [
        bool((row.get("payload") or {}).get("opening_context_ready"))
        for row in rows
    ]
    return bool(flags) and all(flags)


def recovered_decision_minutes(session: str) -> set[str]:
    payload = json.loads(quality_path(session).read_text())
    completed = (payload.get("evidence") or {}).get("recovered_checkpoint_minutes") or []
    decisions: set[str] = set()
    for value in completed:
        timestamp = pd.Timestamp(value) + pd.Timedelta(minutes=1)
        decisions.add(timestamp.strftime("%H:%M"))
    return decisions


def add_causal_history_health(minute: pd.DataFrame) -> pd.DataFrame:
    """Mark rows whose entire causal lookback is continuously healthy."""
    minute = minute.copy()
    minute["_decision_ts"] = pd.to_datetime(
        minute["session_date"].astype(str)
        + " "
        + minute["decision_minute_et"].astype(str),
        format="%Y-%m-%d %H:%M",
    )
    minute = minute.sort_values(
        ["session_date", "_decision_ts"], kind="stable"
    ).reset_index(drop=True)

    bases = (
        "raw_atm_aligned_spx_within_0_75",
        "fully_healthy",
    )
    for base in bases:
        for lookback in HISTORY_WINDOWS_MINUTES:
            column = f"{base}_{lookback}m_history"
            minute[column] = False
            for _, index in minute.groupby("session_date", sort=False).groups.items():
                positions = list(index)
                timestamps = minute.loc[positions, "_decision_ts"].reset_index(drop=True)
                healthy = minute.loc[positions, base].astype(bool).reset_index(drop=True)
                window = lookback + 1
                rolling_healthy = healthy.rolling(window, min_periods=window).sum() == window
                continuous = timestamps.diff(periods=lookback) == pd.Timedelta(
                    minutes=lookback
                )
                minute.loc[positions, column] = (
                    rolling_healthy & continuous
                ).to_numpy(dtype=bool)

    return minute.drop(columns=["_decision_ts"])


def minute_health(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session_date",
        "decision_minute_et",
        "A.context.spx_for_ladder_historical",
        "A.context.spx_for_ladder_ibkr",
        "A.context.atm_strike_historical",
        "A.context.atm_strike_ibkr",
    ]
    minute = frame[columns].groupby(
        ["session_date", "decision_minute_et"], as_index=False
    ).median(numeric_only=True)
    minute["spx_abs_drift_points"] = (
        minute["A.context.spx_for_ladder_historical"]
        - minute["A.context.spx_for_ladder_ibkr"]
    ).abs()
    minute["atm_aligned"] = (
        minute["A.context.atm_strike_historical"]
        == minute["A.context.atm_strike_ibkr"]
    )
    recovered = {
        session: recovered_decision_minutes(session)
        for session in SESSIONS
    }
    opening = {
        session: replay_opening_context_ready(session)
        for session in SESSIONS
    }
    minute["recovered_checkpoint"] = [
        str(row.decision_minute_et) in recovered[str(row.session_date)]
        for row in minute.itertuples(index=False)
    ]
    minute["raw_checkpoint"] = ~minute["recovered_checkpoint"]
    minute["opening_context_ready_for_session"] = minute["session_date"].map(opening)
    for tolerance in SPX_TOLERANCES:
        label = str(tolerance).replace(".", "_")
        minute[f"spx_within_{label}"] = minute["spx_abs_drift_points"] <= tolerance
    minute["raw_atm_aligned_spx_within_0_75"] = (
        minute["raw_checkpoint"]
        & minute["atm_aligned"]
        & minute["spx_within_0_75"]
    )
    minute["fully_healthy"] = (
        minute["raw_atm_aligned_spx_within_0_75"]
        & minute["opening_context_ready_for_session"]
    )
    minute["degraded"] = ~minute["fully_healthy"]
    return add_causal_history_health(minute)


def build_frozen_population(
    eval_frame: pd.DataFrame,
    admitted: list[str],
    args: SimpleNamespace,
) -> pd.DataFrame:
    contract = json.loads(battery.FROZEN["contract"][0].read_text())
    l1_summary = json.loads((battery.ATTEMPT001_DIR / "l1_summary.json").read_text())
    l1_thresholds = battery.frozen_l1_thresholds(contract, l1_summary)
    previous_l3 = battery.build_l3_reference(
        contract,
        battery.ATTEMPT001_DIR / "l3_transfer_probe_results.csv",
    )
    train_frame, _, train_readiness = battery.build_frame(
        args,
        battery.BURNED_PREFIX,
        battery.BURNED,
    )
    if train_readiness["blockers"]:
        raise RuntimeError(f"burned training frame blockers: {train_readiness['blockers']}")
    population = battery.build_populations_cross_day(
        eval_frame=eval_frame,
        train_frame=train_frame,
        admitted=admitted,
        l1_thresholds=l1_thresholds,
        previous_l3=previous_l3,
        epsilon=contract["frozen_epsilons"],
    )
    decisions = v14.select_decisions(
        population,
        k=2.0,
        fallback_rule="score_independent_nearest_atm",
    )
    return v14.enrich_disagreement_classes(v14.merge_plane_decisions(decisions))


def health_strata(minute: pd.DataFrame) -> dict[str, set[tuple[str, str]]]:
    definitions = {
        "all": pd.Series(True, index=minute.index),
        "raw_checkpoint": minute["raw_checkpoint"],
        "recovered_checkpoint": minute["recovered_checkpoint"],
        "atm_aligned": minute["atm_aligned"],
        "atm_mismatched": ~minute["atm_aligned"],
        "raw_atm_aligned_spx_within_0_75": minute["raw_atm_aligned_spx_within_0_75"],
        "raw_atm_aligned_spx_within_0_75_5m_history": minute[
            "raw_atm_aligned_spx_within_0_75_5m_history"
        ],
        "raw_atm_aligned_spx_within_0_75_15m_history": minute[
            "raw_atm_aligned_spx_within_0_75_15m_history"
        ],
        "fully_healthy": minute["fully_healthy"],
        "fully_healthy_5m_history": minute["fully_healthy_5m_history"],
        "fully_healthy_15m_history": minute["fully_healthy_15m_history"],
        "degraded": minute["degraded"],
    }
    for tolerance in SPX_TOLERANCES:
        label = str(tolerance).replace(".", "_")
        definitions[f"atm_aligned_spx_within_{label}"] = (
            minute["atm_aligned"] & minute[f"spx_within_{label}"]
        )
    for session in SESSIONS:
        slug = session.replace("-", "_")
        session_mask = minute["session_date"].astype(str) == session
        definitions[f"{slug}_all"] = session_mask
        definitions[f"{slug}_raw_atm_aligned_spx_within_0_75_15m_history"] = (
            session_mask
            & minute["raw_atm_aligned_spx_within_0_75_15m_history"]
        )
        definitions[f"{slug}_fully_healthy_15m_history"] = (
            session_mask & minute["fully_healthy_15m_history"]
        )
    out: dict[str, set[tuple[str, str]]] = {}
    for name, mask in definitions.items():
        subset = minute.loc[mask, ["session_date", "decision_minute_et"]]
        out[name] = set(map(tuple, subset.astype(str).itertuples(index=False, name=None)))
    return out


def summarize_probe_strata(
    merged: pd.DataFrame,
    strata: dict[str, set[tuple[str, str]]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    keys = list(
        zip(
            merged["session_date"].astype(str),
            merged["decision_minute_et"].astype(str),
            strict=True,
        )
    )
    tables: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for name, allowed in strata.items():
        subset = merged[[key in allowed for key in keys]].copy()
        if subset.empty:
            continue
        table = v14.summarize_all(subset)
        table.insert(0, "stratum", name)
        tables.append(table)
        non_random = table[table["probe"] != "random_with_guards_null"]
        action_pass = non_random["action_agreement"].astype(float) >= 0.98
        eligible_slot = pd.to_numeric(
            non_random["mutually_confident_enter_n"], errors="coerce"
        ) >= 30
        slot_pass = (
            pd.to_numeric(
                non_random["selected_slot_agreement_mutually_confident_enter"],
                errors="coerce",
            )
            >= 0.99
        )
        summaries.append(
            {
                "stratum": name,
                "minutes": len(allowed),
                "probe_rows": len(subset),
                "non_random_probe_count": int(len(non_random)),
                "action_pass_count": int(action_pass.sum()),
                "action_fail_probes": non_random.loc[~action_pass, "probe"].tolist(),
                "worst_action_agreement": float(
                    non_random["action_agreement"].astype(float).min()
                ),
                "slot_gate_eligible_probe_count": int(eligible_slot.sum()),
                "slot_fail_probes": non_random.loc[
                    eligible_slot & ~slot_pass, "probe"
                ].tolist(),
                "material_true_reorderings": int(
                    (
                        (subset["true_score_reordering"] == True)  # noqa: E712
                        & (subset["economically_material"] == True)  # noqa: E712
                    ).sum()
                ),
            }
        )
    return pd.concat(tables, ignore_index=True), summaries


def summarize_flip_mechanisms(
    merged: pd.DataFrame,
    strata: dict[str, set[tuple[str, str]]],
) -> pd.DataFrame:
    keys = list(
        zip(
            merged["session_date"].astype(str),
            merged["decision_minute_et"].astype(str),
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    for name, allowed in strata.items():
        subset = merged[[key in allowed for key in keys]].copy()
        for probe, group in subset.groupby("probe", sort=True):
            score_drift = (
                pd.to_numeric(group["selected_score_historical"], errors="coerce")
                - pd.to_numeric(group["selected_score_ibkr"], errors="coerce")
            ).abs()
            flips = group[group["action_flip"] == True]  # noqa: E712
            rows.append(
                {
                    "stratum": name,
                    "probe": probe,
                    "minutes": len(group),
                    "action_flips": len(flips),
                    "threshold_adjacent_action_flips": int(
                        (flips["threshold_adjacent_action_flip"] == True).sum()  # noqa: E712
                    ),
                    "non_threshold_action_flips": int(
                        (flips["non_threshold_action_flip"] == True).sum()  # noqa: E712
                    ),
                    "median_abs_selected_score_drift": float(score_drift.median()),
                    "p95_abs_selected_score_drift": float(score_drift.quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)


def l0_by_stratum(
    eval_frame: pd.DataFrame,
    admitted: list[str],
    strata: dict[str, set[tuple[str, str]]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    keys = list(
        zip(
            eval_frame["session_date"].astype(str),
            eval_frame["decision_minute_et"].astype(str),
            strict=True,
        )
    )
    out: list[dict[str, Any]] = []
    tables: list[pd.DataFrame] = []
    names = [
        "all",
        "raw_checkpoint",
        "recovered_checkpoint",
        "raw_atm_aligned_spx_within_0_75",
        "raw_atm_aligned_spx_within_0_75_5m_history",
        "raw_atm_aligned_spx_within_0_75_15m_history",
        "fully_healthy",
        "fully_healthy_5m_history",
        "fully_healthy_15m_history",
        "degraded",
    ]
    names.extend(
        name
        for name in strata
        if name.startswith("2026_07_")
    )
    for name in names:
        allowed = strata[name]
        subset = eval_frame[[key in allowed for key in keys]]
        if subset.empty:
            continue
        table, summary = battery.compact_l0(subset, admitted)
        table.insert(0, "stratum", name)
        tables.append(table)
        out.append({"stratum": name, "slot_rows": len(subset), **summary})
    return pd.concat(tables, ignore_index=True), out


def main() -> None:
    cli = parse_args()
    if cli.out_dir.exists():
        if not cli.force:
            raise SystemExit(f"{cli.out_dir} exists; pass --force")
        shutil.rmtree(cli.out_dir)
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "status": "started",
        "evidence_grade": "post_open_failure_attribution_only",
        "model_training_executed": False,
        "threshold_tuning_executed": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
    }
    write_json(cli.out_dir / "progress.json", progress)

    build_args = SimpleNamespace(
        l0_l2_dir=battery.l1l3.DEFAULT_L0_L2_DIR,
        group1_dir=battery.l1l3.DEFAULT_GROUP1_DIR,
        group1_definition=battery.l1l3.DEFAULT_GROUP1_DEFINITION,
    )
    eval_frame, admitted, readiness = battery.build_frame(
        build_args,
        PREFIX,
        SESSIONS,
    )
    blockers = list(readiness["blockers"])
    blockers.extend(battery.full_session_trace_blockers(readiness, SESSIONS))
    if blockers:
        write_json(cli.out_dir / "routing_decision.json", {
            "routing_decision": "attribution_insufficient_artifacts",
            "blockers": blockers,
        })
        return

    minute = minute_health(eval_frame)
    minute.to_csv(cli.out_dir / "minute_health.csv", index=False)
    strata = health_strata(minute)

    merged = build_frozen_population(eval_frame, admitted, build_args)
    merged.to_parquet(cli.out_dir / "frozen_probe_decisions_merged.parquet", index=False)
    probe_table, probe_summaries = summarize_probe_strata(merged, strata)
    probe_table.to_csv(cli.out_dir / "probe_agreement_by_health_stratum.csv", index=False)
    flip_table = summarize_flip_mechanisms(merged, strata)
    flip_table.to_csv(cli.out_dir / "action_flip_mechanisms_by_health_stratum.csv", index=False)
    l0_table, l0_summaries = l0_by_stratum(eval_frame, admitted, strata)
    l0_table.to_csv(cli.out_dir / "l0_by_health_stratum.csv", index=False)

    minute_counts = []
    for name, allowed in strata.items():
        per_session = {}
        for session in SESSIONS:
            per_session[session] = sum(key[0] == session for key in allowed)
        minute_counts.append({
            "stratum": name,
            "minutes": len(allowed),
            "per_session": per_session,
        })

    fully_healthy = next(
        item for item in probe_summaries if item["stratum"] == "fully_healthy"
    )
    fully_healthy_15m = next(
        item
        for item in probe_summaries
        if item["stratum"] == "fully_healthy_15m_history"
    )
    july15_clean_15m = next(
        item
        for item in probe_summaries
        if item["stratum"]
        == "2026_07_15_raw_atm_aligned_spx_within_0_75_15m_history"
    )
    july22_clean_15m = next(
        item
        for item in probe_summaries
        if item["stratum"] == "2026_07_22_fully_healthy_15m_history"
    )
    fully_healthy_sessions = sorted(
        {
            session
            for session, _ in strata["fully_healthy"]
        }
    )
    routing = {
        "schema_version": "Protocol101EarlySealedFailureAttributionV1",
        "routing_decision": (
            "cannot_stop_collection_with_high_confidence"
            if len(fully_healthy_sessions) < 2
            else "healthy_minute_transfer_requires_owner_review"
        ),
        "reason": (
            "Only one session contributes fully healthy minutes, and its 15-minute "
            "history-clean subset passes only "
            f"{fully_healthy_15m['action_pass_count']}/"
            f"{fully_healthy_15m['non_random_probe_count']} action-agreement gates "
            f"with worst agreement {fully_healthy_15m['worst_action_agreement']:.4f}. "
            "Recorder degradation explains substantial failure, but does not "
            "explain all residual model-transfer instability."
        ),
        "sessions": list(SESSIONS),
        "fully_healthy_sessions": fully_healthy_sessions,
        "fully_healthy_probe_summary": fully_healthy,
        "fully_healthy_15m_history_probe_summary": fully_healthy_15m,
        "july15_raw_15m_history_probe_summary": july15_clean_15m,
        "july22_fully_healthy_15m_history_probe_summary": july22_clean_15m,
        "hill_climbing_recommendation": (
            "do_not_start_governed_hill_climbing_under_current_transfer_contract"
        ),
        "parallel_work_recommendation": (
            "repair_residual_canonical_model_transfer_on_opened_evidence_while_"
            "continuing_scheduled_recorder_collection"
        ),
        "prior_battery_materiality_reporting_issue": (
            "The earlier battery referenced absent column "
            "selected_slot_swap_economically_material; current merged frames use "
            "economically_material. Its reported zero material reorderings is invalid."
        ),
        "minute_counts": minute_counts,
        "probe_summaries": probe_summaries,
        "l0_summaries": l0_summaries,
        "spx_tolerances_reported_points": list(SPX_TOLERANCES),
        "frozen_contract_sha256": battery.FROZEN["contract"][1],
        "no_confirmation_claim": True,
        "side_effects": {
            "model_training_executed": False,
            "threshold_tuning_executed": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "real_money_path_changed": False,
        },
    }
    write_json(cli.out_dir / "routing_decision.json", routing)

    lines = [
        "# Protocol101 Early Sealed Failure Attribution",
        "",
        f"- Routing: `{routing['routing_decision']}`",
        f"- Fully healthy sessions: `{fully_healthy_sessions}`",
        f"- Fully healthy minutes: `{len(strata['fully_healthy'])}`",
        f"- Fully healthy 15-minute-history rows: "
        f"`{len(strata['fully_healthy_15m_history'])}`",
        f"- Degraded minutes: `{len(strata['degraded'])}`",
        f"- Hill-climbing recommendation: `{routing['hill_climbing_recommendation']}`",
        "",
        "## Probe summary by health stratum",
        "",
    ]
    for item in probe_summaries:
        lines.append(
            f"- `{item['stratum']}`: minutes `{item['minutes']}`, "
            f"action passes `{item['action_pass_count']}/{item['non_random_probe_count']}`, "
            f"worst action agreement `{item['worst_action_agreement']:.4f}`, "
            f"slot failures `{item['slot_fail_probes']}`"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- July 15 raw/ATM/SPX-aligned 15-minute-history slice: "
        f"`{july15_clean_15m['action_pass_count']}/"
        f"{july15_clean_15m['non_random_probe_count']}` action gates pass.",
        f"- July 22 fully healthy 15-minute-history slice: "
        f"`{july22_clean_15m['action_pass_count']}/"
        f"{july22_clean_15m['non_random_probe_count']}` action gates pass; "
        f"worst agreement `{july22_clean_15m['worst_action_agreement']:.4f}`.",
        "- The recorder/index failures are causal contributors, but clean-history "
        "filtering does not restore the frozen transfer contract.",
        "- The prior two-day battery's zero material-reordering count is invalid "
        "because it referenced a materiality column absent from the merged frame.",
        "",
        "This is post-open causal attribution, not a sealed confirmation rerun.",
        "No frozen threshold, feature, model, epsilon, or selection rule changed.",
    ])
    (cli.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    progress["status"] = "complete"
    progress["routing_decision"] = routing["routing_decision"]
    write_json(cli.out_dir / "progress.json", progress)
    print(json.dumps({
        "routing_decision": routing["routing_decision"],
        "fully_healthy_sessions": fully_healthy_sessions,
        "fully_healthy_minutes": len(strata["fully_healthy"]),
        "degraded_minutes": len(strata["degraded"]),
    }, indent=2))


if __name__ == "__main__":
    main()
