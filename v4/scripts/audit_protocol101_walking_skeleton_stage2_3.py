"""Independent delta-scoped audit for the quarantined Stage 2–3 packet."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from v4.model.protocol101_walking_skeleton_lifecycle import (
    FEATURE_NAMES,
    QUARANTINE_LABELS,
    QUARANTINE_TEXT,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage2_3"
REVIEW = OUT / "delta_scoped_review.json"
AUTHORITY_SHA256 = "82d9573e120d6395825aa8a5f2d66fdac9bf32d825190737876b204dd112e2f2"
GRAPH_SHA256 = "9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text())


def main(*, browser_render_verified: bool = False) -> int:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "pass": bool(passed), "evidence": evidence})

    receipt = load_json("receipt.json")
    manifest = load_json("forced_buy_manifest.json")
    serial = load_json("serial_replay.json")
    safety = load_json("entry_safety_audit.json")
    model_manifest = load_json("lifecycle_model_manifest.json")
    floor = load_json("protective_floor_artifact.json")
    floor_ablation = load_json("floor_ablation_report.json")
    early = load_json("d58_early_drawdown_report_only.json")
    four = load_json("four_bucket_distribution.json")

    authority = ROOT / receipt["authority_path"]
    graph = ROOT / receipt["graph_path"]
    check("authority_hash_exact", sha256_path(authority) == AUTHORITY_SHA256, sha256_path(authority))
    check("graph_hash_exact", sha256_path(graph) == GRAPH_SHA256, sha256_path(graph))
    check(
        "stage1_real_entry_still_abstains",
        receipt["stage1_real_entry_outcome"] == "ABSTAINS",
        receipt["stage1_real_entry_outcome"],
    )
    check(
        "frozen_entry_composer_gate_contract_bytes_unchanged",
        manifest.get("frozen_bytes_unchanged") is True
        and manifest.get("frozen_before") == manifest.get("frozen_after")
        and all(
            sha256_path(ROOT / path) == digest
            for path, digest in manifest["frozen_after"].items()
        ),
        manifest.get("frozen_after"),
    )

    artifact_results: dict[str, Any] = {}
    for relative, expected in receipt["artifacts"].items():
        path = ROOT / relative
        artifact_results[relative] = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_path(path) if path.is_file() else None,
            "expected": expected,
        }
    check(
        "receipt_artifact_hashes_and_sizes_exact",
        all(
            result["exists"]
            and result["bytes"] == result["expected"]["bytes"]
            and result["sha256"] == result["expected"]["sha256"]
            for result in artifact_results.values()
        ),
        artifact_results,
    )

    intents = manifest["forced_intents"]
    per_session = Counter(item["session"] for item in intents)
    replay_intents = [item for item in intents if item["split"] == "plumbing_replay"]
    check("forced_intents_exactly_six_per_13_sessions", len(intents) == 78 and len(per_session) == 13 and set(per_session.values()) == {6}, dict(per_session))
    check("real_replay_A6_proposal_count_exact", manifest["real_replay_a6_proposal_count"] == 552, manifest["real_replay_a6_proposal_count"])
    check(
        "override_is_downstream_only",
        all(
            item["composer_action"] == "WAIT"
            and item["override_boundary"] == "downstream_confidence_abstention_only"
            and item["composer_proposed_contract_id"] == item["contract_id"]
            for item in intents
        ),
        Counter(item["composer_wait_reason"] for item in intents),
    )
    check(
        "all_payload_intents_have_four_quarantine_labels",
        all(tuple(item["quarantine_labels"]) == QUARANTINE_LABELS for item in intents),
        list(QUARANTINE_LABELS),
    )

    exit_tensor_path = next(OUT.glob("*forced_buy_plumbing_harness_exit_tensor.parquet"))
    tensor = pd.read_parquet(
        exit_tensor_path,
        columns=[
            "session",
            "split",
            "contract_id",
            "decision_time_ns",
            "sample_time_ns",
            "source_event_time_ns",
            "current_bid",
            "entry_ask",
            "exit_target",
            "exit_probability",
            "quarantine_labels",
        ],
    )
    check("exit_tensor_uses_exact_13_sessions", tensor["session"].nunique() == 13 and len(tensor) == receipt["exit_tensor_rows"], {"rows": len(tensor), "sessions": tensor["session"].nunique()})
    check("exit_tensor_quarantine_label_column_exact", set(tensor["quarantine_labels"].unique()) == {QUARANTINE_TEXT}, tensor["quarantine_labels"].unique().tolist())
    check("exit_tensor_has_valid_executable_prices", bool((tensor["entry_ask"] > 0).all() and (tensor["current_bid"] >= 0).all()), {"minimum_entry_ask": float(tensor["entry_ask"].min()), "minimum_bid": float(tensor["current_bid"].min())})
    check("exit_tensor_has_both_HOLD_and_EXIT_targets", set(tensor["exit_target"].unique()) == {0, 1}, tensor["exit_target"].value_counts().to_dict())
    check("exit_tensor_probabilities_finite_and_bounded", bool(tensor["exit_probability"].between(0.0, 1.0).all()), {"min": float(tensor["exit_probability"].min()), "max": float(tensor["exit_probability"].max())})

    payload = joblib.load(OUT / Path(model_manifest["model_artifact"]).name)
    forbidden = {"exit_target", "future_max_bid_300s_label", "future_min_bid_300s_label"}
    check("model_artifact_is_quarantined_HGB", payload["model"].__class__.__name__ == "HistGradientBoostingClassifier" and tuple(payload["quarantine_labels"]) == QUARANTINE_LABELS, payload["model"].__class__.__name__)
    check("runtime_features_exclude_future_labels", set(payload["feature_names"]) == set(FEATURE_NAMES) and not set(payload["feature_names"]) & forbidden, payload["feature_names"])
    check("floor_order_matches_authority", floor["ordering"] == ["check_floor_committed_at_t_minus_1_at_current_executable_bid", "evaluate_lifecycle_model_if_floor_not_crossed", "after_HOLD_compute_upward_only_floor_for_t_plus_1"], floor["ordering"])
    check(
        "D52_learned_exit_without_floor_comparator_present",
        floor_ablation["same_frozen_entries"] is True
        and floor_ablation["same_model_and_threshold"] is True
        and len(floor_ablation["rows"]) == len(intents)
        and floor_ablation["all_13_sessions"]["floor_on"]["count"] == len(intents)
        and floor_ablation["all_13_sessions"]["floor_off"]["count"] == len(intents),
        {
            "all_13_sessions": floor_ablation["all_13_sessions"],
            "owned_validation_slice": floor_ablation["owned_validation_slice"],
        },
    )
    check(
        "D52_harvest_tripwire_not_overclaimed_or_tuned",
        floor_ablation["harvest_tripwire_disposition"].startswith("REPORT_ONLY_NO_THRESHOLD_PREREGISTERED")
        and floor_ablation["formal_model_quality_or_alpha_claim"] is False,
        floor_ablation["harvest_tripwire_disposition"],
    )

    action_path = next(OUT.glob("*forced_buy_plumbing_harness_lifecycle_actions.parquet"))
    actions = pd.read_parquet(action_path)
    terminal_actions = actions[actions["action"] == "EXIT"]
    trigger_counts_all = terminal_actions["trigger"].value_counts().to_dict()
    hold_count = int((actions["action"] == "HOLD").sum())
    monotone = True
    for _, group in actions[actions["action"] == "HOLD"].groupby(["session", "decision_time_ns", "contract_id"]):
        values = pd.to_numeric(group["floor_committed_for_next_second"], errors="coerce").dropna()
        monotone &= bool((values.diff().dropna() >= -1e-12).all())
    check("HOLD_action_fired", hold_count > 0, hold_count)
    check("all_terminal_paths_fired_across_13_sessions", {"learned_exit", "floor_trigger", "forced_flat"}.issubset(trigger_counts_all), trigger_counts_all)
    check("protective_floor_never_decreases", monotone, monotone)

    trade_rows = serial["trades"]
    replay_triggers = Counter(row["metadata"]["terminal_trigger"] for row in trade_rows)
    check("all_terminal_paths_reach_serial_simulator_v5", {"learned_exit", "floor_trigger", "forced_flat"}.issubset(replay_triggers), dict(replay_triggers))
    check("serial_simulator_v5_zero_internal_skips", not any(serial["skipped"].values()), serial["skipped"])
    pnl_exact = all(
        math.isclose(
            float(row["raw_label_pnl_after_campaign_fee"]),
            (float(row["label_executable_exit_bid"]) - float(row["entry_ask"])) * 100.0 - 3.0,
            abs_tol=1e-9,
            rel_tol=0.0,
        )
        for row in trade_rows
    )
    check("ask_entry_bid_exit_fee_PnL_exact", pnl_exact, serial["estimated_pnl_dollars"])
    accepted_by_identity = {
        (row["session"], int(row["decision_time_ns"]), row["contract_id"]): row
        for row in safety["accepted"]
    }
    exit_provenance_results: list[dict[str, Any]] = []
    for row in trade_rows:
        identity = (row["session"], int(row["decision_time_ns"]), row["contract_id"])
        source = accepted_by_identity[identity]
        simulator_age_ms = float(row["label_exit_quote_age_ms"])
        metadata = row["metadata"]
        recorded_event_time_ns = int(metadata["exit_source_event_time_ns"])
        recorded_age_ms = float(metadata["exit_market_quote_age_ms"])
        expected_age_ms = float(source["exit_market_quote_age_ms"])
        source_time_exact = recorded_event_time_ns == int(
            source["exit_source_event_time_ns"]
        )
        age_exact = math.isclose(recorded_age_ms, expected_age_ms, abs_tol=1e-9, rel_tol=0.0)
        clock_arithmetic_exact = math.isclose(
            (int(row["label_realized_exit_time_ns"]) - recorded_event_time_ns)
            / 1_000_000.0,
            recorded_age_ms,
            abs_tol=1e-6,
            rel_tol=0.0,
        )
        simulator_v5_clock_contract_exact = (
            int(row["label_source_exit_quote_time_ns"])
            == int(row["label_realized_exit_time_ns"])
            and math.isclose(simulator_age_ms, 0.0, abs_tol=1e-12, rel_tol=0.0)
        )
        exit_provenance_results.append(
            {
                "identity": identity,
                "source_time_exact": source_time_exact,
                "age_exact": age_exact,
                "clock_arithmetic_exact": clock_arithmetic_exact,
                "simulator_v5_clock_contract_exact": simulator_v5_clock_contract_exact,
                "market_quote_age_ms": recorded_age_ms,
                "fresh_under_90000ms_contract": recorded_age_ms <= 90_000.0,
            }
        )
    check(
        "serial_exit_two_clock_provenance_exact_and_fresh",
        all(
            item["source_time_exact"]
            and item["age_exact"]
            and item["clock_arithmetic_exact"]
            and item["simulator_v5_clock_contract_exact"]
            and item["fresh_under_90000ms_contract"]
            for item in exit_provenance_results
        ),
        exit_provenance_results,
    )
    check("D48_D49_all_accepted_pass", safety["summary"]["all_accepted_d48_pass"] and safety["summary"]["all_accepted_d49_pass"], safety["summary"])
    check("one_to_six_trades_each_replay_session", safety["summary"]["all_sessions_one_to_six"] and len(safety["summary"]["sessions"]) == 3, safety["summary"]["sessions"])
    check("one_open_position_enforced", all(row["entry_safety_reason"] == "one_open_position_overlap" or row["entry_safety_reason"].startswith("d49_") or row["entry_safety_reason"] in {"six_trade_session_cap", "d48_premium_cap_mask", "unaffordable", "at_or_after_15_30_entry_cutoff"} for row in safety["rejected"]), Counter(row["entry_safety_reason"] for row in safety["rejected"]))
    accepted_before_cutoff = all(
        (timestamp.hour, timestamp.minute) < (15, 30)
        for timestamp in (
            pd.Timestamp(int(row["decision_time_ns"]), unit="ns", tz="UTC").tz_convert(
                "America/New_York"
            )
            for row in safety["accepted"]
        )
    )
    check(
        "accepted_entry_decisions_are_strictly_before_15_30",
        accepted_before_cutoff,
        [row["decision_time_ns"] for row in safety["accepted"]],
    )

    check("D58_early_drawdown_is_report_only", early["status"] == "REPORT_ONLY" and early["formal_entry_ranking_or_alpha_claim"] is False, {key: early[key] for key in ("sample_count", "spearman_rank_correlation", "worst_quartile_overlap_fraction")})
    check("four_bucket_distribution_covers_every_trade", sum(four["counts"].values()) == len(trade_rows) and four["trade_count"] == len(trade_rows), four["counts"])

    html_results: dict[str, Any] = {}
    trade_chart_payload: dict[str, Any] | None = None
    for name in ("equity.html", "trades-on-spx.html", "trades.html", "four_bucket_distribution.html"):
        text = (OUT / name).read_text()
        external_script = "<script src=\"http" in text or "<script src='http" in text
        if name == "trades-on-spx.html":
            structural = "<canvas" in text and '"spx":' in text and all(row["contract_id"] in text for row in trade_rows)
            payload_text = text.split(
                '<script id="payload" type="application/json">', 1
            )[1].split("</script>", 1)[0]
            trade_chart_payload = json.loads(payload_text)
        elif name == "equity.html":
            structural = "<canvas" in text and '"trades":' in text and all(row["contract_id"] in text for row in trade_rows)
        elif name == "four_bucket_distribution.html":
            structural = all(bucket in text for bucket in four["counts"])
        else:
            structural = "<canvas" in text and '"trades":' in text
        html_results[name] = {
            "bytes": len(text.encode()),
            "quarantine_labels_present": QUARANTINE_TEXT in text,
            "self_contained_no_external_script": not external_script,
            "required_payload_and_structure_present": structural,
        }
    check("D60_HTML_payloads_structurally_complete", all(all((value["quarantine_labels_present"], value["self_contained_no_external_script"], value["required_payload_and_structure_present"])) for value in html_results.values()), html_results)
    chart_spx = (trade_chart_payload or {}).get("spx", [])
    chart_trades = (trade_chart_payload or {}).get("trades", [])
    dense_bars = [int(row["bar"]) for row in chart_spx]
    check(
        "D60_trade_chart_dense_coordinates_and_three_session_span",
        dense_bars == list(range(len(chart_spx)))
        and {row["session"] for row in chart_spx}
        == {"2025-03-06", "2025-03-07", "2025-03-10"}
        and all(
            0 <= float(row["entry_bar"]) <= float(row["exit_bar"]) < len(chart_spx)
            for row in chart_trades
        ),
        {
            "bar_count": len(chart_spx),
            "first_bar": dense_bars[0] if dense_bars else None,
            "last_bar": dense_bars[-1] if dense_bars else None,
            "sessions": sorted({row["session"] for row in chart_spx}),
            "trade_count": len(chart_trades),
        },
    )

    runner_text = (ROOT / "v4/scripts/run_protocol101_walking_skeleton_stage2_3.py").read_text()
    forbidden_imports = [line for line in runner_text.splitlines() if line.startswith(("import databento", "from databento", "import ib_insync", "from ib_insync", "import requests", "from requests"))]
    check("runner_has_no_broker_download_or_network_imports", not forbidden_imports, forbidden_imports)
    check("no_Stage4_side_effect_or_start", receipt["stage4_started"] is False and receipt["broker_contacted"] is False and receipt["download_or_purchase_performed"] is False and receipt["runtime_promotion_or_default_changed"] is False, {key: receipt[key] for key in ("stage4_started", "broker_contacted", "download_or_purchase_performed", "runtime_promotion_or_default_changed")})
    check("highest_claim_is_exactly_bounded", receipt["highest_allowed_claim"] == "Downstream plumbing was exercised end-to-end on a quarantined forced-BUY harness; the real entry composer remains frozen and still abstains.", receipt["highest_allowed_claim"])
    check("producer_stop_line_exact", receipt["next_action"] == "STOP_FOR_CLAUDE_VERIFICATION", receipt["next_action"])

    failed = [row for row in checks if not row["pass"]]
    review = {
        "schema_version": "Protocol101WalkingSkeletonStages2And3DeltaScopedReviewV1",
        "outcome": "PASS" if not failed else "FAIL",
        "scope": [
            "v4/model/protocol101_walking_skeleton_lifecycle.py",
            "v4/scripts/run_protocol101_walking_skeleton_stage2_3.py",
            "v4/scripts/audit_protocol101_walking_skeleton_stage2_3.py",
            "v4/tests/test_protocol101_walking_skeleton_stage2_3.py",
            "v4/scripts/export_protocol101_trade_charts.py",
            "v4/tests/test_protocol113_trade_charts.py",
            "v4/audit/autoresearch/protocol101_walking_skeleton_stage2_3/",
            "append-only v4/ledger/RESEARCH_LEDGER.md entry",
            "v4/docs/protocol101/training/execution/PROTOCOL101_WALKING_SKELETON_LEARNINGS_LEDGER_2026_07_30.md",
        ],
        "checks_passed": len(checks) - len(failed),
        "checks_failed": len(failed),
        "checks": checks,
        "render_verification": {
            "status": (
                "PASS_LOCAL_HTTP_BROWSER_VISUAL_QA"
                if browser_render_verified
                else "STRUCTURAL_PAYLOAD_PASS_BROWSER_VISUAL_QA_NOT_ATTESTED"
            ),
            "browser_policy_respected": True,
            "browser_render_verified": browser_render_verified,
            "verified_pages": (
                ["equity.html", "trades-on-spx.html", "four_bucket_distribution.html"]
                if browser_render_verified
                else []
            ),
            "verified_trade_chart_date_span": (
                "2025-03-06 through 2025-03-10 across three sessions"
                if browser_render_verified
                else None
            ),
            "overclaim_of_rendered_visual_QA": False,
            "html_results": html_results,
        },
        "highest_allowed_claim": receipt["highest_allowed_claim"],
        "next_action": (
            "STOP_BEFORE_STAGE4_AWAIT_OWNER_RECORDER_AND_IBKR_PAPER_AUTHORIZATIONS"
            if not failed and browser_render_verified
            else "STOP_FOR_CLAUDE_VERIFICATION"
        ),
        "quarantine_labels": list(QUARANTINE_LABELS),
    }
    REVIEW.write_text(json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"outcome": review["outcome"], "checks_passed": review["checks_passed"], "checks_failed": review["checks_failed"], "failed": [row["name"] for row in failed]}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--browser-render-verified",
        action="store_true",
        help="attest only after separately viewing the three owner-facing HTML pages",
    )
    arguments = parser.parse_args()
    raise SystemExit(main(browser_render_verified=arguments.browser_render_verified))
