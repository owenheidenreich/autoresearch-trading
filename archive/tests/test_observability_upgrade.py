from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ingest_evidence import ingest
from tools.monitor import _resolve_local_run_dir
from training.run_loop import (
    SCHEMA_VERSION,
    detect_anomaly_flags,
    load_promoted_history,
    record_promotion_event,
    run_training,
    validate_safety,
    validate_experiment_v2_schema,
)


def test_validate_safety_rejects_feature_group_width_drift(tmp_path, monkeypatch) -> None:
    import training.run_loop as rl

    monkeypatch.setattr(rl, "BEST_TRAIN_PY", str(tmp_path / "missing_best_train.py"))
    code = """
BATCH_SIZE = 128
D_MODEL = 96
N_HEADS = 4
DEPTH = 6
FEATURE_GROUPS = {
    "all": (0, 64),
}
"""
    err = validate_safety(code)
    assert err is not None
    assert "covers 64 features" in err


def test_run_training_parse_fails_when_required_metric_missing(tmp_path) -> None:
    script = tmp_path / "train_missing_metric.py"
    script.write_text(
        "\n".join(
            [
                "print('score: 1.0')",
                "print('profit_factor: 1.2')",
                "print('trades_per_day: 2.3')",
                "print('trade_sharpe: 0.4')",
                "print('stop_loss_rate: 0.1')",
                # missing required key: worst_chunk_pf
            ]
        )
        + "\n"
    )
    out = run_training(str(script), timeout=5, time_budget=1)
    assert out.get("error_type") == "parse"
    assert "worst_chunk_pf" in out.get("error", "")


def test_experiment_v2_schema_validation() -> None:
    valid = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": 1,
        "timestamp": "2026-03-17T10:00:00",
        "train_py_before_hash": "c",
        "train_py_after_hash": "d",
        "failure_type": "none",
        "anomaly_flags": [],
        "kept": True,
        "score": 1.23,
    }
    assert validate_experiment_v2_schema(valid) == []

    bad_failure = dict(valid)
    bad_failure["failure_type"] = "weird"
    assert any(err.startswith("failure_type:") for err in validate_experiment_v2_schema(bad_failure))

    missing_field = dict(valid)
    del missing_field["train_py_before_hash"]
    errs = validate_experiment_v2_schema(missing_field)
    assert "missing:train_py_before_hash" in errs


def test_anomaly_detector_flags_expected_conditions() -> None:
    flags = detect_anomaly_flags(
        {
            "do_nothing_pct": 0.0,
            "exit_pct": 0.995,
            "trades_per_day": 99.0,
            "num_trades": 10,
            "win_rate": 1.2,  # invalid
            "profit_factor": -1.0,  # invalid
            "stop_loss_rate": 2.0,  # invalid
        }
    )
    assert "do_nothing_zero" in flags
    assert "exit_pct_extreme" in flags
    assert "trades_per_day_extreme" in flags
    assert "metric_inconsistent" in flags


def test_anomaly_detector_flags_realism_conditions() -> None:
    flags = detect_anomaly_flags(
        {
            "trades_per_day": 2.0,
            "num_trades": 20,
            "win_rate": 0.55,
            "profit_factor": 1.4,
            "stop_loss_rate": 0.2,
            "cost_realism_coverage": 0.10,
            "avg_entry_quality": 0.10,
            "high_cost_entry_rate": 0.80,
            "low_quality_entry_rate": 0.90,
            "actionable_bar_rate": 0.01,
        }
    )
    assert "cost_realism_low_coverage" in flags
    assert "entry_quality_too_low" in flags
    assert "high_cost_entry_rate_high" in flags
    assert "low_quality_entry_rate_high" in flags
    assert "actionable_bar_rate_low" in flags


def test_run_training_parses_realism_metrics(tmp_path) -> None:
    script = tmp_path / "train_realism_metrics.py"
    script.write_text(
        "\n".join(
            [
                "print('score: 1.0')",
                "print('profit_factor: 1.2')",
                "print('trades_per_day: 2.3')",
                "print('trade_sharpe: 0.4')",
                "print('stop_loss_rate: 0.1')",
                "print('worst_chunk_pf: 1.1')",
                "print('avg_entry_cost_bps: 150.0')",
                "print('avg_entry_quality: 0.6')",
                "print('cost_realism_coverage: 0.8')",
                "print('high_cost_entry_rate: 0.2')",
                "print('low_quality_entry_rate: 0.1')",
                "print('actionable_bar_rate: 0.4')",
                "print('risk_off_bar_rate: 0.05')",
            ]
        )
        + "\n"
    )
    out = run_training(str(script), timeout=5, time_budget=1)
    assert out.get("error") is None
    assert out.get("avg_entry_cost_bps") == pytest.approx(150.0)
    assert out.get("avg_entry_quality") == pytest.approx(0.6)
    assert out.get("cost_realism_coverage") == pytest.approx(0.8)
    assert out.get("actionable_bar_rate") == pytest.approx(0.4)
    assert out.get("risk_off_bar_rate") == pytest.approx(0.05)


def test_results_dir_is_repo_root_results() -> None:
    import training.run_loop as rl

    # In local repo execution, run_loop lives at <repo>/training/run_loop.py.
    assert Path(rl.RESULTS_DIR).name == "results"
    assert Path(rl.RESULTS_DIR).parent == REPO_ROOT


def test_promotion_ledger_roundtrip(tmp_path, monkeypatch) -> None:
    import training.run_loop as rl

    promoted_dir = tmp_path / "results" / "promoted"
    promoted_history = promoted_dir / "history.jsonl"

    monkeypatch.setattr(rl, "PROMOTED_DIR", str(promoted_dir))
    monkeypatch.setattr(rl, "PROMOTED_HISTORY_JSONL", str(promoted_history))
    monkeypatch.setattr(rl, "BEST_MODEL_PT", str(tmp_path / "best_model.pt"))
    monkeypatch.setattr(rl, "BEST_TRAIN_PY", str(tmp_path / "best_train.py"))

    exp = {
        "run_name": "run-2026-03-18-010203",
        "experiment_id": 7,
        "score": 1.234,
        "model_fingerprint_after": "mh",
        "train_py_after_hash": "th",
        "profit_factor": 1.9,
        "trades_per_day": 2.1,
        "trade_sharpe": 1.4,
        "change_summary": "loss tweak",
    }
    event = record_promotion_event(exp)

    assert event["run_name"] == "run-2026-03-18-010203"
    assert event["experiment_id"] == 7
    assert promoted_history.exists()

    hist = load_promoted_history()
    assert len(hist) == 1
    assert hist[0]["score"] == pytest.approx(1.234)


def test_ingest_evidence_integration(tmp_path) -> None:
    results_root = tmp_path / "results"
    run_dir = results_root / "run-2026-03-17"
    run_dir.mkdir(parents=True, exist_ok=True)
    live_dir = results_root / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = results_root / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)

    exp_v2 = run_dir / "experiments.v2.jsonl"
    exp_v2.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": 2,
                        "experiment_id": 1,
                        "timestamp": "2026-03-17T09:30:00",
                        "prompt_fingerprint": "a",
                        "program_md_fingerprint": "b",
                        "train_py_before_hash": "c",
                        "train_py_after_hash": "d",
                        "data_fingerprint": "e",
                        "model_fingerprint_before": "f",
                        "model_fingerprint_after": "g",
                        "failure_type": "none",
                        "anomaly_flags": [],
                        "kept": True,
                        "score": 1.5,
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 2,
                        "experiment_id": 2,
                        "timestamp": "2026-03-17T09:45:00",
                        "prompt_fingerprint": "a2",
                        "program_md_fingerprint": "b2",
                        "train_py_before_hash": "c2",
                        "train_py_after_hash": "d2",
                        "data_fingerprint": "e2",
                        "model_fingerprint_before": "f2",
                        "model_fingerprint_after": "g2",
                        "failure_type": "regression",
                        "anomaly_flags": ["do_nothing_zero"],
                        "kept": False,
                        "score": 1.49,
                    }
                ),
            ]
        )
        + "\n"
    )

    (live_dir / "audit.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-03-17T10:00:00",
                "event": "entitlement_probe",
                "payload": {"passed": False},
            }
        )
        + "\n"
    )

    (replay_dir / "sample_journal.json").write_text(
        json.dumps(
            {
                "replay_date": "2026-03-17",
                "trades": [],
                "session_stats": {"total_bars": 390, "avg_gate_prob": 0.1},
                "bar_log": [],
            }
        )
    )

    output_root = results_root / "analysis"
    (run_dir / "loop.log").write_text("[09:50:00] FAILED: parse error in metrics\n")
    summary = ingest(results_root=results_root, output_root=output_root)

    assert summary["events"] > 0
    assert Path(summary["evidence_path"]).exists()
    assert Path(summary["incidents_path"]).exists()
    assert Path(summary["digest_path"]).exists()

    evidence = pd.read_parquet(summary["evidence_path"])
    assert set(["source", "subsystem", "event_type", "signature"]).issubset(set(evidence.columns))

    incidents = [json.loads(line) for line in Path(summary["incidents_path"]).read_text().splitlines() if line.strip()]
    sigs = {row["signature"] for row in incidents}
    assert "exp.failure_type:regression" in sigs
    assert "live.entitlement_probe.issue" in sigs
    assert "loop.log.failure.issue" in sigs
    for row in incidents:
        assert set(["id", "subsystem", "severity", "signature", "first_seen", "last_seen", "count", "latest_ref"]).issubset(
            set(row.keys())
        )


def test_ingest_ignores_legacy_experiments_jsonl(tmp_path) -> None:
    results_root = tmp_path / "results"
    run_dir = results_root / "run-2026-03-17-111111"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiments.jsonl").write_text(
        json.dumps({"experiment_id": 1, "score": 1.0}) + "\n"
    )

    out = results_root / "analysis"
    summary = ingest(results_root=results_root, output_root=out)
    assert summary["events"] == 0


def test_ingest_replay_csv_filter_ignores_training_trade_log(tmp_path) -> None:
    results_root = tmp_path / "results"
    replay_dir = results_root / "analysis" / "nightly-replay"
    run_dir = results_root / "run-2026-03-17-111111"
    replay_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    (replay_dir / "replay-2026-03-17.csv").write_text("a,b\n1,2\n")
    (run_dir / "trade_log.csv").write_text("a,b\n3,4\n")

    out = results_root / "analysis-out"
    summary = ingest(results_root=results_root, output_root=out)
    evidence = pd.read_parquet(summary["evidence_path"])
    replay_rows = evidence[evidence["event_type"] == "replay_trade_log"]
    assert len(replay_rows) == 1
    assert "nightly-replay" in replay_rows.iloc[0]["ref_path"]


def test_ingest_replay_qa_artifacts(tmp_path) -> None:
    results_root = tmp_path / "results"
    replay_dir = results_root / "analysis" / "nightly-replay"
    replay_dir.mkdir(parents=True, exist_ok=True)

    (replay_dir / "replay-2026-03-17_qa.json").write_text(
        json.dumps(
            {
                "schema_version": "replay_qa_v1",
                "passed": False,
                "critical_count": 1,
                "warning_count": 0,
                "anomalies": [{"severity": "critical", "code": "bad_trade_time_format"}],
            }
        )
    )
    (replay_dir / "replay-2026-03-17_ledger_days.csv").write_text(
        "replay_date,num_trades,total_bars,total_pnl_pct,qa_passed,qa_critical_count,qa_warning_count\n"
        "2026-03-17,2,390,1.2,false,1,0\n"
    )

    out = results_root / "analysis-out"
    summary = ingest(results_root=results_root, output_root=out)
    evidence = pd.read_parquet(summary["evidence_path"])
    qa_rows = evidence[evidence["event_type"] == "replay_qa"]
    assert len(qa_rows) == 1

    incidents = [json.loads(line) for line in Path(summary["incidents_path"]).read_text().splitlines() if line.strip()]
    sigs = {row["signature"] for row in incidents}
    assert "replay.qa_failed" in sigs
    assert "replay.qa.bad_trade_time_format" in sigs
    assert "replay.day_qa_failed" in sigs


def test_monitor_resolves_run_dir_from_current_run_pointer(tmp_path) -> None:
    results_root = tmp_path / "results"
    run_name = "run-2026-03-17-101010"
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (results_root / "current_run.txt").write_text(run_name + "\n")

    resolved = _resolve_local_run_dir(str(results_root), None)
    assert resolved == run_dir


def test_monitor_pointer_target_missing_returns_none(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "current_run.txt").write_text("run-2026-03-17-111111\n")

    resolved = _resolve_local_run_dir(str(results_root), None)
    assert resolved is None
