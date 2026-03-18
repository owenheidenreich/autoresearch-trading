from __future__ import annotations

import json
import time
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
    _initialize_run_layout,
    detect_anomaly_flags,
    load_promoted_history,
    prompt_contract_check,
    record_promotion_event,
    validate_experiment_v2_schema,
    write_status,
)


def _contract_text(include_two_head: bool = True, include_worst_chunk: bool = True) -> str:
    bits = [
        "Direction head: [CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10]",
        "8 effective actions",
        "Current stabilization phase locked to 60 features",
        "score:",
        "profit_factor:",
        "trades_per_day:",
        "trade_sharpe:",
        "stop_loss_rate:",
    ]
    if include_two_head:
        bits.insert(0, "two-head architecture")
    if include_worst_chunk:
        bits.append("worst_chunk_pf:")
    return "\n".join(bits)


def test_prompt_contract_check_catches_stale_action_text() -> None:
    system = "Loop system prompt"
    good = prompt_contract_check(system, _contract_text())
    assert good["ok"] is True

    bad = prompt_contract_check(system, _contract_text() + "\n4 effective actions")
    assert bad["ok"] is False
    rules = {v["rule"] for v in bad["violations"]}
    assert "stale_four_action_semantics" in rules


def test_prompt_contract_check_catches_missing_two_head_and_metrics() -> None:
    system = "Loop system prompt"
    missing_two_head = prompt_contract_check(system, _contract_text(include_two_head=False))
    assert missing_two_head["ok"] is False
    rules = {v["rule"] for v in missing_two_head["violations"]}
    assert "two_head_output_contract" in rules

    missing_metric = prompt_contract_check(system, _contract_text(include_worst_chunk=False))
    assert missing_metric["ok"] is False
    rules = {v["rule"] for v in missing_metric["violations"]}
    assert "missing_metric_key_worst_chunk_pf" in rules


def test_experiment_v2_schema_validation() -> None:
    valid = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": 1,
        "timestamp": "2026-03-17T10:00:00",
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
        "score": 1.23,
    }
    assert validate_experiment_v2_schema(valid) == []

    bad_failure = dict(valid)
    bad_failure["failure_type"] = "weird"
    assert any(err.startswith("failure_type:") for err in validate_experiment_v2_schema(bad_failure))

    missing_field = dict(valid)
    del missing_field["prompt_fingerprint"]
    errs = validate_experiment_v2_schema(missing_field)
    assert "missing:prompt_fingerprint" in errs


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


def test_write_status_backward_compatible_with_new_reliability_fields(tmp_path, monkeypatch) -> None:
    import training.run_loop as rl

    status_path = tmp_path / "status.json"
    monkeypatch.setattr(rl, "STATUS_JSON", str(status_path))

    deadline = time.time() + 3600
    write_status(
        "between_experiments",
        experiment_id=7,
        best_score=1.2345,
        kept=2,
        failed=1,
        total=9,
        deadline=deadline,
        last_exp={"failure_type": "regression", "anomaly_flags": ["exit_pct_extreme"], "score": 1.0},
        contract_checksum="abcdef1234567890",
    )
    status = json.loads(status_path.read_text())
    # Legacy fields must remain for deploy.sh parser.
    assert status["kept"] == 2
    assert status["failed"] == 1
    assert status["total"] == 9
    assert status["phase"] == "between_experiments"
    assert "best_score" in status
    # New reliability indicators.
    assert status["contract_checksum"] == "abcdef123456"
    assert status["last_failure_type"] == "regression"
    assert status["last_anomaly_flags"] == ["exit_pct_extreme"]


def test_initialize_run_layout_writes_current_run_pointer(tmp_path, monkeypatch) -> None:
    import training.run_loop as rl

    run_name = "run-2026-03-17-123456"
    results_dir = tmp_path / "results"
    run_dir = results_dir / run_name
    artifacts_dir = run_dir / "artifacts"
    pointer = results_dir / "current_run.txt"
    promoted_dir = results_dir / "promoted"

    monkeypatch.setattr(rl, "RUN_NAME", run_name)
    monkeypatch.setattr(rl, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(rl, "RUN_DIR", str(run_dir))
    monkeypatch.setattr(rl, "ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setattr(rl, "CURRENT_RUN_TXT", str(pointer))
    monkeypatch.setattr(rl, "PROMOTED_DIR", str(promoted_dir))

    _initialize_run_layout()

    assert run_dir.exists()
    assert artifacts_dir.exists()
    assert pointer.exists()
    assert pointer.read_text().strip() == run_name
    assert promoted_dir.exists()


def test_results_dir_is_repo_root_results() -> None:
    import training.run_loop as rl

    # In local repo execution, run_loop lives at <repo>/training/run_loop.py.
    assert Path(rl.RESULTS_DIR).name == "results"
    assert Path(rl.RESULTS_DIR).parent == REPO_ROOT


def test_promotion_ledger_roundtrip(tmp_path, monkeypatch) -> None:
    import training.run_loop as rl

    promoted_dir = tmp_path / "results" / "promoted"
    promoted_current = promoted_dir / "current.txt"
    promoted_history = promoted_dir / "history.jsonl"

    monkeypatch.setattr(rl, "RUN_NAME", "run-2026-03-18-010203")
    monkeypatch.setattr(rl, "PROMOTED_DIR", str(promoted_dir))
    monkeypatch.setattr(rl, "PROMOTED_CURRENT_TXT", str(promoted_current))
    monkeypatch.setattr(rl, "PROMOTED_HISTORY_JSONL", str(promoted_history))
    monkeypatch.setattr(rl, "BEST_MODEL_PT", str(tmp_path / "best_model.pt"))
    monkeypatch.setattr(rl, "BEST_TRAIN_PY", str(tmp_path / "best_train.py"))

    exp = {
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
    assert promoted_current.read_text().strip() == "run-2026-03-18-010203"
    assert promoted_history.exists()

    prompt_hist = load_promoted_history()
    assert len(prompt_hist) == 1
    assert prompt_hist[0]["kept"] is True
    assert prompt_hist[0]["score"] == pytest.approx(1.234)


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


def test_monitor_resolves_run_dir_from_current_run_pointer(tmp_path) -> None:
    results_root = tmp_path / "results"
    run_name = "run-2026-03-17-101010"
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (results_root / "current_run.txt").write_text(run_name + "\n")

    resolved = _resolve_local_run_dir(str(results_root))
    assert resolved == str(run_dir)


def test_monitor_pointer_target_missing_fails_fast(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "current_run.txt").write_text("run-2026-03-17-111111\n")

    with pytest.raises(RuntimeError):
        _resolve_local_run_dir(str(results_root))
