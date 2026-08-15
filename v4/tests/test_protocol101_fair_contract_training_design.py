"""Tests for the Protocol101 fair-contract training design packet."""
from __future__ import annotations

from pathlib import Path

from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
from v4.scripts.run_protocol101_fair_contract_training_design import (
    REQUIRED_EXPANDING_FOLD_COUNT,
    build_packet,
    build_split_policy,
)


def test_split_policy_excludes_protected_holdout_sessions() -> None:
    sessions = [
        "2025-05-15",
        "2025-05-16",
        "2025-06-30",
        "2026-02-27",
        "2026-03-02",
        "2026-03-03",
        "2026-03-04",
    ]

    split = build_split_policy(
        sessions,
        protected_sessions=["2025-05-16", "2025-06-30"],
    )

    assert "2025-05-15" in split["train_sessions"]
    assert "2025-05-16" not in split["train_sessions"]
    assert "2025-06-30" not in split["train_sessions"]
    split_sessions = (
        set(split["train_sessions"])
        | set(split["validation_sessions"])
        | set(split["diagnostic_test_sessions"])
    )
    assert "2025-05-16" not in split_sessions
    assert "2025-06-30" not in split_sessions
    assert split["protected_holdout_status"] == (
        "excluded_from_train_validation_diagnostic_splits"
    )
    assert {row["session"] for row in split["excluded_from_splits"]} == {
        "2025-05-16",
        "2025-06-30",
    }


def test_split_policy_builds_five_expanding_window_folds() -> None:
    sessions = [f"2025-01-{day:02d}" for day in range(1, 31)]

    split = build_split_policy(sessions)

    assert split["schema_version"] == "Protocol101ChronologicalExpandingWindowSplitPolicyV2"
    assert split["required_expanding_window_cv"] is True
    assert split["fold_count"] == REQUIRED_EXPANDING_FOLD_COUNT
    assert len(split["folds"]) == REQUIRED_EXPANDING_FOLD_COUNT
    previous_train_count = 0
    validation_sessions = set()
    for fold in split["folds"]:
        assert len(fold["embargoed_sessions"]) == 1
        assert fold["train_sessions"] == sorted(fold["train_sessions"])
        assert fold["validation_sessions"] == sorted(fold["validation_sessions"])
        assert max(fold["train_sessions"]) < min(fold["validation_sessions"])
        assert len(fold["train_sessions"]) > previous_train_count
        previous_train_count = len(fold["train_sessions"])
        assert not (set(fold["validation_sessions"]) & validation_sessions)
        validation_sessions.update(fold["validation_sessions"])


def test_design_packet_records_protected_holdout_exclusion() -> None:
    preflight = {
        "status": "ready_for_owner_authorized_fair_contract_training_design",
        "selected_feature_contract": FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
    }
    manifest = {
        "included_sessions": [
            {
                "session": "2025-05-15",
                "processed_file": "/tmp/protocol101/2025-05-15.pkl",
            },
            {
                "session": "2025-05-16",
                "processed_file": "/tmp/protocol101/2025-05-16.pkl",
            },
            {
                "session": "2026-03-03",
                "processed_file": "/tmp/protocol101/2026-03-03.pkl",
            },
            {
                "session": "2026-03-04",
                "processed_file": "/tmp/protocol101/2026-03-04.pkl",
            },
        ],
    }
    protected_holdout = {
        "status": "pass",
        "protected_holdout_hash": "unit-test-hash",
        "sessions": ["2025-05-16"],
    }

    packet = build_packet(
        preflight,
        manifest,
        manifest_path=Path("/tmp/protocol101/manifest.json"),
        protected_holdout=protected_holdout,
    )

    assert "2025-05-16" not in packet.split_policy["train_sessions"]
    assert packet.training_target["label_menu"] == (
        "protocol101_trade_shape_menu_v2_all_seven_shapes"
    )
    assert len(packet.training_target["labels"]) == 7
    assert "ask_to_bid_stop100_target9900_hold384m" in packet.training_target["labels"]
    assert packet.excluded_data[1]["source"] == "protected_holdout_artifact"
    assert packet.excluded_data[1]["sessions"] == ["2025-05-16"]
    assert packet.model_training_authorized is False
