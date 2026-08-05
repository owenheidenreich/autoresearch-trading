from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from v4.research.pathd_phase1_replay import Phase1ReplayError, run_four_box_replay
from v4.research.phase1_exit_model import stable_hash
from v4.research.phase1_exit_model import sensitivity_label_path


def _campaign(path: Path, payload: dict) -> Path:
    payload = {**payload, "protected_holdout_opened": False}
    payload["campaign_sha256"] = stable_hash(payload)
    path.write_text(json.dumps(payload))
    return path


def test_four_box_replay_is_hash_bound_and_reports_underpowered(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    exit_root = tmp_path / "artifacts" / "exit_v1"
    index_rows = []
    for trajectory_id, role in (("learned-1", "LEARNED_OOF"), ("control-1", "DETERMINISTIC_CONTROL_OOF")):
        session = "2026-01-02"
        identity = pd.DataFrame(
            {
                "session": [session] * 3,
                "trajectory_id": [trajectory_id] * 3,
                "decision_time_ns": [1, 2, 3],
            }
        )
        features = identity.assign(
            entry_fill_option_price=4.0,
            current_net_pnl_dollars=[0.0, 1.0, 2.0],
            current_return_on_entry_premium=[0.0, 0.1, 0.2],
            seconds_held=[0.0, 60.0, 300.0],
        )
        labels = identity.assign(
            hold_to_1555_value_dollars=10.0,
            exit_until_filled_value_dollars=[0.0, 2.0, 4.0],
        )
        predictions = identity.assign(utility_hold=[1.0, 1.0, 1.0])
        feature_dir = scratch / "exit_features" / f"session={session}"
        label_dir = scratch / "exit_labels" / f"session={session}"
        prediction_dir = exit_root / "oof_predictions/fold=0" / f"session={session}"
        feature_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        features.to_parquet(feature_dir / f"{trajectory_id}.parquet", index=False)
        labels.to_parquet(label_dir / f"{trajectory_id}.parquet", index=False)
        for fee in (1.5, 2.0):
            for latency in (0, 1, 2, 5):
                path = sensitivity_label_path(
                    scratch,
                    session=session,
                    trajectory_id=trajectory_id,
                    fee_per_side_dollars=fee,
                    latency_seconds=latency,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                labels.to_parquet(path, index=False)
        predictions.to_parquet(prediction_dir / f"{trajectory_id}.parquet", index=False)
        index_rows.append(
            {
                "trajectory_id": trajectory_id,
                "session": session,
                "outer_fold": 0,
                "entry_policy_role": role,
                "receipt_path": "unused",
                "receipt_sha256": "a" * 64,
            }
        )
    index_path = tmp_path / "trajectory_index.parquet"
    pd.DataFrame(index_rows).to_parquet(index_path, index=False)
    entry_path = _campaign(
        tmp_path / "entry_campaign.json", {"trajectory_index_path": str(index_path)}
    )
    exit_path = _campaign(
        exit_root / "campaign.json", {"positive_target_skill_folds": 1}
    )
    # The gate is superseded and refuses by default; this test reproduces the
    # historical behaviour, which is the only sanctioned use of the escape hatch.
    with pytest.raises(Phase1ReplayError, match="SUPERSEDED 2026-08-05"):
        run_four_box_replay(
            scratch_root=scratch,
            entry_campaign_path=entry_path,
            exit_campaign_path=exit_path,
            report_root=tmp_path / "report",
        )

    result = run_four_box_replay(
        scratch_root=scratch,
        entry_campaign_path=entry_path,
        exit_campaign_path=exit_path,
        report_root=tmp_path / "report",
        i_understand_this_gate_is_defective=True,
    )
    assert result["verdict"] == "UNDERPOWERED"
    assert set(result["four_boxes"]) == {
        "learned_entry/control_exit",
        "learned_entry/learned_exit",
        "control_entry/control_exit",
        "control_entry/learned_exit",
    }
    assert result["protected_holdout_opened"] is False
