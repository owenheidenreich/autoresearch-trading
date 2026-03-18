from __future__ import annotations

import json

from tools.live_feature_parity_report import build_report


def test_build_report_from_bar_snapshots(tmp_path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    rows = [
        {"event": "session_start", "payload": {}},
        {
            "event": "bar_snapshot",
            "payload": {
                "non_nan_mask": [1, 1, 0] + [1] * 57,
            },
        },
        {
            "event": "bar_snapshot",
            "payload": {
                "missing_feature_indices": [1, 2],
            },
        },
    ]
    with open(audit_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report = build_report(str(audit_path))
    assert report["bar_snapshots"] == 2
    f0 = report["features"][0]
    f1 = report["features"][1]
    f2 = report["features"][2]
    assert f0["present_bars"] == 2
    assert f1["present_bars"] == 1
    assert f2["present_bars"] == 0
    assert f2["status"] == "missing"


def test_build_report_raises_without_feature_masks(tmp_path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    with open(audit_path, "w") as f:
        f.write(json.dumps({"event": "bar_snapshot", "payload": {"completeness": 0.9}}) + "\n")

    try:
        build_report(str(audit_path))
    except RuntimeError as exc:
        assert "No usable bar_snapshot rows" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing per-feature masks")

