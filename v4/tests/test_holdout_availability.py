from __future__ import annotations

import json
from pathlib import Path

from v4.foundation.holdout_availability import build_holdout_availability
from v4.model.neural_training_readiness import make_default_holdout_reservation


def test_holdout_availability_blocks_when_reserved_data_is_pending(tmp_path: Path) -> None:
    path = tmp_path / "v4/audit/autoresearch/unified_untouched_holdout_reservation/summary.json"
    path.parent.mkdir(parents=True)
    reservation = make_default_holdout_reservation(reserved_on="2026-05-24").to_dict()
    path.write_text(json.dumps({"reservation": reservation}) + "\n")

    payload = build_holdout_availability(tmp_path)

    assert payload["decision"] == "untouched_holdout_data_pending_collection"
    assert payload["data_available"] is False
    assert payload["protected_holdout_scored"] is False


def test_holdout_availability_passes_only_when_data_is_frozen(tmp_path: Path) -> None:
    path = tmp_path / "v4/audit/autoresearch/unified_untouched_holdout_reservation/summary.json"
    path.parent.mkdir(parents=True)
    reservation = make_default_holdout_reservation(reserved_on="2026-05-24").to_dict()
    reservation["data_status"] = "available_frozen"
    path.write_text(json.dumps({"reservation": reservation}) + "\n")

    payload = build_holdout_availability(tmp_path)

    assert payload["decision"] == "untouched_holdout_data_available_frozen"
    assert payload["data_available"] is True
