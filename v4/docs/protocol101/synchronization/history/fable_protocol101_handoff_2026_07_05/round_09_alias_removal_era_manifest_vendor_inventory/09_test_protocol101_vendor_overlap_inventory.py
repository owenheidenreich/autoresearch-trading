"""Tests for Protocol101 vendor overlap inventory helpers."""
from __future__ import annotations

from pathlib import Path

from v4.scripts.run_protocol101_vendor_overlap_inventory import (
    ProductRoot,
    file_format,
    inventory_product,
    session_from_path,
)


def test_session_from_path_and_file_format() -> None:
    path = Path("data/raw/databento/opra_spxw_cbbo_1m/2026-07-01.cbbo-1m.dbn.zst")

    assert session_from_path(path) == "2026-07-01"
    assert file_format(path) == "dbn.zst"
    assert file_format(Path("2026-07-01.parquet")) == "parquet"


def test_inventory_product_counts_sessions_and_formats(tmp_path: Path) -> None:
    root = tmp_path / "cbbo"
    root.mkdir()
    (root / "2026-07-01.cbbo-1m.dbn.zst").write_text("placeholder")
    (root / "2026-07-01.cbbo-1m.parquet").write_text("placeholder")
    (root / "2026-07-02.cbbo-1m.dbn.zst").write_text("placeholder")
    (root / "not-a-session.dbn.zst").write_text("placeholder")

    payload = inventory_product(
        ProductRoot("databento", "opra_spxw_cbbo_1m", "option", root, "*"),
        max_sample_files=0,
    )

    assert payload["file_count"] == 3
    assert payload["session_count"] == 2
    assert payload["sessions"] == ["2026-07-01", "2026-07-02"]
    assert payload["formats"] == {"dbn.zst": 2, "parquet": 1}
