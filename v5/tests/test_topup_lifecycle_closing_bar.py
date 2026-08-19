"""Guards for the closing-bar top-up.

The tool spends money and writes into a data tree, so every refusal it claims is
pinned here: the window must contain exactly the closing stamp, the ladder must
be filtered by each symbol's own expiry, a wrong bar or a foreign symbol must be
refused after download, and the ceiling must bind.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from v5.ops import topup_lifecycle_closing_bar as topup
from v5.ops.acquire_lifecycle_backfill import _payload_sha256

SESSION = "2024-03-15"
STAMP = "240315"


def _symbol(right: str = "C", strike: int = 5100, expiry: str = STAMP) -> str:
    return f"SPXW  {expiry}{right}{strike * 1000:08d}"


def _definitions(tmp_path: Path, symbols: list[str]) -> Path:
    root = tmp_path / "definitions"
    root.mkdir()
    pd.DataFrame({"raw_symbol": symbols}).to_parquet(
        root / f"{SESSION}.definition.parquet", index=False
    )
    return root


def _bars(symbols: list[str], stamp: str = "16:00:00") -> pd.DataFrame:
    ts = pd.to_datetime([f"{SESSION} {stamp}"] * len(symbols)).tz_localize(
        "America/New_York"
    ).tz_convert("UTC")
    return pd.DataFrame({"ts_recv": ts, "symbol": symbols, "bid_px_00": 1.0})


def test_window_is_half_open_around_the_closing_stamp() -> None:
    start, end = topup._window(SESSION, seconds=1)
    # 16:00:00 ET in March is 20:00Z; the exclusive end must sit just past it so
    # the 16:00:00 stamp is inside and the 15:59:00 bar is not.
    assert start == "2024-03-15T20:00:00Z"
    assert end == "2024-03-15T20:00:01Z"


def test_ladder_reapplies_the_same_day_rule(tmp_path: Path) -> None:
    good, foreign = _symbol(), _symbol(expiry="240328")
    root = _definitions(tmp_path, [good, foreign, "GARBAGE"])
    assert topup.session_ladder(root, SESSION) == [good]


def test_ladder_refuses_a_session_with_no_same_day_symbol(tmp_path: Path) -> None:
    root = _definitions(tmp_path, [_symbol(expiry="240328")])
    with pytest.raises(topup.TopupError, match="no same-day"):
        topup.session_ladder(root, SESSION)


def test_verify_refuses_the_wrong_bar() -> None:
    with pytest.raises(topup.TopupError, match="only the 16:00:00 bar"):
        topup._verify_topup(_bars([_symbol()], stamp="15:59:00"), SESSION)


def test_verify_refuses_a_foreign_expiry() -> None:
    with pytest.raises(topup.TopupError, match="non-same-day symbol"):
        topup._verify_topup(_bars([_symbol(expiry="240328")]), SESSION)


def test_verify_accepts_the_closing_bar() -> None:
    checked = topup._verify_topup(_bars([_symbol(), _symbol("P", 5000)]), SESSION)
    assert checked == {"session": SESSION, "rows": 2, "contracts": 2}


def _sealed(payload: dict) -> dict:
    payload["receipt_sha256"] = _payload_sha256(payload, "receipt_sha256")
    return payload


def test_sessions_come_from_a_self_hash_valid_receipt(tmp_path: Path) -> None:
    path = tmp_path / "acq.json"
    path.write_text(
        json.dumps(
            _sealed({"files": [{"session": SESSION, "schema": "cbbo-1m"}]})
        )
    )
    assert topup.sessions_from_receipt(path) == [SESSION]

    tampered = json.loads(path.read_text())
    tampered["files"].append({"session": "2024-03-18", "schema": "cbbo-1m"})
    path.write_text(json.dumps(tampered))
    with pytest.raises(topup.TopupError, match="self-hash mismatch"):
        topup.sessions_from_receipt(path)


class _Client:
    """Records calls so a test can prove what was and was not requested."""

    def __init__(self, cost: float, frame: pd.DataFrame | None = None) -> None:
        self.metadata = self
        self.timeseries = self
        self.symbology = self
        self._cost, self._frame = cost, frame
        self.downloads: list[str] = []

    def get_cost(self, **kwargs) -> float:
        return self._cost

    def get_range(self, **kwargs):
        self.downloads.append(kwargs["start"])

        class _Data:
            def __init__(self, frame): self._frame = frame
            def to_df(self): return self._frame

        return _Data(self._frame)

    def resolve(self, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("the top-up must not spend symbology calls")


def _preflight(tmp_path: Path, cost: float, cap: float) -> tuple[Path, Path, Path]:
    acq = tmp_path / "acq.json"
    acq.write_text(
        json.dumps(_sealed({"files": [{"session": SESSION, "schema": "cbbo-1m"}]}))
    )
    root = _definitions(tmp_path, [_symbol()])
    out = tmp_path / "pre.json"
    topup.write_preflight(
        receipt_path=acq,
        definition_root=root,
        output_path=out,
        cap_usd=cap,
        seconds=1,
        client=_Client(cost),
    )
    return acq, root, out


def test_preflight_stops_over_the_cap_and_spends_no_symbology(tmp_path: Path) -> None:
    _, _, out = _preflight(tmp_path, cost=0.10, cap=0.05)
    receipt = json.loads(out.read_text())
    assert receipt["gate"] == "STOP_OVER_HARD_CAP"
    assert receipt["integrity"] == {
        "download_performed": False,
        "money_spent": False,
        "broker_contacted": False,
        "reserved_sessions_used": False,
    }


def test_acquire_refuses_a_failed_preflight(tmp_path: Path) -> None:
    acq, root, out = _preflight(tmp_path, cost=0.10, cap=0.05)
    with pytest.raises(topup.TopupError, match="did not pass"):
        topup.acquire(
            receipt_path=acq,
            definition_root=root,
            preflight_path=out,
            out_root=tmp_path / "data",
            acquisition_out=tmp_path / "topup.json",
            seconds=1,
            client=_Client(0.10, _bars([_symbol()])),
        )


def test_acquire_saves_and_receipts_the_closing_bar(tmp_path: Path) -> None:
    acq, root, out = _preflight(tmp_path, cost=0.01, cap=0.05)
    receipt = topup.acquire(
        receipt_path=acq,
        definition_root=root,
        preflight_path=out,
        out_root=tmp_path / "data",
        acquisition_out=tmp_path / "topup.json",
        seconds=1,
        client=_Client(0.01, _bars([_symbol(), _symbol("P", 5000)])),
    )
    assert receipt["completion"]["downloaded"] == 1
    assert receipt["completion"]["contracts_total"] == 2
    assert (tmp_path / "data" / f"{SESSION}.cbbo-1m.closing.parquet").is_file()
    assert receipt["receipt_sha256"] == _payload_sha256(receipt, "receipt_sha256")


def test_acquire_is_idempotent_and_never_rebuys(tmp_path: Path) -> None:
    acq, root, out = _preflight(tmp_path, cost=0.01, cap=0.05)
    common = dict(
        receipt_path=acq,
        definition_root=root,
        preflight_path=out,
        out_root=tmp_path / "data",
        seconds=1,
    )
    topup.acquire(
        **common,
        acquisition_out=tmp_path / "one.json",
        client=_Client(0.01, _bars([_symbol()])),
    )
    second = _Client(0.01, _bars([_symbol()]))
    receipt = topup.acquire(
        **common, acquisition_out=tmp_path / "two.json", client=second
    )
    assert second.downloads == []  # the existing file must not be re-bought
    assert receipt["completion"]["skipped_existing"] == 1
    assert receipt["spent_this_run_usd"] == 0.0


def test_acquire_refuses_a_bar_the_vendor_stamped_wrong(tmp_path: Path) -> None:
    acq, root, out = _preflight(tmp_path, cost=0.01, cap=0.05)
    with pytest.raises(topup.TopupError, match="only the 16:00:00 bar"):
        topup.acquire(
            receipt_path=acq,
            definition_root=root,
            preflight_path=out,
            out_root=tmp_path / "data",
            acquisition_out=tmp_path / "topup.json",
            seconds=1,
            client=_Client(0.01, _bars([_symbol()], stamp="15:59:00")),
        )
    assert not (tmp_path / "data" / f"{SESSION}.cbbo-1m.closing.parquet").exists()
