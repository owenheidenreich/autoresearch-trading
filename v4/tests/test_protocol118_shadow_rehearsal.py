from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from v4.live.shadow_parity import ShadowParityConfig, summarize_shadow_parity
from v4.scripts.run_protocol101_event_history_policy import FEATURE_COLUMNS
from v4.scripts.run_protocol118_protocol101_shadow_rehearsal import (
    _decision,
    _load_highres_rows,
    _paper_account_check,
    _terminal_action,
    build_shadow_observations,
)
from v4.sim.shadow_paper import ShadowPaperConfig, replay_shadow_paper


def _highres_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_file": "protocol101.json",
        "split": "q3_2025",
        "seed": 1,
        "candidate_uid": "candidate-1",
        "contract_id": "SPXW-20250701-06200.000-C",
        "decision_time": "2025-07-01T14:35:00+00:00",
        "session": "2025-07-01",
        "audit_status": "audited",
        "entry_time_1s": "2025-07-01T14:34:59.500000000+00:00",
        "mandatory_exit_time_1s": "2025-07-01T14:40:00+00:00",
        "entry_bid_1s": 1.0,
        "entry_ask_1s": 1.2,
        "exit_bid_1s": 1.8,
        "exit_ask_1s": 2.0,
        "exit_reason_1s": "target",
        "score": 2.5,
        "threshold": 0.5,
    }
    row.update(overrides)
    return row


def _write_context(root: Path, symbol: str, session: str) -> Path:
    out = root / f"{symbol.lower()}_1m"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2025-07-01T14:35:00+00:00", "2025-07-01T14:40:00+00:00"],
                utc=True,
            ),
            "symbol": [symbol, symbol],
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [6200.0, 6201.0] if symbol == "SPX" else [18.0, 18.2],
            "volume": [0, 0],
            "context_source": ["test_official", "test_official"],
            "is_official_index_data": [True, True],
        }
    )
    df.to_parquet(out / f"{session}.official_{symbol.lower()}.parquet")
    return out


def test_protocol118_dedupes_duplicate_highres_rows(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    row = _highres_row()
    path.write_text(json.dumps({"rows": [row, dict(row), _highres_row(seed=2)]}))

    rows = _load_highres_rows(path, paper_seed=1)

    assert len(rows) == 1
    assert rows[0]["candidate_uid"] == "candidate-1"


def test_protocol118_builds_no_order_shadow_stream_that_passes_strict_checks(tmp_path: Path) -> None:
    row = _highres_row()
    feature_map = {
        ("q3_2025", 1, "candidate-1"): {
            column: 1.0 for column in FEATURE_COLUMNS
        }
    }
    spx_dir = _write_context(tmp_path, "SPX", "2025-07-01")
    vix_dir = _write_context(tmp_path, "VIX", "2025-07-01")

    observations, build_summary = build_shadow_observations(
        highres_rows=[row],
        feature_map=feature_map,
        spx_dirs=(spx_dir,),
        vix_dirs=(vix_dir,),
        paper_seed=1,
    )

    assert len(observations) == 2
    assert build_summary["official_spx_fraction"] == 1.0
    assert build_summary["official_vix_fraction"] == 1.0
    assert observations[0]["order_intent"] is None
    assert observations[0]["decision"]["action"] == "hold"
    assert observations[1]["decision"]["action"] == "exit"
    assert observations[0]["features"]["entry_ask"] == 1.2

    parity = summarize_shadow_parity(
        observations,
        config=ShadowParityConfig(
            protocol_id="protocol101",
            max_quote_age_ms=5_000,
            max_context_age_ms=125_000,
            required_feature_columns=tuple(FEATURE_COLUMNS),
            allowed_actions=("hold", "exit", "stop", "forced_flat"),
        ),
    )
    paper = replay_shadow_paper(
        observations,
        config=ShadowPaperConfig(
            protocol_id="protocol101",
            require_all_closed=True,
            require_terminal_final=True,
            enforce_global_one_position=True,
        ),
    )

    assert parity["status"] == "pass"
    assert paper["status"] == "pass"
    assert paper["max_concurrent_positions"] == 1


def test_protocol118_blocks_unaffordable_contracts() -> None:
    account = _paper_account_check([_highres_row(entry_ask_1s=101.0)], starting_cash=10_000.0)

    assert account["summary"]["skipped_unaffordable_trades"] == 1


def test_protocol118_decision_requires_all_gates_to_pass() -> None:
    assert _terminal_action("stop") == "stop"
    assert _terminal_action("target") == "exit"
    decision = _decision(
        {"status": "pass"},
        {"status": "pass"},
        {"summary": {"skipped_unaffordable_trades": 1}},
        {"missing_feature_rows": [], "missing_context_rows": [], "official_spx_fraction": 1.0, "official_vix_fraction": 1.0},
    )

    assert decision == "blocked_unaffordable_trades"
