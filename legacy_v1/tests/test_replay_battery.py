from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from tools.replay_battery import _parse_dates, run_battery


def test_parse_dates_range_business_days() -> None:
    dates = _parse_dates(
        dates_csv=None,
        dates_file=None,
        start_date="2026-03-16",
        end_date="2026-03-20",
        include_weekends=False,
    )
    assert dates == ["2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20"]


def _write_stub_replay(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            import argparse
            import csv
            import json
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--date", required=True)
            parser.add_argument("--output", required=True)
            args, _ = parser.parse_known_args()

            out_csv = Path(args.output)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out_csv.write_text("num,date\\n")
            base = out_csv.with_suffix("")
            qa_passed = not args.date.endswith("18")
            qa = {
                "schema_version": "replay_qa_v1",
                "passed": qa_passed,
                "critical_count": 0 if qa_passed else 1,
                "warning_count": 0,
                "anomalies": [] if qa_passed else [{"severity": "critical", "code": "stub_failure"}],
            }
            (Path(str(base) + "_qa.json")).write_text(json.dumps(qa))
            with open(Path(str(base) + "_ledger_days.csv"), "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "replay_date",
                        "num_trades",
                        "total_bars",
                        "total_pnl_pct",
                        "qa_passed",
                        "qa_critical_count",
                        "qa_warning_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "replay_date": args.date,
                        "num_trades": 1,
                        "total_bars": 2,
                        "total_pnl_pct": 1.25,
                        "qa_passed": qa_passed,
                        "qa_critical_count": qa["critical_count"],
                        "qa_warning_count": qa["warning_count"],
                    }
                )
            """
        )
    )


def test_run_battery_gate_from_stub_replay(tmp_path) -> None:
    stub_replay = tmp_path / "stub_replay.py"
    _write_stub_replay(stub_replay)

    output_root = tmp_path / "battery"
    summary = run_battery(
        dates=["2026-03-17", "2026-03-18"],
        output_root=output_root,
        replay_script=stub_replay,
        python_bin=sys.executable,
        model=None,
        train_py=None,
        no_download=True,
        ib_port=4002,
        warmup_days=1,
        min_trade_prob=0.55,
        risk_mode="live_like",
        enforce_max_hold=False,
    )
    assert summary["total_dates"] == 2
    assert summary["passed_dates"] == 1
    assert summary["failed_dates"] == 1
    assert summary["release_gate_passed"] is False
    assert (output_root / "battery_summary.json").exists()
    assert (output_root / "battery_results.csv").exists()
    loaded = json.loads((output_root / "battery_summary.json").read_text())
    assert loaded["release_gate_passed"] is False
