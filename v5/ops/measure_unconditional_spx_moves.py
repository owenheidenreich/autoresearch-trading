"""Create immutable evidence for the unconditional SPX move-terrain census.

The wrapper deliberately has no knobs for horizons, thresholds, clock anchors,
or option assumptions.  Those declarations live in the reviewed analysis
module and are hashed into every attempt.  A failure after the attempt
directory is created leaves an archived wrapper, module, log, and self-hashed
failure receipt; a successful attempt never overwrites an existing directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

# Direct ``python v5/ops/...py`` execution starts with v5/ops on sys.path.
# Add the repository root before importing the package, matching the required
# operator invocation rather than requiring an undocumented PYTHONPATH.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v5.research import unconditional_spx_moves as terrain


SCHEMA_VERSION = "v5.unconditional-spx-move-terrain-receipt.v1"
CREATED_ON = "2026-08-23"
DEFAULT_DATA_ROOT = Path(
    "/Volumes/AR_TRADING_DATA/spx_parity_tape_2022-06-01_2026-07-31"
)
DEFAULT_OUTPUT_DIR = Path(
    "v4/audit/autoresearch/unconditional_spx_move_terrain_2026_08_23_attempt001"
)

UPSTREAM_EVIDENCE = {
    "parity_tape_build_receipt": Path(
        "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/"
        "parity_spot_tape_receipt.json"
    ),
    "corpus_build_receipt": Path(
        "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/"
        "corpus_build_spx_tape_receipt.json"
    ),
    "interior_freeze_audit_receipt": Path(
        "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/"
        "interior_full_book_freeze_audit_2026_08_22_attempt002.json"
    ),
    "two_era_structural_audit_receipt": Path(
        "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/"
        "two_era_structural_audit_2026_08_22_attempt001.json"
    ),
}


class TerrainRunError(RuntimeError):
    """The evidence wrapper cannot preserve its declared audit contract."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _signed_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in payload:
        raise TerrainRunError("receipt payload is already signed")
    signed = dict(payload)
    signed["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return signed


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _source_paths() -> tuple[Path, Path]:
    return Path(__file__).resolve(), Path(terrain.__file__).resolve()


def _archive_sources(output_dir: Path) -> dict[str, Path]:
    wrapper, module = _source_paths()
    paths = {
        "archived_wrapper": output_dir / wrapper.name,
        "archived_analysis_module": output_dir / module.name,
    }
    shutil.copyfile(wrapper, paths["archived_wrapper"])
    shutil.copyfile(module, paths["archived_analysis_module"])
    return paths


def _upstream_hashes(repo_root: Path) -> dict[str, dict[str, Any]]:
    hashes: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, relative in UPSTREAM_EVIDENCE.items():
        path = repo_root / relative
        if not path.is_file():
            missing.append(str(relative))
            continue
        hashes[name] = {
            "path": str(relative),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    if missing:
        raise TerrainRunError(f"upstream evidence is missing: {missing}")
    return hashes


def _declared_grid() -> dict[str, Any]:
    valid = [
        {
            "start_time_et": start,
            "horizon_minutes": horizon,
            "status": "OBSERVABLE",
        }
        for start, _, horizon in terrain.valid_configurations()
    ]
    valid_keys = {(row["start_time_et"], row["horizon_minutes"]) for row in valid}
    unavailable = [
        {
            "start_time_et": start,
            "horizon_minutes": horizon,
            "status": "UNOBSERVABLE_PAST_16_00",
        }
        for start in terrain.START_TIMES
        for horizon in terrain.HORIZONS_MINUTES
        if (start, horizon) not in valid_keys
    ]
    declaration = {
        "start_times_et": list(terrain.START_TIMES),
        "horizons_minutes": list(terrain.HORIZONS_MINUTES),
        "move_thresholds_points": list(terrain.MOVE_THRESHOLDS_POINTS),
        "adverse_race_thresholds_points": list(terrain.ADVERSE_THRESHOLDS_POINTS),
        "directions": list(terrain.DIRECTIONS),
        "observable_clock_horizon_cells": valid,
        "unobservable_clock_horizon_cells": unavailable,
    }
    declaration["grid_sha256"] = hashlib.sha256(canonical_json(declaration)).hexdigest()
    return declaration


def _table_artifacts(output_dir: Path, corpus: terrain.TapeCorpus, tables: terrain.AnalysisTables) -> dict[str, dict[str, Any]]:
    paths = {
        "input_manifest": output_dir / "input_manifest.csv",
        "defect_disposition": output_dir / "defect_disposition.csv",
        "excursion_grid": output_dir / "excursion_grid.csv",
        "race_grid": output_dir / "race_grid.csv",
        "contract_necessary_condition_grid": output_dir
        / "contract_necessary_condition_grid.csv",
        "comparison_grid": output_dir / "comparison_grid.csv",
        "readable_tables": output_dir / "readable_tables.md",
    }
    corpus.input_manifest.to_csv(paths["input_manifest"], index=False)
    defect_frame = pd.DataFrame(corpus.defect_disposition)
    defect_frame.to_csv(paths["defect_disposition"], index=False)
    tables.excursion.to_csv(paths["excursion_grid"], index=False)
    tables.race.to_csv(paths["race_grid"], index=False)
    tables.contract.to_csv(paths["contract_necessary_condition_grid"], index=False)
    tables.comparisons.to_csv(paths["comparison_grid"], index=False)
    paths["readable_tables"].write_text(
        terrain.render_readable_tables(corpus, tables), encoding="utf-8"
    )
    rows = {
        "input_manifest": len(corpus.input_manifest),
        "defect_disposition": len(defect_frame),
        "excursion_grid": len(tables.excursion),
        "race_grid": len(tables.race),
        "contract_necessary_condition_grid": len(tables.contract),
        "comparison_grid": len(tables.comparisons),
    }
    return {
        name: _artifact(path, rows=rows.get(name))
        for name, path in paths.items()
    }


def _success_payload(
    *,
    data_root: Path,
    corpus: terrain.TapeCorpus,
    tables: terrain.AnalysisTables,
    artifacts: dict[str, dict[str, Any]],
    upstream: dict[str, dict[str, Any]],
    quality_control: dict[str, Any],
) -> dict[str, Any]:
    era_counts = {
        "backfill": int(sum(value < terrain.ERA_CUTOFF for value in corpus.dates)),
        "owned": int(sum(value >= terrain.ERA_CUTOFF for value in corpus.dates)),
    }
    year_counts = {
        str(year): int(sum(value.year == year for value in corpus.dates))
        for year in range(2022, 2027)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_on": CREATED_ON,
        "status": "PASS",
        "purpose": "unconditional SPX parity-snapshot move terrain; not a strategy test",
        "data_root": str(data_root.resolve()),
        "population": {
            "discovered_sessions": len(corpus.input_manifest),
            "analyzed_sessions": len(corpus.sessions),
            "excluded_present_sessions": int(
                corpus.input_manifest["status"].eq("EXCLUDED_KNOWN_DEFECT").sum()
            ),
            "first_analyzed_session": corpus.sessions[0],
            "last_analyzed_session": corpus.sessions[-1],
            "source_era_counts": era_counts,
            "calendar_year_counts": year_counts,
            "confirmation_reservation_start": terrain.CONFIRMATION_START.isoformat(),
            "reserved_sessions_used": 0,
        },
        "grid": _declared_grid(),
        "scope_law": {
            "primary": ["pooled", "backfill era", "owned era", "calendar year"],
            "endpoint_comparisons": [
                "2026 Jan-Jul minus 2022 Jun-Dec",
                "2026 Jun-Jul minus 2022 Jun-Jul common-calendar support",
            ],
            "pooled_is_descriptive_only": True,
            "source_and_date_are_confounded": True,
        },
        "clock_law": {
            "tape_bar_labels_et": "09:30 through 15:59",
            "observation_minutes_et": "09:31 through 16:00",
            "bar_t_contains_snapshot_from": "t+1 minute",
            "window_path": "exact observation snapshots t+1 through t+N, entry at t",
            "late_cells": "reported in the receipt as structurally unobservable, not misses",
        },
        "uncertainty": {
            "unit": "session",
            "windows_per_session_per_cell": 1,
            "pointwise_probability_interval": "two-sided 95% Wilson score over session Bernoulli outcomes",
            "difference_interval": "two-sided 95% Newcombe score over disjoint sessions",
            "cluster_statement": "each cell has one exact clock window per session, so no within-session windows are treated as independent",
            "cross_grid_multiplicity_adjusted": False,
            "permitted_interpretation": "descriptive cellwise intervals only; no best-cell, family, drift-detected, or equivalence claim",
        },
        "companion_contract_scenario": {
            "status": "necessary-condition arithmetic only; not historical expected P&L",
            "spot": terrain.COMPANION_SPOT,
            "sigma": terrain.COMPANION_SIGMA,
            "otm_points": list(terrain.COMPANION_OTM_POINTS),
            "atm_spread_points": terrain.COMPANION_ATM_SPREAD,
            "otm_spread_points": terrain.COMPANION_OTM_SPREAD,
            "signed_ticket_cap_usd": terrain.SIGNED_TICKET_CAP_USD,
            "friction_law": "$3.08 option fees plus one crossed spread; $13.08 ATM and $23.08 OTM under this scenario",
            "status_conflict_resolution": "$17.92 is the STATUS-controlled ES futures friction, not the option round trip",
            "time_mapping_correction": "330 minutes to 16:00 is 10:30 ET; 09:31 ET has 389 minutes remaining",
            "fixed_points_hit": "historical SPX favourable excursion compared with scenario required points",
            "scale_adjusted_hit": "historical excursion/entry spot compared with scenario required move/6800",
        },
        "claim_classification": {
            "unconditional_market_structure": True,
            "feature_conditioning": False,
            "signal_test": False,
            "model_fit": False,
            "threshold_tuning": False,
            "strategy_proposed": False,
            "option_economics_measured_on_historical_contracts": False,
            "adoption": "ADOPT NOTHING",
        },
        "limitations": [
            "Parity OHLC is one snapshot per minute, so touches and race order are first observed snapshot crossings; intraminute touches can be missed.",
            "A required-move hit is necessary but not sufficient for nonnegative option P&L; losses, overshoots, IV, spread variation, fillability, and the whole payoff distribution are not joined historically.",
            "No unconditional hit rate alone defines a break-even probability, so this census cannot prove that the long-option class is profitable or structurally dead.",
            "An unconditional base rate does not prove that no separately authorized conditional signal could enrich an event.",
            "The contract join fixes spot, volatility, spread, and moneyness rather than reconstructing each historical option surface.",
            "2022 and 2026 are partial years; the common June-July contrast reduces but does not remove time/source confounding.",
            "Wilson/Newcombe intervals use sessions as the observation unit but do not model adjacent-session serial dependence; they are pointwise and not simultaneous grid-wide bands.",
        ],
        "table_rows": {
            "excursion": len(tables.excursion),
            "race": len(tables.race),
            "contract_necessary_condition": len(tables.contract),
            "comparisons": len(tables.comparisons),
        },
        "quality_control": quality_control,
        "upstream_evidence": upstream,
        "artifacts": artifacts,
    }


def run(data_root: Path, output_dir: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Run one immutable attempt and return its PASS/FAIL receipt."""

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise TerrainRunError(f"refusing to overwrite existing attempt: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    archived = _archive_sources(output_dir)
    log_path = output_dir / "run.log"
    root = Path.cwd() if repo_root is None else Path(repo_root)
    try:
        upstream = _upstream_hashes(root)
        corpus = terrain.load_tape_corpus(data_root, strict_population=True)
        tables = terrain.analyze_corpus(corpus)
        quality_control = terrain.validate_tables(corpus, tables)
        artifacts = _table_artifacts(output_dir, corpus, tables)
        log_path.write_text(
            "PASS\n"
            f"discovered_sessions={len(corpus.input_manifest)}\n"
            f"analyzed_sessions={len(corpus.sessions)}\n"
            f"excursion_rows={len(tables.excursion)}\n"
            f"race_rows={len(tables.race)}\n"
            f"contract_rows={len(tables.contract)}\n"
            f"comparison_rows={len(tables.comparisons)}\n",
            encoding="utf-8",
        )
        artifacts.update(
            {
                name: _artifact(path)
                for name, path in {**archived, "run_log": log_path}.items()
            }
        )
        receipt = _signed_receipt(
            _success_payload(
                data_root=data_root,
                corpus=corpus,
                tables=tables,
                artifacts=artifacts,
                upstream=upstream,
                quality_control=quality_control,
            )
        )
        _write_json(output_dir / "receipt.json", receipt)
        return receipt
    except Exception as exc:  # noqa: BLE001 - preserve every failed attempt
        failure_traceback = traceback.format_exc()
        log_path.write_text(f"FAIL\n{failure_traceback}", encoding="utf-8")
        failure_artifacts = {
            name: _artifact(path)
            for name, path in {**archived, "run_log": log_path}.items()
            if path.is_file()
        }
        failure = _signed_receipt(
            {
                "schema_version": SCHEMA_VERSION,
                "created_on": CREATED_ON,
                "status": "FAIL",
                "purpose": "preserved failed unconditional SPX move-terrain attempt",
                "data_root": str(data_root.resolve()),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "failure_behavior": "attempt directory, source archive, traceback log, and self-hashed failure receipt preserved",
                "artifacts": failure_artifacts,
            }
        )
        _write_json(output_dir / "failure_receipt.json", failure)
        return failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    receipt = run(args.data_root, args.output_dir)
    receipt_name = "receipt.json" if receipt["status"] == "PASS" else "failure_receipt.json"
    print(args.output_dir / receipt_name)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
