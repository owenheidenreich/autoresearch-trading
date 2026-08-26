"""Offline guards for the Job-49 CMBP catalogue foundation."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from v5.ops import prepare_cmbp_catalogue_preflight as cli
from v5.research import cmbp_catalogue_preflight as catalogue


def _symbol(session: str, right: str = "C", strike: int = 4_000) -> str:
    return f"SPXW  {session[2:].replace('-', '')}{right}{strike * 1000:08d}"


def _code_hashes() -> dict[str, str]:
    return {"v5/research/cmbp_catalogue_preflight.py": "a" * 64}


def _dependencies() -> dict[str, object]:
    return {"python": "test", "packages": {"pyarrow": "test"}}


def _manifest(tmp_path: Path) -> dict:
    root = tmp_path / "ladder"
    root.mkdir()
    sessions = ("2023-03-27", "2023-03-28", "2023-03-29")
    for session in sessions:
        (root / f"{session}.parquet").touch()

    def reader(path: Path) -> list[str]:
        session = path.stem
        # Repetition is expected in a minute ladder.  The frozen symbol set is
        # unique and sorted only after the raw-symbol-only projection.
        return [_symbol(session, "P", 3_995), _symbol(session), _symbol(session)]

    return catalogue.manifest_from_ladder_directory(
        root,
        coverage_start="2023-03-28",
        expected_source_session_count=3,
        raw_symbol_reader=reader,
    )


def _declaration(tmp_path: Path) -> dict:
    return catalogue.prepare_catalogue_declaration(
        _manifest(tmp_path),
        code_hashes=_code_hashes(),
        dependency_manifest=_dependencies(),
    )


def _responses(declaration: dict, costs: tuple[object, ...] = ("0", "0.1850")) -> dict:
    value = catalogue.response_template(declaration)
    value["dataset_range"].update(
        status="OK", dataset_start="2013-04-01", dataset_end="2026-08-25"
    )
    assert len(value["sessions"]) == len(costs)
    declared_by_session = {row["session"]: row for row in declaration["sessions"]}
    for index, (row, cost) in enumerate(zip(value["sessions"], costs, strict=True), start=1):
        row["status"] = "OK"
        row["cost_usd"] = cost
        row["record_count"] = index * 100
        row["resolved_symbols"] = {
            symbol: [f"synthetic-{index}-{offset}"]
            for offset, symbol in enumerate(
                declared_by_session[row["session"]]["symbols"], start=1
            )
        }
        row["unresolved_symbols"] = []
    return value


def test_scope_uses_exact_raw_osi_band_and_never_parent_symbol(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    symbols = [
        symbol for row in declaration["sessions"] for symbol in row["symbols"]
    ]
    assert symbols
    assert all(symbol.startswith("SPXW  ") and len(symbol) == 21 for symbol in symbols)
    assert all(symbol != "SPXW.OPT" for symbol in symbols)


def test_root_expiry_right_strike_uniqueness_and_sorted_order_are_required(
    tmp_path: Path,
) -> None:
    declaration = _declaration(tmp_path)
    for row in declaration["sessions"]:
        assert row["symbols"] == sorted(set(row["symbols"]))
        for symbol in row["symbols"]:
            assert catalogue.validate_raw_spxw_osi(symbol, row["session"]) == symbol


def test_request_is_one_session_rth_half_open_and_cmbp_only(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    for row in declaration["sessions"]:
        for request in row["requests"]:
            parameters = request["parameters"]
            assert parameters["dataset"] == "OPRA.PILLAR"
            if request["method"] == "symbology.resolve":
                assert parameters["start"] == row["session"]
                assert parameters["stype_out"] == "instrument_id"
            else:
                assert parameters["start"] == row["rth_open_utc"]
                assert parameters["end"] == row["rth_close_utc"]
            if request["method"] in {
                "metadata.get_cost",
                "metadata.get_record_count",
            }:
                assert parameters["schema"] == "cmbp-1"


def test_only_dataset_range_symbology_cost_and_count_are_allowlisted() -> None:
    assert catalogue.ALLOWED_FUTURE_METHODS == (
        "metadata.get_dataset_range",
        "symbology.resolve",
        "metadata.get_cost",
        "metadata.get_record_count",
    )
    assert all(
        "timeseries" not in method.lower() and "download" not in method.lower()
        for method in catalogue.ALLOWED_FUTURE_METHODS
    )


def test_precoverage_date_is_excluded_without_schema_fallback(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    row = declaration["sessions"][0]
    assert row["session"] == "2023-03-27"
    assert row["coverage_disposition"] == "EXCLUDED_KNOWN_PRE_COVERAGE"
    assert row["requests"] == []
    assert declaration["schema"] == "cmbp-1"


def test_byte_exact_osi_requires_two_spaces_and_same_day_expiry() -> None:
    good = "SPXW  230328C04000000"
    assert len(good) == 21
    assert catalogue.validate_raw_spxw_osi(good, "2023-03-28") == good

    with pytest.raises(catalogue.CataloguePreflightError) as one_space:
        catalogue.validate_raw_spxw_osi("SPXW 230328C04000000", "2023-03-28")
    assert one_space.value.status == catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH

    with pytest.raises(catalogue.CataloguePreflightError) as wrong_expiry:
        catalogue.validate_raw_spxw_osi("SPXW  230329C04000000", "2023-03-28")
    assert wrong_expiry.value.status == catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH


def test_scope_retains_all_sessions_and_requests_only_event_era(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert manifest["manifest_sha256"] == catalogue.self_hash(manifest, "manifest_sha256")
    declaration = catalogue.prepare_catalogue_declaration(
        manifest,
        code_hashes=_code_hashes(),
        dependency_manifest=_dependencies(),
    )
    assert declaration["status"] == catalogue.STATUS_CATALOGUE_READY_LOCAL
    assert declaration["source_session_count"] == 3
    assert declaration["excluded_pre_event_era_session_count"] == 1
    assert declaration["request_session_count"] == 2
    assert declaration["sessions"][0]["coverage_disposition"].startswith("EXCLUDED_")
    assert declaration["sessions"][0]["requests"] == []
    assert [request["method"] for request in declaration["sessions"][1]["requests"]] == list(
        catalogue.SESSION_REQUEST_METHODS
    )
    assert declaration["declaration_sha256"] == catalogue.self_hash(
        declaration, "declaration_sha256"
    )


def test_utc_bounds_are_dst_aware_half_open_and_pin_early_close() -> None:
    assert catalogue.rth_utc_bounds("2023-03-28", early_close=False) == (
        "2023-03-28T13:30:00Z",
        "2023-03-28T20:00:00Z",
    )
    assert catalogue.rth_utc_bounds("2024-07-03", early_close=True) == (
        "2024-07-03T13:30:00Z",
        "2024-07-03T17:00:00Z",
    )


def test_default_ladder_reader_projects_only_raw_symbol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "ladder"
    root.mkdir()
    session = "2023-03-28"
    path = root / f"{session}.parquet"
    pq.write_table(
        pa.table(
            {
                "raw_symbol": [_symbol(session), _symbol(session, "P")],
                "future_return": [999.0, -999.0],
                "pnl": [1000.0, -1000.0],
            }
        ),
        path,
    )
    original = pq.read_table
    observed: list[list[str]] = []

    def guarded_read_table(*args: object, **kwargs: object) -> pa.Table:
        observed.append(list(kwargs["columns"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", guarded_read_table)
    manifest = catalogue.manifest_from_ladder_directory(
        root,
        coverage_start="2023-03-28",
        expected_source_session_count=1,
    )
    assert observed == [["raw_symbol"]]
    assert manifest["source_policy"]["parquet_columns_read"] == ["raw_symbol"]
    assert manifest["source_policy"]["outcome_columns_read"] == []


def test_declaration_tampering_fails_scope_drift(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    declaration["sessions"][1]["symbols"][0] = _symbol("2023-03-28", "C", 4_010)
    with pytest.raises(catalogue.CataloguePreflightError) as caught:
        catalogue.validate_catalogue_declaration(declaration)
    assert caught.value.status == catalogue.STOP_SCOPE_OR_CODE_DRIFT


def test_success_receipt_sums_decimal_costs_exactly_and_authorizes_nothing(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    responses = _responses(declaration, costs=("0", "0.1850"))
    receipt = catalogue.build_preflight_receipt(
        declaration,
        responses,
        hard_cap_usd="0.20",
    )
    assert receipt["status"] == catalogue.STATUS_PREFLIGHT_PASS_ONLY
    assert receipt["exact_total_cost_usd"] == "0.185"
    assert receipt["exact_total_record_count"] == 300
    assert receipt["request_session_count"] == receipt["response_session_count"] == 2
    assert receipt["integrity"]["timeseries_requested"] is False
    assert receipt["integrity"]["download_requested"] is False
    assert receipt["authorization_effect"] == "NONE"
    catalogue.validate_preflight_receipt(receipt, declaration=declaration)


def test_actual_per_session_values_are_summed_not_extrapolated(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    receipt = catalogue.build_preflight_receipt(
        declaration,
        _responses(declaration, costs=("0.101", "0.209")),
        hard_cap_usd="0.31",
    )
    assert receipt["exact_total_cost_usd"] == "0.31"
    assert [row["cost_usd"] for row in receipt["sessions"]] == ["0.101", "0.209"]


def test_zero_cost_requires_positive_record_count(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    responses = _responses(declaration)
    responses["sessions"][0]["record_count"] = 0
    receipt = catalogue.build_preflight_receipt(declaration, responses, hard_cap_usd="1")
    assert receipt["status"] == catalogue.STOP_ZERO_RECORDS
    assert receipt["sessions"] == []
    catalogue.validate_preflight_receipt(receipt, declaration=declaration)


def test_no_missing_or_silent_session_drop(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    responses = _responses(declaration)
    missing = responses["sessions"].pop()["session"]
    receipt = catalogue.build_preflight_receipt(declaration, responses, hard_cap_usd="1")
    assert receipt["status"] == catalogue.STOP_MISSING_SESSION_RESPONSE
    assert receipt["failure"]["details"]["missing_sessions"] == [missing]

    responses = _responses(declaration)
    responses["sessions"].append(
        {"session": "2023-03-27", "status": "OK", "schema": "cmbp-1"}
    )
    receipt = catalogue.build_preflight_receipt(declaration, responses, hard_cap_usd="1")
    assert receipt["status"] == catalogue.STOP_SCOPE_OR_CODE_DRIFT
    assert receipt["failure"]["details"]["extra_sessions"] == ["2023-03-27"]


def test_nonfinite_cost_missing_count_and_session_drop_fail_closed(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    nonfinite = _responses(declaration)
    nonfinite["sessions"][0]["cost_usd"] = float("inf")
    assert catalogue.build_preflight_receipt(
        declaration, nonfinite, hard_cap_usd="1"
    )["status"] == catalogue.STOP_NONFINITE_COST

    missing_count = _responses(declaration)
    missing_count["sessions"][0]["record_count"] = None
    assert catalogue.build_preflight_receipt(
        declaration, missing_count, hard_cap_usd="1"
    )["status"] == catalogue.STOP_ZERO_RECORDS

    dropped = _responses(declaration)
    dropped["sessions"].pop()
    assert catalogue.build_preflight_receipt(
        declaration, dropped, hard_cap_usd="1"
    )["status"] == catalogue.STOP_MISSING_SESSION_RESPONSE


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), "-0.01"])
def test_nonfinite_or_negative_cost_fails_closed(tmp_path: Path, cost: object) -> None:
    declaration = _declaration(tmp_path)
    responses = _responses(declaration)
    responses["sessions"][0]["cost_usd"] = cost
    receipt = catalogue.build_preflight_receipt(declaration, responses, hard_cap_usd="1")
    assert receipt["status"] == catalogue.STOP_NONFINITE_COST


def test_exact_total_binds_hard_cap(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    responses = _responses(declaration, costs=("0.10", "0.20"))
    receipt = catalogue.build_preflight_receipt(declaration, responses, hard_cap_usd="0.299999")
    assert receipt["status"] == catalogue.STOP_OVER_HARD_CAP
    assert receipt["failure"]["details"]["exact_total_cost_usd"] == "0.3"


def test_schema_and_method_firewalls_fail_closed(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    responses = _responses(declaration)
    responses["sessions"][0]["schema"] = "tcbbo"
    receipt = catalogue.build_preflight_receipt(declaration, responses, hard_cap_usd="1")
    assert receipt["status"] == catalogue.STOP_SCHEMA_UNAVAILABLE

    responses = _responses(declaration)
    responses["method_attestation"]["methods_used"] = ["timeseries.get_range"]
    responses["method_attestation"]["timeseries_calls"] = 1
    receipt = catalogue.build_preflight_receipt(declaration, responses, hard_cap_usd="1")
    assert receipt["status"] == catalogue.STOP_SCOPE_OR_CODE_DRIFT
    assert receipt["integrity"]["timeseries_requested"] is False


def test_explicit_session_stop_is_preserved(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    responses = _responses(declaration)
    responses["sessions"][0]["status"] = catalogue.STOP_SCHEMA_UNAVAILABLE
    responses["sessions"][0]["reason"] = "coverage endpoint refused this date"
    receipt = catalogue.build_preflight_receipt(declaration, responses, hard_cap_usd="1")
    assert receipt["status"] == catalogue.STOP_SCHEMA_UNAVAILABLE


def test_receipt_tampering_breaks_self_hash(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path)
    receipt = catalogue.build_preflight_receipt(
        declaration, _responses(declaration), hard_cap_usd="1"
    )
    tampered = copy.deepcopy(receipt)
    tampered["exact_total_record_count"] += 1
    with pytest.raises(catalogue.CataloguePreflightError, match="self-hash mismatch"):
        catalogue.validate_preflight_receipt(tampered, declaration=declaration)


def test_request_manifest_is_deterministic_and_hash_bound(tmp_path: Path) -> None:
    first = _declaration(tmp_path)
    second_root = tmp_path / "second"
    second_root.mkdir()
    second = _declaration(second_root)
    assert first["request_manifest_sha256"] == second["request_manifest_sha256"]
    assert first["symbol_manifest_sha256"] == second["symbol_manifest_sha256"]
    tampered = copy.deepcopy(first)
    tampered["global_requests"][0]["parameters"]["dataset"] = "OTHER"
    with pytest.raises(catalogue.CataloguePreflightError):
        catalogue.validate_catalogue_declaration(tampered)


def test_cli_builds_and_validates_offline_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "ladder"
    root.mkdir()
    for session in ("2023-03-27", "2023-03-28"):
        pq.write_table(pa.table({"raw_symbol": [_symbol(session)]}), root / f"{session}.parquet")
    declaration_path = tmp_path / "declaration.json"
    assert (
        cli.main(
            [
                "prepare-ladder",
                "--ladder-root",
                str(root),
                "--coverage-start",
                "2023-03-28",
                "--expected-source-sessions",
                "2",
                "--output",
                str(declaration_path),
            ]
        )
        == 0
    )
    assert cli.main(["validate-declaration", "--declaration", str(declaration_path)]) == 0

    template_path = tmp_path / "responses.json"
    assert (
        cli.main(
            [
                "response-template",
                "--declaration",
                str(declaration_path),
                "--output",
                str(template_path),
            ]
        )
        == 0
    )
    responses = json.loads(template_path.read_text())
    declaration = json.loads(declaration_path.read_text())
    declared = next(row for row in declaration["sessions"] if row["requests"])
    responses["dataset_range"].update(
        status="OK", dataset_start="2013-04-01", dataset_end="2026-08-25"
    )
    responses["sessions"][0].update(
        status="OK",
        cost_usd="0",
        record_count=123,
        resolved_symbols={symbol: [index] for index, symbol in enumerate(declared["symbols"], 1)},
        unresolved_symbols=[],
    )
    template_path.write_text(json.dumps(responses))
    receipt_path = tmp_path / "receipt.json"
    assert (
        cli.main(
            [
                "receipt",
                "--declaration",
                str(declaration_path),
                "--responses",
                str(template_path),
                "--hard-cap-usd",
                "0",
                "--output",
                str(receipt_path),
            ]
        )
        == 0
    )
    assert cli.main(["validate-receipt", "--receipt", str(receipt_path), "--declaration", str(declaration_path)]) == 0


def test_cli_has_no_network_or_live_client_import() -> None:
    tree = ast.parse(Path(cli.__file__).read_text())
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert not imported_roots & {"databento", "requests", "httpx", "socket", "urllib"}
    assert catalogue.ALLOWED_FUTURE_METHODS == (
        "metadata.get_dataset_range",
        "symbology.resolve",
        "metadata.get_cost",
        "metadata.get_record_count",
    )


def test_job49_validator_accepts_offline_artifacts_and_has_no_vendor_client(
    tmp_path: Path,
) -> None:
    declaration = _declaration(tmp_path)
    catalogue.validate_catalogue_declaration(declaration)
    module_text = Path(catalogue.__file__).read_text(encoding="utf-8")
    assert "import databento" not in module_text
    assert declaration["network_policy"]["offline_processor_network_calls"] == 0
    assert declaration["local_gate_status"] == "CMBP_CATALOGUE_READY_LOCAL"


def test_selected_64_session_fixture_is_marked_parser_only() -> None:
    repo = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (
            repo
            / "v5/work/lifecycle-training/CMBP_SEMANTIC_GATE_MANIFEST_2026_08_23.json"
        ).read_text(encoding="utf-8")
    )
    contract = json.loads(
        (
            repo / "v5/work/human-policy-foundation/PROGRAM_CONTRACT_V1.json"
        ).read_text(encoding="utf-8")
    )
    declared = contract["evidence_gates"]["C_offline_cmbp_catalogue"][
        "frozen_request_semantics"
    ]
    selected = contract["evidence_gates"]["C_offline_cmbp_catalogue"][
        "selected_semantic_fixture"
    ]
    assert len(fixture["sessions"]) == selected["sessions"] == 64
    assert fixture["session_symbol_pairs"] == selected["session_symbols"] == 248
    assert selected["population_or_economic_inference_forbidden"] is True
    assert declared["source_session_count"] == 1014


def test_no_availability_cost_free_boundary_or_population_claim_is_emitted(
    tmp_path: Path,
) -> None:
    declaration = _declaration(tmp_path)
    serialized = json.dumps(declaration, sort_keys=True).lower()
    assert "free boundary is" not in serialized
    assert "availability confirmed" not in serialized
    assert declaration["status_meaning"].endswith(
        "download, acquisition, or outcome access is authorized."
    )
    assert declaration["claim_policy"]["zero_price_boundary"] == "UNKNOWN"
    assert declaration["claim_policy"]["population_event_prevalence"] == "UNKNOWN"
