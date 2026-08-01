"""Independent mechanical validation for the scoped-round FT2-08 packet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from v4.model.protocol101_canonical_stage1_contract import (
    FEATURE_NAMES,
    feature_matrix,
)


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
AUTHORITY = REPO / (
    "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
AUTHORITY_HASH = (
    "1d215845cf7b853550c5cf27af5bafca66db2355e0f12493e2c5a8922278d4bc"
)
LAW_PRODUCT_CONTRACT_HASH = (
    "2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a"
)
FT204 = REPO / "v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze"
FT205 = REPO / "v4/audit/autoresearch/protocol101_ft2_05_opportunity_census"
CROSSWALK = REPO / (
    "v4/audit/autoresearch/protocol101_ft2_final_repair_attempt002/"
    "findings_crosswalk.json"
)
INTENT_LAW_HASH = (
    "5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name: str) -> dict[str, Any]:
    payload = json.loads((ROOT / name).read_text())
    assert isinstance(payload, dict)
    return payload


def main() -> int:
    checks: dict[str, bool] = {}
    checks["authority_hash"] = sha256(AUTHORITY) == AUTHORITY_HASH

    ft204_receipt = json.loads((FT204 / "receipt.json").read_text())
    ft205_receipt = json.loads((FT205 / "receipt.json").read_text())
    checks["ft204_v2"] = (
        ft204_receipt["outcome"] == "labels_frozen_v2"
        and ft204_receipt["product_contract_hash"] == AUTHORITY_HASH
    )
    checks["ft205_current"] = (
        ft205_receipt["schema_version"] == "Protocol101FT205NodeReceiptV5"
        and ft205_receipt["product_contract_hash"] == AUTHORITY_HASH
        and ft205_receipt["session_count"] == 45
    )
    intent_law = load("intent_fill_recheck_law.json")
    checks["intent_law_hash"] = (
        sha256(ROOT / "intent_fill_recheck_law.json") == INTENT_LAW_HASH
        and intent_law["product_contract_hash"] == LAW_PRODUCT_CONTRACT_HASH
    )

    tensor = load("tensor_schema.json")
    market = tensor["market_history"]["values"]["feature_order"]
    contract = tensor["contract_path_core"]["feature_order"]
    alpha_union = set(market[:-2]) | set(contract[2:])
    checks["signed_17_exact_once_by_axis"] = (
        alpha_union == set(FEATURE_NAMES)
        and len(market[:-2]) + len(contract[2:]) == len(FEATURE_NAMES)
    )
    checks["tensor_shapes"] = (
        tensor["market_history"]["values"]["shape"] == [90, 11]
        and tensor["contract_path_core"]["shape"] == [90, 42, 10]
        and tensor["ladder"]["action_count"] == 43
    )
    checks["contract_dependent_features_not_global"] = not (
        set(FEATURE_NAMES[9:]) & set(market)
    )
    semantics = load("field_semantics_manifest.json")
    legacy_crosswalk = semantics["ft2_20_legacy_field_crosswalk"]
    checks["all_19_plus_4_reviewed_fields_crosswalked"] = (
        len(legacy_crosswalk["legacy_market_history_19"]) == 19
        and len(legacy_crosswalk["legacy_contract_path_core_4"]) == 4
        and legacy_crosswalk["coverage"]["unmapped_fields"] == 0
        and legacy_crosswalk["coverage"][
            "signed_alpha_channels_after_deduplication"
        ]
        == 17
        and "No contract-free E-family Greek is permitted"
        in legacy_crosswalk["global_e_family_resolution"]
    )
    checks["owner_signed_extension_required"] = (
        "owner-signed synchronization amendment hash"
        in tensor["admitted_price_extension"]["activation_requires_all"]
    )
    checks["complete_ladder_wait"] = (
        tensor["flat_state_masks"]["action_mask"]["incomplete_ladder_rule"]
        == "all 42 BUY actions false; WAIT true"
    )
    checks["open_contract_outside_ladder"] = (
        tensor["open_state"]["dedicated_open_contract_record"][
            "outside_current_ladder"
        ]
        == "record remains required and valid"
    )
    checks["terminal_minute_not_learned_action"] = (
        tensor["open_state"]["last_learned_exit_decision_et"] == "15:54"
        and tensor["open_state"]["last_learned_exit_fill_et"] == "15:55"
        and tensor["open_state"]["actions_at_1555"] == []
        and tensor["open_state"]["terminal_minute_1555"]["model_invoked"] is False
    )

    labels = load("label_join_spec.json")
    checks["target_valid_firewalled"] = (
        labels["runtime_label_firewall"]["target_valid_available_to_composer"]
        is False
        and labels["runtime_label_firewall"][
            "historical_action_without_labels_partition"
        ]
        == "must be bit-identical"
    )
    checks["tplus1_and_no_bid"] = (
        labels["timing"]["entry_fill_time"] == "e=t+1"
        and labels["timing"]["exit_fill_time"] == "x=v+1"
        and labels["no_bid"]["primary"]
        == "full-loss state with option value zero"
    )

    folds = load("fold_roles.json")
    checks["role_firewall"] = (
        folds["census"]["session_count"] == 45
        and folds["census"]["outer_test_union_intersection_count"] == 0
        and folds["census"]["protected_holdout_intersection_count"] == 0
        and folds["census"]["embargo_intersection_count"] == 0
    )

    account = load("account_state_ledger_spec.json")
    checks["cent_ledger_and_fee_path_soft_close"] = (
        account["numeric_unit"] == "integer USD cents"
        and account["d49"]["fee_path_floor_cost_cents"]["fee_3"] == 10300
        and account["d49"]["fee_path_floor_cost_cents"]["fee_4"] == 10400
        and account["d49"]["shared_hard_coded_threshold_forbidden"] is True
        and account["d48"]["future_quote_in_intent_mask"] is False
    )
    checks["deterministic_session_start_equity"] = (
        account["session_start"]["canonical_frozen_timestamp"]
        == "session date 09:30:00 America/New_York"
        and account["session_start"]["live_paper_value"][
            "maximum_source_age_at_freeze_seconds"
        ]
        == 60
        and "block all new entries"
        in account["session_start"]["live_paper_value"][
            "missing_stale_ambiguous_or_late"
        ]
    )
    replay = load("replay_authority_v5_1_spec.json")
    checks["separate_fee_trajectories"] = (
        "separate complete causal replay"
        in replay["fee_trajectories"]["four_dollar"]
        and replay["base_simulator"]["modified_by_ft2_08_repair"] is False
    )

    crosswalk = json.loads(CROSSWALK.read_text())
    findings = crosswalk["findings"]
    checks["all_findings_crosswalked"] = (
        len(findings) == 47
        and all(item["verdict"] == "ANSWERED" for item in findings)
        and sum(item["finding_class"] == "fresh" for item in findings) == 19
        and sum(item["finding_class"] == "original" for item in findings) == 28
    )

    golden = load("synthetic_golden_vectors.json")
    vector = golden["vectors"][0]
    offsets = np.arange(-50, 51, 5, dtype=float)
    option_ladder = np.empty((21, 2, 1), dtype=float)
    option_ladder[:, 0, 0] = 10.0 - 0.2 * (offsets / 5.0)
    option_ladder[:, 1, 0] = 9.0 + 0.2 * (offsets / 5.0)
    row = {
        "feature_names": ("mid",),
        "market_feature_names": (
            "spx_close",
            "spx_vwap",
            "omar",
            "session_range",
            "momentum_5m",
            "momentum_15m",
        ),
        "option_ladder": option_ladder,
        "strike_offsets": offsets,
        "market_window": np.asarray([[6000, 5990, 4, 50, 6, -12]], dtype=float),
        "rights": ("C", "P"),
        "decision_time": "2025-01-02T15:00:00Z",
        "atm_strike": 6000,
    }
    actual = feature_matrix(row)
    index = {name: position for position, name in enumerate(FEATURE_NAMES)}
    expected_call = vector["expected_call_contract_channels"]
    expected_put = vector["expected_put_contract_channels"]
    checks["golden_feature_vector"] = all(
        np.isclose(actual[10, 0, index[name]], value, atol=1e-12, rtol=0.0)
        for name, value in expected_call.items()
    ) and all(
        np.isclose(actual[10, 1, index[name]], value, atol=1e-12, rtol=0.0)
        for name, value in expected_put.items()
    )
    intent_vector = next(
        item
        for item in golden["vectors"]
        if item["id"] == "intent_pass_fill_recheck_rejects_without_charge"
    )
    intent_input = intent_vector["input"]
    intent_expected = intent_vector["expected"]
    intent_cost = (
        int(intent_input["A_t_quote_cents"]) * 100
        + int(intent_input["fee_cents"])
    )
    fill_cost = (
        int(intent_input["A_tplus1_quote_cents"]) * 100
        + int(intent_input["fee_cents"])
    )
    checks["golden_intent_fill_recheck"] = (
        intent_cost == intent_expected["intent_cost_cents"]
        and fill_cost == intent_expected["fill_cost_cents"]
        and intent_cost <= int(intent_input["daily_budget_cents"])
        and fill_cost > int(intent_input["daily_budget_cents"])
        and intent_expected["position_opened"] is False
        and intent_expected["fee_charged_cents"] == 0
    )
    soft_close_vector = next(
        item
        for item in golden["vectors"]
        if item["id"] == "d49_soft_close_fee_path_and_ladder_boundary"
    )
    remaining = (
        int(soft_close_vector["input"]["daily_budget_cents"])
        - int(soft_close_vector["input"]["realized_loss_cents"])
    )
    checks["golden_fee_path_soft_close"] = (
        remaining == soft_close_vector["expected"]["remaining_budget_cents"]
        and remaining
        == soft_close_vector["expected"]["fee_3"]["computed_floor_cost_cents"]
        and soft_close_vector["expected"]["fee_3"]["soft_close"] is False
        and remaining
        < soft_close_vector["expected"]["fee_4"]["computed_floor_cost_cents"]
        and soft_close_vector["expected"]["fee_4"]["soft_close"] is True
    )

    failed = sorted(name for name, passed in checks.items() if not passed)
    output = {
        "schema_version": "Protocol101FT208ContractValidationV4",
        "passed": not failed,
        "checks": checks,
        "failed": failed,
        "input_hashes": {
            name: sha256(ROOT / name)
            for name in (
                "account_state_ledger_spec.json",
                "field_semantics_manifest.json",
                "fold_roles.json",
                "identity_mapping_spec.json",
                "intent_fill_recheck_law.json",
                "label_join_spec.json",
                "replay_authority_v5_1_spec.json",
                "storage_spec.json",
                "synthetic_golden_vectors.json",
                "tensor_schema.json",
            )
        },
    }
    (ROOT / "validation.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    if failed:
        raise SystemExit(f"FT2-08 contract validation failed: {failed}")
    print(json.dumps({"passed": True, "check_count": len(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
