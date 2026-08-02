"""Path-D Tier-S entry/exit research constants and preregistration freeze.

This module is deliberately quarantined from live, paper, registry, and promotion
code.  It may read only the owner-provided historical corpus and repository inputs.
The first callable stage freezes a complete preregistration before any fitting.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, ClassVar, Iterable, Mapping
from xml.etree import ElementTree

from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES
from v4.model.protocol101_walking_skeleton import HGBConfig
from v4.research.pathd_feature_live_twin import (
    ENTRY17_LIVE_TWIN_INVENTORY,
    EXIT47_CORRECTED_FEATURE_NAMES,
    EXIT47_CORRECTED_LIVE_TWIN_INVENTORY,
    EXIT48_INTERMEDIATE_FEATURE_NAMES,
    EXIT48_INTERMEDIATE_LIVE_TWIN_INVENTORY,
    EXIT49_FEATURE_NAMES,
    EXIT49_LIVE_TWIN_INVENTORY,
    INVENTORY_SCHEMA_VERSION,
    DROP_UNTIL_EXACT_ADAPTER_RECEIPT,
    NO_INTRADAY_LIVE_TWIN,
    live_twin_record,
    validate_inventory_contracts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    REPO_ROOT
    / "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_PATHD_ENTRY_EXIT_MODEL_PLAN_2026_08_01.md"
)
OWNER_DECISIONS_PATH = (
    REPO_ROOT
    / "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_PATHD_ENTRY_EXIT_OWNER_DECISIONS_2026_08_01.json"
)
OWNER_DECISIONS_SHA256 = (
    "8d9078e5549776b6a92b018cec7f01a89b7a8d290359900aafb62760771d7c86"
)
CORPUS_ROOT = Path.home() / ".autoresearch-trading/pathd_2025-08-01_2026-07-31"
INTEGRITY_MANIFEST_PATH = (
    REPO_ROOT
    / "v4/audit/autoresearch/protocol101_pathd_data_acquisition/integrity_sha256.jsonl"
)
CORRECTION_PROPOSAL_PATH = (
    REPO_ROOT
    / "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_PATHD_BUILD_CORRECTION_PROPOSAL_2026_08_01.md"
)
TERMINAL_RULE_RECONCILIATION_PATH = (
    REPO_ROOT
    / "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_PATHD_TERMINAL_RULE_RECONCILIATION_AND_AREF_BLOCKER_2026_08_01.md"
)
CORRECTED_V3_OWNER_AUTHORIZATION_PATH = (
    REPO_ROOT
    / "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_PATHD_CORRECTED_V3_OWNER_AUTHORIZATION_2026_08_01.json"
)
TERMINAL_RULE_RECONCILIATION_SHA256 = (
    "194ebc35df8018e6f8290b869da4fbcfb785161028d40d1502e094efa1bd4bc0"
)
CORRECTED_V3_OWNER_AUTHORIZATION_SHA256 = (
    "dc3d6b70a491b0e764f18cbf93d8cdeed1295c517ff94b7f98fcb6850bef9038"
)
FIELD_PARITY_FINDING_PATH = (
    REPO_ROOT
    / "v4/audit/autoresearch/protocol101_pathd_feature_research/"
    "FINDING_databento_live_field_parity_2026_08_01.md"
)
ENTRY_MICROSTRUCTURE_FINDING_PATH = (
    REPO_ROOT
    / "v4/audit/autoresearch/protocol101_pathd_feature_research/"
    "FINDING_entry_microstructure_signal_2026_08_01.md"
)
SUPERSEDED_AUDIT_ROOT = (
    REPO_ROOT
    / "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_2026_08_01"
)
INTERMEDIATE_CORRECTED_AUDIT_ROOT = (
    REPO_ROOT
    / "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_corrected_2026_08_01"
)
PREVIOUS_CORRECTED_V2_AUDIT_ROOT = (
    REPO_ROOT
    / "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_corrected_v2_2026_08_01"
)
TEST_CONTAMINATION_QUARANTINE_ROOT = (
    REPO_ROOT
    / "v4/audit/autoresearch/pathd_test_contamination_quarantine_2026_08_01"
)
AUDIT_ROOT = (
    REPO_ROOT
    / "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_corrected_v3_2026_08_01"
)
PREREG_PATH = AUDIT_ROOT / "preregistration.json"
PREREG_HASH_PATH = AUDIT_ROOT / "preregistration.sha256"
SESSION_PATH = AUDIT_ROOT / "session_assignments.json"
LINEAGE_PATH = AUDIT_ROOT / "feature_lineage.json"
FREEZE_RECEIPT_PATH = AUDIT_ROOT / "preregistration_freeze_receipt.json"
LINEAGE_IMPLEMENTATION_RECEIPT_PATH = (
    AUDIT_ROOT / "feature_lineage_implementation_receipt.json"
)
ENTRY_MACHINERY_RECEIPT_PATH = AUDIT_ROOT / "entry_machinery_receipt.json"
ENTRY_MACHINERY_TEST_EVIDENCE_PATH = AUDIT_ROOT / "entry_machinery_test_evidence.json"
ENTRY_MACHINERY_JUNIT_PATH = AUDIT_ROOT / "entry_machinery_frozen_gate.junit.xml"
ENTRY_PREREG_SUPPLEMENTAL_JUNIT_PATH = (
    AUDIT_ROOT / "entry_prereg_supplemental.junit.xml"
)
ENTRY_EVIDENCE_GATE_JUNIT_PATH = AUDIT_ROOT / "entry_evidence_gate.junit.xml"
PROTECTED_HOLDOUT_GATE_JUNIT_PATH = AUDIT_ROOT / "protected_holdout_gate.junit.xml"
FIXED_SCIENCE_GATE_JUNIT_PATH = AUDIT_ROOT / "fixed_science_gate.junit.xml"
FEATURE_LIVE_TWIN_JUNIT_PATH = AUDIT_ROOT / "feature_live_twin.junit.xml"
EVIDENCE_CAPABILITY_HARDENING_JUNIT_PATH = (
    AUDIT_ROOT / "evidence_capability_hardening.junit.xml"
)
FOUNDATION_CORRECTION_JUNIT_PATH = AUDIT_ROOT / "foundation_correction.junit.xml"
CORPUS_INTEGRITY_RECEIPT_PATH = AUDIT_ROOT / "corpus_integrity_verification_receipt.json"
FOUNDATION_RESTORATION_RECEIPT_PATH = (
    AUDIT_ROOT / "foundation_restoration_receipt.json"
)
FOUNDATION_STABILITY_RECEIPT_PATH = AUDIT_ROOT / "foundation_stability_receipt.json"
CONTEXT_DIAGNOSTIC_INVENTORY_RECEIPT_PATH = (
    AUDIT_ROOT / "entry_context_diagnostic_inventory_receipt.json"
)
ENTRY_FOLD_ARTIFACT_ROOT = AUDIT_ROOT / "entry_outer_folds"
ENTRY_POOLED_ACCEPTANCE_RESULT_PATH = AUDIT_ROOT / "entry_pooled_acceptance_result.json"
ENTRY_POOLED_ACCEPTANCE_RECEIPT_PATH = AUDIT_ROOT / "entry_pooled_acceptance_receipt.json"
ENTRY_CONTEXT_DIAGNOSTICS_AUTHORIZATION_PATH = (
    AUDIT_ROOT / "entry_context_diagnostics_authorization.json"
)
ENTRY_CONTEXT_DIAGNOSTICS_ACCESS_RECEIPT_PATH = (
    AUDIT_ROOT / "entry_context_diagnostics_access_receipt.json"
)
ENTRY_POOLED_CONTEXT_DIAGNOSTICS_PATH = (
    AUDIT_ROOT / "entry_pooled_context_diagnostics.json"
)
ENTRY_POOLED_CONTEXT_DIAGNOSTICS_RECEIPT_PATH = (
    AUDIT_ROOT / "entry_pooled_context_diagnostics_receipt.json"
)
ENTRY_CONTEXT_DIAGNOSTICS_ABORT_RECEIPT_PATH = (
    AUDIT_ROOT / "entry_context_diagnostics_abort_receipt.json"
)
PRE_HOLDOUT_PACKET_PATH = AUDIT_ROOT / "complete_pre_holdout_packet.json"
PROTECTED_HOLDOUT_ROOT = AUDIT_ROOT / "protected_holdout"
HOLDOUT_LOCK_PATH = PROTECTED_HOLDOUT_ROOT / ".transaction.lock"
HOLDOUT_ACCESS_RECEIPT_PATH = PROTECTED_HOLDOUT_ROOT / "holdout_access_receipt.json"
HOLDOUT_RESULT_PATH = PROTECTED_HOLDOUT_ROOT / "holdout_result.json"
HOLDOUT_SEAL_RECEIPT_PATH = PROTECTED_HOLDOUT_ROOT / "holdout_seal_receipt.json"
HOLDOUT_ABORT_RECEIPT_PATH = PROTECTED_HOLDOUT_ROOT / "holdout_abort_receipt.json"
HOLDOUT_TRACE_PATH = PROTECTED_HOLDOUT_ROOT / "holdout_trace.jsonl"
HOLDOUT_EVALUATOR_RECEIPT_PATH = (
    PROTECTED_HOLDOUT_ROOT / "holdout_evaluator_receipt.json"
)

# Raw byte bindings for the superseded generation are evidence, never templates to
# copy or mutate.  The corrected generation preserves this history while restoring
# a distinct five-fold namespace.  The burn's semantic receipt hash is separately
# bound because its canonical file hash includes the trailing newline.
SUPERSEDED_FOUNDATION_FILE_BINDINGS = (
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_2026_08_01/feature_lineage.json",
        "bytes": 28_916,
        "sha256": "1534bd8784715e257e6a89701b05e36512f2dfaf66cc50cfdb2fa4d3506f970a",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_2026_08_01/preregistration.json",
        "bytes": 415_531,
        "sha256": "c811acf57f8d57c9067e9534314957da64b3a1f3860b877c3cdc2a842fcdf60a",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_2026_08_01/preregistration.sha256",
        "bytes": 87,
        "sha256": "3928fd34b4f19d1f2acd7af453e3db168bb5c6b29e122761fea85dce411c3ef7",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_2026_08_01/preregistration_freeze_receipt.json",
        "bytes": 1_962,
        "sha256": "27937f6a923d7de58e33049149f3f24a2da3ca33fd784b1140f8af5fbdb916ac",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_2026_08_01/session_assignments.json",
        "bytes": 183_209,
        "sha256": "8455127b950269e966f378790ff77b2c3d7f8565518c2ad581fbd9dfc288f9a2",
    },
)

# The first corrected generation is also immutable history.  It lawfully
# restored five folds and dropped OI, but its parity audit still described the
# signed-17 option ladder as CBBO-1s and retained minute volume without a real
# shared adapter receipt.  The v2 generation supersedes it rather than mutating
# its frozen files in place.
INTERMEDIATE_CORRECTED_FOUNDATION_FILE_BINDINGS = (
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_2026_08_01/feature_lineage.json",
        "bytes": 116_912,
        "sha256": "0cd8059141074a9fb35e5c64c68a3c515cebb3e0b48ae0bbbeab6242b9de5adb",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_2026_08_01/foundation_restoration_receipt.json",
        "bytes": 5_129,
        "sha256": "20e32e44843976ae3ff1985a05e389f0c1c87f0504c1079b018457e6ca0cacbd",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_2026_08_01/preregistration.json",
        "bytes": 347_371,
        "sha256": "4a01f145abfb2a5da664af4f3de1c122069678dc7c3aa1b953745672f2822ec8",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_2026_08_01/preregistration.sha256",
        "bytes": 87,
        "sha256": "32bc571be0a201d29000ab846139f8c9d1c600cf4c1d11568be41e40fa91a692",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_2026_08_01/preregistration_freeze_receipt.json",
        "bytes": 2_642,
        "sha256": "79958c44a5b47b6db78ca8cfcd7ea5bd650c11ba471f65803df4b6c9db36a1ce",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_2026_08_01/session_assignments.json",
        "bytes": 103_609,
        "sha256": "431cd14879ad6a14b278cb2683e4f3b860eef960e6b74a4f0edb3e620cf82475",
    },
)

# Corrected-v2 is the immediate immutable predecessor.  Corrected-v3 resolves
# the A_ref role contradiction and terminal-rule ambiguity in a new namespace;
# it must never rewrite or regenerate these six files in place.
PREVIOUS_CORRECTED_V2_FOUNDATION_FILE_BINDINGS = (
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_v2_2026_08_01/feature_lineage.json",
        "bytes": 156_333,
        "sha256": "4521b3e62982f672daaba73d98ce5a9ce72df170b083cd40df7dc979983d1ef0",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_v2_2026_08_01/foundation_restoration_receipt.json",
        "bytes": 6_619,
        "sha256": "b7b370a2779900bf72e2bf825ed2fdc94ead246fe3c863daf35261d94da11f88",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_v2_2026_08_01/preregistration.json",
        "bytes": 350_870,
        "sha256": "f5e8ed8b99da21b6a4b1d83cdf3e23729ced1eb0e038da1467b5aa64d833c04e",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_v2_2026_08_01/preregistration.sha256",
        "bytes": 87,
        "sha256": "0acdba512a2705f5dd841c426e5c6af62d79e5baae192654a02a1afbf2ac08e9",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_v2_2026_08_01/preregistration_freeze_receipt.json",
        "bytes": 2_788,
        "sha256": "fddd07870fd8400b71a59c0ccef5f073810580fcffa030bac9dc4969bac76f7d",
    },
    {
        "path": "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_v2_2026_08_01/session_assignments.json",
        "bytes": 103_609,
        "sha256": "431cd14879ad6a14b278cb2683e4f3b860eef960e6b74a4f0edb3e620cf82475",
    },
)
PREVIOUS_CORRECTED_V2_FOUNDATION_GENERATION_SHA256 = (
    "c1a0b31383524197a383281f9fb43526d4a20174b6557def2794b2eb7994b79a"
)
QUARANTINED_TEST_BURN_BINDING = {
    "path": (
        "v4/audit/autoresearch/pathd_test_contamination_quarantine_2026_08_01/"
        "outer_primary_burned_receipt.frozen-gate-transaction.json"
    ),
    "bytes": 601,
    "sha256": "48aa120cbe1c9917bc92206bbeb96bb0662a066a8e5e80f02a385713a7625aeb",
    "receipt_sha256": "5aa970c413d4b963641b076d74294432fd4950a1f45301af3615abe8e29b7d58",
    "transaction_id": "frozen-gate-transaction",
    "status": "BURNED_NO_REOPEN",
    "reason": "FROZEN_FOUNDATION_DRIFT_BEFORE_DECODE",
    "role": "outer_test_primary",
    "outer_fold": 1,
    "access_count": 0,
    "access_receipt_sha256": None,
    "dataset_receipt_sha256": None,
    "result_sha256": None,
    "reopen_permitted": False,
    "holdout_open_count": 0,
}

CORE_INTEGRITY_PARTITION = "CORE_AUTHORITATIVE"
CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION = "POST_ENTRY_CONTEXT_DIAGNOSTICS"
UNCONSUMED_VIX_CACHE_PARTITION = "UNCONSUMED_NONAUTHORITATIVE_VIX_CACHE"

# These grammars, counts, byte totals, and semantic hashes describe the exact
# immutable manifest rows.  A path is loader-authorized only by its partition;
# the aggregate manifest membership alone does not grant read authority.
CORE_INTEGRITY_PATH_GRAMMARS = (
    ("ALIGNED_EXIT_LABELS", r"aligned/exit_labels/pathd_exit_labels_12m\.parquet", 1, 315_473_967, "ceb9544e0eebba10bd3cfac0e0da170d3c7d896901e42cab950f77fee5e2a228"),
    ("ALIGNED_NORMALIZED_BASE", r"aligned/normalized/databento_spxw_0dte_\d{4}-\d{2}-\d{2}\.parquet", 251, 633_034_069, "af4981d31621554ad50d514a861d193a0c0fd097c0abac22d82e74ddb19cbddf"),
    ("ALIGNED_NORMALIZED_OFFICIAL_CONTEXT", r"aligned/normalized/databento_spxw_0dte_\d{4}-\d{2}-\d{2}_official_context\.parquet", 251, 633_034_069, "a6c635f62c53a99b6ec48f8d1f71b479137455703b158e6c5d03577289ee0d87"),
    ("ALIGNED_PROCESSED_MINUTE_ENTRY", r"aligned/processed/minute_entry/\d{4}-\d{2}-\d{2}\.pkl", 251, 4_786_788_428, "7639b3f2fc4450d6f7c00d6e851c178d569110e8b08d58cf599a004524e5c305"),
    ("OPRA_CBBO_1M", r"raw/databento/opra_spxw_cbbo_1m/\d{4}-\d{2}-\d{2}\.cbbo-1m\.parquet", 251, 506_320_442, "1ac909113536e88fc06ff085d16235583874f77f157d5faf883d6d0200b1bcd4"),
    ("OPRA_CBBO_1S", r"raw/databento/opra_spxw_cbbo_1s/\d{4}-\d{2}-\d{2}\.cbbo-1s\.parquet", 251, 14_271_506_482, "3ff6c9d374082d30ea9b3f4a15c7a70002a719f719355271a848392962ab090b"),
    ("OPRA_DEFINITION", r"raw/databento/opra_spxw_definition/\d{4}-\d{2}-\d{2}\.definition\.parquet", 251, 198_519_321, "ed9f227d422471c6c2c1a28b4ece1435882e7d464c79e79ca8917c8b8e154ebb"),
    ("OPRA_OHLCV_1M", r"raw/databento/opra_spxw_ohlcv_1m/\d{4}-\d{2}-\d{2}\.ohlcv-1m\.parquet", 251, 100_788_503, "d63e5df0925c793f37f63b9e40a5c7abdc0b8f5736267a9ede44c18fe959e599"),
    ("OPRA_STATISTICS", r"raw/databento/opra_spxw_statistics/\d{4}-\d{2}-\d{2}\.statistics\.parquet", 251, 11_163_938, "0d7d16430f4b464490b8b600d877bb8e16f249bfca3b31058272238d72c0d2ca"),
    ("RAW_INDEX_SPX_OFFICIAL", r"raw/index/spx_1m/\d{4}-\d{2}-\d{2}\.official_spx\.parquet", 251, 4_692_218, "52dca3985ba3366fa249bd16a1dc7937b56c1f72c9e449d9f3d778cba1ce7c7e"),
    ("RAW_INDEX_SPX_PROXY", r"raw/index/spx_1m/\d{4}-\d{2}-\d{2}\.proxy_es_fut\.parquet", 254, 3_820_638, "676e34743452c3272c2332ca7740571d5c80af09f176651f8ac445181bb9aee2"),
    ("VALIDATION_PROBE", r"validation_probe/pathd_exit_labels_2025-08-01\.parquet", 1, 1_265_480, "f8612793d98bff0eecc41a394692cbf3df5f882f3f4c6676c49f16747f224b3b"),
    ("THETADATA_OFFICIAL_SPX", r"vendor/thetadata/index/spx_1m/\d{4}-\d{2}-\d{2}\.parquet", 251, 4_877_839, "6b893e168dc389b72a077e7ac062ab8480ab3129ecacc938837522b210530738"),
)
CONTEXT_DIAGNOSTIC_INTEGRITY_PATH_GRAMMARS = (
    ("THETADATA_OFFICIAL_VIX", r"vendor/thetadata/index/vix_1m/\d{4}-\d{2}-\d{2}\.parquet", 255, 3_577_015, "d7e2e503f7e34e18cd5439717f3f3d7c3c0d87eb49fd2d22d3fd34a86601c779"),
    ("DATABENTO_GLBX_ES", r"raw/databento/glbx_es_ohlcv_1m/\d{4}-\d{2}-\d{2}\.es_c_0\.ohlcv-1m\.parquet", 261, 3_923_463, "ffe8a8cedc97078efa57c452d1190da253e4495057f33c7301b461d83c82e519"),
    ("DATABENTO_XCBF_VX", r"raw/databento/xcbf_vx_ohlcv_1m/\d{4}-\d{2}-\d{2}\.vx_c_0\.ohlcv-1m\.parquet", 88, 1_343_111, "6e34502fbd1d04fa7d88d5b269b82fe76624d185bf66495ba9c427b2bdf495a8"),
)
UNCONSUMED_VIX_CACHE_PATH_GRAMMARS = (
    ("RAW_INDEX_VIX_OFFICIAL_CACHE", r"raw/index/vix_1m/\d{4}-\d{2}-\d{2}\.official_vix\.parquet", 251, 3_354_634, "c5b8da48d8a62bc3f2355206f50667a5edc176c92dd6a63cb6757298d6870b64"),
    ("RAW_INDEX_VX_PROXY_CACHE", r"raw/index/vix_1m/\d{4}-\d{2}-\d{2}\.proxy_vx_fut\.parquet", 87, 1_309_061, "eb630123437cf67809591f7ebcf2b7cbea3070980c03811eae9eefcfde44714b"),
)

QUARANTINE_LABELS = (
    "tier-s",
    "research",
    "quarantined",
    "not-promotable",
    "no-paper",
    "no-real-money",
)
HOLDOUT_CAVEAT = (
    "The final 30-session firewall is protected from model/economic inspection "
    "but is not pristine from the already published full-corpus aggregate label statistics."
)
CLAIM_BOUNDARY = (
    "Tier-S feasibility evidence for a Path-D entry+exit trader; guarded and "
    "holdout-honest; no promotion/paper/real-money claim."
)
RESEARCH_FORCED_FLAT_REASON_CODES = ("PATHD_RESEARCH_TERMINAL",)
RESEARCH_SESSION_TERMINAL_REASON_CODES = ("PATHD_RESEARCH_SESSION_COMPLETE",)

BURNED_SMOKE_DATES = (
    "2026-06-30",
    "2026-07-01",
    "2026-07-02",
    "2026-07-10",
    "2026-07-13",
    "2026-07-14",
)
OPRA_DEGRADED_DATES = ("2025-10-22", "2026-07-31")
SCHEDULED_SHORT_SESSIONS = ("2025-11-28", "2025-12-24")
GLBX_DIAGNOSTIC_DEGRADED_DATES = (
    "2025-09-17",
    "2025-09-24",
    "2025-11-28",
    "2026-03-16",
    "2026-04-10",
)

GATE_HORIZONS = (10, 20, 45, 90, "remaining_session")
DIAGNOSTIC_HORIZONS = (3, 5)
TARGET_AXES = (
    "mfe_dollars",
    "mfe_return",
    "profit_area_dollars",
    "profit_area_return",
)
ENTRY_HEAD_TARGETS = tuple(
    f"h{horizon}_{axis}" if isinstance(horizon, int) else f"session_{axis}"
    for horizon in GATE_HORIZONS
    for axis in TARGET_AXES
)
ENTRY_FUTURE_PATH_KEYS = (
    "h3", "h5", "h10", "h20", "h45", "h90", "remaining_session"
)

MATCHED_RANDOM_SEEDS = (
    10121001,
    10121002,
    10121003,
    10121004,
    10121005,
    10121006,
    10121007,
    10121008,
)
SHUFFLED_TARGET_SEEDS = MATCHED_RANDOM_SEEDS
EXIT_MATCHED_RANDOM_SEEDS = tuple(value + 20_000 for value in MATCHED_RANDOM_SEEDS)
SESSION_BOOTSTRAP_SEED = 101_950_001
CALIBRATION_BOOTSTRAP_SEED = 101_900_001
INNER_FOLD_COUNT = 5
ENTRY_NEGATIVE_CONTROL_IDS = (
    "CONSTANT",
    "SIGN_REVERSED",
    "TIME_SHIFTED_FEATURES",
    *(f"SHUFFLED_TARGET_{index:02d}" for index in range(1, 9)),
)
ENTRY_REPLAY_CHANNELS = (
    "HOLD_TO_FLAT_FEE3",
    "HOLD_TO_FLAT_FEE4",
    "SHARED_TRANSPARENT_FEE3",
    "SHARED_TRANSPARENT_FEE4",
)
ENTRY_ABLATION_IDS = (
    "REMOVE_CONTEXT_12_KEEP_D_E",
    "REMOVE_D_KEEP_CONTEXT_E",
    "REMOVE_E_KEEP_CONTEXT_D",
    "GEOMETRY_AND_MASKS_ONLY",
)
ENTRY_LARGE_IMPROVEMENT_CHECK_SLUGS = (
    "entry_exit_fill_rates",
    "action_rates_and_retries",
    "fee_slippage_hard_limit_reconciliation",
    "duplicate_identity_audit",
    "feature_label_clock_leakage_audit",
    "protected_holdout_access_audit",
    "concentration_audit",
    "comparator_exposure_occupancy_matching",
)
FROZEN_GATE_TEST_NAMES = (
    "test_public_api_surface_exact",
    "test_signed17_historical_source_neutral_byte_equality",
    "test_signed17_lineage_fail_closed",
    "test_mutate_future_is_causal_and_live",
    "test_frozen_roles_history_and_model_input_boundary",
    "test_shared_fill_law_exact",
    "test_default_simulator_governor_unchanged_and_research_opt_in",
    "test_corpus_hash_precedes_decode",
    "test_result_envelope_and_tamper",
    "test_fit_gate_tamper_matrix",
)
ENTRY_PREREG_SUPPLEMENTAL_TEST_NAMES = (
    "test_pathd_session_assignment_is_exact_and_firewalled",
    "test_pathd_signed_entry_feature_contract_remains_exactly_17",
    "test_pathd_locked_fill_and_floor_laws",
    "test_pathd_all_mandatory_lineage_negative_fixtures_fail_closed",
    "test_pathd_fold_calibration_is_strictly_before_outer_test[0]",
    "test_pathd_fold_calibration_is_strictly_before_outer_test[1]",
    "test_pathd_fold_calibration_is_strictly_before_outer_test[2]",
    "test_pathd_fold_calibration_is_strictly_before_outer_test[3]",
    "test_pathd_fold_calibration_is_strictly_before_outer_test[4]",
    "test_pathd_inner_blocks_are_forward_chaining_and_embargoed",
    "test_pathd_lineage_requires_implementation_receipt_before_fit",
    "test_pathd_semantic_preregistration_validates_and_has_no_wall_clock",
    "test_pathd_evidence_guard_blocks_all_36_firewall_sessions",
    "test_pathd_holdout_guard_is_narrower_for_permitted_schema_audits",
    "test_pathd_freeze_is_idempotent_and_prereg_alone_cannot_fit",
    "test_pathd_existing_freeze_detects_tampered_sidecar",
    "test_pathd_fit_receipt_paths_reject_noncanonical_labels[../outside.py]",
    "test_pathd_fit_receipt_paths_reject_noncanonical_labels[/tmp/outside.py]",
    "test_pathd_fit_receipt_paths_reject_noncanonical_labels[./v4/research/pathd_entry_exit.py]",
    "test_pathd_fit_receipt_paths_reject_noncanonical_labels[v4/research/../research/pathd_entry_exit.py]",
    r"test_pathd_fit_receipt_paths_reject_noncanonical_labels[v4\\research\\pathd_entry_exit.py]",
    "test_pathd_fit_receipt_path_accepts_one_exact_repo_regular_file",
    "test_pathd_closed_fit_roles_resolve_only_frozen_sessions[outer_weights-1-None]",
    "test_pathd_closed_fit_roles_resolve_only_frozen_sessions[outer_calibration-5-None]",
    "test_pathd_closed_fit_roles_resolve_only_frozen_sessions[nested_weights-3-4]",
    "test_pathd_closed_fit_roles_resolve_only_frozen_sessions[nested_calibration-5-4]",
    "test_pathd_closed_fit_roles_resolve_only_frozen_sessions[full_weights-None-None]",
    "test_pathd_closed_fit_roles_resolve_only_frozen_sessions[full_calibration-None-None]",
    "test_pathd_invalid_or_under_calibrated_fit_roles_fail_closed",
    "test_pathd_fit_receipt_rejects_symlinked_source",
    "test_pathd_implementation_receipt_requires_exact_unique_coverage",
    "test_pathd_immutable_source_drift_cannot_be_receipted",
    "test_pathd_production_dependency_closure_rejects_external_escape_imports",
    "test_pathd_production_dependency_closure_rejects_os_process_launches",
    "test_pathd_dependency_closure_allows_offline_imports_and_scans_extensible_tests",
    "test_pathd_test_evidence_rejects_arbitrary_success_command",
    "test_pathd_semantic_validator_rejects_role_or_receipt_contract_drift",
    "test_pathd_result_envelope_fails_closed",
    "test_pathd_stale_or_altered_fit_authorization_is_rejected",
    "test_pathd_owner_locked_operational_contract_tamper_matrix",
    "test_pathd_vix_es_vx_diagnostic_contract_is_closed_and_nonalpha",
    "test_pathd_every_postfreeze_result_carries_exact_holdout_caveat",
    "test_pathd_immutable_pooled_gate_rejects_runner_approved_forged_pass",
)
ENTRY_EVIDENCE_GATE_TEST_NAMES = (
    "test_access_count_zero_to_one_and_decode_precedence",
    "test_concurrent_and_repeated_open_are_impossible",
    "test_invalid_nested_block_seals_skip_without_decode",
    "test_result_is_bound_to_exact_dataset_and_revokes_capability",
    "test_invalid_typed_result_burns_scope",
    "test_later_block_outer_and_diagnostic_order_are_fail_closed",
    "test_future_outer_and_premature_shortened_artifacts_are_rejected",
    "test_shortened_diagnostic_opens_only_after_primary",
    "test_crash_without_complete_result_is_permanently_burned",
    "test_crash_with_durable_unsealed_result_is_permanently_burned",
    "test_access_receipt_is_canonical_nofollow_and_tamper_evident",
    "test_foundation_drift_after_open_burns_before_decode",
    "test_sealed_source_bytes_are_rehashed_without_decode",
)
PROTECTED_HOLDOUT_GATE_TEST_NAMES = (
    "test_access_is_exclusive_durable_and_precedes_decode",
    "test_result_is_bound_and_durably_sealed",
    "test_crash_after_access_burns_and_recovery_never_decodes",
    "test_post_access_validation_failure_releases_live_lease",
    "test_crash_after_durable_result_burns_without_seal_or_reopen",
    "test_decode_failure_aborts_once_and_never_reopens",
    "test_forged_verdict_is_recomputed_and_burned",
    "test_arbitrary_evaluation_object_cannot_enter_seal",
    "test_crash_with_partial_result_stays_corrupt_and_never_decodes",
    "test_symlink_partial_reserved_and_unexpected_paths_burn_fail_closed[broken_access_symlink]",
    "test_symlink_partial_reserved_and_unexpected_paths_burn_fail_closed[partial_access]",
    "test_symlink_partial_reserved_and_unexpected_paths_burn_fail_closed[premature_result]",
    "test_symlink_partial_reserved_and_unexpected_paths_burn_fail_closed[unexpected_file]",
    "test_packet_and_artifact_tamper_block_before_access",
    "test_packet_rejects_declared_pass_or_policy_without_semantic_proof",
    "test_preregistration_revalidation_semantically_validates_corpus_receipt",
    "test_parent_directory_fsync_failure_prevents_access_receipt",
    "test_nonblocking_flock_excludes_a_second_process",
    "test_exclusive_writer_rejects_nonfinite_and_every_occupied_inode",
    "test_opaque_types_reject_reconstruction_and_bad_abort_reason",
)
FIXED_SCIENCE_GATE_TEST_NAMES = (
    "test_authorized_dependency_closure_rejects_unregistered_v4_and_test_imports",
    "test_entry_family_selection_is_globally_frozen_before_any_outer_open",
    "test_neural_entry_fit_requires_frozen_hgb_baseline_for_same_scope",
    "test_entry_composer_scores_wait_and_all_42_actions_under_physical_mask",
    "test_entry_negative_controls_transform_complete_bundles_and_reject_tamper",
    "test_matched_random_schedules_enforce_hash_quota_collision_and_no_redraw",
    "test_calibration_statistics_fixed_vectors_cover_seed_lcb_q10_and_action_deciles",
    "test_nested_and_outer_evidence_scopes_open_and_seal_exactly_once",
    "test_nested_and_outer_results_bind_dataset_example_membership_authorization_and_ledger",
    "test_frame_coverage_visits_every_ordered_example_and_rejects_omitted_eligible_frame",
    "test_entry_policy_evaluation_reconstructs_nonzero_economics_from_full_journal",
    "test_fixed_holdout_trace_reconstructs_synthetic_30_session_result",
    "test_fixed_holdout_trace_rejects_all_payload_identity_partition_and_hash_tamper",
    "test_fixed_holdout_evaluator_rejects_caller_authored_summary",
    "test_preholdout_packet_runs_all_six_semantic_validators_and_rejects_declared_pass",
    "test_aref_decision_critical_support_closure_and_gradient_boundaries",
    "test_aref_hgb_and_neural_topologies_are_exact",
    "test_aref_exit_coverage_identity_and_dependency_tamper_fail_closed",
    "test_context_diagnostics_are_fixed_one_dimensional_post_primary_nonalpha_tables",
)
FEATURE_LIVE_TWIN_TEST_NAMES = (
    "test_exact_signed17_original_exit49_intermediate_exit48_and_corrected_exit47_coverage",
    "test_every_signed17_feature_is_live_derivable_without_changing_signed17",
    "test_entry_option_features_bind_historical_cbbo_1m",
    "test_exit49_has_an_explicit_classification_for_every_feature",
    "test_corrected_exit47_drops_open_interest_and_unproved_minute_volume",
    "test_open_interest_is_classified_prior_day_static_but_absent_from_exit47",
    "test_feature_without_live_twin_fixture_rejects_intraday_open_interest_even_with_adapter",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change0-concrete live adapter]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change1-implementation SHA-256]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change2-receipt SHA-256]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change3-live source]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change4-bar-open]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change5-bar close]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change6-available_at at or before]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change7-unavailable at decision]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change8-age bound]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change9-completed observations]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change10-exact-contract]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change11-90-second]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change12-nonnegative]",
    "test_minute_volume_fails_without_exact_completed_bar_semantics[change13-Cross-session|cross-session]",
    "test_minute_volume_is_conditionally_live_derivable_with_exact_receipt[0-DATABENTO_OPRA_OHLCV_1M_COMPLETED]",
    "test_minute_volume_is_conditionally_live_derivable_with_exact_receipt[0-DATABENTO_OPRA_TRADES_AGGREGATED_TO_1M_COMPLETED]",
    "test_minute_volume_is_conditionally_live_derivable_with_exact_receipt[90-DATABENTO_OPRA_OHLCV_1M_COMPLETED]",
    "test_minute_volume_is_conditionally_live_derivable_with_exact_receipt[90-DATABENTO_OPRA_TRADES_AGGREGATED_TO_1M_COMPLETED]",
)
EVIDENCE_CAPABILITY_HARDENING_TEST_NAMES = (
    "test_untrusted_exact_authorization_cannot_read_or_burn_fixed_scope[detached]",
    "test_untrusted_exact_authorization_cannot_read_or_burn_fixed_scope[wrong_identity]",
    "test_untrusted_exact_authorization_cannot_read_or_burn_fixed_scope[wrong_pid]",
    "test_untrusted_exact_authorization_cannot_read_or_burn_fixed_scope[wrong_hash]",
    "test_untrusted_exact_authorization_cannot_read_or_burn_fixed_scope[inactive]",
    "test_active_capability_foundation_drift_burns_and_releases_transaction",
    "test_scope_id_is_bound_to_frozen_foundation_generation",
)
FOUNDATION_CORRECTION_TEST_NAMES = (
    "test_p1_forensic_contract_is_benign_zero_access_and_restores_five",
    "test_corrected_generation_binds_restoration_and_stability_receipts",
    "test_v3_preserves_corrected_v2_bytes_and_remains_build_only",
    "test_refined_br_requires_exact_five_valid_folds_and_rejects_rescue",
    "test_refined_br_preserves_entry_verdict_on_exit_power_stop",
    "test_foundation_byte_root_changes_on_any_file_or_generation_change",
    "test_stability_seal_requires_all_corrected_fold_namespaces_pristine",
    "test_p2_live_twin_inventory_keeps_signed17_and_corrects_exit47",
    "test_fold_stage_preflight_stops_before_dispatch_on_foundation_mismatch",
    "test_p3_widen_entry_lead_is_forward_only",
    "test_nested_dataset_membership_uses_only_current_authorization_segment",
)
REGISTERED_MACHINERY_TEST_NAMES = (
    *FROZEN_GATE_TEST_NAMES,
    *ENTRY_PREREG_SUPPLEMENTAL_TEST_NAMES,
    *ENTRY_EVIDENCE_GATE_TEST_NAMES,
    *PROTECTED_HOLDOUT_GATE_TEST_NAMES,
    *FIXED_SCIENCE_GATE_TEST_NAMES,
    *FEATURE_LIVE_TWIN_TEST_NAMES,
    *EVIDENCE_CAPABILITY_HARDENING_TEST_NAMES,
    *FOUNDATION_CORRECTION_TEST_NAMES,
)

IMMUTABLE_SOURCE_PATHS_AT_FIT = (
    "v4/__init__.py",
    "v4/docs/protocol101/training/execution/PROTOCOL101_PATHD_ENTRY_EXIT_MODEL_PLAN_2026_08_01.md",
    "v4/docs/protocol101/training/execution/PROTOCOL101_PATHD_ENTRY_EXIT_OWNER_DECISIONS_2026_08_01.json",
    "v4/docs/protocol101/training/execution/PROTOCOL101_PATHD_BUILD_CORRECTION_PROPOSAL_2026_08_01.md",
    "v4/docs/protocol101/training/execution/PROTOCOL101_PATHD_TERMINAL_RULE_RECONCILIATION_AND_AREF_BLOCKER_2026_08_01.md",
    "v4/docs/protocol101/training/execution/PROTOCOL101_PATHD_CORRECTED_V3_OWNER_AUTHORIZATION_2026_08_01.json",
    "v4/audit/autoresearch/protocol101_pathd_feature_research/FINDING_databento_live_field_parity_2026_08_01.md",
    "v4/audit/autoresearch/protocol101_pathd_feature_research/FINDING_entry_microstructure_signal_2026_08_01.md",
    "v4/research/__init__.py",
    "v4/research/pathd_feature_live_twin.py",
    "v4/research/pathd_entry_exit.py",
    "v4/research/pathd_evidence_gate.py",
    "v4/research/pathd_holdout_gate.py",
    "v4/tests/test_pathd_entry_exit_gate_frozen.py",
    "v4/tests/test_pathd_entry_exit_research.py",
    "v4/tests/test_pathd_evidence_gate.py",
    "v4/tests/test_pathd_holdout_gate.py",
    "v4/tests/test_pathd_fixed_science_contract.py",
    "v4/tests/test_pathd_feature_live_twin.py",
    "v4/tests/test_pathd_evidence_gate_capability_hardening.py",
    "v4/tests/test_pathd_foundation_correction.py",
    "v4/model/__init__.py",
    "v4/model/protocol101_canonical_stage1_contract.py",
    "v4/model/protocol101_regimen_repair.py",
    "v4/model/protocol101_walking_skeleton.py",
    "v4/path_d/__init__.py",
    "v4/path_d/contracts/__init__.py",
    "v4/path_d/contracts/_base.py",
    "v4/path_d/contracts/broker_state.py",
    "v4/path_d/contracts/execution_event.py",
    "v4/path_d/contracts/execution_intent.py",
    "v4/path_d/contracts/executor_port.py",
    "v4/path_d/contracts/fixtures/execution_intent_v1.json",
    "v4/path_d/contracts/feature_snapshot.py",
    "v4/path_d/contracts/governor_decision.py",
    "v4/path_d/contracts/market_event.py",
    "v4/path_d/execution/__init__.py",
    "v4/path_d/execution/latency.py",
    "v4/path_d/execution/simulated.py",
    "v4/path_d/risk/governor.py",
    "v4/path_d/execution/state_machine.py",
    "v4/path_d/risk/__init__.py",
    "v4/audit/autoresearch/protocol101_pathd_data_acquisition/acquisition_manifest.json",
    "v4/audit/autoresearch/protocol101_pathd_data_acquisition/integrity_sha256.jsonl",
    "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/controls_spec.json",
    "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json",
    "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/matched_random_generator_spec.json",
    "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/objective_spec.json",
    "v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/label_spec.json",
)

# These paths are frozen as the only implementation surface that may authorize an
# entry fit.  Some do not exist at preregistration time by design; the build must
# create them and bind every current byte hash in the post-build machinery receipt.
LINEAGE_IMPLEMENTATION_PATHS = (
    "v4/research/pathd_entry_features.py",
)
MACHINERY_IMPLEMENTATION_PATHS = (
    "v4/research/pathd_entry_dataset.py",
    "v4/research/pathd_entry_models.py",
    "v4/path_d/execution/research_fill_law.py",
    "v4/path_d/execution/research_replay.py",
    "v4/scripts/run_pathd_entry_exit_research.py",
)
REQUIRED_ENTRY_FIT_IMPLEMENTATION_PATHS = (
    *LINEAGE_IMPLEMENTATION_PATHS,
    *MACHINERY_IMPLEMENTATION_PATHS,
)
AUTHORIZED_IN_REPO_DEPENDENCY_PATHS = tuple(
    dict.fromkeys((*IMMUTABLE_SOURCE_PATHS_AT_FIT, *REQUIRED_ENTRY_FIT_IMPLEMENTATION_PATHS))
)
AUTHORIZED_RUNTIME_DEPENDENCY_PATHS = tuple(
    path for path in AUTHORIZED_IN_REPO_DEPENDENCY_PATHS if not path.startswith("v4/tests/")
)

# Exact direct import roots permitted in the frozen Path-D runtime. This is an
# import/source-indirection boundary, not an operating-system network sandbox;
# corpus/data validators separately require canonical local, hash-sealed files.
# Repo-local execution is still independently constrained by
# ``AUTHORIZED_RUNTIME_DEPENDENCY_PATHS`` and its transitive closure below.
FROZEN_OFFLINE_IMPORT_ROOTS = (
    "__future__",
    "argparse",
    "ast",
    "contextlib",
    "dataclasses",
    "datetime",
    "enum",
    "errno",
    "fcntl",
    "hashlib",
    "joblib",
    "json",
    "math",
    "numpy",
    "os",
    "pandas",
    "pathlib",
    "platform",
    "pyarrow",
    "random",
    "re",
    "scipy",
    "sklearn",
    "stat",
    "sys",
    "threadpoolctl",
    "torch",
    "types",
    "typing",
    "uuid",
    "v4",
    "xml",
    "zoneinfo",
)
FROZEN_EXTENSIBLE_TEST_IMPORT_ROOTS = (
    "copy",
    "dataclasses",
    "hashlib",
    "inspect",
    "json",
    "pathlib",
    "pytest",
    "types",
    "typing",
    "v4",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def canonical_session_hash(sessions: Iterable[str]) -> str:
    """Hash one ordered session identity per line, including the final newline."""

    values = [str(value) for value in sessions]
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def forward_inner_folds(
    model_fit_sessions: list[str], *, count: int = INNER_FOLD_COUNT
) -> dict[str, Any]:
    """Freeze five blocks and four expanding scores with one boundary embargo."""

    values = list(model_fit_sessions)
    if count != 5:
        raise ValueError("Path-D preregistration freezes exactly five inner blocks")
    quotient, remainder = divmod(len(values), count)
    sizes = [quotient + (1 if index < remainder else 0) for index in range(count)]
    blocks: list[list[str]] = []
    cursor = 0
    for size in sizes:
        blocks.append(values[cursor : cursor + size])
        cursor += size
    if cursor != len(values) or min(sizes) < 2:
        raise ValueError("insufficient chronological history for frozen inner blocks")
    folds: list[dict[str, Any]] = []
    for index in range(1, count):
        earlier = [session for block in blocks[:index] for session in block]
        train = earlier[:-1]
        embargo = earlier[-1:]
        validation = blocks[index]
        if not train or not validation or not max(train) < embargo[0] < min(validation):
            raise AssertionError("inner-fold chronology drifted")
        nested_calibration_count = int(math.ceil(len(train) * 0.20))
        nested_calibration = train[-nested_calibration_count:]
        nested_before_calibration = train[:-nested_calibration_count]
        nested_calibration_embargo = nested_before_calibration[-1:]
        nested_model_fit = nested_before_calibration[:-1]
        if not nested_model_fit or not nested_calibration_embargo:
            raise AssertionError("inner nested calibration history is empty")
        folds.append(
            {
                "inner_fold": index,
                "available_earlier_history": train,
                "model_fit": nested_model_fit,
                "calibration_embargo": nested_calibration_embargo,
                "calibration": nested_calibration,
                "calibration_minimum_sessions": 10,
                "calibration_valid": len(nested_calibration) >= 10,
                "outer_validation_embargo": embargo,
                "validation": validation,
                "model_fit_sha256_newline": canonical_session_hash(nested_model_fit),
                "calibration_embargo_sha256_newline": canonical_session_hash(
                    nested_calibration_embargo
                ),
                "calibration_sha256_newline": canonical_session_hash(nested_calibration),
                "outer_validation_embargo_sha256_newline": canonical_session_hash(embargo),
                "validation_sha256_newline": canonical_session_hash(validation),
            }
        )
    return {
        "partition_rule": "five contiguous blocks; sizes differ by <=1; remainder assigned to earlier blocks",
        "block_1_role": "warmup_no_score",
        "blocks": blocks,
        "block_sizes": sizes,
        "scored_forward_folds": folds,
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )


def read_json(path: Path) -> Any:
    return strict_json_loads(path.read_text(encoding="utf-8"))


def validated_owner_authorization() -> dict[str, Any]:
    """Load the one exact owner decision/execution receipt bound to this run."""

    expected = {
        "schema_version": "pathd.entry_exit.owner_decisions.v1",
        "status": "OWNER_CONFIRMED_BEFORE_RESEARCH_PREREGISTRATION",
        "recorded_date": "2026-08-01",
        "binding_plan_path": (
            "v4/docs/protocol101/training/execution/"
            "PROTOCOL101_PATHD_ENTRY_EXIT_MODEL_PLAN_2026_08_01.md"
        ),
        "binding_plan_sha256": (
            "6b694226d0fbb84c5b4ae2cc05009daf5d847d2d94f1bcf38923cfb622be4ec9"
        ),
        "authority": (
            "The active user-created Codex execution goal is the separately authorized "
            "build/train goal contemplated by the earlier plan-only instruction. It "
            "authorizes this quarantined Tier-S research run and resolves the five open "
            "questions in section 10 of the reviewed plan."
        ),
        "authorization_source": {
            "kind": "active_user_created_codex_goal",
            "thread_id": "019fbd34-55cc-7563-b571-c16788cff353",
            "objective_title": (
                "EXECUTE THE PATH-D ENTRY+EXIT MODEL PLAN — Tier-S research build + "
                "feasibility gate"
            ),
            "objective_scope": (
                "Build, fit, validate, conditionally run four-box replay, and "
                "conditionally open the protected holdout exactly once under the "
                "binding plan and owner locks."
            ),
            "relationship_to_plan_only_instruction": (
                "This is the later separate build/train goal expressly deferred by the "
                "earlier plan-only instruction."
            ),
        },
        "scope": {
            "research_only": True,
            "quarantined_not_promotable": True,
            "governance_contract": False,
            "promotion_paper_real_money_authority": False,
        },
        "execution_authorization": {
            "authorized": True,
            "purpose": (
                "Execute the reviewed Path-D Tier-S entry+exit research build and "
                "feasibility gate under the owner-confirmed decisions in this receipt."
            ),
            "authorized_stages_in_order": [
                "freeze_research_preregistration",
                "build_and_verify_entry_machinery",
                "fit_entry_hgb_then_nonselectable_neural_challenger",
                "freeze_each_fold_specific_oof_entry_artifact_before_its_outer_open",
                "evaluate_complete_entry_gate_from_the_five_frozen_outer_results",
                "fit_and_freeze_full_fit_entry_artifact_only_if_complete_entry_gate_passes",
                "build_exit_trajectories_from_fold_specific_oof_entry_artifacts_only_if_complete_entry_gate_passes",
                "fit_exit_only_if_entry_and_exit_minimum_power_gates_allow",
                "run_four_box_serial_replay_only_if_entry_exit_and_guard_gates_allow",
                "open_protected_holdout_exactly_once_only_if_every_outer_exit_four_box_minimum_power_survival_and_preopen_freeze_gate_passes",
            ],
            "forbidden": [
                "broker_or_ibkr_connectivity",
                "live_market_data_capture",
                "paper_or_real_money_orders",
                "training_registry_runtime_launchd_or_default_changes",
                "Protocol158_or_Protocol160_edits",
                "parent_authority_or_governance_contract_freeze",
                "cmbp_1_or_Tier_T_acquisition",
                "paid_data_downloads",
                "v4_env_access_or_edit",
                "promotion_or_deployment_claim",
            ],
        },
        "decisions": {
            "entry_alpha": "exactly_signed_17_exit_microstructure_only",
            "entry_self_computed_greek_grandfathering": (
                "E.bs.delta and E.bs.gamma are two fields inside the owner-locked "
                "signed17; they remain entry alpha; the exit-only restriction applies "
                "to new enrichment and raw vendor Greeks remain forbidden"
            ),
            "exit_local_horizon_seconds": 300,
            "exit_downside_penalty": 0.25,
            "calibration_level": 0.9,
            "catastrophic_floor": "max(0,0.50*P_entry+3/100)",
            "vix_es_vx": "diagnostics_and_strata_only",
            "holdout_not_pristine_caveat_accepted": True,
        },
        "required_holdout_caveat": HOLDOUT_CAVEAT,
        "claim_boundary": CLAIM_BOUNDARY,
        "terminal_marker": "STOP_FOR_CLAUDE_VERIFICATION",
    }
    if sha256_path(OWNER_DECISIONS_PATH) != OWNER_DECISIONS_SHA256:
        raise ValueError("owner authorization receipt byte hash drift")
    record = read_json(OWNER_DECISIONS_PATH)
    if type(record) is not dict or record != expected:
        raise ValueError("owner authorization receipt semantic drift")
    if record["binding_plan_path"] != repo_path_label(PLAN_PATH):
        raise ValueError("owner authorization plan path drift")
    if record["binding_plan_sha256"] != sha256_path(PLAN_PATH):
        raise ValueError("owner authorization plan hash drift")
    return record


def validated_corrected_v3_owner_authorization() -> dict[str, Any]:
    """Load the exact owner approval for the corrected-v3 foundation-only build."""

    if (
        sha256_path(TERMINAL_RULE_RECONCILIATION_PATH)
        != TERMINAL_RULE_RECONCILIATION_SHA256
    ):
        raise ValueError("terminal-rule reconciliation byte hash drift")
    if (
        sha256_path(CORRECTED_V3_OWNER_AUTHORIZATION_PATH)
        != CORRECTED_V3_OWNER_AUTHORIZATION_SHA256
    ):
        raise ValueError("corrected-v3 owner authorization byte hash drift")
    record = read_json(CORRECTED_V3_OWNER_AUTHORIZATION_PATH)
    if (
        type(record) is not dict
        or record.get("schema_version")
        != "pathd.corrected_v3.owner_authorization.v1"
        or record.get("status") != "OWNER_CONFIRMED_CORRECTED_V3_BUILD_ONLY"
        or record.get("aref_action_calibration_role", {}).get("decision")
        != "ADOPT_WITH_ADVERSARIAL_CONDITIONS"
        or record.get("composite_calibration_terminal_rule", {}).get("decision")
        != "ADOPT_REFINED_B_R"
        or record.get("composite_calibration_terminal_rule", {}).get(
            "failure_precedence"
        )
        != [
            "invalid_result",
            "insufficient_evidence",
            "owner_decision_required",
            "no_genuine_signal",
            "PASS",
        ]
        or record.get("build_permissions")
        != {
            "source_spec_and_test_changes": True,
            "distinct_corrected_v3_preregistration_freeze": True,
            "distinct_corrected_v3_restoration_receipt": True,
            "preserve_corrected_v2_byte_for_byte": True,
            "machinery_seal": False,
            "foundation_stability_seal": False,
            "model_fit": False,
            "corpus_decode": False,
            "nested_or_outer_evidence_open": False,
            "protected_holdout_open": False,
            "live_or_broker_action": False,
        }
        or record.get("required_terminal_marker")
        != "STOP_FOR_CLAUDE_VERIFICATION"
    ):
        raise ValueError("corrected-v3 owner authorization semantic drift")
    return record


def repo_path_label(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _integrity_manifest_entries() -> list[dict[str, Any]]:
    entries = [
        strict_json_loads(line)
        for line in INTEGRITY_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paths = [row.get("relative_path") for row in entries]
    if (
        not entries
        or len(paths) != len(set(paths))
        or paths != sorted(paths)
        or any(
            set(row) != {"relative_path", "bytes", "sha256"}
            or type(row["relative_path"]) is not str
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
            or type(row["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
            for row in entries
        )
    ):
        raise ValueError("Path-D integrity manifest schema/order drift")
    return entries


def _integrity_partition_rows_and_contract(
    entries: list[dict[str, Any]],
    *,
    partition_id: str,
    grammars: tuple[tuple[str, str, int, int, str], ...],
    loader_authority: str,
    expected_file_count: int,
    expected_total_bytes: int,
    expected_semantic_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grammar_contracts: list[dict[str, Any]] = []
    for grammar_id, path_regex, file_count, total_bytes, semantic_sha256 in grammars:
        matched = [
            row
            for row in entries
            if re.fullmatch(path_regex, row["relative_path"]) is not None
        ]
        observed = {
            "grammar_id": grammar_id,
            "path_regex": path_regex,
            "file_count": len(matched),
            "total_bytes": sum(row["bytes"] for row in matched),
            "ordered_entries_semantic_sha256": stable_hash(matched),
        }
        expected = {
            "grammar_id": grammar_id,
            "path_regex": path_regex,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "ordered_entries_semantic_sha256": semantic_sha256,
        }
        if observed != expected:
            raise ValueError(
                f"Path-D integrity grammar drift: {partition_id}:{grammar_id}"
            )
        grammar_contracts.append(observed)
        rows.extend(matched)
    rows.sort(key=lambda row: row["relative_path"])
    if len(rows) != len({row["relative_path"] for row in rows}):
        raise ValueError(f"Path-D integrity grammar overlap: {partition_id}")
    contract = {
        "partition_id": partition_id,
        "loader_authority": loader_authority,
        "path_grammars": grammar_contracts,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "ordered_entries_semantic_sha256": stable_hash(rows),
        "lexicographically_sorted": True,
    }
    if (
        contract["file_count"] != expected_file_count
        or contract["total_bytes"] != expected_total_bytes
        or contract["ordered_entries_semantic_sha256"]
        != expected_semantic_sha256
    ):
        raise ValueError(f"Path-D integrity partition drift: {partition_id}")
    return rows, contract


def integrity_manifest_contract() -> dict[str, Any]:
    entries = _integrity_manifest_entries()
    core_rows, core = _integrity_partition_rows_and_contract(
        entries,
        partition_id=CORE_INTEGRITY_PARTITION,
        grammars=CORE_INTEGRITY_PATH_GRAMMARS,
        loader_authority="CORE_ONLY_FIT_EVIDENCE_HOLDOUT",
        expected_file_count=2_766,
        expected_total_bytes=21_471_285_394,
        expected_semantic_sha256=(
            "31e1abb88b9a5539f9445a63c9af650385f8ded207a94021422a53a364a29497"
        ),
    )
    diagnostic_rows, diagnostics = _integrity_partition_rows_and_contract(
        entries,
        partition_id=CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION,
        grammars=CONTEXT_DIAGNOSTIC_INTEGRITY_PATH_GRAMMARS,
        loader_authority="POST_ENTRY_CONTEXT_DIAGNOSTICS_ONLY",
        expected_file_count=604,
        expected_total_bytes=8_843_589,
        expected_semantic_sha256=(
            "11605b5f8cd88957b5b0227ef45ec8380d80cfbfae16ff0f0111c55857d8ae1c"
        ),
    )
    cache_rows, cache = _integrity_partition_rows_and_contract(
        entries,
        partition_id=UNCONSUMED_VIX_CACHE_PARTITION,
        grammars=UNCONSUMED_VIX_CACHE_PATH_GRAMMARS,
        loader_authority="NEVER_LOADER_AUTHORIZED",
        expected_file_count=338,
        expected_total_bytes=4_663_695,
        expected_semantic_sha256=(
            "61bfb0daf39de84879ee31df9d9023d0563b04c41e36196c91580c35155df56f"
        ),
    )
    partition_paths = [
        {row["relative_path"] for row in core_rows},
        {row["relative_path"] for row in diagnostic_rows},
        {row["relative_path"] for row in cache_rows},
    ]
    if (
        any(partition_paths[left] & partition_paths[right] for left in range(3) for right in range(left + 1, 3))
        or set().union(*partition_paths)
        != {row["relative_path"] for row in entries}
    ):
        raise ValueError("Path-D integrity partitions are not exact and disjoint")
    if (
        sum(partition["file_count"] for partition in (core, diagnostics, cache))
        != len(entries)
        or sum(partition["total_bytes"] for partition in (core, diagnostics, cache))
        != sum(row["bytes"] for row in entries)
    ):
        raise ValueError("Path-D integrity partition arithmetic drift")
    return {
        "manifest_path": repo_path_label(INTEGRITY_MANIFEST_PATH),
        "manifest_sha256": sha256_path(INTEGRITY_MANIFEST_PATH),
        "ordered_entries_semantic_sha256": stable_hash(entries),
        "file_count": len(entries),
        "total_bytes": sum(row["bytes"] for row in entries),
        "unique_relative_paths": True,
        "lexicographically_sorted": True,
        "partitions_in_order": [
            CORE_INTEGRITY_PARTITION,
            CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION,
            UNCONSUMED_VIX_CACHE_PARTITION,
        ],
        "partitions": {
            CORE_INTEGRITY_PARTITION: core,
            CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION: diagnostics,
            UNCONSUMED_VIX_CACHE_PARTITION: cache,
        },
        "partition_rule": (
            "every manifest row matches exactly one frozen path grammar; only the "
            "core receipt gates fit/evidence/holdout, diagnostics require a later "
            "read-only context authorization, and raw/index/vix cache rows are never "
            "loader-authorized"
        ),
    }


def integrity_manifest_entries_for_partition(
    partition_id: str, /
) -> tuple[dict[str, Any], ...]:
    """Return manifest rows for one exact authority partition, never file contents."""

    if partition_id not in {
        CORE_INTEGRITY_PARTITION,
        CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION,
    }:
        raise RuntimeError("Path-D integrity partition is not loader-authorized")
    entries = _integrity_manifest_entries()
    contract = integrity_manifest_contract()["partitions"][partition_id]
    patterns = [row["path_regex"] for row in contract["path_grammars"]]
    selected = tuple(
        dict(row)
        for row in entries
        if any(re.fullmatch(pattern, row["relative_path"]) is not None for pattern in patterns)
    )
    if (
        len(selected) != contract["file_count"]
        or stable_hash(list(selected)) != contract["ordered_entries_semantic_sha256"]
    ):
        raise RuntimeError("Path-D loader partition reconstruction drift")
    return selected


def context_diagnostic_inventory_receipt_spec() -> dict[str, Any]:
    partition = integrity_manifest_contract()["partitions"][
        CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION
    ]
    return {
        "schema_version": "pathd.entry_exit.context_diagnostic_inventory_receipt.v1",
        "status": "PASS_POST_ENTRY_DIAGNOSTIC_INVENTORY",
        "fixed_path": repo_path_label(CONTEXT_DIAGNOSTIC_INVENTORY_RECEIPT_PATH),
        "partition_id": CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION,
        "partition_contract_sha256": stable_hash(partition),
        "fields_in_order": [
            "schema_version", "status", "verification_completed_at_utc",
            "preregistration_sha256", "source_hash_policy_sha256", "command",
            "manifest_sha256", "partition_id", "partition_contract_sha256",
            "outer_result_receipt_sha256s",
            "pooled_acceptance_receipt_sha256", "verified_file_count",
            "verified_total_bytes",
            "verified_ordered_entries_semantic_sha256", "mismatches",
            "unreceipted_paths", "cache_paths_opened",
            "source_decode_performed", "receipt_sha256",
        ],
        "verification": {
            "file_count": 604,
            "total_bytes": 8_843_589,
            "ordered_entries_semantic_sha256": (
                "11605b5f8cd88957b5b0227ef45ec8380d80cfbfae16ff0f0111c55857d8ae1c"
            ),
            "mismatches": [],
            "unreceipted_paths": [],
            "cache_paths_opened": [],
            "source_decode_performed": False,
        },
        "precondition": "all five primary outer result receipts and pooled entry acceptance are frozen and revalidated before hashing; parquet content is not decoded",
        "durability": "canonical JSON O_EXCL|O_NOFOLLOW mode 0600 with file and parent-directory fsync; no overwrite or replacement",
    }


def core_corpus_integrity_receipt_spec() -> dict[str, Any]:
    manifest = integrity_manifest_contract()
    partition = manifest["partitions"][CORE_INTEGRITY_PARTITION]
    return {
        "schema_version": "pathd.entry_exit.corpus_integrity_verification_receipt.v1",
        "status": "PASS_CORE_AUTHORITATIVE",
        "fixed_path": repo_path_label(CORPUS_INTEGRITY_RECEIPT_PATH),
        "partition_id": CORE_INTEGRITY_PARTITION,
        "partition_contract_sha256": stable_hash(partition),
        "fields_in_order": [
            "schema_version", "status", "verification_completed_at_utc",
            "partition_id", "source_hash_policy_sha256",
            "machinery_receipt_sha256", "command", "corpus_root",
            "integrity_contract", "manifest_contract_sha256",
            "verified_file_count", "verified_total_bytes",
            "verified_ordered_entries_semantic_sha256",
            "diagnostic_files_verified", "cache_files_verified", "mismatches",
            "unreceipted_paths", "loader_rehash_required",
            "quarantine_labels", "claim_boundary", "holdout_caveat",
            "plan_sha256", "preregistration_sha256", "fill_law_hash",
            "holdout_open_count", "receipt_sha256",
        ],
        "verification": {
            "file_count": 2_766,
            "total_bytes": 21_471_285_394,
            "ordered_entries_semantic_sha256": (
                "31e1abb88b9a5539f9445a63c9af650385f8ded207a94021422a53a364a29497"
            ),
            "diagnostic_files_verified": 0,
            "cache_files_verified": 0,
            "mismatches": [],
            "unreceipted_paths": [],
            "loader_rehash_required": True,
        },
        "authority": "this receipt alone gates fit/evidence/holdout; it grants no diagnostic or raw/index/vix cache read authority",
    }


def fit_session_role_spec() -> dict[str, str]:
    return {
        "outer_weights": "folds[outer_fold].model_fit",
        "outer_calibration": "folds[outer_fold].calibration_last_20_percent",
        "nested_weights": "folds[outer_fold].inner_forward_folds.scored_forward_folds[inner_fold].model_fit",
        "nested_calibration": "folds[outer_fold].inner_forward_folds.scored_forward_folds[inner_fold].calibration; calibration_valid must be true",
        "full_weights": "full_fit_pre_holdout_bundle.entry_model_fit",
        "full_calibration": "full_fit_pre_holdout_bundle.entry_calibration",
    }


def evidence_session_role_spec() -> dict[str, str]:
    return {
        "nested_validation": "folds[outer_fold].inner_forward_folds.scored_forward_folds[inner_fold].validation; evaluation-only for mandatory HGB-versus-neural challenger attribution and machinery checks, never family selection or outer evidence",
        "outer_test_primary": "folds[outer_fold].outer_test_primary_1555_complete",
        "outer_test_shortened_diagnostic": "folds[outer_fold].outer_test_shortened_diagnostic_only; never economic/action/model-family/comparator evidence",
    }


def result_aggregation_role_spec() -> dict[str, str]:
    return {
        "pooled_outer_primary": "immutable typed primary evaluation/result/trace receipts from outer folds 1..5 in order; raw corpus/dataset loading and replay are forbidden",
    }


def entry_dataset_seal_spec() -> dict[str, Any]:
    return {
        "future_path_keys_in_order": list(ENTRY_FUTURE_PATH_KEYS),
        "future_path_window_rule": "marks_by_horizon has exactly the seven string keys h3,h5,h10,h20,h45,h90,remaining_session in that semantic order. If arrival+N minutes<=15:55, hN contains exactly N integer executable SELL-until-filled mark micros for arrival+1..arrival+N; if unavailable by decision clock it is an empty list and its four targets are invalid without any future read. remaining_session contains exactly floor((15:55-arrival)/60s) marks strictly after arrival through 15:55 and is nonempty for every legal entry. No truncation, endpoint reuse, extra mark, mixed int/string key, or inferred validity is accepted",
        "source_file_row_fields": ["relative_path", "bytes", "sha256"],
        "source_file_rule": "repo-external corpus-relative POSIX path under the fixed corpus root, regular non-symlink file, byte count exact nonnegative int, lowercase SHA-256 exact; rows lexicographically sorted by relative_path, unique, and every hash is streamed/rechecked before decode against the frozen integrity manifest",
        "source_receipt_fields": [
            "session", "source_files", "source_files_sha256",
            "example_count", "ordered_example_list_sha256", "session_content_sha256",
            "receipt_sha256",
        ],
        "source_receipt_rule": "one receipt for every authorization session in identical order and no others; source_files_sha256=stable_hash(source_files); example_count>=1; ordered_example_list_sha256=stable_hash(the ordered list of EntryExampleV1.canonical_sha256 values for that session); session_content_sha256 is the exact EntrySessionV1 content hash below; receipt_sha256=stable_hash of every source-receipt field except receipt_sha256 and is the only value quote/context rows may bind",
        "entry_session_content_digest": "EntrySessionV1.content_sha256 and the matching source receipt session_content_sha256 both equal stable_hash of exactly {schema_version,session,source_files,source_files_sha256,ordered_example_hashes}, where ordered_example_hashes is the canonical ordered list of EntryExampleV1.canonical_sha256 values; the examples objects and content_sha256 field itself are not JSON-serialized directly",
        "example_identity": "(model_input.session,model_input.decision_time_ns,each model_input.contract_ids in canonical action order); session must equal an authorized session; frame identities strictly sort by (session,decision_time_ns), are unique, and each frame contract list is unique/exactly 42",
        "dataset_digest_object_fields": [
            "schema_version", "authorization_sha256", "role", "sessions",
            "sessions_sha256_newline", "source_receipts", "ordered_example_hashes",
        ],
        "dataset_digest": "SHA-256 of UTF-8 compact sorted-key JSON allow_nan=false over exactly the digest object fields, where ordered_example_hashes is the full canonical ordered list of EntryExampleV1.canonical_sha256 values; dataset_sha256 itself and object memory/pickle bytes are excluded",
        "fit_loader": "load_authorized_entry_dataset alone constructs EntryFitDatasetV1; every fit wrapper calls it internally and validates the seal immediately before implementation",
        "evidence_loader": "load_authorized_entry_evidence_dataset alone calls claim_entry_evidence_decode_once exactly once and constructs EntryEvidenceDatasetV1 from FrozenEvidenceAuthorization; validate_entry_evidence_access is repeatable and non-consuming while the live transaction remains active so composer/quote/journal/replay can revalidate the same authorization after decode; primary/diagnostic roles stay distinct",
        "mutation_rule": "cross-session examples, missing/extra/duplicate sessions or files, reordering, any source hash/byte edit, and any model-input/target/validity/audit mutation change or invalidate the dataset seal",
    }


def _outer_fold_artifact_path(outer_fold: int, name: str) -> Path:
    return ENTRY_FOLD_ARTIFACT_ROOT / f"fold_{outer_fold}" / name


def outer_evidence_open_gate_spec() -> dict[str, Any]:
    return {
        "composite_calibration_terminal_rule_sha256": stable_hash(
            composite_calibration_terminal_rule_spec()
        ),
        "outer_calibration_scope_gate_receipts": [
            repo_path_label(
                _outer_fold_artifact_path(
                    fold, "entry_calibration_scope_gate_receipt.json"
                )
            )
            for fold in range(1, 6)
        ],
        "nested_calibration_scope_gate_receipts": [
            repo_path_label(
                _outer_fold_artifact_path(
                    fold,
                    f"nested_inner_{inner}_calibration_scope_gate_receipt.json",
                )
            )
            for fold in range(1, 6)
            for inner in range(1, 5)
        ],
        "preopen_receipts": [
            repo_path_label(_outer_fold_artifact_path(fold, "preopen_receipt.json"))
            for fold in range(1, 6)
        ],
        "nested_preopen_receipts": [
            repo_path_label(
                _outer_fold_artifact_path(fold, f"nested_inner_{inner}_preopen_receipt.json")
            )
            for fold in range(1, 6)
            for inner in range(1, 5)
        ],
        "nested_result_receipts": [
            repo_path_label(
                _outer_fold_artifact_path(fold, f"nested_inner_{inner}_result_receipt.json")
            )
            for fold in range(1, 6)
            for inner in range(1, 5)
        ],
        "nested_skip_receipts": [
            repo_path_label(
                _outer_fold_artifact_path(fold, f"nested_inner_{inner}_skip_receipt.json")
            )
            for fold in range(1, 6)
            for inner in range(1, 5)
        ],
        "nested_access_receipts": [
            repo_path_label(
                _outer_fold_artifact_path(fold, f"nested_inner_{inner}_access_receipt.json")
            )
            for fold in range(1, 6)
            for inner in range(1, 5)
        ],
        "outer_primary_access_receipts": [
            repo_path_label(
                _outer_fold_artifact_path(fold, "outer_primary_access_receipt.json")
            )
            for fold in range(1, 6)
        ],
        "outer_shortened_preopen_receipts": [
            repo_path_label(
                _outer_fold_artifact_path(fold, "shortened_preopen_receipt.json")
            )
            for fold in range(1, 6)
        ],
        "outer_shortened_access_receipts": [
            repo_path_label(
                _outer_fold_artifact_path(fold, "shortened_access_receipt.json")
            )
            for fold in range(1, 6)
        ],
        "outer_shortened_result_receipts": [
            repo_path_label(
                _outer_fold_artifact_path(fold, "shortened_result_receipt.json")
            )
            for fold in range(1, 6)
        ],
        "outer_result_receipts": [
            repo_path_label(_outer_fold_artifact_path(fold, "outer_result_receipt.json"))
            for fold in range(1, 6)
        ],
        "artifact_names": [
            "hgb_model_bundle.json", "hgb_calibration_bundle.json",
            "hgb_composer.json",
            "neural_model_bundle.json", "neural_calibration_bundle.json",
            "neural_composer.json", "control_exit.json",
            "negative_control_manifest.json",
            "control_replay_config.json",
        ],
        "postopen_artifact_names": [
            "negative_control_panel.json", "control_replay_result.json",
            "outer_primary_result.json",
        ],
        "nested_rule": "process inner blocks 1..4 in order. A calibration-valid block has one preopen receipt binding exact HGB+neural bundles/calibrators/composers and replay config, then a durable exclusive access receipt changing its validation_access_count from zero to exactly one before decode, then one result receipt binding the exact sealed dataset, complete journal traces, and paired HGB/neural serial results. A calibration-invalid block has instead exactly one self-hashed INSUFFICIENT_CALIBRATION skip receipt proving no preopen/access/dataset/result exists. Each block requires the correct result-or-skip receipts for every prior block and forbids every artifact/receipt for later blocks. Nested HGB/neural outcomes are mandatory attribution and multiplicity evidence only and can never select or replace the preregistered global HGB candidate",
        "preopen_rule": "before outer fold k primary authorization, every required calibration node for OUTER_k must be VALID and a write-once calibration-scope gate receipt must bind the exact ordered node manifest with evidence_access_count=0; both HGB and mandatory nonselectable neural model/calibration/composer triplets, control-exit, complete preopen negative-control manifest, outcome-free replay config, and a self-hashed preopen receipt must then exist. The receipt binds only HGB as the selected candidate plus its exact model/calibrator/composer hashes, exact prior-fold result receipts, exact four nested result-or-lawful-structural-skip receipts, environment and source hashes, outer_evidence_access_count=0, and every path/hash/session role. An O_EXCL durable outer_primary_access_receipt changes count to exactly one before any primary source decode; reissue/reopen is forbidden",
        "sequential_rule": "outer fold k preopen may be sealed only after folds <k outer result receipts are frozen; no later fold outcome or artifact may exist or enter selection",
        "result_rule": "the primary outer result receipt validates the durable access receipt, exact evidence dataset/source receipt root, both typed HGB and mandatory neural-challenger evaluations embedded inside outer_primary_result.json (there are no standalone hgb/neural evaluation files), full journal/trace coverage, post-open negative-control panel and replay result, globally frozen HGB candidate identity, result envelope, and every preopen artifact hash. Neural outcomes cannot select, replace, or rescue HGB. Post-open outcomes may not appear in a preopen manifest/config",
        "pooled_rule": "pooled entry statistics use a FrozenResultAggregationAuthorization over the five immutable outer result/evaluation/trace receipts only. No pooled EntryEvidenceDatasetV1 exists and raw corpus rows may not be decoded or replayed a second time for pooled selection/statistics",
        "pooled_acceptance_rule": "freeze_entry_pooled_acceptance_once alone recomputes the complete entry verdict from the exact five frozen outer result/evaluation/trace receipts under FrozenResultAggregationAuthorization, validates every negative-control/action/economic/power input, and durably writes the typed result then its O_EXCL terminal receipt. No caller may supply a verdict, metric, receipt path, trace, policy id, gate input, or result payload. Full fit requires a revalidated PASS receipt; context authorization requires the frozen receipt regardless of verdict",
        "context_diagnostics_rule": "only after all five outer result receipts and the pooled entry-acceptance terminal receipt are immutable may issue_frozen_context_diagnostics_authorization create one exact read-only authorization for the five primary outer session lists. The diagnostics runner resolves VIX/ES/VX paths internally from the frozen manifest/spec, accepts no caller path/data/policy/tables, writes fold_1..fold_5 artifacts and receipts in order with O_EXCL, then derives the pooled artifact only from the five validated per-fold artifacts and seals its receipt. Diagnostics cannot enter the already frozen pooled acceptance, entry family, model, calibrator, comparator, threshold, action, or gate",
        "shortened_rule": "a shortened diagnostic can open only after that fold's primary outer result receipt is durable. It has a dedicated preopen, O_EXCL access, sealed dataset, result, and result receipt; it is never part of economic/action/model-family/comparator evidence and cannot alter any primary artifact",
        "crash_rule": "any lease loss after the count-one access receipt permanently burns that nested/outer/diagnostic block, including when dataset/result bytes are durable but the terminal result receipt is absent. Recovery never decodes, validates, or seals unsealed result bytes, never replaces an artifact or recreates authorization, and never resets count to zero",
        "protected_holdout_rule": "none of these receipts authorizes firewall/protected sessions; the protected set remains reserved for a distinct atomic count=1 transaction after the complete entry+exit/four-box packet",
    }


def entry_fit_environment_spec() -> dict[str, Any]:
    """Exact deterministic CPU process contract required before any estimator fit."""

    return {
        "schema_version": "pathd.entry_fit_environment.v1",
        "python_version": "3.12.13",
        "platform": {"system": "Darwin", "machine": "arm64"},
        "python_packages_exact": {
            "numpy": "2.0.2",
            "pandas": "3.0.3",
            "scikit_learn": "1.5.2",
            "torch": "2.5.1",
            "scipy": "1.14.1",
            "joblib": "1.5.3",
            "threadpoolctl": "3.6.0",
        },
        "required_environment": {
            "PYTHONHASHSEED": "0",
            "PATHD_ENTRY_DEVICE": "cpu",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "MKL_DYNAMIC": "FALSE",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "LOKY_MAX_CPU_COUNT": "1",
            "BLIS_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        },
        "active_python_hash_probe": {
            "input": "pathd-entry-fit-probe-v1",
            "expected_hash": -4808614846103707838,
        },
        "torch_runtime": {
            "device": "cpu",
            "num_threads": 1,
            "num_interop_threads": 1,
            "deterministic_algorithms": True,
            "deterministic_debug_mode": 2,
            "default_dtype": "torch.float32",
            "cudnn_benchmark": False,
            "cuda_allowed": False,
            "mps_allowed": False,
        },
        "threadpool_runtime": {
            "all_loaded_pools_num_threads": 1,
            "required_multiset": [
                {"user_api": "openmp", "internal_api": "openmp", "prefix": "libomp", "version": None},
                {"user_api": "openmp", "internal_api": "openmp", "prefix": "libomp", "version": None},
            ],
        },
        "startup_rule": "launch a fresh process with every required environment variable already set before Python imports NumPy/sklearn/Torch; then call torch.set_num_threads(1), torch.set_num_interop_threads(1), torch.use_deterministic_algorithms(True,warn_only=False), set deterministic debug mode 2, and never change them",
        "drift": "any version, platform, variable, device, Torch flag/thread, or loaded threadpool mismatch blocks authorization before corpus read or fit",
    }


def current_entry_fit_environment_record() -> dict[str, Any]:
    """Capture the deterministic runtime state without changing it."""

    import joblib
    import numpy
    import pandas
    import scipy
    import sklearn
    import threadpoolctl
    import torch

    pools = []
    for row in threadpoolctl.threadpool_info():
        pools.append(
            {
                "user_api": row.get("user_api"),
                "internal_api": row.get("internal_api"),
                "prefix": row.get("prefix"),
                "version": row.get("version"),
                "num_threads": row.get("num_threads"),
            }
        )
    pools.sort(key=lambda row: json.dumps(row, sort_keys=True, default=str))
    return {
        "schema_version": "pathd.entry_fit_environment_record.v1",
        "python_version": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "python_packages_exact": {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
            "threadpoolctl": threadpoolctl.__version__,
        },
        "environment": {
            name: os.environ.get(name)
            for name in entry_fit_environment_spec()["required_environment"]
        },
        "active_python_hash_probe": {
            "input": "pathd-entry-fit-probe-v1",
            "observed_hash": hash("pathd-entry-fit-probe-v1"),
        },
        "torch_runtime": {
            "device": os.environ.get("PATHD_ENTRY_DEVICE"),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
            "default_dtype": str(torch.get_default_dtype()),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_allowed": False,
            "mps_allowed": False,
        },
        "observed_accelerators": {
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "threadpools": pools,
    }


def validate_entry_fit_environment_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one current record against the preregistered deterministic contract."""

    spec = entry_fit_environment_spec()
    if type(record) is not dict:
        raise RuntimeError("entry fitting blocked: fit environment record is malformed")
    if record.get("schema_version") != "pathd.entry_fit_environment_record.v1":
        raise RuntimeError("entry fitting blocked: fit environment record schema drift")
    for field in ("python_version", "platform", "python_packages_exact"):
        if record.get(field) != spec[field]:
            raise RuntimeError(f"entry fitting blocked: fit environment {field} drift")
    if record.get("environment") != spec["required_environment"]:
        raise RuntimeError("entry fitting blocked: deterministic process environment drift")
    probe = record.get("active_python_hash_probe")
    if (
        type(probe) is not dict
        or probe.get("input") != spec["active_python_hash_probe"]["input"]
        or type(probe.get("observed_hash")) is not int
        or probe.get("observed_hash")
        != spec["active_python_hash_probe"]["expected_hash"]
    ):
        raise RuntimeError("entry fitting blocked: active PYTHONHASHSEED drift")
    if record.get("torch_runtime") != spec["torch_runtime"]:
        raise RuntimeError("entry fitting blocked: deterministic Torch runtime drift")
    accelerators = record.get("observed_accelerators")
    if not isinstance(accelerators, dict) or accelerators.get("cuda_available") is not False:
        raise RuntimeError("entry fitting blocked: CUDA is visible in CPU-only process")
    pools = record.get("threadpools")
    if not isinstance(pools, list) or any(
        type(row) is not dict or row.get("num_threads") != 1 for row in pools
    ):
        raise RuntimeError("entry fitting blocked: loaded threadpool is not single-threaded")
    multiset = [
        {key: row.get(key) for key in ("user_api", "internal_api", "prefix", "version")}
        for row in pools
    ]
    multiset.sort(key=lambda row: json.dumps(row, sort_keys=True, default=str))
    expected = list(spec["threadpool_runtime"]["required_multiset"])
    expected.sort(key=lambda row: json.dumps(row, sort_keys=True, default=str))
    if multiset != expected:
        raise RuntimeError("entry fitting blocked: loaded threadpool implementation drift")
    json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return record


def assert_entry_fit_environment_current() -> dict[str, Any]:
    return validate_entry_fit_environment_record(current_entry_fit_environment_record())


def entry_future_api_contract() -> dict[str, Any]:
    """Frozen post-preregistration API surface exercised by the immutable gate."""

    return {
        "schema_versions": {
            "EntrySnapshotV1": "pathd.entry_snapshot.v1",
            "Signed17FrameV1": "pathd.signed17_frame.v1",
            "Signed17HistoryV1": "pathd.signed17_history.v1",
            "EntryFuturePathV1": "pathd.entry_future_path.v1",
            "EntryExampleV1": "pathd.entry_example.v1",
            "EntryActionExecutionFactV1": "pathd.entry_action_execution_fact.v1",
            "EntrySessionV1": "pathd.entry_session.v1",
            "EntryFitDatasetV1": "pathd.entry_fit_dataset.v1",
            "EntryEvidenceDatasetV1": "pathd.entry_evidence_dataset.v1",
            "EntryModelInputV1": "pathd.entry_model_input.v1",
            "EntryPredictionV1": "pathd.entry_prediction.v1",
            "EntryModelBundleV1": "pathd.entry_model_bundle.v1",
            "EntryCalibrationBundleV1": "pathd.entry_calibration_bundle.v1",
            "EntryComposerBundleV1": "pathd.entry_composer_bundle.v1",
            "EntryNegativeControlBundleV1": "pathd.entry_negative_control_bundle.v1",
            "EntryNegativeControlCalibrationBundleV1": "pathd.entry_negative_control_calibration_bundle.v1",
            "EntryNegativeControlComposerBundleV1": "pathd.entry_negative_control_composer_bundle.v1",
            "EntryNegativeControlManifestV1": "pathd.entry_negative_control_manifest.v1",
            "EntryReplayEvaluationCellV1": "pathd.entry_replay_evaluation_cell.v1",
            "EntryActionCalibrationObservationV1": "pathd.entry_action_calibration_observation.v1",
            "EntryTime300TrajectoryEvidenceV1": "pathd.entry_time300_trajectory_evidence.v1",
            "EntrySurvivalAuditRecordV1": "pathd.entry_survival_audit_record.v1",
            "EntryNegativeControlPanelV1": "pathd.entry_negative_control_panel.v1",
            "EntryPolicyEvaluationV1": "pathd.entry_policy_evaluation.v1",
            "EntryNestedFamilyEvaluationV1": "pathd.entry_nested_family_evaluation.v1",
            "EntryOuterPrimaryResultV1": "pathd.entry_outer_primary_result.v1",
            "EntryPooledAcceptanceResultV1": "pathd.entry_pooled_acceptance_result.v1",
            "FrozenContextDiagnosticsAuthorizationV1": "pathd.frozen_context_diagnostics_authorization.v1",
            "EntryContextSourceSelectionV1": "pathd.entry_context_source_selection.v1",
            "EntryContextAnchorRowV1": "pathd.entry_context_anchor_row.v1",
            "EntryOuterContextDiagnosticsV1": "pathd.entry_outer_context_diagnostics.v1",
            "EntryPooledContextDiagnosticsV1": "pathd.entry_pooled_context_diagnostics.v1",
            "EntryControlReplayConfigV1": "pathd.entry_control_replay_config.v1",
            "EntryControlReplayResultV1": "pathd.entry_control_replay_result.v1",
            "EntryControlExitSelectionV1": "pathd.entry_control_exit_selection.v1",
            "EntryNestedReplayConfigV1": "pathd.entry_nested_replay_config.v1",
            "EntryActionDecisionV1": "pathd.entry_action_decision.v1",
            "EntryActionObservationV1": "pathd.entry_action_observation.v1",
            "EntryFrameCoverageV1": "pathd.entry_frame_coverage.v1",
            "ResearchFillLawV1": "pathd.research_fill_law.v1",
            "VerifiedResearchQuoteRowV1": "pathd.verified_research_quote_row.v1",
            "VerifiedOfficialSpxRowV1": "pathd.verified_official_spx_row.v1",
            "ResearchQuoteV1": "pathd.research_quote.v1",
            "ResearchFillDecisionV1": "pathd.research_fill_decision.v1",
            "ResearchExecutionResultV1": "pathd.research_execution_result.v1",
            "ResearchDecisionContextV1": "pathd.research_decision_context.v1",
            "ResearchLedgerStateV1": "pathd.research_ledger_state.v1",
            "ResearchLedgerControlEventV1": "pathd.research_ledger_control_event.v1",
            "ResearchLedgerTransitionV1": "pathd.research_ledger_transition.v1",
            "ResearchLedgerJournalV1": "pathd.research_ledger_journal.v1",
            "ResearchPreparedExecutionV1": "pathd.research_prepared_execution.v1",
            "ResearchResultEnvelopeV1": "pathd.research_result_envelope.v1",
            "NonOrderSafetyEventV1": "pathd.research.non_order_safety_event.v1",
            "ResearchExecutionOutcomeV1": "pathd.research_execution_outcome.v1",
            "ProtectedHoldoutStateV1": "pathd.protected_holdout_state.v1",
            "ProtectedHoldoutEvaluationV1": "pathd.protected_holdout_evaluation.v1",
            "ProtectedHoldoutTraceRecordV1": "pathd.protected_holdout_trace_record.v1",
        },
        "dataclass_fields": {
            "EntrySnapshotV1": [
                "schema_version", "session", "decision_time_ns", "decision_watermark_ns",
                "atm_strike", "strike_offsets", "rights", "option_ladder",
                "option_feature_names", "market_window", "market_feature_names",
                "contract_ids", "contract_quote_watermark_ns", "declared_alpha_sources",
            ],
            "Signed17FrameV1": [
                "schema_version", "feature_names", "values", "finite", "contract_ids",
                "decision_time_ns", "decision_watermark_ns",
            ],
            "Signed17HistoryV1": [
                "schema_version", "values", "finite", "contract_present",
                "minute_available", "current_contract_ids", "decision_time_ns",
            ],
            "EntryFuturePathV1": [
                "schema_version", "arrival_time_ns", "arrival_contract_id",
                "arrival_bid_micros", "arrival_ask_micros", "arrival_watermark_ns",
                "marks_by_horizon", "future_audit_nonce",
            ],
            "EntryExampleV1": [
                "schema_version", "model_input", "action_execution_facts",
                "targets", "target_validity", "audit",
            ],
            "EntryActionExecutionFactV1": [
                "schema_version", "session", "decision_time_ns",
                "source_neutral_contract_id", "contract",
                "represented_interval_end_ns", "ts_recv_ns", "available_at_ns",
                "bid_micros", "ask_micros", "source_receipt_sha256",
                "source_record_sha256", "physical_eligible", "fact_sha256",
            ],
            "EntrySessionV1": [
                "schema_version", "session", "source_files", "source_files_sha256",
                "examples", "content_sha256",
            ],
            "EntryFitDatasetV1": [
                "schema_version", "authorization_sha256", "role", "sessions",
                "sessions_sha256_newline", "source_receipts", "examples", "dataset_sha256",
            ],
            "EntryEvidenceDatasetV1": [
                "schema_version", "authorization_sha256", "role", "sessions",
                "sessions_sha256_newline", "source_receipts", "examples", "dataset_sha256",
            ],
            "EntryModelInputV1": [
                "schema_version", "session", "decision_time_ns", "signed17_frame",
                "signed17_history", "hgb_summaries", "current_offsets", "current_rights",
                "contract_ids", "physical_action_mask",
            ],
            "EntryPredictionV1": [
                "schema_version", "family", "model_artifact_sha256",
                "authorization_sha256", "dataset_sha256", "input_sha256",
                "prediction_values", "prediction_sha256",
            ],
            "EntryModelBundleV1": [
                "schema_version", "family", "fit_role", "outer_fold", "inner_fold",
                "authorization_sha256", "sessions_sha256_newline", "dataset_sha256",
                "artifact_sha256", "payload",
            ],
            "EntryCalibrationBundleV1": [
                "schema_version", "model_artifact_sha256", "calibration_role", "outer_fold",
                "inner_fold", "authorization_sha256", "sessions_sha256_newline",
                "dataset_sha256", "artifact_sha256", "payload",
            ],
            "EntryComposerBundleV1": [
                "schema_version", "model_artifact_sha256", "calibration_artifact_sha256",
                "outer_fold", "inner_fold", "composer_spec_sha256", "artifact_sha256", "payload",
            ],
            "EntryNegativeControlBundleV1": [
                "schema_version", "control_id", "base_family", "fit_role", "outer_fold",
                "inner_fold", "authorization_sha256", "dataset_sha256", "seed_key_sha256",
                "artifact_sha256", "payload",
            ],
            "EntryNegativeControlCalibrationBundleV1": [
                "schema_version", "control_artifact_sha256", "control_id",
                "base_family", "calibration_role", "outer_fold", "inner_fold",
                "authorization_sha256", "dataset_sha256", "artifact_sha256", "payload",
            ],
            "EntryNegativeControlComposerBundleV1": [
                "schema_version", "control_artifact_sha256",
                "calibration_artifact_sha256", "control_id", "base_family",
                "outer_fold", "inner_fold", "transform_stage",
                "composer_spec_sha256", "artifact_sha256", "payload",
            ],
            "EntryNegativeControlManifestV1": [
                "schema_version", "outer_fold", "required_control_ids",
                "required_base_families", "control_bundle_sha256s",
                "control_calibration_sha256s", "control_composer_sha256s",
                "sign_reversed_base_composer_sha256s", "seed_receipt_sha256s",
                "replay_config_sha256", "artifact_sha256",
            ],
            "EntryReplayEvaluationCellV1": [
                "schema_version", "outer_fold", "owner_policy_id",
                "replay_policy_id", "channel", "matched_random_seed",
                "authorization_sha256", "dataset_sha256", "evaluation",
                "evaluation_sha256", "terminal_journal_sha256",
                "candidate_budget_sha256", "ordered_schedule_sha256",
                "structural_skip_trace_sha256",
                "realized_intents_and_fills_sha256", "cell_sha256",
            ],
            "EntryActionCalibrationObservationV1": [
                "schema_version", "outer_fold", "action", "session",
                "trajectory_or_episode_id", "predicted_mean_micros",
                "predicted_lower_micros", "realized_micros",
                "source_policy_evaluation_sha256",
                "source_transition_sha256", "observation_sha256",
            ],
            "EntryTime300TrajectoryEvidenceV1": [
                "schema_version", "outer_fold", "session", "trajectory_id",
                "source_policy_evaluation_sha256",
                "entry_fill_transition_sha256", "h300_mark_receipt_sha256",
                "evidence_sha256",
            ],
            "EntrySurvivalAuditRecordV1": [
                "schema_version", "outer_fold", "session", "policy_id",
                "overlap", "unaffordable_fill", "quantity_breach",
                "nonflat_close", "duplicate_terminal_consumption",
                "d48_breach", "d49_breach", "source_transition_sha256s",
                "record_sha256",
            ],
            "EntryNegativeControlPanelV1": [
                "schema_version", "holdout_caveat", "outer_fold", "manifest_sha256",
                "required_control_ids", "channels",
                "control_bundle_sha256s", "control_channel_evaluations",
                "control_action_calibration_observations",
                "control_time300_trajectory_evidence", "artifact_sha256",
            ],
            "EntryPolicyEvaluationV1": [
                "schema_version", "holdout_caveat", "evidence_role",
                "outer_fold", "inner_fold",
                "authorization_sha256", "dataset_sha256", "policy_id", "sessions",
                "sessions_sha256_newline", "session_pnl_micros",
                "session_terminal_journal_sha256s", "completed_trade_ids",
                "completed_trade_sessions", "zero_trade_sessions",
                "execution_trace_root_sha256", "terminal_journal",
                "terminal_journal_sha256",
                "session_coverage_sha256", "metrics", "result_sha256",
            ],
            "EntryNestedFamilyEvaluationV1": [
                "schema_version", "holdout_caveat", "outer_fold", "inner_fold",
                "authorization_sha256", "dataset_sha256",
                "source_receipts_root_sha256", "access_receipt_sha256",
                "hgb", "neural", "result_sha256",
            ],
            "EntryOuterPrimaryResultV1": [
                "schema_version", "outer_fold", "evidence_role",
                "authorization_sha256", "dataset_sha256",
                "source_receipts_root_sha256", "access_receipt_sha256",
                "hgb", "neural", "selected_family",
                "selected_evaluation_sha256", "negative_control_panel_sha256",
                "control_replay_result_sha256", "session_coverage_root_sha256",
                "result_sha256",
            ],
            "EntryPooledAcceptanceResultV1": [
                "schema_version", "holdout_caveat",
                "aggregation_authorization_sha256",
                "outer_result_receipts_sha256", "sessions",
                "sessions_sha256_newline", "candidate_family",
                "candidate_outer_evaluation_sha256s", "control_exit_sha256s",
                "control_replay_result_sha256s", "negative_control_panel_sha256s",
                "gate_spec_sha256", "reconstructed_gate_inputs",
                "gate_inputs_sha256", "pass_criteria_recomputed", "verdict",
                "stop_reason", "full_fit_authorized", "result_sha256",
            ],
            "FrozenContextDiagnosticsAuthorizationV1": [
                "schema_version", "status", "preregistration_sha256",
                "plan_sha256", "session_assignments_sha256",
                "source_hash_policy_sha256", "corpus_integrity_receipt_sha256",
                "machinery_receipt_sha256", "holdout_open_count",
                "diagnostic_inventory_receipt_path",
                "diagnostic_inventory_receipt_sha256",
                "context_access_receipt_path", "context_access_count",
                "pooled_acceptance_receipt_path",
                "pooled_acceptance_receipt_sha256",
                "outer_result_receipt_paths", "outer_result_receipt_sha256s",
                "outer_sessions_by_fold", "outer_sessions_root_sha256",
                "policy_journal_binding_roots_by_fold",
                "policy_journal_bindings_root_sha256",
                "diagnostic_spec_sha256", "authorization_sha256",
            ],
            "EntryContextSourceSelectionV1": [
                "schema_version", "source", "status", "requested",
                "source_degraded", "manifest_relative_path",
                "source_file_sha256", "current_component_locators",
                "lag15_component_locators", "current_available_at_ns",
                "lag15_available_at_ns", "current_age_ns", "lag15_age_ns",
                "current_close_float64_hex", "lag15_close_float64_hex",
                "selection_sha256",
            ],
            "EntryContextAnchorRowV1": [
                "schema_version", "outer_fold", "policy_id",
                "policy_order_index", "policy_evaluation_sha256",
                "terminal_journal_sha256", "anchor_kind", "anchor_ordinal",
                "anchor_id", "session", "anchor_time_ns",
                "source_transition_sha256", "action", "filled", "trade_id",
                "trade_pnl_micros", "session_pnl_micros",
                "source_selections", "source_selections_root_sha256",
                "derived_values_float64_hex", "bin_ids", "row_sha256",
            ],
            "EntryOuterContextDiagnosticsV1": [
                "schema_version", "holdout_caveat", "status", "outer_fold",
                "authorization_sha256", "access_receipt_sha256",
                "primary_result_receipt_sha256",
                "core_integrity_receipt_sha256",
                "diagnostic_inventory_receipt_sha256",
                "policy_journal_bindings",
                "policy_journal_bindings_root_sha256",
                "sessions", "sessions_sha256_newline", "source_receipts",
                "source_receipts_root_sha256", "anchor_context_rows",
                "anchor_context_rows_root_sha256",
                "age_samples_ns_by_source", "age_samples_root_sha256",
                "tables", "tables_sha256", "artifact_sha256",
            ],
            "EntryPooledContextDiagnosticsV1": [
                "schema_version", "holdout_caveat", "status",
                "authorization_sha256", "access_receipt_sha256",
                "core_integrity_receipt_sha256",
                "diagnostic_inventory_receipt_sha256",
                "outer_artifact_paths",
                "outer_artifact_sha256s", "outer_receipt_paths",
                "outer_receipt_sha256s",
                "policy_journal_binding_roots_by_fold",
                "policy_journal_bindings_root_sha256",
                "anchor_context_rows", "anchor_context_rows_root_sha256",
                "age_samples_ns_by_source", "age_samples_root_sha256",
                "tables", "tables_sha256", "artifact_sha256",
            ],
            "EntryControlReplayConfigV1": [
                "schema_version", "outer_fold", "fill_law_hash",
                "control_exit_sha256", "required_policy_ids",
                "matched_random_seeds", "fee_paths", "sell_delay_rungs_ms",
                "headline_config_sha256", "artifact_sha256",
            ],
            "EntryControlReplayResultV1": [
                "schema_version", "holdout_caveat", "outer_fold", "replay_config_sha256",
                "candidate_outer_evaluation_sha256",
                "neural_outer_evaluation_sha256",
                "channels", "candidate_channel_evaluations",
                "p5_channel_evaluations", "matched_random_owner_policy_ids",
                "matched_random_seeds", "matched_random_channel_evaluations",
                "candidate_action_calibration_observations",
                "candidate_time300_trajectory_evidence",
                "candidate_budget_sha256s",
                "ordered_schedule_sha256s", "structural_skip_trace_sha256s",
                "realized_intents_and_fills_sha256s", "artifact_sha256",
            ],
            "EntryControlExitSelectionV1": [
                "schema_version", "holdout_caveat", "outer_fold", "model_fit_sessions_sha256_newline",
                "candidate_policy_ids", "candidate_evaluation_sha256s",
                "valid_session_counts", "total_net_pnl_micros",
                "selected_policy_id", "selection_rule", "artifact_sha256",
            ],
            "EntryNestedReplayConfigV1": [
                "schema_version", "outer_fold", "inner_fold", "fill_law_hash",
                "policy_ids", "fee_path", "sell_delay_ms", "floor_on",
                "exit_policy_id", "nested_account_scope_sha256", "artifact_sha256",
            ],
            "EntryActionDecisionV1": [
                "schema_version", "action", "reason", "selected_action_index",
                "selected_source_neutral_contract_id", "selected_contract",
                "reference_bid_micros", "reference_ask_micros",
                "buy_hard_limit_micros", "available_horizons",
                "mean_lcb_dollars", "mean_lcb_return", "q10_dollars",
                "q10_return", "physical_action_mask", "dynamic_account_mask",
                "combined_action_mask", "authorization_sha256", "dataset_sha256",
                "example_sha256", "model_input_sha256", "prediction_sha256",
                "composer_sha256", "prior_journal_root_sha256",
                "fill_law_hash", "decision_sha256",
            ],
            "EntryActionObservationV1": [
                "schema_version", "authorization_sha256", "dataset_sha256",
                "session", "trajectory_id", "episode_id", "example_sha256",
                "decision_sha256", "action", "predicted_mean_cash_micros",
                "predicted_lower_cash_micros", "realized_value_cash_micros",
                "coverage", "filled", "execution_result_sha256",
                "observation_sha256",
            ],
            "EntryFrameCoverageV1": [
                "schema_version", "authorization_sha256", "dataset_sha256",
                "session", "decision_time_ns", "example_sha256",
                "disposition", "reason_codes", "prior_journal_root_sha256",
                "coverage_sha256",
            ],
            "ResearchFillLawV1": [
                "schema_version", "fill_law_hash", "headline_delay_ms",
                "delay_sensitivity_ms", "paired_quote_latency_bounds_ms",
                "entry_fee_micros", "exit_fee_micros", "fill_price_mode",
            ],
            "VerifiedResearchQuoteRowV1": [
                "schema_version", "session", "source_relative_path", "source_file_sha256",
                "row_group", "row_index", "canonical_row_sha256", "contract",
                "source_vendor", "represented_interval_end_ns", "ts_recv_ns",
                "available_at_ns", "bid_micros", "ask_micros",
                "source_receipt_sha256", "query_at_or_before_ns", "query_maximum_age_ms",
                "query_require_actionable", "eligible_row_count", "selection_key",
                "selection_proof_sha256", "record_sha256",
            ],
            "VerifiedOfficialSpxRowV1": [
                "schema_version", "session", "source_relative_path", "source_file_sha256",
                "row_group", "row_index", "canonical_row_sha256", "source_vendor",
                "represented_interval_end_ns", "ts_recv_ns", "available_at_ns",
                "spx_micros", "source_receipt_sha256", "query_at_or_before_ns",
                "query_maximum_age_ms", "eligible_row_count", "selection_key",
                "selection_proof_sha256", "record_sha256",
            ],
            "ResearchQuoteV1": [
                "schema_version", "session", "contract", "source_vendor",
                "represented_interval_end_ns", "ts_recv_ns", "available_at_ns",
                "bid_micros", "ask_micros", "actionable", "invalid_reason",
                "source_relative_path", "source_file_sha256", "row_group", "row_index",
                "canonical_row_sha256", "source_receipt_sha256", "source_record_sha256",
                "quote_sha256",
            ],
            "ResearchFillDecisionV1": [
                "schema_version", "filled", "fill_price_micros", "hard_limit_micros",
                "reason", "position_unchanged",
            ],
            "ResearchDecisionContextV1": [
                "schema_version", "intent_kind", "session", "contract", "decision_time_ns", "event_interval_end_ns",
                "option_watermark_ns", "spx_watermark_ns", "authorization_sha256",
                "dataset_sha256", "entry_example_sha256",
                "entry_action_decision_sha256", "option_source_receipt_sha256",
                "spx_source_receipt_sha256", "decision_quote", "official_spx_row",
                "context_sha256",
            ],
            "ResearchLedgerStateV1": [
                "schema_version", "policy_id", "fee_path", "authorization_sha256",
                "sessions_sha256_newline", "account_scope_sha256",
                "account_session_index", "session_index", "session",
                "session_start_equity_micros", "cash_micros",
                "realized_session_pnl_micros", "position", "pending_intent_id",
                "sequence", "last_decision_time_ns", "prior_transition_sha256",
                "ledger_sha256",
            ],
            "ResearchLedgerControlEventV1": [
                "schema_version", "event_kind", "event_time_ns",
                "prior_authorization_sha256", "next_authorization_sha256",
                "prior_session", "next_session", "prior_session_index",
                "next_session_index", "prior_account_session_index",
                "next_account_session_index",
                "intervening_skip_receipt_sha256s", "reason",
                "event_sha256",
            ],
            "ResearchLedgerTransitionV1": [
                "schema_version", "transition_kind", "event_time_ns",
                "prior_authorization_sha256", "next_authorization_sha256",
                "prior_ledger_sha256", "event_schema_version",
                "event_sha256", "event_payload", "next_ledger",
                "transition_sha256",
            ],
            "ResearchLedgerJournalV1": [
                "schema_version", "account_scope_sha256", "policy_id", "fee_path",
                "transitions", "tip_ledger", "journal_root_sha256",
            ],
            "ResearchPreparedExecutionV1": [
                "schema_version", "intent", "broker_state", "feed_health", "authorization",
                "decision_context", "decision_context_sha256",
                "decision_quote_sha256", "source_receipt_sha256",
                "prior_journal", "prior_journal_root_sha256",
                "governor_config_sha256",
                "arrival_base_time_ns", "prepared_sha256",
            ],
            "ResearchExecutionResultV1": [
                "schema_version", "prepared_execution_sha256", "decision_quote_sha256",
                "arrival_quote_sha256", "source_receipt_sha256", "delay_ms", "events",
                "final_state", "filled_quantity", "fill_price_micros", "fee_micros",
                "cash_delta_micros", "position_before", "position_after",
                "prior_ledger_sha256", "result_ledger", "result_sha256",
            ],
            "ResearchExecutionOutcomeV1": [
                "schema_version", "execution_result", "result_journal",
                "outcome_sha256",
            ],
            "ResearchResultEnvelopeV1": [
                "schema_version", "authorization_sha256", "sessions_sha256_newline",
                "dataset_sha256", "preregistration_sha256", "plan_sha256", "fill_law_hash",
                "quarantine_labels", "claim_boundary", "holdout_caveat", "holdout_open_count",
                "payload_sha256", "payload",
            ],
            "NonOrderSafetyEventV1": [
                "schema_version", "event_id", "event_type", "session",
                "policy_id", "fee_path", "authorization_sha256",
                "event_time_utc", "event_time_ns",
                "source_neutral_contract_id", "osi_symbol", "state_from",
                "state_to", "invalid_reason", "provenance",
                "last_actual_option_watermark_utc",
                "zero_proceeds_cash_micros",
                "modeled_close_penalty_cash_micros", "cash_before_micros",
                "cash_after_micros", "equity_before_micros",
                "equity_after_micros", "pending_intent_id_consumed",
                "fill_law_hash", "prior_ledger_sha256", "result_ledger",
                "event_sha256",
            ],
            "ProtectedHoldoutStateV1": [
                "schema_version", "state", "holdout_open_count",
                "transaction_id", "detail", "access_receipt_sha256",
                "result_sha256", "terminal_receipt_sha256",
            ],
            "ProtectedHoldoutEvaluationV1": [
                "schema_version", "authorization_sha256",
                "access_receipt_sha256", "preopen_packet_sha256",
                "artifact_root_sha256", "fit_environment_sha256",
                "sessions_sha256_newline",
                "primary_sessions_sha256_newline", "dataset_sha256",
                "source_receipts_root_sha256", "box_d_policy_id",
                "comparator_policy_id", "trace_root_sha256",
                "trace_artifact_sha256", "evaluator_receipt_sha256",
                "payload_sha256", "payload",
            ],
            "ProtectedHoldoutTraceRecordV1": [
                "schema_version", "holdout_caveat", "session", "authorization_sha256",
                "dataset_sha256", "box_d_policy_id", "comparator_policy_id",
                "box_d_terminal_journal", "comparator_terminal_journal",
                "box_d_session_pnl_micros",
                "comparator_session_pnl_micros",
                "box_d_completed_trade_ids", "guard_results",
                "survival_violations", "record_sha256",
            ],
        },
        "functions": [
            {
                "qualified_name": "v4.research.pathd_entry_features.historical_snapshot_from_processed_row",
                "parameters": [{"name": "row", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_features.source_neutral_snapshot_from_causal_inputs",
                "parameters": [
                    {"name": name, "kind": "KEYWORD_ONLY", "required": True}
                    for name in (
                        "session", "decision_time_ns", "decision_watermark_ns", "atm_strike",
                        "strike_offsets", "rights", "option_ladder", "option_feature_names",
                        "market_window", "market_feature_names", "contract_ids",
                        "contract_quote_watermark_ns", "declared_alpha_sources",
                    )
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_features.validate_alpha_source_record",
                "parameters": [{"name": "record", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_features.signed17_from_snapshot",
                "parameters": [{"name": "snapshot", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_features.build_identity_joined_history",
                "parameters": [
                    {"name": "frames", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "current_session", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "current_decision_time_ns", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "current_contract_ids", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "history_minutes", "kind": "KEYWORD_ONLY", "required": False, "default": 90},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_features.hgb_signed17_summaries",
                "parameters": [
                    {"name": "history", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "current_offsets", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "current_rights", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.build_entry_example",
                "parameters": [
                    {"name": "snapshot", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "future_path", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "fill_law", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.available_entry_horizons",
                "parameters": [
                    {"name": "decision_time_ns", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "terminal_time_ns", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.entry_targets_from_executable_marks",
                "parameters": [
                    {"name": "entry_fill_price_micros", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "round_trip_fee_micros", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "marks_by_horizon", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "available_horizons", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.read_verified_entry_session",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "session", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.load_authorized_entry_dataset",
                "parameters": [{"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.canonical_entry_dataset_sha256",
                "parameters": [
                    {"name": "schema_version", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization_sha256", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "role", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "sessions", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "sessions_sha256_newline", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "source_receipts", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "examples", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.validate_entry_fit_dataset",
                "parameters": [
                    {"name": "dataset", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.read_verified_entry_evidence_session",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "session", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.load_authorized_entry_evidence_dataset",
                "parameters": [{"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.validate_entry_evidence_dataset",
                "parameters": [
                    {"name": "dataset", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.load_authorized_protected_holdout_dataset",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.validate_protected_holdout_dataset",
                "parameters": [
                    {"name": "dataset", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.read_verified_research_quote_row",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "session", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "contract", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "at_or_before_ns", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "maximum_age_ms", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "require_actionable", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_dataset.read_verified_official_spx_row",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "session", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "at_or_before_ns", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "maximum_age_ms", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.model_input_from_example",
                "parameters": [{"name": "example", "kind": "POSITIONAL_ONLY", "required": True}],
                "annotation_rule": "accepts EntryExampleV1 and returns EntryModelInputV1 after discarding action_execution_facts, targets, target_validity, and audit; only the signed-17 representation, non-alpha geometry, and immutable physical mask remain",
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.causal_probe_predict",
                "parameters": [{"name": "inputs", "kind": "POSITIONAL_ONLY", "required": True}],
                "annotation_rule": "accepts only EntryModelInputV1; no future/example/target argument",
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.predict_entry",
                "parameters": [
                    {"name": "bundle", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "inputs", "kind": "POSITIONAL_ONLY", "required": True},
                ],
                "annotation_rule": "second argument is EntryModelInputV1; no future/example/target argument",
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.fit_hgb_entry_bundle",
                "parameters": [{"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True}],
                "annotation_rule": "authorization role must be outer_weights, nested_weights, or full_weights; the wrapper internally invokes the verified loader, accepts no caller dataset, and artifact binds exact authorization/session/dataset hashes",
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.fit_neural_entry_bundle",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "hgb_bundle", "kind": "KEYWORD_ONLY", "required": True},
                ],
                "annotation_rule": "authorization role must be outer_weights, nested_weights, or full_weights; hgb_bundle must be the already sealed and validated HGB baseline for this exact authorization, scope, sessions, and sealed dataset. The wrapper internally invokes the verified loader, accepts no caller dataset, and binds the HGB baseline hash into the neural challenger artifact",
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.fit_entry_calibrators",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "model_bundle", "kind": "KEYWORD_ONLY", "required": True},
                ],
                "annotation_rule": "authorization role must be matching outer/nested/full calibration; wrapper internally loads its exact verified dataset, and model weights artifact comes from the paired earlier weights role",
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.seal_entry_model_bundle",
                "parameters": [
                    {"name": "family", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset_sha256", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "payload", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_model_bundle",
                "parameters": [{"name": "bundle", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.seal_entry_calibration_bundle",
                "parameters": [
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "model_bundle", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset_sha256", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "payload", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_calibration_bundle",
                "parameters": [
                    {"name": "bundle", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "model_bundle", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.apply_entry_calibration",
                "parameters": [
                    {"name": "bundle", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "prediction", "kind": "POSITIONAL_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.seal_entry_composer_bundle",
                "parameters": [
                    {"name": "model_bundle", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "calibration_bundle", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "payload", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_composer_bundle",
                "parameters": [
                    {"name": "bundle", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "model_bundle", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "calibration_bundle", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.compose_entry_action",
                "parameters": [
                    {"name": "bundle", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "model_bundle", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "example", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "journal", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "fill_law", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.fit_entry_negative_control_bundle",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "base_family", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "control_id", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_negative_control_bundle",
                "parameters": [{"name": "bundle", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.fit_entry_negative_control_calibrators",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "control_bundle", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_negative_control_calibration_bundle",
                "parameters": [
                    {"name": "bundle", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "control_bundle", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.build_entry_negative_control_composer",
                "parameters": [
                    {"name": "control_bundle", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "calibration_bundle", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.build_entry_sign_reversed_composer",
                "parameters": [
                    {"name": "model_bundle", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "calibration_bundle", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "composer_bundle", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_negative_control_composer_bundle",
                "parameters": [{"name": "bundle", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.build_entry_negative_control_manifest",
                "parameters": [
                    {"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "control_bundles", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "calibration_bundles", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "composer_bundles", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "replay_config_sha256", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_negative_control_manifest",
                "parameters": [{"name": "manifest", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_negative_control_panel",
                "parameters": [{"name": "panel", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_control_replay_config",
                "parameters": [{"name": "config", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_control_exit_selection",
                "parameters": [{"name": "selection", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_nested_replay_config",
                "parameters": [{"name": "config", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.validate_entry_control_replay_result",
                "parameters": [
                    {"name": "result", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "config", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.session_balanced_population_mean_std",
                "parameters": [
                    {"name": "values", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "sessions", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "valid", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.weighted_lower_conformal_correction",
                "parameters": [
                    {"name": "raw_lower", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "realized", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "sessions", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "identities", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "alpha", "kind": "KEYWORD_ONLY", "required": False, "default": 0.1},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.compose_enter_statistics",
                "parameters": [
                    {"name": "mean_dollars", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "mean_returns", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "q10_dollars", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "q10_returns", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "available", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.wait_raw_lower_statistic",
                "parameters": [{"name": "legal_action_q10_dollar_composites", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.build_hgb_regressor",
                "parameters": [
                    {"name": "seed", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "loss", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "quantile", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.build_entry_neural_module",
                "parameters": [{"name": "seed", "kind": "KEYWORD_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.path_d.execution.research_fill_law.research_fill_law_from_preregistration",
                "parameters": [{"name": "payload", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.path_d.execution.research_fill_law.option_tick_micros",
                "parameters": [{"name": "reference_price_micros", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.path_d.execution.research_fill_law.evaluate_research_arrival",
                "parameters": [
                    {"name": "intent", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "arrival", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "law", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_fill_law.seal_research_quote",
                "parameters": [{"name": "source_row", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.replay_research_intent",
                "parameters": [
                    {"name": "prepared", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "law", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "delay_ms", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.prepare_research_execution",
                "parameters": [
                    {"name": "intent_kind", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "context", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "journal", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "law", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "reason_code", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.entry_decision_context_from_verified_example",
                "parameters": [
                    {"name": "example", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "decision", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.exit_decision_context_from_verified_journal",
                "parameters": [
                    {"name": "journal", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.genesis_research_ledger_journal",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "policy_id", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "fee_path", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.append_research_action_decision",
                "parameters": [
                    {"name": "journal", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "decision", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.entry_frame_coverage_from_verified_example",
                "parameters": [
                    {"name": "example", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "journal", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.append_entry_frame_coverage",
                "parameters": [
                    {"name": "journal", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "coverage", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.entry_action_observation_from_verified_dataset",
                "parameters": [
                    {"name": "decision", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "trajectory_id", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "episode_id", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "execution_outcome", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.append_entry_action_observation",
                "parameters": [
                    {"name": "journal", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "observation", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.mark_research_session_terminal",
                "parameters": [
                    {"name": "journal", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "reason_code", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.advance_research_ledger_session",
                "parameters": [
                    {"name": "journal", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.handoff_research_ledger_authorization",
                "parameters": [
                    {"name": "journal", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "current_authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "next_authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.validate_research_ledger_journal",
                "parameters": [
                    {"name": "journal", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.validate_research_ledger_journal_against_dataset",
                "parameters": [
                    {"name": "journal", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.reconstruct_entry_policy_evaluation_from_journal",
                "parameters": [
                    {"name": "journal", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset_sha256", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "sessions", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "evidence_role", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "inner_fold", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "policy_id", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.validate_research_execution_outcome",
                "parameters": [
                    {"name": "outcome", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "law", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.seal_research_result",
                "parameters": [
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "fill_law", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "payload", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.validate_research_result",
                "parameters": [
                    {"name": "envelope", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "fill_law", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_evidence_gate.begin_entry_evidence_once",
                "parameters": [
                    {"name": "role", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": False, "default": None},
                    {"name": "inner_fold", "kind": "KEYWORD_ONLY", "required": False, "default": None},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_evidence_gate.validate_entry_evidence_access",
                "parameters": [{"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_evidence_gate.claim_entry_evidence_decode_once",
                "parameters": [{"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True}],
            },
            {
                "qualified_name": "v4.research.pathd_evidence_gate.read_frozen_entry_evidence_authorization",
                "parameters": [
                    {"name": "role", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": False, "default": None},
                    {"name": "inner_fold", "kind": "KEYWORD_ONLY", "required": False, "default": None},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_evidence_gate.seal_invalid_nested_block_skip",
                "parameters": [
                    {"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "inner_fold", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_evidence_gate.seal_entry_evidence_result",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "evaluation", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_evidence_gate.inspect_entry_evidence_state",
                "parameters": [
                    {"name": "role", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": False, "default": None},
                    {"name": "inner_fold", "kind": "KEYWORD_ONLY", "required": False, "default": None},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_evidence_gate.recover_entry_evidence_after_crash",
                "parameters": [
                    {"name": "role", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": False, "default": None},
                    {"name": "inner_fold", "kind": "KEYWORD_ONLY", "required": False, "default": None},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_exit.assert_entry_result_aggregation_ready",
                "parameters": [
                    {"name": "role", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_holdout_gate.write_json_exclusive_durable",
                "parameters": [
                    {"name": "path", "kind": "POSITIONAL_OR_KEYWORD", "required": True},
                    {"name": "payload", "kind": "POSITIONAL_OR_KEYWORD", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_holdout_gate._freeze_complete_pre_holdout_packet",
                "parameters": [
                    {"name": "packet", "kind": "POSITIONAL_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_holdout_gate.inspect_protected_holdout_state",
                "parameters": [],
            },
            {
                "qualified_name": "v4.research.pathd_holdout_gate.execute_protected_holdout_once",
                "parameters": [],
            },
            {
                "qualified_name": "v4.research.pathd_holdout_gate.recover_protected_holdout_after_crash",
                "parameters": [],
            },
            {
                "qualified_name": "v4.path_d.risk.governor.DeterministicGovernor.forced_flat_intent",
                "parameters": [
                    {"name": "self", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "state", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "feed_health", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "now_utc", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "reference_vendor", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "reference_bid_micros", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "reference_ask_micros", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "reason_code", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.entry_evidence_role_for_session",
                "parameters": [
                    {"name": "assignments", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "inner_fold", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "session", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.run_entry_policy_evaluation",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "policy_bundle", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "composer_bundle", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "fill_law", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "opening_journal", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.validate_entry_policy_evaluation",
                "parameters": [
                    {"name": "evaluation", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.validate_entry_nested_family_evaluation",
                "parameters": [
                    {"name": "evaluation", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.validate_entry_outer_primary_result",
                "parameters": [
                    {"name": "result", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.freeze_entry_pooled_acceptance_once",
                "parameters": [],
            },
            {
                "qualified_name": "v4.research.pathd_entry_exit.validate_entry_pooled_acceptance_result",
                "parameters": [
                    {"name": "result", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_exit.read_frozen_entry_pooled_acceptance",
                "parameters": [
                    {"name": "require_pass", "kind": "KEYWORD_ONLY", "required": False, "default": False},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_exit.assert_entry_context_diagnostics_ready",
                "parameters": [],
            },
            {
                "qualified_name": "v4.research.pathd_entry_exit.assert_context_diagnostics_authorization_current",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_exit.read_validated_context_policy_journals",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.run_fixed_entry_context_diagnostics",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.validate_entry_outer_context_diagnostics",
                "parameters": [
                    {"name": "diagnostics", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.validate_entry_pooled_context_diagnostics",
                "parameters": [
                    {"name": "diagnostics", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            *[
                {
                    "qualified_name": (
                        "v4.scripts.run_pathd_entry_exit_research."
                        + function_name
                    ),
                    "parameters": [
                        {"name": "path", "kind": "POSITIONAL_ONLY", "required": True},
                    ],
                }
                for function_name in (
                    "validate_outer_entry_exit_artifacts_for_holdout",
                    "validate_four_box_packet_for_holdout",
                    "validate_guard_panel_for_holdout",
                    "validate_outer_acceptance_packet_for_holdout",
                    "validate_full_fit_entry_artifacts_for_holdout",
                    "validate_full_fit_exit_artifacts_for_holdout",
                )
            ],
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.run_fixed_protected_holdout_evaluator",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "POSITIONAL_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.validate_protected_holdout_trace",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "POSITIONAL_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.reconstruct_protected_holdout_evaluation_from_trace",
                "parameters": [
                    {"name": "authorization", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "dataset", "kind": "POSITIONAL_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.validate_protected_holdout_evaluation",
                "parameters": [
                    {"name": "evaluation", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.select_entry_action_index",
                "parameters": [
                    {"name": "mean_lcb_dollars", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "mean_lcb_returns", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "q10_dollars", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "q10_returns", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "physical_action_mask", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dynamic_account_mask", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "tie_break_keys", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.transform_entry_negative_control_bundle",
                "parameters": [
                    {"name": "control_id", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "target_bundles", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "target_validity", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "feature_histories", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "row_identities", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "seed", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.reverse_entry_calibrated_statistics",
                "parameters": [
                    {"name": "mean_lcb_dollars", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "mean_lcb_returns", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "q10_dollars", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "q10_returns", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.build_matched_random_schedule",
                "parameters": [
                    {"name": "proposals", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "quotas", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "seed", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "policy_id", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.realize_frozen_matched_random_schedule",
                "parameters": [
                    {"name": "schedule", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "proposal_population", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "outcomes", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.session_bootstrap_mean_lcb_correction",
                "parameters": [
                    {"name": "residuals", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "sessions", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "seed", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "resamples", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.research.pathd_entry_models.reconstruct_entry_action_calibration_gate_inputs",
                "parameters": [
                    {"name": "observations", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "action", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "evidence_sessions", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "seed", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.path_d.execution.research_replay.reconstruct_entry_policy_economics_from_validated_transitions",
                "parameters": [
                    {"name": "transitions", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "starting_cash_micros", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.validate_entry_outer_selection_binding",
                "parameters": [
                    {"name": "result", "kind": "POSITIONAL_ONLY", "required": True},
                    {"name": "authorization_sha256", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset_sha256", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "source_receipts_root_sha256", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "access_receipt_sha256", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.validate_protected_holdout_trace_records",
                "parameters": [
                    {"name": "records", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "authorization_sha256", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "dataset_sha256", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "sessions", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "box_d_policy_id", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "comparator_policy_id", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.reconstruct_protected_holdout_payload_from_validated_trace",
                "parameters": [
                    {"name": "records", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "primary_sessions", "kind": "KEYWORD_ONLY", "required": True},
                    {"name": "degradation_session", "kind": "KEYWORD_ONLY", "required": True},
                ],
            },
            {
                "qualified_name": "v4.scripts.run_pathd_entry_exit_research.run_canonical_entry_fold",
                "parameters": [{"name": "outer_fold", "kind": "KEYWORD_ONLY", "required": True}],
            },
        ],
        "immutable_gate": {
            "path": "v4/tests/test_pathd_entry_exit_gate_frozen.py",
            "test_names": list(FROZEN_GATE_TEST_NAMES),
            "forbidden_pytest_controls": ["skip", "skipif", "xfail", "importorskip"],
        },
    }


def result_artifact_classification() -> dict[str, Any]:
    """Classify every frozen future API schema once, with no universal-receipt claim."""

    schema_names = set(entry_future_api_contract()["schema_versions"])
    standalone = {
        "EntryNegativeControlPanelV1",
        "EntryPolicyEvaluationV1",
        "EntryNestedFamilyEvaluationV1",
        "EntryControlReplayResultV1",
        "EntryControlExitSelectionV1",
        "EntryPooledAcceptanceResultV1",
        "EntryOuterContextDiagnosticsV1",
        "EntryPooledContextDiagnosticsV1",
        "ProtectedHoldoutTraceRecordV1",
    }
    embedded = {
        "EntryOuterPrimaryResultV1",
        "ProtectedHoldoutEvaluationV1",
    }
    non_result = schema_names - standalone - embedded
    if standalone & embedded or standalone & non_result or embedded & non_result:
        raise AssertionError("result artifact classes overlap")
    if standalone | embedded | non_result != schema_names:
        raise AssertionError("result artifact classification is incomplete")

    fixed_paths: dict[str, str] = {}
    for fold in range(1, 6):
        for inner in range(1, 5):
            fixed_paths[
                repo_path_label(
                    _outer_fold_artifact_path(
                        fold, f"nested_inner_{inner}_result.json"
                    )
                )
            ] = "standalone_scientific_result"
        for name, classification in (
            ("shortened_result.json", "standalone_scientific_result"),
            ("negative_control_panel.json", "standalone_scientific_result"),
            ("control_replay_result.json", "standalone_scientific_result"),
            ("control_exit.json", "standalone_scientific_result"),
            ("outer_primary_result.json", "enveloped_scientific_result"),
            ("entry_context_diagnostics.json", "standalone_scientific_result"),
        ):
            fixed_paths[repo_path_label(_outer_fold_artifact_path(fold, name))] = (
                classification
            )
    fixed_paths[repo_path_label(ENTRY_POOLED_ACCEPTANCE_RESULT_PATH)] = (
        "standalone_scientific_result"
    )
    fixed_paths[repo_path_label(ENTRY_POOLED_CONTEXT_DIAGNOSTICS_PATH)] = (
        "standalone_scientific_result"
    )
    fixed_paths[repo_path_label(HOLDOUT_TRACE_PATH)] = (
        "standalone_scientific_result_jsonl_records"
    )
    fixed_paths[repo_path_label(HOLDOUT_RESULT_PATH)] = "enveloped_scientific_result"
    return {
        "schema_version": "pathd.result_artifact_classification.v1",
        "classes": {
            "standalone_scientific_result": sorted(standalone),
            "embedded_scientific_result": sorted(embedded),
            "non_result_contract_control_or_evidence_artifact": sorted(non_result),
        },
        "standalone_rule": "the standalone schema carries holdout_caveat exactly and its fixed writer/terminal result receipt supplies and validates the complete quarantine, claim, plan, preregistration, fill-law, and holdout-open metadata before durable write",
        "embedded_rule": "the schema may appear only as the payload of a fully caveated and validated ResearchResultEnvelopeV1 or protected-holdout result envelope; it is never written as a bare scientific result",
        "non_result_rule": "access, lease, burn, abort, seal, skip, preopen, authorization, dataset, source, integrity, test, machinery, model, configuration, manifest, journal, and execution evidence are control/provenance artifacts rather than scientific findings and are not falsely required to be universal result envelopes",
        "fixed_named_result_paths": dict(sorted(fixed_paths.items())),
        "bootstrap_non_results": [
            "preregistration.json",
            "session_assignments.json",
            "feature_lineage.json",
            "hash-only sidecars",
            "owner authorization receipt",
            "pre-existing corpus integrity manifest",
        ],
    }


def validate_immutable_gate_source() -> None:
    """Prove the pre-freeze gate has exactly ten unconditional tests and no bypass."""

    path = REPO_ROOT / "v4/tests/test_pathd_entry_exit_gate_frozen.py"
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValueError("immutable gate source does not parse") from exc
    tests = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    if tests != list(FROZEN_GATE_TEST_NAMES) or len(tests) != len(set(tests)):
        raise ValueError("immutable gate testcase identity drift")
    forbidden_attributes = {"skip", "skipif", "xfail", "importorskip"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            if node.decorator_list:
                raise ValueError("immutable gate test decorators are forbidden")
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Attribute) and target.attr in forbidden_attributes:
                raise ValueError("immutable gate bypass control is forbidden")
            if isinstance(target, ast.Name) and target.id in forbidden_attributes:
                raise ValueError("immutable gate bypass control is forbidden")


def validate_fixed_science_gate_source() -> None:
    """Freeze the exact seventeen-test scientific oracle without bypass controls."""

    label = "v4/tests/test_pathd_fixed_science_contract.py"
    path = _canonical_repo_regular_file(label)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ValueError("fixed-science gate source does not parse") from exc
    tests = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    if tests != list(FIXED_SCIENCE_GATE_TEST_NAMES) or len(tests) != len(set(tests)):
        raise ValueError("fixed-science gate testcase identity drift")
    forbidden = {"skip", "skipif", "xfail", "importorskip", "parametrize"}
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            and node.decorator_list
        ):
            raise ValueError("fixed-science gate test decorators are forbidden")
        if isinstance(node, ast.Call):
            target = node.func
            if (
                isinstance(target, ast.Attribute) and target.attr in forbidden
            ) or (isinstance(target, ast.Name) and target.id in forbidden):
                raise ValueError("fixed-science gate bypass control is forbidden")
    _reject_dynamic_dependency_escape(label, tree)
    escaped = sorted(
        _resolved_v4_import_paths(label, tree)
        - set(AUTHORIZED_IN_REPO_DEPENDENCY_PATHS)
    )
    if escaped:
        raise ValueError(
            f"fixed-science gate imports unregistered in-repo dependency: {escaped}"
        )
def receipt_contract_spec() -> dict[str, Any]:
    fit_test_env = {
        "PYTHONPATH": ".",
        **entry_fit_environment_spec()["required_environment"],
    }

    def pytest_command(
        test_path: str, junit_path: Path, test_names: tuple[str, ...]
    ) -> dict[str, Any]:
        return {
            "argv": [
                ".venv/bin/python",
                "-m",
                "pytest",
                test_path,
                "-q",
                "--junitxml=" + repo_path_label(junit_path),
            ],
            "cwd": ".",
            "env": dict(fit_test_env),
            "junit_path": repo_path_label(junit_path),
            "expected_junit_summary": {
                "tests": len(test_names),
                "passed": len(test_names),
                "failures": 0,
                "errors": 0,
                "skipped": 0,
            },
            "required_testcase_names": list(test_names),
        }

    return {
        "preregistration_path": repo_path_label(PREREG_PATH),
        "preregistration_hash_path": repo_path_label(PREREG_HASH_PATH),
        "session_assignments_path": repo_path_label(SESSION_PATH),
        "feature_lineage_path": repo_path_label(LINEAGE_PATH),
        "freeze_receipt_path": repo_path_label(FREEZE_RECEIPT_PATH),
        "lineage_receipt_path": repo_path_label(LINEAGE_IMPLEMENTATION_RECEIPT_PATH),
        "machinery_receipt_path": repo_path_label(ENTRY_MACHINERY_RECEIPT_PATH),
        "machinery_test_evidence_path": repo_path_label(
            ENTRY_MACHINERY_TEST_EVIDENCE_PATH
        ),
        "machinery_junit_path": repo_path_label(ENTRY_MACHINERY_JUNIT_PATH),
        "prereg_supplemental_junit_path": repo_path_label(
            ENTRY_PREREG_SUPPLEMENTAL_JUNIT_PATH
        ),
        "entry_evidence_gate_junit_path": repo_path_label(
            ENTRY_EVIDENCE_GATE_JUNIT_PATH
        ),
        "protected_holdout_gate_junit_path": repo_path_label(
            PROTECTED_HOLDOUT_GATE_JUNIT_PATH
        ),
        "fixed_science_gate_junit_path": repo_path_label(
            FIXED_SCIENCE_GATE_JUNIT_PATH
        ),
        "feature_live_twin_junit_path": repo_path_label(
            FEATURE_LIVE_TWIN_JUNIT_PATH
        ),
        "evidence_capability_hardening_junit_path": repo_path_label(
            EVIDENCE_CAPABILITY_HARDENING_JUNIT_PATH
        ),
        "foundation_correction_junit_path": repo_path_label(
            FOUNDATION_CORRECTION_JUNIT_PATH
        ),
        "corpus_integrity_receipt_path": repo_path_label(CORPUS_INTEGRITY_RECEIPT_PATH),
        "foundation_restoration_receipt_path": repo_path_label(
            FOUNDATION_RESTORATION_RECEIPT_PATH
        ),
        "foundation_stability_receipt_path": repo_path_label(
            FOUNDATION_STABILITY_RECEIPT_PATH
        ),
        "context_diagnostic_inventory_receipt_path": repo_path_label(
            CONTEXT_DIAGNOSTIC_INVENTORY_RECEIPT_PATH
        ),
        "entry_pooled_acceptance_paths": {
            "result": repo_path_label(ENTRY_POOLED_ACCEPTANCE_RESULT_PATH),
            "receipt": repo_path_label(ENTRY_POOLED_ACCEPTANCE_RECEIPT_PATH),
        },
        "entry_context_diagnostics_paths": {
            "authorization": repo_path_label(
                ENTRY_CONTEXT_DIAGNOSTICS_AUTHORIZATION_PATH
            ),
            "access_receipt": repo_path_label(
                ENTRY_CONTEXT_DIAGNOSTICS_ACCESS_RECEIPT_PATH
            ),
            "outer_artifacts": [
                repo_path_label(
                    _outer_fold_artifact_path(
                        fold, "entry_context_diagnostics.json"
                    )
                )
                for fold in range(1, 6)
            ],
            "outer_receipts": [
                repo_path_label(
                    _outer_fold_artifact_path(
                        fold, "entry_context_diagnostics_receipt.json"
                    )
                )
                for fold in range(1, 6)
            ],
            "pooled_artifact": repo_path_label(
                ENTRY_POOLED_CONTEXT_DIAGNOSTICS_PATH
            ),
            "pooled_receipt": repo_path_label(
                ENTRY_POOLED_CONTEXT_DIAGNOSTICS_RECEIPT_PATH
            ),
            "abort_receipt": repo_path_label(
                ENTRY_CONTEXT_DIAGNOSTICS_ABORT_RECEIPT_PATH
            ),
        },
        "complete_pre_holdout_packet_path": repo_path_label(PRE_HOLDOUT_PACKET_PATH),
        "protected_holdout_paths": {
            "root": repo_path_label(PROTECTED_HOLDOUT_ROOT),
            "lock": repo_path_label(HOLDOUT_LOCK_PATH),
            "access_receipt": repo_path_label(HOLDOUT_ACCESS_RECEIPT_PATH),
            "result": repo_path_label(HOLDOUT_RESULT_PATH),
            "seal_receipt": repo_path_label(HOLDOUT_SEAL_RECEIPT_PATH),
            "abort_receipt": repo_path_label(HOLDOUT_ABORT_RECEIPT_PATH),
            "trace": repo_path_label(HOLDOUT_TRACE_PATH),
            "evaluator_receipt": repo_path_label(HOLDOUT_EVALUATOR_RECEIPT_PATH),
        },
        "lineage_schema_version": "pathd.entry_exit.lineage_implementation_receipt.v1",
        "machinery_schema_version": "pathd.entry_exit.entry_machinery_receipt.v1",
        "machinery_test_evidence_schema_version": "pathd.entry_exit.entry_machinery_test_evidence.v1",
        "corpus_integrity_receipt_schema_version": "pathd.entry_exit.corpus_integrity_verification_receipt.v1",
        "foundation_restoration_receipt_schema_version": (
            "pathd.entry_exit.foundation_restoration_receipt.v3"
        ),
        "foundation_stability_receipt_schema_version": (
            "pathd.entry_exit.foundation_stability_receipt.v3"
        ),
        "context_diagnostic_inventory_receipt_schema_version": "pathd.entry_exit.context_diagnostic_inventory_receipt.v1",
        "core_corpus_integrity_receipt": core_corpus_integrity_receipt_spec(),
        "implementation_sha256_format": "lowercase_hex_64",
        "required_test_commands": [
            pytest_command(
                "v4/tests/test_pathd_entry_exit_gate_frozen.py",
                ENTRY_MACHINERY_JUNIT_PATH,
                FROZEN_GATE_TEST_NAMES,
            ),
            pytest_command(
                "v4/tests/test_pathd_entry_exit_research.py",
                ENTRY_PREREG_SUPPLEMENTAL_JUNIT_PATH,
                ENTRY_PREREG_SUPPLEMENTAL_TEST_NAMES,
            ),
            pytest_command(
                "v4/tests/test_pathd_evidence_gate.py",
                ENTRY_EVIDENCE_GATE_JUNIT_PATH,
                ENTRY_EVIDENCE_GATE_TEST_NAMES,
            ),
            pytest_command(
                "v4/tests/test_pathd_holdout_gate.py",
                PROTECTED_HOLDOUT_GATE_JUNIT_PATH,
                PROTECTED_HOLDOUT_GATE_TEST_NAMES,
            ),
            pytest_command(
                "v4/tests/test_pathd_fixed_science_contract.py",
                FIXED_SCIENCE_GATE_JUNIT_PATH,
                FIXED_SCIENCE_GATE_TEST_NAMES,
            ),
            pytest_command(
                "v4/tests/test_pathd_feature_live_twin.py",
                FEATURE_LIVE_TWIN_JUNIT_PATH,
                FEATURE_LIVE_TWIN_TEST_NAMES,
            ),
            pytest_command(
                "v4/tests/test_pathd_evidence_gate_capability_hardening.py",
                EVIDENCE_CAPABILITY_HARDENING_JUNIT_PATH,
                EVIDENCE_CAPABILITY_HARDENING_TEST_NAMES,
            ),
            pytest_command(
                "v4/tests/test_pathd_foundation_correction.py",
                FOUNDATION_CORRECTION_JUNIT_PATH,
                FOUNDATION_CORRECTION_TEST_NAMES,
            ),
        ],
        "required_integrity_command": {
            "argv": [
                ".venv/bin/python",
                "-m",
                "v4.scripts.run_pathd_entry_exit_research",
                "verify-corpus-integrity",
            ],
            "cwd": ".",
            "env": {"PYTHONPATH": "."},
        },
        "required_context_diagnostic_inventory_command": {
            "argv": [
                ".venv/bin/python",
                "-m",
                "v4.scripts.run_pathd_entry_exit_research",
                "verify-context-diagnostic-inventory",
            ],
            "cwd": ".",
            "env": {"PYTHONPATH": "."},
        },
        "lineage_evidence_checks": [
            "all_17_positive_fixtures",
            "five_transitive_negative_fixtures",
            "historical_online_replay_byte_equality",
            "clock_watermark",
            "field_parity_inventory_exact_17_49_48_47",
            "entry_cbbo_1m_lineage",
            "intraday_open_interest_nonempty_adapter_rejected",
            "minute_volume_exact_clock_carry_and_receipt",
            "entry_signed17_unchanged_after_correction",
        ],
        "machinery_evidence_checks": [
            "schema",
            "negative_fixtures",
            "mutate_future_features",
            "mutate_future_predictions",
            "mutate_future_labels_live",
            "inner_folds",
            "holdout_access_zero",
            "shared_fill_law_hash",
            "fit_authorization_tamper_matrix",
            "result_envelope_fail_closed",
            "corpus_loader_hash_before_read",
            "foundation_default_preservation_and_delta",
            "closed_in_repo_dependency_graph",
            "prohibited_external_dependency_imports",
            "prohibited_os_process_launches",
            "entry_evidence_access_receipt_precedes_decode",
            "entry_evidence_foundation_revalidated_after_open",
            "entry_evidence_source_bytes_rehashed_before_decode",
            "entry_evidence_crash_burns_unsealed_result_no_reopen",
            "nested_evidence_exclusive_count_one",
            "outer_evidence_exclusive_count_one",
            "nested_result_binding_fail_closed",
            "outer_result_binding_fail_closed",
            "policy_journal_dataset_example_membership_and_order",
            "policy_journal_complete_state_aware_frame_coverage",
            "global_hgb_family_frozen_before_fit_or_outer_open",
            "neural_entry_requires_same_scope_hgb_baseline",
            "entry_composer_full_action_mask",
            "negative_control_transformations_complete_and_tamper_evident",
            "matched_random_schedule_quota_collision_no_redraw",
            "calibration_statistics_fixed_vectors",
            "ledger_authorization_cursor_coverage",
            "protected_holdout_access_receipt_precedes_decode",
            "protected_holdout_no_live_authorization_public_api",
            "protected_holdout_result_binding_fail_closed",
            "protected_holdout_exclusive_count_one",
            "protected_holdout_crash_burns_unsealed_result_no_reopen",
            "protected_holdout_corpus_receipt_semantically_revalidated",
            "protected_holdout_trace_reconstruction_scientific_exact",
            "protected_holdout_trace_tamper_rejected",
            "protected_holdout_caller_summary_rejected",
            "preholdout_all_six_semantic_validators_fail_closed",
            "aref_decision_critical_support_closure",
            "aref_hgb_neural_topology_and_gradient_boundaries",
            "aref_exit_coverage_identity_and_dependency_tamper",
            "context_diagnostics_closed_post_primary_nonalpha",
            "context_diagnostics_prereg_exact",
            "owner_locked_operational_contract_cross_binding",
            "postfreeze_result_caveat_classification",
            "benign_test_contamination_restores_five_folds",
            "corrected_v2_immutable_predecessor_and_build_only_pause",
            "refined_br_exact_five_valid_and_no_rescue",
            "refined_br_entry_verdict_preserved_on_exit_power_stop",
            "foundation_byte_root_is_mutation_sensitive",
            "foundation_stability_seal_requires_pristine_folds",
            "detached_authorization_cannot_touch_fixed_scope",
            "active_foundation_drift_burns_and_releases",
            "evidence_scope_id_is_generation_bound",
            "nested_journal_membership_is_current_segment_only",
            "widen_entry_followup_is_nonalpha",
        ],
        "evidence_check_to_testcase": {
            "schema": "test_public_api_surface_exact",
            "all_17_positive_fixtures": "test_signed17_historical_source_neutral_byte_equality",
            "historical_online_replay_byte_equality": "test_signed17_historical_source_neutral_byte_equality",
            "five_transitive_negative_fixtures": "test_signed17_lineage_fail_closed",
            "negative_fixtures": "test_signed17_lineage_fail_closed",
            "clock_watermark": "test_mutate_future_is_causal_and_live",
            "field_parity_inventory_exact_17_49_48_47": "test_exact_signed17_original_exit49_intermediate_exit48_and_corrected_exit47_coverage",
            "entry_cbbo_1m_lineage": "test_entry_option_features_bind_historical_cbbo_1m",
            "intraday_open_interest_nonempty_adapter_rejected": "test_feature_without_live_twin_fixture_rejects_intraday_open_interest_even_with_adapter",
            "minute_volume_exact_clock_carry_and_receipt": "test_minute_volume_is_conditionally_live_derivable_with_exact_receipt[90-DATABENTO_OPRA_OHLCV_1M_COMPLETED]",
            "entry_signed17_unchanged_after_correction": "test_every_signed17_feature_is_live_derivable_without_changing_signed17",
            "mutate_future_features": "test_mutate_future_is_causal_and_live",
            "mutate_future_predictions": "test_mutate_future_is_causal_and_live",
            "mutate_future_labels_live": "test_mutate_future_is_causal_and_live",
            "inner_folds": "test_frozen_roles_history_and_model_input_boundary",
            "shared_fill_law_hash": "test_shared_fill_law_exact",
            "foundation_default_preservation_and_delta": "test_default_simulator_governor_unchanged_and_research_opt_in",
            "corpus_loader_hash_before_read": "test_corpus_hash_precedes_decode",
            "result_envelope_fail_closed": "test_pathd_result_envelope_fails_closed",
            "holdout_access_zero": "test_fit_gate_tamper_matrix",
            "fit_authorization_tamper_matrix": "test_pathd_stale_or_altered_fit_authorization_is_rejected",
            "closed_in_repo_dependency_graph": "test_authorized_dependency_closure_rejects_unregistered_v4_and_test_imports",
            "prohibited_external_dependency_imports": "test_pathd_production_dependency_closure_rejects_external_escape_imports",
            "prohibited_os_process_launches": "test_pathd_production_dependency_closure_rejects_os_process_launches",
            "entry_evidence_access_receipt_precedes_decode": "test_access_count_zero_to_one_and_decode_precedence",
            "entry_evidence_foundation_revalidated_after_open": "test_foundation_drift_after_open_burns_before_decode",
            "entry_evidence_source_bytes_rehashed_before_decode": "test_sealed_source_bytes_are_rehashed_without_decode",
            "entry_evidence_crash_burns_unsealed_result_no_reopen": "test_crash_with_durable_unsealed_result_is_permanently_burned",
            "nested_evidence_exclusive_count_one": "test_concurrent_and_repeated_open_are_impossible",
            "outer_evidence_exclusive_count_one": "test_nested_and_outer_evidence_scopes_open_and_seal_exactly_once",
            "nested_result_binding_fail_closed": "test_result_is_bound_to_exact_dataset_and_revokes_capability",
            "outer_result_binding_fail_closed": "test_nested_and_outer_results_bind_dataset_example_membership_authorization_and_ledger",
            "policy_journal_dataset_example_membership_and_order": "test_nested_and_outer_results_bind_dataset_example_membership_authorization_and_ledger",
            "policy_journal_complete_state_aware_frame_coverage": "test_frame_coverage_visits_every_ordered_example_and_rejects_omitted_eligible_frame",
            "global_hgb_family_frozen_before_fit_or_outer_open": "test_entry_family_selection_is_globally_frozen_before_any_outer_open",
            "neural_entry_requires_same_scope_hgb_baseline": "test_neural_entry_fit_requires_frozen_hgb_baseline_for_same_scope",
            "entry_composer_full_action_mask": "test_entry_composer_scores_wait_and_all_42_actions_under_physical_mask",
            "negative_control_transformations_complete_and_tamper_evident": "test_entry_negative_controls_transform_complete_bundles_and_reject_tamper",
            "matched_random_schedule_quota_collision_no_redraw": "test_matched_random_schedules_enforce_hash_quota_collision_and_no_redraw",
            "calibration_statistics_fixed_vectors": "test_calibration_statistics_fixed_vectors_cover_seed_lcb_q10_and_action_deciles",
            "ledger_authorization_cursor_coverage": "test_entry_policy_evaluation_reconstructs_nonzero_economics_from_full_journal",
            "protected_holdout_access_receipt_precedes_decode": "test_access_is_exclusive_durable_and_precedes_decode",
            "protected_holdout_no_live_authorization_public_api": "test_access_is_exclusive_durable_and_precedes_decode",
            "protected_holdout_result_binding_fail_closed": "test_result_is_bound_and_durably_sealed",
            "protected_holdout_exclusive_count_one": "test_access_is_exclusive_durable_and_precedes_decode",
            "protected_holdout_crash_burns_unsealed_result_no_reopen": "test_crash_after_durable_result_burns_without_seal_or_reopen",
            "protected_holdout_corpus_receipt_semantically_revalidated": "test_preregistration_revalidation_semantically_validates_corpus_receipt",
            "protected_holdout_trace_reconstruction_scientific_exact": "test_fixed_holdout_trace_reconstructs_synthetic_30_session_result",
            "protected_holdout_trace_tamper_rejected": "test_fixed_holdout_trace_rejects_all_payload_identity_partition_and_hash_tamper",
            "protected_holdout_caller_summary_rejected": "test_fixed_holdout_evaluator_rejects_caller_authored_summary",
            "preholdout_all_six_semantic_validators_fail_closed": "test_preholdout_packet_runs_all_six_semantic_validators_and_rejects_declared_pass",
            "aref_decision_critical_support_closure": "test_aref_decision_critical_support_closure_and_gradient_boundaries",
            "aref_hgb_neural_topology_and_gradient_boundaries": "test_aref_hgb_and_neural_topologies_are_exact",
            "aref_exit_coverage_identity_and_dependency_tamper": "test_aref_exit_coverage_identity_and_dependency_tamper_fail_closed",
            "context_diagnostics_closed_post_primary_nonalpha": "test_context_diagnostics_are_fixed_one_dimensional_post_primary_nonalpha_tables",
            "context_diagnostics_prereg_exact": "test_pathd_vix_es_vx_diagnostic_contract_is_closed_and_nonalpha",
            "owner_locked_operational_contract_cross_binding": "test_pathd_owner_locked_operational_contract_tamper_matrix",
            "postfreeze_result_caveat_classification": "test_pathd_every_postfreeze_result_carries_exact_holdout_caveat",
            "benign_test_contamination_restores_five_folds": "test_p1_forensic_contract_is_benign_zero_access_and_restores_five",
            "corrected_v2_immutable_predecessor_and_build_only_pause": "test_v3_preserves_corrected_v2_bytes_and_remains_build_only",
            "refined_br_exact_five_valid_and_no_rescue": "test_refined_br_requires_exact_five_valid_folds_and_rejects_rescue",
            "refined_br_entry_verdict_preserved_on_exit_power_stop": "test_refined_br_preserves_entry_verdict_on_exit_power_stop",
            "foundation_byte_root_is_mutation_sensitive": "test_foundation_byte_root_changes_on_any_file_or_generation_change",
            "foundation_stability_seal_requires_pristine_folds": "test_stability_seal_requires_all_corrected_fold_namespaces_pristine",
            "foundation_mismatch_stops_before_fold_dispatch": "test_fold_stage_preflight_stops_before_dispatch_on_foundation_mismatch",
            "detached_authorization_cannot_touch_fixed_scope": "test_untrusted_exact_authorization_cannot_read_or_burn_fixed_scope[detached]",
            "active_foundation_drift_burns_and_releases": "test_active_capability_foundation_drift_burns_and_releases_transaction",
            "evidence_scope_id_is_generation_bound": "test_scope_id_is_bound_to_frozen_foundation_generation",
            "nested_journal_membership_is_current_segment_only": "test_nested_dataset_membership_uses_only_current_authorization_segment",
            "widen_entry_followup_is_nonalpha": "test_p3_widen_entry_lead_is_forward_only",
        },
    }


def corpus_sessions(corpus_root: Path = CORPUS_ROOT) -> list[str]:
    raw = corpus_root / "raw/databento/opra_spxw_cbbo_1s"
    sessions: list[str] = []
    for path in raw.glob("*.parquet"):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
        if match:
            sessions.append(match.group(1))
    sessions = sorted(set(sessions))
    if len(sessions) != 251:
        raise ValueError(f"expected 251 paired Path-D sessions, found {len(sessions)}")
    if sessions[0] != "2025-08-01" or sessions[-1] != "2026-07-31":
        raise ValueError("Path-D corpus endpoints drifted")
    return sessions


def session_assignments(sessions: list[str] | None = None) -> dict[str, Any]:
    values = list(sessions or corpus_sessions())
    fold_specs = (
        (1, 70, 71, 72, 99),
        (2, 99, 100, 101, 128),
        (3, 128, 129, 130, 157),
        (4, 157, 158, 159, 186),
        (5, 186, 187, 188, 215),
    )
    folds: list[dict[str, Any]] = []
    prior_embargoes: set[str] = set()
    degraded = set(OPRA_DEGRADED_DATES)
    shortened = set(SCHEDULED_SHORT_SESSIONS)
    for fold, train_end, embargo_index, test_start, test_end in fold_specs:
        embargo = values[embargo_index - 1]
        train_all = values[:train_end]
        train_primary = [
            session
            for session in train_all
            if session not in prior_embargoes
            and session not in degraded
            and session not in shortened
        ]
        calibration_count = max(1, int(math.ceil(len(train_primary) * 0.20)))
        calibration = train_primary[-calibration_count:]
        before_calibration = train_primary[:-calibration_count]
        calibration_embargo = before_calibration[-1:]
        model_fit = before_calibration[:-1]
        inner_manifest = forward_inner_folds(model_fit)
        inner_scores = inner_manifest["scored_forward_folds"]
        exit_weight_sessions = [
            session
            for inner in inner_scores[:3]
            if inner["calibration_valid"]
            for session in inner["validation"]
        ]
        exit_calibration_sessions = (
            list(inner_scores[3]["validation"])
            if inner_scores[3]["calibration_valid"]
            else []
        )
        outer_test = values[test_start - 1 : test_end]
        outer_test_primary = [session for session in outer_test if session not in shortened]
        outer_test_shortened = [session for session in outer_test if session in shortened]
        if degraded.intersection(outer_test):
            raise ValueError(f"OPRA-degraded date entered fold {fold} outer test")
        folds.append(
            {
                "fold": fold,
                "train_index_range": [1, train_end],
                "train_all": train_all,
                "train_primary_excluding_prior_embargo_and_opra_degraded": train_primary,
                "model_fit": model_fit,
                "calibration_embargo": calibration_embargo,
                "calibration_last_20_percent": calibration,
                "calibration_count_rule": "ceil(0.20*N_primary)",
                "inner_forward_folds": inner_manifest,
                "exit_training_oof_mapping": {
                    "entry_bundle_source": "nested inner bundle frozen before each validation block",
                    "exit_model_weight_sessions": exit_weight_sessions,
                    "exit_calibration_sessions": exit_calibration_sessions,
                    "exit_outer_evidence_sessions": outer_test_primary,
                    "rule": "entry OOF validation blocks 2-4 fit exit weights; block 5 calibrates exit; outer-test positions are exit evidence only",
                    "weight_session_count": len(exit_weight_sessions),
                    "calibration_session_count": len(exit_calibration_sessions),
                    "invalid_inner_blocks": [
                        inner["inner_fold"]
                        for inner in inner_scores
                        if not inner["calibration_valid"]
                    ],
                },
                "embargo": [embargo],
                "outer_test_index_range": [test_start, test_end],
                "outer_test": outer_test,
                "outer_test_primary_1555_complete": outer_test_primary,
                "outer_test_shortened_diagnostic_only": outer_test_shortened,
            }
        )
        prior_embargoes.add(embargo)

    firewall_raw = values[215:]
    burned = set(BURNED_SMOKE_DATES)
    protected = [session for session in firewall_raw if session not in burned]
    if len(firewall_raw) != 36 or len(protected) != 30:
        raise AssertionError("Path-D firewall arithmetic drifted")
    if protected[:1] != ["2026-06-10"] or protected[-1:] != ["2026-07-31"]:
        raise AssertionError("Path-D protected holdout endpoints drifted")
    full_primary = [
        session
        for session in values[:215]
        if session not in prior_embargoes
        and session not in degraded
        and session not in shortened
    ]
    full_calibration_count = int(math.ceil(len(full_primary) * 0.20))
    full_calibration = full_primary[-full_calibration_count:]
    full_before_calibration = full_primary[:-full_calibration_count]
    full_calibration_embargo = full_before_calibration[-1:]
    full_model_fit = full_before_calibration[:-1]
    full_exit_weight_sessions = [
        session for fold in folds[:4] for session in fold["outer_test_primary_1555_complete"]
    ]
    full_exit_calibration_sessions = list(folds[4]["outer_test_primary_1555_complete"])
    manifest = {
        "schema_version": "pathd.entry_exit.sessions.v1",
        "all_251_sessions": values,
        "initial_history_indices_1_70": values[:70],
        "folds": folds,
        "pre_holdout_session_indices_1_215": values[:215],
        "full_fit_pre_holdout_bundle": {
            "entry_model_fit": full_model_fit,
            "entry_calibration_embargo": full_calibration_embargo,
            "entry_calibration": full_calibration,
            "entry_model_fit_count": len(full_model_fit),
            "entry_calibration_count": len(full_calibration),
            "exit_weight_trajectory_sessions": full_exit_weight_sessions,
            "exit_calibration_trajectory_sessions": full_exit_calibration_sessions,
            "exit_source": "five frozen outer-fold OOF entry traces only; folds 1-4 weights and fold 5 calibration",
            "holdout_use": "full-fit entry/exit bundle may predict protected sessions only after pre-open packet freeze",
        },
        "firewall_raw_indices_216_251": firewall_raw,
        "burned_smoke_dates": list(BURNED_SMOKE_DATES),
        "protected_holdout_30": protected,
        "protected_holdout_primary_non_degraded_29": [
            session for session in protected if session not in degraded
        ],
        "opra_degraded_diagnostic_only": list(OPRA_DEGRADED_DATES),
        "scheduled_short_sessions_diagnostic_only": list(SCHEDULED_SHORT_SESSIONS),
        "glbx_degraded_strata_only": list(GLBX_DIAGNOSTIC_DEGRADED_DATES),
        "holdout_open_count": 0,
        "holdout_caveat": HOLDOUT_CAVEAT,
    }
    manifest["ordered_list_hashes_sha256_newline"] = {
        "all_251_sessions": canonical_session_hash(values),
        "initial_history_indices_1_70": canonical_session_hash(values[:70]),
        "outer_embargo_5": canonical_session_hash(
            [fold["embargo"][0] for fold in folds]
        ),
        **{
            f"outer_fold_{fold['fold']}_test": canonical_session_hash(fold["outer_test"])
            for fold in folds
        },
        "pre_holdout_indices_1_215": canonical_session_hash(values[:215]),
        "firewall_raw_indices_216_251": canonical_session_hash(firewall_raw),
        "burned_smoke_dates": canonical_session_hash(BURNED_SMOKE_DATES),
        "scheduled_short_sessions_diagnostic_only": canonical_session_hash(
            SCHEDULED_SHORT_SESSIONS
        ),
        "protected_holdout_30": canonical_session_hash(protected),
    }
    return manifest


def feature_lineage() -> dict[str, Any]:
    validate_inventory_contracts()
    builder_path = REPO_ROOT / "v4/model/protocol101_canonical_stage1_contract.py"
    live_twin_path = REPO_ROOT / "v4/research/pathd_feature_live_twin.py"
    builder = "v4/model/protocol101_canonical_stage1_contract.py::feature_matrix"
    leaves_by_feature = {
        "spx_vwap_gap_points": ["spx_close", "spx_vwap"],
        "spx_vwap_gap_bps": ["spx_close", "spx_vwap"],
        "spx_vwap_gap_over_session_range": [
            "spx_close",
            "spx_vwap",
            "session_range",
        ],
        "session_range_bps": ["spx_close", "session_range"],
        "momentum_5m_bps": ["spx_close", "momentum_5m"],
        "momentum_15m_bps": ["spx_close", "momentum_15m"],
        "momentum_5m_over_session_range": ["momentum_5m", "session_range"],
        "momentum_15m_over_session_range": ["momentum_15m", "session_range"],
        "omar_clipped_neg3_pos3": ["omar"],
        "vwap_side_alignment_flag": ["spx_close", "spx_vwap", "right"],
        "omar_side_alignment_flag": ["omar", "right"],
        "momentum15_side_alignment_flag": ["momentum_15m", "right"],
        "D.near_atm.straddle_mid_spot_bps": [
            "all_near_atm_quantized_mids",
            "spx_close",
            "strike_offsets",
            "rights",
        ],
        "D.near_atm.put_call_mid_ratio": [
            "all_near_atm_quantized_mids",
            "strike_offsets",
            "rights",
        ],
        "D.near_atm.side_smile_slope_bps_per_5pt": [
            "all_same_side_near_atm_quantized_mids",
            "spx_close",
            "strike_offsets",
            "rights",
        ],
        "E.bs.delta": [
            "action_quantized_mid",
            "spx_close",
            "atm_strike",
            "strike_offset",
            "right",
            "decision_time",
            "risk_free_rate_0.05",
            "dividend_yield_0.0",
            "feature_quantization_tick_0.05",
        ],
        "E.bs.gamma": [
            "action_quantized_mid",
            "spx_close",
            "atm_strike",
            "strike_offset",
            "right",
            "decision_time",
            "risk_free_rate_0.05",
            "dividend_yield_0.0",
            "feature_quantization_tick_0.05",
        ],
    }
    def live_twin_row(record: Any) -> dict[str, Any]:
        value = asdict(record)
        value["permitted_intraday_sources"] = list(
            value["permitted_intraday_sources"]
        )
        return value

    entry_twin_rows = [live_twin_row(record) for record in ENTRY17_LIVE_TWIN_INVENTORY]
    exit49_twin_rows = [live_twin_row(record) for record in EXIT49_LIVE_TWIN_INVENTORY]
    exit48_intermediate_twin_rows = [
        live_twin_row(record) for record in EXIT48_INTERMEDIATE_LIVE_TWIN_INVENTORY
    ]
    exit47_twin_rows = [
        live_twin_row(record) for record in EXIT47_CORRECTED_LIVE_TWIN_INVENTORY
    ]
    rows: list[dict[str, Any]] = []
    for name in FEATURE_NAMES:
        uses_option = name.startswith(("D.", "E."))
        vendors = ["THETADATA_OFFICIAL_SPX"]
        schemas = ["aligned/processed/minute_entry.market_window"]
        if uses_option:
            vendors.append("DATABENTO_OPRA")
            schemas.extend(
                [
                    "raw/databento/opra_spxw_cbbo_1m/{session}.cbbo-1m.parquet",
                    "aligned/processed/minute_entry.option_ladder",
                    "aligned/processed/minute_entry.contract_quote_metadata",
                ]
            )
        rows.append(
            {
                "name": name,
                "namespace": "entry",
                "role": "alpha",
                "transitive_source_leaves": leaves_by_feature[name],
                "source_vendors": vendors,
                "source_schemas": schemas,
                "clock_semantics": (
                    "option receipt clocks and the official ThetaData SPX available_at clock "
                    "must be <=decision_time; ThetaData event_time is a BAR_OPEN_TIMESTAMP "
                    "and official SPX available_at=event_time+60 seconds; completed minute only"
                ),
                "historical_adapter": "v4.research.pathd_entry_features::historical_snapshot_from_processed_row",
                "future_live_twin_adapter": "v4.research.pathd_entry_features::source_neutral_snapshot_from_causal_inputs",
                "shared_kernel": "v4.research.pathd_entry_features::signed17_from_snapshot",
                "reference_implementation": builder,
                "reference_implementation_sha256": sha256_path(builder_path),
                "twin_semantics": (
                    "both adapters create the same canonical snapshot consumed by one pure kernel; "
                    "the online-replay adapter performs no connection and supports no live-parity claim"
                ),
                "raw_vendor_greek": False,
                "ibkr_market_alpha": False,
            }
        )
    return {
        "schema_version": "pathd.entry_exit.feature_lineage.v1",
        "contract_id": "pathd-tier-s-entry-signed17-lineage-v1",
        "entry_feature_names_in_exact_order": list(FEATURE_NAMES),
        "entry_feature_count": len(FEATURE_NAMES),
        "entry_feature_order_sha256": stable_hash(list(FEATURE_NAMES)),
        "additional_properties": False,
        "entry_microstructure_enrichment": False,
        "vix_es_vx_model_alpha": False,
        "live_twin_inventory": {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "implementation_path": repo_path_label(live_twin_path),
            "implementation_sha256": sha256_path(live_twin_path),
            "entry17_records": entry_twin_rows,
            "entry17_records_sha256": stable_hash(entry_twin_rows),
            "original_exit49_feature_names": list(EXIT49_FEATURE_NAMES),
            "original_exit49_records": exit49_twin_rows,
            "original_exit49_records_sha256": stable_hash(exit49_twin_rows),
            "intermediate_exit48_feature_names": list(
                EXIT48_INTERMEDIATE_FEATURE_NAMES
            ),
            "intermediate_exit48_records": exit48_intermediate_twin_rows,
            "intermediate_exit48_records_sha256": stable_hash(
                exit48_intermediate_twin_rows
            ),
            "corrected_exit47_feature_names": list(
                EXIT47_CORRECTED_FEATURE_NAMES
            ),
            "corrected_exit47_records": exit47_twin_rows,
            "corrected_exit47_records_sha256": stable_hash(exit47_twin_rows),
            "correction_delta": {
                "removed": [
                    "last_causal_minute_volume",
                    "last_causal_open_interest",
                ],
                "added": [],
            },
            "minute_volume_current_run_disposition": (
                "DROPPED_UNTIL_EXACT_SHARED_ADAPTER_RECEIPT"
            ),
            "unresolved_live_twin_allowed_in_current_alpha": False,
        },
        "signed17_reference_builder_sha256": sha256_path(builder_path),
        "planned_adapter_contract": {
            "historical_adapter": "v4.research.pathd_entry_features::historical_snapshot_from_processed_row",
            "source_neutral_online_replay_adapter": "v4.research.pathd_entry_features::source_neutral_snapshot_from_causal_inputs",
            "shared_kernel": "v4.research.pathd_entry_features::signed17_from_snapshot",
            "meaning_of_future_twin": "offline-importable source-neutral adapter over a canonical causal snapshot; no broker/live connection and no live-parity claim",
            "byte_equality": {"atol": 0.0, "rtol": 0.0},
        },
        "implementation_receipt_required_before_fit": True,
        "training_allowed_at_preregistration": False,
        "raw_leaf_registry": {
            "official_spx": {
                "vendor": "THETADATA",
                "path": "vendor/thetadata/index/spx_1m/{session}.parquet",
                "integrity_partition": CORE_INTEGRITY_PARTITION,
                "fields": ["event_time", "symbol", "close", "volume", "context_source", "is_derived", "is_proxy", "is_official_index_data"],
                "row_invariants": {
                    "symbol": "SPX",
                    "context_source": "thetadata_index_history_ohlc",
                    "is_derived": False,
                    "is_proxy": False,
                    "is_official_index_data": True,
                },
                "timestamp_semantics": "BAR_OPEN_TIMESTAMP",
                "available_at": "event_time+60 seconds",
                "selection": "latest available_at<=decision_time with decision_time-available_at in [0,90 seconds]; no forward fill",
                "source_receipt": "bind exact manifest relative path/file SHA-256 plus selected parquet row-group, row-index, canonical-row SHA-256, event_time, and available_at",
                "derivations": "completed close; cumulative VWAP=sum(close*volume)/sum(volume), mean-close fallback at zero volume; range=max(close)-min(close); open=first close; OMAR=(close-open)/range else 0; momentum=current-minus-lagged close",
            },
            "option_quotes": {
                "vendor": "DATABENTO_OPRA",
                "raw_historical_path": (
                    "raw/databento/opra_spxw_cbbo_1m/"
                    "{session}.cbbo-1m.parquet"
                ),
                "path": "aligned/processed/minute_entry.contract_quote_metadata",
                "fields": ["contract_id", "bid", "ask", "mid", "received_timestamp_utc", "strike", "right", "offset"],
                "timestamp_semantics": (
                    "Databento CBBO-1m timestamp marks the completed interval end; "
                    "latest completed exact-contract minute at or before decision"
                ),
                "future_live_twin": (
                    "direct completed OPRA CBBO-1m or exact CBBO-1s/CMBP-1 to "
                    "CBBO-1m consolidation with identical minute boundary and ladder sampling"
                ),
                "derivations": "normalized mid from completed CBBO-1m quote; canonical feature transform quantizes at $0.05",
            },
            "structural": ["decision_time", "atm_strike", "strike_offsets", "rights", "source_neutral_contract_id"],
        },
        "feature_transform_tick_note": (
            "the signed-17 authority's fixed $0.05 mid quantization is an alpha transform "
            "and is distinct from the $0.05/$0.10 execution tick law"
        ),
        "features": rows,
        "fail_closed_negative_fixtures": [
            "ibkr_bid_renamed_pnl",
            "vendor_greek_renamed_internal_delta",
            "future_ts_recv",
            "unregistered_feature",
            "feature_without_live_twin",
        ],
        "source_mutation_canaries": {
            "official_spx_available_at_after_decision": "reject",
            "official_spx_event_time_treated_as_completed_without_plus_60": "reject",
            "raw_index_spx_alias_for_official_thetadata": "reject",
            "raw_index_vix_cache_mutation": "signed17_and_prediction_byte_invariant_and_cache_path_never_opened",
            "legacy_processed_vix_mutation": "signed17_and_prediction_byte_invariant",
            "causally_available_official_thetadata_spx_close_mutation": "signed17_hash_must_change",
        },
    }


def entry_representation_spec() -> dict[str, Any]:
    return {
        "history_minutes": 90,
        "tensor_shape": [90, 42, 17],
        "time_order": "oldest_to_current_completed_minute",
        "warmup": "use same-session left padding and masks before 90 minutes; BUY follows the substrate's frozen 30-minute context_ready mask, so no new 90-minute trading blackout is introduced",
        "identity_join": "for each current exact identity, look up that same source-neutral identity at every historical minute; missing/recentered/out-of-ladder is NaN and masked, never slot-carried",
        "session_boundary": "no cross-session carry and no forward fill",
        "artifact_dtype": "float64",
        "neural_dtype": "float32 only after fold-fit transform",
        "sidecars": [
            "feature_finite[90,42,17]",
            "contract_present[90,42]",
            "minute_available[90]",
            "current_exact_identities[42]",
            "current_offsets[42]",
            "current_rights[42]",
            "current_action_eligible[42]",
            "complete_ladder",
        ],
        "hgb_summaries": {
            "per_action_rolling_windows_minutes": [5, 15, 30, 60, 90],
            "per_window_statistics_in_order": [
                "finite_mean",
                "std_population",
                "ols_slope_per_minute",
                "latest_minus_earliest_finite",
                "valid_fraction",
            ],
            "column_order": "for each signed feature in exact order: current, then five statistics for each window in ascending order",
            "signed_summary_count": 442,
            "permitted_non_alpha_geometry_after_summaries": [
                "strike_offset_points_div_50",
                "right_is_call",
            ],
            "total_column_count": 444,
            "statistic_minimums": "mean >=1 finite; std/slope/delta >=2; otherwise NaN; OLS uses actual integer minute offsets",
            "mask_role": "eligibility and complete-ladder are post-prediction action masks, not predictive values",
            "nan_policy": "retain NaN for HGB native missing handling; no scaler",
        },
        "neural_tensor": {
            "signed17_history": [90, 42, 17],
            "finite_flags": [90, 42, 17],
            "token_input": "concatenate 17 fold-fit standardized values and 17 finite flags",
            "normalization": "for feature j use every finite token value at (session,decision frame,history minute,exact action,j) whose minute_available and contract_present flags are true; let S_j be contributing model-fit sessions and n_sj the count in session s, weight each value 1/(S_j*n_sj); compute mean with math.fsum in canonical (session,decision_time_ns,history_minute_oldest_to_newest,contract_id) order and population variance with a second math.fsum pass of w*(x-mean)^2 in that same order, all inputs/weights float64; no finite values invalidates the bundle, exact zero variance uses scale 1.0, negative/nonfinite variance invalidates; create finite flags before normalization, transform in float64, replace nonfinite standardized values by 0, then cast values and flags to float32",
            "shared_token_projection": "Linear(34,32)+GELU",
            "temporal_encoder": "shared one-layer GRU hidden_size=32 over 90 minutes for each exact action",
            "action_projection": "concatenate temporal state with [offset/50,call_flag], then Linear to d_model=64",
            "set_encoder": "one 42-action self-attention block, 4 heads, feedforward=128, dropout=0.10",
            "output_branches": "per-action 20 conditional-mean heads plus 20 q10 heads",
            "output_masks": "BUY masks applied after prediction; WAIT iff no legal gate-passer",
        },
        "forbidden": [
            "stored legacy labels",
            "raw bid/ask/spread/size/volume/open_interest as entry alpha",
            "VIX/ES/VX alpha",
            "slot-index identity carry",
            "future/path-derived horizon availability; the preregistered decision-clock-only deadline mask is permitted",
        ],
    }


def exit_feature_spec() -> dict[str, Any]:
    """Exact first-run exit-alpha subset; VIX/ES/VX are intentionally absent."""

    validate_inventory_contracts()
    features = list(EXIT47_CORRECTED_FEATURE_NAMES)
    return {
        "feature_names_in_exact_order": features,
        "feature_count": len(features),
        "original_exit49_audit_binding": {
            "feature_names_in_exact_order": list(EXIT49_FEATURE_NAMES),
            "feature_names_sha256": stable_hash(list(EXIT49_FEATURE_NAMES)),
            "intermediate_exit48_feature_names_in_exact_order": list(
                EXIT48_INTERMEDIATE_FEATURE_NAMES
            ),
            "removed_for_corrected_v2_generation": [
                "last_causal_minute_volume",
                "last_causal_open_interest",
            ],
        },
        "live_twin_inventory_schema_version": INVENTORY_SCHEMA_VERSION,
        "decision_cadence_seconds": 1,
        "history_seconds": 300,
        "option_freshness_ms": 2_000,
        "official_spx_carry_bound_seconds": 90,
        "last_causal_minute_volume": {
            "intraday_alpha": "DROPPED",
            "reason": (
                "field-level OHLCV/trades derivability is plausible, but no shared "
                "historical/live adapter receipt proves exact completed-bar, sparse/no-trade, "
                "exact-contract, and 90-second carry semantics"
            ),
            "future_research": (
                "may return only in a separately frozen generation after an exact adapter "
                "implementation and parity receipt"
            ),
        },
        "last_causal_open_interest": {
            "intraday_alpha": "DROPPED",
            "reason": "OPRA statistics open interest is daily/EOD with no intraday live twin",
            "prior_day_static_research": "not used in this corrected run",
        },
        "rolling_window_rule": "closed intervals ending at decision second; require >=80% finite seconds; otherwise NaN plus masks",
        "self_greeks": {
            "inputs": ["causal option_mid", "official_spx_close", "strike", "right", "time_to_expiry"],
            "constants": {"risk_free_rate": 0.05, "dividend_yield": 0.0},
            "raw_vendor_greeks": False,
        },
        "diagnostic_only_not_alpha": ["VIX", "ES", "VX"],
        "forbidden": [
            "future_best_bid",
            "future_mfe",
            "future_exit",
            "oracle_advantage",
            "A_ref target",
            "label_validity",
            "trained_policy outcome",
        ],
    }


def entry_global_family_spec() -> dict[str, Any]:
    """The plan-wide entry-family choice frozen before any estimator fit."""

    return {
        "schema_version": "pathd.entry_global_family_freeze.v1",
        "frozen_before_any_model_fit_or_outer_open": True,
        "selectable_candidate": "HGB",
        "selectable_candidates": ["HGB"],
        "mandatory_challenger": "NEURAL",
        "challenger_role": "nonselectable_attribution_multiplicity_and_machinery_only",
        "build_order": "fit and seal the exact-scope HGB baseline before the neural challenger for that same authorization, sessions, and dataset",
        "outer_folds": ["HGB", "HGB", "HGB", "HGB", "HGB"],
        "entry_acceptance_candidate": "HGB",
        "exit_oof_trajectory_entry_family": "HGB",
        "four_box_learned_entry_family": "HGB",
        "full_fit_entry_family": "HGB",
        "protected_holdout_entry_family": "HGB",
        "neural_outcome_can_select_replace_rescue_or_reweight_hgb": False,
        "nested_outer_pooled_postouter_or_holdout_reselection": False,
        "hyperparameter_candidates_per_family": 1,
    }


def aref_decision_critical_topology_spec() -> dict[str, Any]:
    """Freeze the corrected-v3 A_ref roles, losses, gradients, and serialization."""

    diagnostic_families = [
        "downside_300",
        "recovery_300",
        "giveback_300",
        "remaining_tail_300",
    ]
    hgb_constructor = {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "max_depth": 3,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
        "max_features": 1.0,
        "max_bins": 255,
        "categorical_features": None,
        "monotonic_cst": None,
        "interaction_cst": None,
        "warm_start": False,
        "early_stopping": False,
        "scoring": "loss",
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "tol": 1e-7,
        "verbose": 0,
        "random_state": "exact ensemble seed",
    }
    decision_nodes = [
        "A_REF_MEAN_RAW",
        "A_REF_Q10_RAW",
        "A_REF_Q50_CALIBRATION_LOCATION",
        "A_REF_Q90_EXIT_SUPPORT",
        "A_REF_MONOTONE_TRIPLET",
        "A_REF_MEAN_LCB90",
        "A_REF_JOINT_Q10_Q50_Q90_CALIBRATOR",
        "EXIT_COMPOSER_MEAN_Q10",
        "HOLD_RELIABILITY_Q10",
        "EXIT_RELIABILITY_NEGATED_Q90",
    ]
    diagnostic_nodes = [
        f"{family.upper()}_{quantile.upper()}"
        for family in diagnostic_families
        for quantile in ("q10", "q50", "q90")
    ]
    return {
        "schema_version": "pathd.aref_decision_critical_topology.v1",
        "direct_action_inputs": ["A_ref_mean", "A_ref_q10"],
        "decision_critical_calibration_support": [
            "A_ref_q10",
            "A_ref_q50",
            "A_ref_q90",
        ],
        "complete_required_aref_outputs": [
            "A_ref_mean",
            "A_ref_q10",
            "A_ref_q50",
            "A_ref_q90",
        ],
        "roles": {
            "A_ref_mean": "direct composer input and HOLD/EXIT predicted mean",
            "A_ref_q10": "direct downside input and binding HOLD lower-coverage support",
            "A_ref_q50": "joint-calibration location and transitive dependency of calibrated q10",
            "A_ref_q90": "binding EXIT upper-coverage support only; never a direct utility input",
        },
        "decision_critical_nodes": decision_nodes,
        "diagnostic_only_families": diagnostic_families,
        "diagnostic_only_nodes": diagnostic_nodes,
        "complete_output_count": 16,
        "dependency_edges": [
            "A_REF_MEAN_RAW->A_REF_MEAN_LCB90->EXIT_COMPOSER_MEAN_Q10",
            "A_REF_Q10_RAW+A_REF_Q50_CALIBRATION_LOCATION->A_REF_JOINT_Q10_Q50_Q90_CALIBRATOR.q10->EXIT_COMPOSER_MEAN_Q10",
            "A_REF_Q10_RAW+A_REF_Q50_CALIBRATION_LOCATION->A_REF_JOINT_Q10_Q50_Q90_CALIBRATOR.q10->HOLD_RELIABILITY_Q10",
            "A_REF_Q50_CALIBRATION_LOCATION->A_REF_JOINT_Q10_Q50_Q90_CALIBRATOR.q10,q90",
            "A_REF_Q90_EXIT_SUPPORT->A_REF_JOINT_Q10_Q50_Q90_CALIBRATOR.q90->EXIT_RELIABILITY_NEGATED_Q90",
            "LOCAL_PATH_DIAGNOSTIC_NODES->DIAGNOSTIC_REPORTS_ONLY",
        ],
        "forbidden_edges": [
            "A_ref_q90->A_ref_mean",
            "A_ref_q90->calibrated_A_ref_q10",
            "A_ref_q90->EXIT_COMPOSER_MEAN_Q10",
            "A_ref_q90->HOLD_RELIABILITY_Q10",
            "local_path_diagnostic->A_ref_artifact",
            "local_path_diagnostic->action_or_gate_or_verdict_or_family_selection",
        ],
        "shared_read_only_contract": {
            "feature_transform": "same frozen byte-identical payload for all three A_ref bundles",
            "target_scaler": "same frozen A_ref affine scaler payload",
            "target_validity_mask": "same preregistered mask",
            "row_identities_and_weights": "same exact ordered rows and session/trajectory-balanced weights",
            "partition": "same chronologically earlier model-fit/calibration partition",
            "mutable_object_sharing": False,
            "serialization_rule": "independently serialized references must validate byte-identical before any fit or inference",
        },
        "training_order": [
            "A_REF_ACTION_CORE",
            "A_REF_Q50_SUPPORT",
            "A_REF_Q90_SUPPORT",
            "A_REF_JOINT_Q10_Q50_Q90_CALIBRATOR",
        ],
        "action_bundle": {
            "targets": ["A_ref_mean", "A_ref_q10"],
            "losses": {
                "A_ref_mean": "squared_error",
                "A_ref_q10": "pinball_tau_0.10",
            },
            "loss_weights": {"A_ref_mean": 0.5, "A_ref_q10": 0.5},
            "fit_order": 1,
            "frozen_before_support_fit": True,
            "mutable_from_support_losses": False,
        },
        "q50_support_bundle": {
            "target": "A_ref_q50",
            "loss": "pinball_tau_0.50",
            "fit_order": 2,
            "q10_anchor_detached_and_frozen": True,
            "backpropagation_into_action_bundle": False,
        },
        "q90_support_bundle": {
            "target": "A_ref_q90",
            "loss": "pinball_tau_0.90",
            "fit_order": 3,
            "q50_and_upstream_anchors_detached_and_frozen": True,
            "backpropagation_into_action_or_q50_bundles": False,
        },
        "raw_noncrossing": {
            "formula": "q10=frozen_action_q10; q50=q10+softplus(gap50); q90=q50+softplus(gap90)",
            "support_anchor_rule": "support fitting consumes detached frozen anchors",
            "sorting_or_projection": False,
        },
        "hgb": {
            "library": "scikit-learn 1.5.2 HistGradientBoostingRegressor",
            "ensemble_seeds": [301, 302, 303],
            "estimator_fits_per_seed": 16,
            "estimator_fits_total": 48,
            "decision_critical_estimators_per_seed": [
                "A_ref_mean_squared_error",
                "A_ref_q10_quantile_0.10",
                "A_ref_q50_location_quantile_0.50",
                "A_ref_q90_location_quantile_0.90",
            ],
            "diagnostic_estimators_per_seed": [
                f"{family}_{quantile}"
                for family in diagnostic_families
                for quantile in ("q10", "q50", "q90")
            ],
            "constructor_kwargs": hgb_constructor,
            "loss_and_quantile_by_estimator": {
                "A_ref_mean": {"loss": "squared_error", "quantile": None},
                "A_ref_q10": {"loss": "quantile", "quantile": 0.10},
                "A_ref_q50_location": {"loss": "quantile", "quantile": 0.50},
                "A_ref_q90_location": {"loss": "quantile", "quantile": 0.90},
            },
            "assembly": "q10=r10_frozen; gap50=r50-q10; q50=q10+softplus(gap50); gap90=r90-q50; q90=q50+softplus(gap90)",
            "loss_gradient_rule": "independent HGB estimators have no shared gradients; q50/q90 support fitting cannot change the frozen r10 estimator",
            "aggregation": "assemble per seed in common standardized A_ref units, median each output over seeds, inverse the common scaler, then jointly calibrate",
            "serialization_manifest_order": [
                "feature_transform",
                "A_ref_target_scaler",
                "A_ref_mean",
                "A_ref_q10",
                "A_ref_q50_support",
                "A_ref_q90_support",
                "joint_monotone_calibration",
                "four_local_path_diagnostic_bundles",
            ],
        },
        "neural": {
            "ensemble_seeds": [311, 312, 313],
            "independent_modules_per_seed": 7,
            "modules": {
                "A_REF_ACTION_CORE": {
                    "outputs": ["A_ref_mean", "A_ref_q10"],
                    "loss": "0.5*MSE(mean)+0.5*pinball_tau_0.10(q10)",
                },
                "A_REF_Q50_SUPPORT": {
                    "outputs": ["gap50", "A_ref_q50"],
                    "loss": "pinball_tau_0.50(q50)",
                    "detached_inputs": ["A_ref_q10"],
                },
                "A_REF_Q90_SUPPORT": {
                    "outputs": ["gap90", "A_ref_q90"],
                    "loss": "pinball_tau_0.90(q90)",
                    "detached_inputs": ["A_ref_q10", "A_ref_q50"],
                },
                **{
                    family.upper(): {
                        "outputs": [f"{family}_q10", f"{family}_q50", f"{family}_q90"],
                        "loss": "equal_one_third_pinball_tau_0.10_0.50_0.90",
                    }
                    for family in diagnostic_families
                },
            },
            "graph": {
                "input": "301x47 values concatenated with 301x47 finite masks",
                "causal_conv1d_channels": [32, 32, 32],
                "kernel": 3,
                "dilations": [1, 4, 16],
                "explicit_left_pads": [2, 8, 32],
                "pooling": ["masked_mean", "masked_max", "current"],
                "mlp_hidden": [128, 64],
                "activation": "GELU",
                "dropout": 0.10,
            },
            "optimizer": "independent AdamW(beta1=0.9,beta2=0.999,eps=1e-8) per module",
            "optimizer_state_parameter_gradient_or_mutable_scaler_sharing": False,
            "gradient_clip_norm": 1.0,
            "epochs": 20,
            "maximum_gradient_updates": 4_000,
            "batch_size": 4096,
            "data_loader_workers": 0,
            "serialization_manifest_order": [
                "A_REF_ACTION_CORE",
                "A_REF_Q50_SUPPORT",
                "A_REF_Q90_SUPPORT",
                "DOWNSIDE_300",
                "RECOVERY_300",
                "GIVEBACK_300",
                "REMAINING_TAIL_300",
                "A_REF_JOINT_Q10_Q50_Q90_CALIBRATOR",
            ],
        },
        "calibration": {
            "fit_after_all_raw_outputs_frozen": True,
            "q50_affects_calibrated_q10_only_through_declared_joint_calibration_edge": True,
            "q90_affects_only_calibrated_q90_and_exit_reliability": True,
            "complete_triplet_required": True,
            "missing_or_nonfinite_required_output": "INVALID_TARGET_COVERAGE->invalid_result",
            "partial_or_survivor_calibration": False,
        },
        "exit_coverage_identity": "A_ref<=calibrated_q90_upper iff -A_ref>=-calibrated_q90_upper",
        "forbidden_quantile_claim": "literal q10(-A_ref)=-q90(A_ref) without a tie-compatible quantile convention",
        "sharpness_reporting": "q90 width and pinball performance versus unconditional baseline are nonbinding diagnostics; coverage alone is not tail skill",
    }


def exit_action_composer_spec() -> dict[str, Any]:
    topology = aref_decision_critical_topology_spec()
    return {
        "reference_target": "A_ref(t)=H_t-E_t",
        "mean_bound": {
            "kind": "one_sided_session_block_calibrated_lower_confidence_bound",
            "level": 0.90,
            "target": "conditional_mean_A_ref",
        },
        "downside_term": {
            "target": "A_ref_q10",
            "penalty": 0.25,
            "formula": "0.25*min(calibrated_q10_A_ref,0)",
        },
        "utility_formula": (
            "U_hold=LCB90(mean[A_ref|s_t])+0.25*min(q10[A_ref|s_t],0)"
        ),
        "action_rule": "HOLD iff U_hold>0; otherwise request EXIT",
        "action_inputs": ["A_ref_mean", "A_ref_q10"],
        "decision_critical_calibration_support": [
            "A_ref_q10",
            "A_ref_q50",
            "A_ref_q90",
        ],
        "transitive_composer_support": ["A_ref_q50"],
        "exit_reliability_only_support": ["A_ref_q90"],
        "topology_sha256": stable_hash(topology),
        "diagnostic_targets_forbidden": [
            "downside_300",
            "recovery_300",
            "giveback_300",
            "remaining_tail_300",
        ],
    }


def exit_diagnostic_model_isolation_spec() -> dict[str, Any]:
    """Keep path-attribution heads physically separate from the action network."""

    topology = aref_decision_critical_topology_spec()
    return {
        "action_model": {
            "targets": ["A_ref_mean", "A_ref_q10"],
            "action_consumers": ["A_ref_mean", "A_ref_q10"],
            "shared_parameters_with_diagnostics": False,
            "shared_preprocessor_or_target_scaler_with_diagnostics": False,
            "shared_optimizer_or_gradient_graph_with_diagnostics": False,
        },
        "aref_decision_critical_support": {
            "targets": ["A_ref_q10", "A_ref_q50", "A_ref_q90"],
            "complete_required_outputs": topology["complete_required_aref_outputs"],
            "topology_sha256": stable_hash(topology),
            "q50_support_can_backpropagate_into_action": False,
            "q90_support_can_backpropagate_into_action_or_q50": False,
            "q50_calibration_edge_is_binding": True,
            "q90_exit_reliability_edge_is_binding": True,
            "missing_or_nonfinite_effect": "invalid_result",
        },
        "diagnostic_models": {
            target: {
                "targets": targets,
                "separate_model_bundle": True,
                "separate_parameters": True,
                "separate_feature_preprocessor": True,
                "separate_target_scaler": True,
                "separate_calibration": True,
                "separate_optimizer_and_gradient_graph": True,
            }
            for target, targets in (
                ("downside_300", ["downside_300_q10", "downside_300_q50", "downside_300_q90"]),
                ("recovery_300", ["recovery_300_q10", "recovery_300_q50", "recovery_300_q90"]),
                ("giveback_300", ["giveback_300_q10", "giveback_300_q50", "giveback_300_q90"]),
                ("remaining_tail_300", ["remaining_tail_300_q10", "remaining_tail_300_q50", "remaining_tail_300_q90"]),
            )
        },
        "hgb_isolation": "each of the four local-path diagnostic families uses independent estimator/preprocessor/scaler/calibrator artifacts and cannot alter any A_ref estimator parameter or selection input",
        "neural_isolation": "each diagnostic family uses an independent module, optimizer, loss, preprocessor, scaler, calibration bundle, and detached input copy; no shared trunk, parameter, gradient, optimizer state, or action loss",
        "forbidden_effects": [
            "A_ref weights",
            "A_ref feature scaler",
            "A_ref target scaler",
            "A_ref calibration",
            "entry or exit family selection",
            "HOLD/EXIT action",
            "threshold or mask",
        ],
        "diagnostic_family_count": 4,
        "diagnostic_failure_effect": "diagnostic_incomplete_only; never changes the primary action model or verdict",
    }


def entry_negative_control_policy_ids() -> tuple[str, ...]:
    """Canonical negative-control ownership order used by every outer artifact."""

    return tuple(
        f"{family}::{control_id}"
        for family in ("HGB", "NEURAL")
        for control_id in ENTRY_NEGATIVE_CONTROL_IDS
    )


def exit_negative_control_policy_ids() -> tuple[str, ...]:
    """Closed corrected-v3 exit-control identities used by calibration topology."""

    return (
        "CONSTANT",
        "SIGN_REVERSED_HGB",
        "SIGN_REVERSED_NEURAL",
        "TIME_SHIFTED_HGB",
        "TIME_SHIFTED_NEURAL",
        *(f"SHUFFLED_TARGET_HGB_{index:02d}" for index in range(1, 9)),
        *(f"SHUFFLED_TARGET_NEURAL_{index:02d}" for index in range(1, 9)),
    )


def entry_required_calibration_node_ids(scope: str) -> tuple[str, ...]:
    """Expand the closed entry calibration-node grammar for one exact scope."""

    if re.fullmatch(r"NESTED_OUTER_[1-5]_INNER_[1-4]", scope) or scope == (
        "FULL_PRE_HOLDOUT"
    ):
        bundles = ("HGB", "NEURAL")
    elif re.fullmatch(r"OUTER_[1-5]", scope):
        bundles = ("HGB", "NEURAL", *entry_negative_control_policy_ids())
    else:
        raise ValueError("invalid entry calibration scope")
    nodes: list[str] = []
    for bundle in bundles:
        nodes.extend(
            f"ENTRY::{scope}::{bundle}::MEAN_LCB::{head}"
            for head in ENTRY_HEAD_TARGETS
        )
        nodes.extend(
            f"ENTRY::{scope}::{bundle}::Q10_CONFORMAL::{head}"
            for head in ENTRY_HEAD_TARGETS
        )
        nodes.extend(
            (
                f"ENTRY::{scope}::{bundle}::ACTION_COMPOSITE::ENTER",
                f"ENTRY::{scope}::{bundle}::ACTION_COMPOSITE::WAIT",
            )
        )
    if len(nodes) != len(set(nodes)):
        raise AssertionError("duplicate entry calibration node id")
    return tuple(nodes)


def exit_required_calibration_node_ids(scope: str) -> tuple[str, ...]:
    """Expand the closed decision-critical exit calibration-node grammar."""

    if scope == "FULL_PRE_HOLDOUT":
        bundles = ("HGB", "NEURAL")
    elif re.fullmatch(r"OUTER_[1-5]", scope):
        bundles = ("HGB", "NEURAL", *exit_negative_control_policy_ids())
    else:
        raise ValueError("invalid exit calibration scope")
    nodes = tuple(
        node
        for bundle in bundles
        for node in (
            f"EXIT::{scope}::{bundle}::MEAN_LCB::A_REF_MEAN",
            f"EXIT::{scope}::{bundle}::MONOTONE_Q10_Q50_Q90::A_REF",
        )
    )
    if len(nodes) != len(set(nodes)):
        raise AssertionError("duplicate exit calibration node id")
    return nodes


def composite_calibration_terminal_rule_spec() -> dict[str, Any]:
    """Freeze refined B-R without opening evidence or releasing model fitting."""

    entry_manifests = []
    for scope in (
        *(f"NESTED_OUTER_{outer}_INNER_{inner}" for outer in range(1, 6) for inner in range(1, 5)),
        *(f"OUTER_{outer}" for outer in range(1, 6)),
        "FULL_PRE_HOLDOUT",
    ):
        nodes = entry_required_calibration_node_ids(scope)
        entry_manifests.append(
            {
                "scope": scope,
                "required_node_count": len(nodes),
                "required_node_ids_sha256": stable_hash(list(nodes)),
            }
        )
    exit_manifests = []
    for scope in (*(f"OUTER_{outer}" for outer in range(1, 6)), "FULL_PRE_HOLDOUT"):
        nodes = exit_required_calibration_node_ids(scope)
        exit_manifests.append(
            {
                "scope": scope,
                "required_node_count": len(nodes),
                "required_node_ids_sha256": stable_hash(list(nodes)),
            }
        )
    structural_skips = [
        {"outer_fold": outer, "inner_fold": inner}
        for outer, last_invalid in ((1, 4), (2, 3), (3, 2), (4, 1))
        for inner in range(1, last_invalid + 1)
    ] + [{"outer_fold": 5, "inner_fold": 1}]
    return {
        "schema_version": "pathd.composite_calibration_terminal_rule.v1",
        "decision_id": "COMPOSITE_CALIBRATION_TERMINAL_RULE",
        "policy": "B_R_STATUS_PRESERVING_WHOLE_RUN_HARD_STOP",
        "generation_scope": "corrected-v3 exact five-fold design",
        "node_id_grammar": "{PLANE}::{SCOPE}::{BUNDLE_ID}::{CALIBRATOR_KIND}::{TARGET}",
        "node_statuses": [
            "VALID",
            "INSUFFICIENT_EVIDENCE",
            "INVALID_TARGET_COVERAGE",
            "DIAGNOSTIC_INCOMPLETE_ONLY",
            "NOT_EVALUATED_UPSTREAM_TERMINAL",
            "PREREGISTERED_STRUCTURAL_SKIP",
        ],
        "status_precedence": [
            "invalid_result",
            "insufficient_evidence",
            "owner_decision_required",
            "no_genuine_signal",
            "PASS",
        ],
        "failure_mapping": {
            "INVALID_TARGET_COVERAGE": "invalid_result",
            "INSUFFICIENT_EVIDENCE": "insufficient_evidence",
            "ADEQUATELY_POWERED_SIGNAL_OR_ACTION_GATE_FAILURE": "no_genuine_signal",
            "DIAGNOSTIC_INCOMPLETE_ONLY": "diagnostic_incomplete_only",
        },
        "node_registry": {
            "entry": {
                "head_targets_in_exact_order": list(ENTRY_HEAD_TARGETS),
                "nodes_per_bundle": 42,
                "nested_and_full_bundles": ["HGB", "NEURAL"],
                "outer_bundles": [
                    "HGB",
                    "NEURAL",
                    *entry_negative_control_policy_ids(),
                ],
                "manifests": entry_manifests,
                "composite_dependency": "ENTER and WAIT each consume the complete ordered 40-node marginal vector for their bundle; marginal failure makes the composite NOT_EVALUATED_UPSTREAM_TERMINAL",
            },
            "exit": {
                "required_aref_outputs": [
                    "A_ref_mean",
                    "A_ref_q10",
                    "A_ref_q50",
                    "A_ref_q90",
                ],
                "atomic_triplet_outputs": ["A_ref_q10", "A_ref_q50", "A_ref_q90"],
                "nodes_per_bundle": 2,
                "full_bundles": ["HGB", "NEURAL"],
                "outer_bundles": [
                    "HGB",
                    "NEURAL",
                    *exit_negative_control_policy_ids(),
                ],
                "manifests": exit_manifests,
                "diagnostic_only_triplets": [
                    "downside_300",
                    "recovery_300",
                    "giveback_300",
                    "remaining_tail_300",
                ],
            },
            "required_fields_per_node": [
                "node_id",
                "stage",
                "scope",
                "family_or_control_identity",
                "target_head_or_action",
                "consumer",
                "observed_count",
                "minimum_count",
                "decision_critical_or_diagnostic",
                "failure_mapping",
            ],
        },
        "stage_chronology": [
            "for OUTER_1 through OUTER_5 sequentially: process nested inner 1 through 4 sequentially",
            "for each calibration-valid nested scope: fit/calibrate, validate every required node, seal a VALID scope gate, then and only then open nested evidence",
            "fit/calibrate outer entry, validate every required outer-entry node, seal a VALID scope gate, then and only then open that fold evidence",
            "freeze each outer entry result before any later-fold artifact",
            "require an exact all-five VALID manifest before pooled or at-least-four-of-five entry economics",
            "freeze an immutable entry-only pooled verdict",
            "only after entry PASS, fit/calibrate/freeze FULL_PRE_HOLDOUT entry and validate all full-entry nodes",
            "evaluate exit minimum power before any exit weights",
            "only if exit power passes, process exit outer folds 1 through 5 with required-node validation before each evidence open",
            "only after exact five valid exit folds, evaluate/freeze combined verdict and FULL_PRE_HOLDOUT exit",
            "validate the complete pre-holdout packet before a single protected-holdout open",
        ],
        "outer_open_gate": {
            "required_before_decode": "write-once pathd.calibration_scope_gate_receipt.v1 with status VALID for the exact scope and complete ordered node manifest",
            "scope_gate_evidence_access_count": 0,
            "computed_failure_can_be_skip": False,
            "later_fold_before_prior_immutable_terminal": False,
        },
        "all_five_manifest": {
            "outer_folds_in_exact_order": [1, 2, 3, 4, 5],
            "required_status_each": "VALID",
            "valid_fold_count": 5,
            "pooled_population": "exact concatenation of five valid chronological outer results",
            "economic_fold_gate": "at least four positive economic deltas among exactly five valid folds",
            "four_as_five_or_survivor_pooling": False,
        },
        "nested_structural_skip": {
            "source_of_truth": "session_assignments inner block frozen calibration_valid=false",
            "exact_blocks": structural_skips,
            "receipt_status": "SKIPPED_INSUFFICIENT_CALIBRATION",
            "access_count": 0,
            "dataset_opened": False,
            "result_generated": False,
            "computed_failure_in_calibration_valid_block_can_be_relabelled_skip": False,
        },
        "failure_receipts": {
            "pathd.calibration_node_failure_receipt.v1": {
                "write_once": True,
                "failure_only": True,
                "required": [
                    "node_id",
                    "raw_node_status",
                    "closed_failure_code",
                    "counts_and_minima",
                    "scope_bundle_target_consumer_role",
                    "upstream_hashes",
                    "evidence_access_count_zero",
                    "holdout_open_count_zero",
                    "receipt_sha256",
                ],
            },
            "pathd.calibration_scope_gate_receipt.v1": {
                "write_once": True,
                "success_only": True,
                "status": "VALID",
                "condition": "exact ordered required-node vector is complete and every required node is VALID",
            },
            "pathd.calibration_campaign_failure_receipt.v1": {
                "write_once": True,
                "failure_only": True,
                "condition": "references first failing scope, all node-failure receipts there, prior immutable stages, and precedence-resolved terminal status",
            },
            "invalid_result_scientific_success_receipt_allowed": False,
        },
        "current_run_exit_power_stop": {
            "known_exit_weight_session_counts_by_fold": [0, 0, 19, 48, 56],
            "minimum_model_fit_sessions": 60,
            "terminal_status_after_entry_pass": "insufficient_evidence",
            "stop_reason": "insufficient_exit_evidence",
            "entry_only_verdict_preserved_by_hash": True,
            "exit_fit_executed": False,
            "four_box_instantiated": False,
            "full_exit_fit_executed": False,
            "preholdout_packet_frozen": False,
            "holdout_open_count": 0,
        },
        "forbidden_rescue": [
            "fold deletion",
            "node deletion",
            "target or quantile deletion",
            "horizon or action deletion",
            "required control deletion",
            "row deletion after prediction",
            "null imputation",
            "zero imputation",
            "target substitution",
            "composite reweighting",
            "status relabeling",
            "survivor-only pooling",
            "treating four valid folds as five",
            "same-generation refit retry reseed threshold change or alternate calibrator",
            "diagnostic veto or rescue of primary result",
            "evidence open without a VALID scope-gate receipt",
        ],
        "same_generation_retry_after_invalid_result": False,
        "full_preholdout": {
            "entry_instantiation": "only after immutable outer entry PASS",
            "exit_instantiation": "only after immutable combined PASS",
            "validation": "every required full-scope calibration node is VALID before protected-holdout open",
            "entry_only_pass_authorizes_combined_full_exit_or_holdout": False,
        },
    }


def validate_br_five_fold_manifest(rows: Any) -> None:
    """Pure pre-fit validator for the exact all-five VALID entry manifest."""

    required_keys = {
        "outer_fold",
        "scope",
        "status",
        "required_node_count",
        "valid_node_count",
        "required_node_ids_sha256",
        "scope_gate_receipt_sha256",
        "outer_result_receipt_sha256",
        "deleted_node_count",
        "imputed_node_count",
    }
    if type(rows) is not list or len(rows) != 5:
        raise ValueError("B-R requires exactly five outer calibration rows")
    for fold, row in enumerate(rows, 1):
        scope = f"OUTER_{fold}"
        nodes = entry_required_calibration_node_ids(scope)
        if (
            type(row) is not dict
            or set(row) != required_keys
            or row.get("outer_fold") != fold
            or row.get("scope") != scope
            or row.get("status") != "VALID"
            or row.get("required_node_count") != len(nodes)
            or row.get("valid_node_count") != len(nodes)
            or row.get("required_node_ids_sha256") != stable_hash(list(nodes))
            or row.get("deleted_node_count") != 0
            or row.get("imputed_node_count") != 0
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("scope_gate_receipt_sha256", "")))
            is None
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("outer_result_receipt_sha256", "")))
            is None
        ):
            raise ValueError("B-R outer calibration manifest is not exact all-five VALID")


def entry_matched_random_owner_policy_ids() -> tuple[str, ...]:
    """Canonical policy ownership order for independent random schedules."""

    return (
        "HGB",
        "NEURAL",
        *entry_negative_control_policy_ids(),
        *(
            f"{family}::ABLATION::{ablation_id}"
            for family in ("HGB", "NEURAL")
            for ablation_id in ENTRY_ABLATION_IDS
        ),
    )


def entry_large_improvement_check_paths() -> tuple[str, ...]:
    """Fixed canonical check receipts; callers cannot choose audit evidence paths."""

    return tuple(
        repo_path_label(
            AUDIT_ROOT
            / "entry_large_improvement_audit"
            / f"{index:02d}_{slug}.json"
        )
        for index, slug in enumerate(ENTRY_LARGE_IMPROVEMENT_CHECK_SLUGS, 1)
    )


def entry_pooled_gate_input_spec() -> dict[str, Any]:
    """Closed reconstructed inputs and Boolean formulas for the pooled entry gate."""

    per_fold_int_keys = [
        "outer_fold", "primary_session_count",
        "shared_exit_candidate_completed_trades",
        "time300_oof_trajectory_count", "holdflat_hgb_minus_p5_micros",
        "holdflat_hgb_minus_random_eighth_micros",
        "shared_hgb_minus_p5_micros",
        "shared_hgb_minus_random_eighth_micros",
        "fee4_shared_hgb_minus_p5_micros",
        "fee4_shared_hgb_minus_random_eighth_micros",
    ]
    pooled_int_keys = per_fold_int_keys[1:]
    action_gate_keys = [
        "distinct_trajectory_count", "coverage_numerator",
        "coverage_denominator", "decile_trajectory_counts",
        "decile_distinct_session_counts",
        "adjacent_valid_bootstrap_replicates", "adjacent_total_draws",
        "adjacent_upper_bounds_micros",
    ]
    criteria = {
        "outer_session_power_each": "all five per_fold.primary_session_count>=25",
        "entry_trade_power_each": "all five per_fold.shared_exit_candidate_completed_trades>=40",
        "entry_trade_power_pooled": "pooled.shared_exit_candidate_completed_trades>=200",
        "time300_trajectory_power_each": "all five per_fold.time300_oof_trajectory_count>=50",
        "time300_trajectory_power_pooled": "pooled.time300_oof_trajectory_count>=300",
        "holdflat_vs_p5_pooled_positive": "pooled.holdflat_hgb_minus_p5_micros>0",
        "holdflat_vs_p5_positive_at_least_4_of_5": "count(per_fold.holdflat_hgb_minus_p5_micros>0)>=4",
        "holdflat_vs_random_pooled_positive": "pooled.holdflat_hgb_minus_random_eighth_micros>0",
        "holdflat_vs_random_positive_at_least_4_of_5": "count(per_fold.holdflat_hgb_minus_random_eighth_micros>0)>=4",
        "shared_vs_p5_pooled_positive": "pooled.shared_hgb_minus_p5_micros>0",
        "shared_vs_p5_positive_at_least_4_of_5": "count(per_fold.shared_hgb_minus_p5_micros>0)>=4",
        "shared_vs_random_pooled_positive": "pooled.shared_hgb_minus_random_eighth_micros>0",
        "shared_vs_random_positive_at_least_4_of_5": "count(per_fold.shared_hgb_minus_random_eighth_micros>0)>=4",
        "enter_action_gate": "ENTER satisfies the exact action-gate formula below",
        "wait_action_gate": "WAIT satisfies the exact action-gate formula below",
        "matched_random_all_complete": "every exact policy×5-fold×8-seed×4-channel completion cell is true and every same-shape receipt hash validates with policy-specific ownership",
        "negative_controls_all_complete": "all 440 negative_controls.complete_by_control_fold_channel cells are true and all 440 evaluation hashes validate",
        "negative_controls_none_pass_full_gate": "all 22 negative_controls.full_gate_passed_by_control values are false",
        "fee4_vs_p5_pooled_positive": "pooled.fee4_shared_hgb_minus_p5_micros>0",
        "fee4_vs_random_pooled_positive": "pooled.fee4_shared_hgb_minus_random_eighth_micros>0",
        "large_improvement_audit_complete_if_triggered": "large_improvement_audit.complete is true under the exact trigger/check law",
        "survival_all_zero": "every survival_violation_counts exact-int value==0",
    }
    return {
        "schema_version": "pathd.entry_pooled_gate_inputs.v1",
        "br_all_five_valid_prerequisite": {
            "manifest_validator": "v4.research.pathd_entry_exit::validate_br_five_fold_manifest",
            "fold_order": [1, 2, 3, 4, 5],
            "status_each": "VALID",
            "required_before_reconstruction": True,
            "survivor_only_pooling": False,
        },
        "top_level_keys_in_order": [
            "schema_version", "fold_order", "per_fold", "pooled",
            "action_calibration", "matched_random", "negative_controls",
            "large_improvement_audit", "survival_violation_counts",
        ],
        "closed_shapes": {
            "fold_order": "exact int list [1,2,3,4,5]",
            "per_fold": {
                "row_count": 5,
                "row_keys_in_order": per_fold_int_keys,
                "types": "every cell exact int (bool forbidden); outer_fold equals fold_order and every other cell is reconstructed from validated journals/results",
            },
            "pooled": {
                "keys_in_order": pooled_int_keys,
                "types": "every cell exact int (bool forbidden)",
                "formula": "each pooled value is exactly the arithmetic sum of its five same-named per_fold values",
            },
            "random_eighth": "each *_minus_random_eighth_micros equals 8*HGB paired net PnL minus the exact sum of the eight registered matched-random policy paired net PnLs; no division, float average, rounding, or aggregate substitution",
            "action_calibration": {
                "keys_in_order": ["ENTER", "WAIT"],
                "action_keys_in_order": action_gate_keys,
                "scalar_types": "distinct_trajectory_count, coverage_numerator, and coverage_denominator are nonnegative exact ints",
                "vector_shapes": "decile_trajectory_counts and decile_distinct_session_counts are 10 nonnegative exact ints; adjacent_valid_bootstrap_replicates and adjacent_total_draws are 9 nonnegative exact ints; adjacent_upper_bounds_micros is 9 finite float64 values with bool/int/NaN/infinity rejected",
                "exact_pass_formula": "distinct_trajectory_count>=30; coverage_denominator==distinct_trajectory_count>0; 100*coverage_numerator>=85*coverage_denominator; every decile trajectory count>=3; every decile distinct-session count>=3; every adjacent valid replicate count==5000; every adjacent total draw count in [5000,50000]; every adjacent upper bound micros<=0",
                "source": "reconstructed from every complete unique ENTER intent or maximal flat WAIT episode under action_calibration_spec; no stored sub-pass Boolean",
            },
            "matched_random": {
                "keys_in_order": ["policy_ids", "seeds", "channels", "complete_by_policy_fold_seed_channel", "evaluation_receipt_sha256s_by_policy_fold_seed_channel"],
                "policy_ids": list(entry_matched_random_owner_policy_ids()),
                "seeds": list(MATCHED_RANDOM_SEEDS),
                "channels": list(ENTRY_REPLAY_CHANNELS),
                "matrices": "both exact policy_ids x 5 x 8 x 4 arrays in policy, fold, seed, then channel order; completion cells are reconstructed exact bools and receipt cells are lowercase hex64 when complete or null when incomplete, bound to policy-owned independent budgets/schedules/ledgers/evaluations; no policy reuses another policy's random receipts",
            },
            "negative_controls": {
                "keys_in_order": ["control_ids", "channels", "complete_by_control_fold_channel", "full_gate_passed_by_control", "evaluation_sha256s_by_control_fold_channel"],
                "control_ids": list(entry_negative_control_policy_ids()),
                "channels": list(ENTRY_REPLAY_CHANNELS),
                "arrays": "completion and evaluation hashes are exact 22x5x4 arrays in control_ids, fold, then channel order; incomplete cells carry false/null and route insufficient_evidence; pass is an exact 22-bool vector recomputed by immutable main from every registered channel/fee journal plus owned random schedules, action observations, time-300 evidence, and survival records; no stored control pass Boolean or favorable cell may substitute for another",
            },
            "large_improvement_audit": {
                "keys_in_order": ["ratio_trigger", "bootstrap_trigger", "triggered", "required_check_names", "required_check_statuses", "required_check_receipt_sha256s", "complete"],
                "fixed_receipt_paths": list(entry_large_improvement_check_paths()),
                "trigger_formula": "immutable main recomputes ratio_trigger and bootstrap_trigger from the validated shared-transparent $3 candidate/P5/eight-random session ledgers under metrics_and_gates.large_improvement_audit; triggered is their logical OR",
                "checks": "required_check_names exactly equals that spec's ordered required_checks and paths are exactly entry_large_improvement_check_paths(); if triggered, statuses and hashes come only from re-opened self-hashed fixed receipts and complete requires every PASS; if not triggered, every status is NOT_TRIGGERED and every receipt hash is null",
            },
            "survival_violation_counts": {
                "keys_in_order": ["overlap", "unaffordable_fill", "quantity_breach", "nonflat_close", "duplicate_terminal_consumption", "d48_breach", "d49_breach"],
                "types": "nonnegative exact int counts reconstructed from all terminal journals",
            },
        },
        "hash_rule": "gate_inputs_sha256=stable_hash of the exact reconstructed object; list order is semantic and no additional key, missing row, coercion, NaN, infinity, aggregate-only replacement, or caller-supplied Boolean is accepted",
        "pass_criteria_exact_keys_and_formulas": criteria,
        "verdict_precedence": [
            "invalid structure/reconstruction/hash => raise invalid_result and persist nothing",
            "any of the first five power criteria false, or an action distinct/decile/bootstrap power component under its minimum => insufficient_evidence",
            "survival_all_zero false, either fee4 criterion false, or large_improvement_audit_complete_if_triggered false => owner_decision_required",
            "adequately powered holdflat/shared/action/random/control criterion false => no_genuine_signal with stop_reason no_genuine_entry_signal",
            "all exact criteria true => PASS with full_fit_authorized=true",
        ],
    }


def entry_pooled_acceptance_spec() -> dict[str, Any]:
    """Exact post-outer entry verdict and downstream full-fit authority."""

    return {
        "schema_version": "pathd.entry_pooled_acceptance_spec.v1",
        "candidate_family": "HGB",
        "prerequisites": {
            "outer_result_receipts": [
                repo_path_label(
                    _outer_fold_artifact_path(fold, "outer_result_receipt.json")
                )
                for fold in range(1, 6)
            ],
            "ordered_primary_session_count": 138,
            "aggregation_authorization": "FrozenResultAggregationAuthorization over exactly those five revalidated result/evaluation/trace receipts",
            "br_all_five_valid_manifest": "validate_br_five_fold_manifest succeeds on exact folds 1..5 before any pooled or >=4/5 economic reconstruction",
        },
        "fixed_paths": {
            "result": repo_path_label(ENTRY_POOLED_ACCEPTANCE_RESULT_PATH),
            "receipt": repo_path_label(ENTRY_POOLED_ACCEPTANCE_RECEIPT_PATH),
        },
        "reconstruction": {
            "caller_supplied_fields": [],
            "embedded_replay_cells": "EntryControlReplayResultV1 carries four mandatory non-null HGB cells and four mandatory non-null P5 cells in exact ENTRY_REPLAY_CHANNELS order plus an exact owner-policy x eight-seed x four-channel nullable matched-random tensor; EntryNegativeControlPanelV1 carries an exact 22-control x four-channel nullable tensor. Every non-null EntryReplayEvaluationCellV1 embeds a complete EntryPolicyEvaluationV1 and terminal journal, is revalidated by immutable main against the exact fold authorization/dataset, and has unique outcome-free budget/schedule/skip/realized receipts for random cells. Null random/control cells reconstruct completion=false and route insufficient_evidence; HGB/P5 null or malformed cells are invalid_result",
            "journal_fact_rows": "ENTER/WAIT action observations and TIME300 facts are complete ordered typed journal events bound only to HGB::SHARED_TRANSPARENT_FEE3 (or the corresponding negative-control shared-$3 journal). Immutable main rejects any omitted, duplicate, reordered, unsealed-session, wrong-transition, or self-hash-mismatched row and computes every action/time power cell itself",
            "survival_records": "every embedded evaluation journal contains exactly one EntrySurvivalAuditRecordV1 per authorized session in session order. The frozen research-replay validator reconstructs each Boolean from its ledger transitions; immutable main requires the record/event equality, exact source-transition coverage, and self hash, then sums the seven counts without accepting a caller summary",
            "large_improvement_checks": {
                "fixed_receipt_paths": list(entry_large_improvement_check_paths()),
                "trigger_source": "immutable main recomputes both triggers from HGB/P5/eight-random shared-transparent-$3 session PnL and the registered 10,000-draw PCG64 bootstrap",
                "missing": "a missing fixed receipt reconstructs MISSING/null and complete=false; a present receipt must be a canonical non-symlink self-hashed object binding exact outer receipts, economics, evidence files, findings file, check name/index, and PASS/FAIL",
            },
            "exact_fold_arrays": [
                "candidate_outer_evaluation_sha256s",
                "control_exit_sha256s",
                "control_replay_result_sha256s",
                "negative_control_panel_sha256s",
            ],
            "gate_inputs": entry_pooled_gate_input_spec(),
            "gate_spec_sha256": "stable_hash of the exact entry acceptance, metrics/action calibration, control, matched-random, power, fill-law, and account-law specifications",
            "pass_criteria": "closed exact booleans recomputed from reconstructed_gate_inputs; no caller-authored summary or declared pass",
            "verdicts": ["PASS", "no_genuine_signal", "insufficient_evidence", "owner_decision_required"],
            "invalid_structure_or_reconstruction": "raise invalid_result and write no scientific result or terminal success receipt",
            "stop_reason": "closed verdict_vocabulary.entry_failure_mapping; minimum-power misses use insufficient_entry_evidence and adequately powered signal/action failures use no_genuine_entry_signal",
            "full_fit_authorized": "true if and only if verdict is PASS and every recomputed criterion is true",
        },
        "durability": "freeze_entry_pooled_acceptance_once writes canonical result then canonical terminal receipt with O_EXCL, mode 0600, file fsync, and parent-directory fsync; neither path may preexist in any inode form; no overwrite, repair, or second freeze",
        "read_rule": "read_frozen_entry_pooled_acceptance reopens and reconstructs the five upstream results and validates exact result/receipt bytes; require_pass=True rejects every non-PASS result",
        "full_fit_binding": "full_weights and full_calibration authorizations alone carry the exact pooled acceptance receipt SHA-256 and require revalidated PASS; every nested/outer authorization carries null",
        "context_binding": "context diagnostics require the same revalidated frozen receipt but do not require PASS, so post-verdict reporting also occurs after an honest entry failure",
    }


def context_age_quantile_type7(age_samples_ns: Iterable[int], q: float, /) -> float | None:
    """Exact fold/pooled age statistic; pooled callers concatenate raw samples."""

    samples = list(age_samples_ns)
    if type(q) is not float or q not in {0.5, 0.95}:
        raise ValueError("context age quantile must be exact float 0.5 or 0.95")
    if any(type(value) is not int or value < 0 for value in samples):
        raise ValueError("context age sample must be a nonnegative exact integer ns")
    if not samples:
        return None
    ordered = sorted(samples)
    h = (len(ordered) - 1) * q
    lower = math.floor(h)
    fraction = h - lower
    upper = min(lower + 1, len(ordered) - 1)
    value_ns = (1.0 - fraction) * ordered[lower] + fraction * ordered[upper]
    return value_ns / 1_000_000_000.0


def context_policy_journal_spec() -> dict[str, Any]:
    policy_ids = [
        "HGB_SHARED_TRANSPARENT_FEE3",
        "NEURAL_PRIMARY",
        "P5_SHARED_TRANSPARENT_FEE3",
        *[
            f"MATCHED_RANDOM_HGB_SEED_{seed}_SHARED_TRANSPARENT_FEE3"
            for seed in MATCHED_RANDOM_SEEDS
        ],
    ]
    return {
        "schema_version": "pathd.entry_context_policy_journals.v1",
        "channel": "SHARED_TRANSPARENT_FEE3",
        "channel_index": 2,
        "matched_random_owner_policy_id": "HGB",
        "matched_random_owner_policy_index": 0,
        "policy_ids_in_order": policy_ids,
        "binding_fields_in_order": [
            "outer_fold", "policy_id", "policy_order_index", "source_kind",
            "outer_result_receipt_sha256", "source_artifact_sha256",
            "replay_cell_sha256", "evaluation_sha256",
            "terminal_journal_sha256", "matched_random_seed",
            "binding_sha256",
        ],
        "sources": {
            "HGB_SHARED_TRANSPARENT_FEE3": "validated control_replay_result.candidate_channel_evaluations[2]",
            "NEURAL_PRIMARY": "validated outer_primary_result.payload.neural",
            "P5_SHARED_TRANSPARENT_FEE3": "validated control_replay_result.p5_channel_evaluations[2]",
            "MATCHED_RANDOM": "validated control_replay_result.matched_random_channel_evaluations[owner HGB index 0][seed index 0..7][channel index 2]",
        },
        "immutable_reader": "read_validated_context_policy_journals accepts only the exact typed authorization and keyword-only outer_fold, derives every fixed path internally, reruns outer/replay/evaluation/journal validators, and returns no caller-authored policy, row, path, table, or summary",
    }


def context_diagnostics_transaction_spec() -> dict[str, Any]:
    return {
        "schema_version": "pathd.entry_context_diagnostics_transaction.v1",
        "states": [
            "UNOPENED", "AUTHORIZED_NOT_OPEN", "OPEN_ACTIVE",
            "OUTPUT_PREFIX_ACTIVE", "POOLED_ARTIFACT_DURABLE_UNSEALED",
            "SEALED", "BURNED", "CORRUPT_BURNED",
        ],
        "abort_reason_codes": [
            "RESERVED_PATH_OCCUPIED", "DIAGNOSTIC_INVENTORY_INVALID",
            "PRIMARY_UPSTREAM_DRIFT", "POLICY_JOURNAL_REVALIDATION_FAILED",
            "CONTEXT_SOURCE_VERIFY_FAILED", "CONTEXT_SOURCE_DECODE_FAILED",
            "ANCHOR_RECONSTRUCTION_FAILED", "OUTER_ARTIFACT_VALIDATION_FAILED",
            "OUTER_RECEIPT_COMMIT_FAILED", "POOLED_RECONSTRUCTION_FAILED",
            "POOLED_ARTIFACT_VALIDATION_FAILED", "POOLED_RECEIPT_COMMIT_FAILED",
            "CRASH_RECOVERY_INCOMPLETE", "CORRUPT_PARTIAL_STATE",
        ],
        "success_statuses": {
            "access": "OPENED_POST_ENTRY_CONTEXT_DIAGNOSTICS_ONCE",
            "outer": "FROZEN_ENTRY_OUTER_CONTEXT_DIAGNOSTICS",
            "pooled": "FROZEN_ENTRY_POOLED_CONTEXT_DIAGNOSTICS",
            "abort": "ABORTED_NO_REDECODE",
        },
        "access_receipt_fields_in_order": [
            "schema_version", "status", "opened_at_utc", "transaction_id",
            "authorization_path", "authorization_sha256",
            "diagnostic_inventory_receipt_sha256", "context_access_count",
            "holdout_open_count", "receipt_sha256",
        ],
        "outer_receipt_fields_in_order": [
            "schema_version", "status", "frozen_at_utc", "outer_fold",
            "authorization_path", "authorization_sha256",
            "access_receipt_path", "access_receipt_sha256",
            "primary_result_receipt_path", "primary_result_receipt_sha256",
            "core_integrity_receipt_sha256",
            "diagnostic_inventory_receipt_sha256", "artifact_path",
            "artifact_sha256", "policy_journal_bindings_root_sha256",
            "sessions_sha256_newline", "source_receipts_root_sha256",
            "anchor_context_rows_root_sha256", "age_samples_root_sha256",
            "tables_sha256", "preregistration_sha256", "plan_sha256",
            "holdout_caveat", "holdout_open_count",
            "primary_artifacts_unchanged", "receipt_sha256",
        ],
        "pooled_receipt_fields_in_order": [
            "schema_version", "status", "frozen_at_utc",
            "authorization_path", "authorization_sha256",
            "access_receipt_path", "access_receipt_sha256",
            "core_integrity_receipt_sha256",
            "diagnostic_inventory_receipt_sha256", "outer_artifact_paths",
            "outer_artifact_sha256s", "outer_receipt_paths",
            "outer_receipt_sha256s",
            "policy_journal_binding_roots_by_fold",
            "policy_journal_bindings_root_sha256", "artifact_path",
            "artifact_sha256", "anchor_context_rows_root_sha256",
            "age_samples_root_sha256", "tables_sha256",
            "preregistration_sha256", "plan_sha256", "holdout_caveat",
            "holdout_open_count", "primary_artifacts_unchanged",
            "receipt_sha256",
        ],
        "abort_receipt_fields_in_order": [
            "schema_version", "status", "aborted_at_utc", "reason_code",
            "state_before_abort", "authorization_path", "authorization_sha256",
            "access_receipt_path", "access_receipt_sha256",
            "diagnostic_inventory_receipt_sha256", "last_completed_outer_fold",
            "partial_artifacts", "primary_upstream_receipt_sha256s",
            "pooled_acceptance_receipt_sha256", "holdout_open_count",
            "no_redecode", "no_reauthorization", "receipt_sha256",
        ],
        "durability": "authorization, access, artifacts, outer receipts, pooled receipt, and abort use canonical JSON with O_EXCL|O_NOFOLLOW mode 0600, file fsync, and parent fsync; fold artifact/receipt pairs commit 1..5 before pooled artifact/receipt",
        "crash_rule": "an access receipt without the pooled terminal receipt is never resumed. A later invocation writes CRASH_RECOVERY_INCOMPLETE by O_EXCL and permanently forbids re-decode, repair, overwrite, and reauthorization",
    }


def context_diagnostic_payload_spec() -> dict[str, Any]:
    """Closed actual source, anchor, age-sample, and table payload contract."""

    policy_columns = [
        "HGB_trades", "HGB_net_pnl_micros", "neural_trades",
        "neural_net_pnl_micros", "P5_trades", "P5_net_pnl_micros",
        "matched_random_trades", "matched_random_net_pnl_eighth_micros",
        "action_count", "fill_count", "no_fill_count",
        "HGB_minus_P5_paired_delta_micros",
        "HGB_minus_matched_random_paired_delta_eighth_micros",
        "descriptive_action_calibration",
    ]
    leading = [
        "unique_session_count", "distinct_trajectory_count", "decision_row_count"
    ]
    return {
        "schema_version": "pathd.entry_context_diagnostic_payload.v1",
        "source_order": ["VIX", "ES", "VX", "SPX_REFERENCE"],
        "source_receipt_row": {
            "fields_in_order": [
                "source", "session", "manifest_relative_path", "requested",
                "exists", "bytes", "sha256", "decoded_row_count",
                "timestamp_count", "publisher_ids",
                "valid_cross_publisher_rows_consolidated",
                "invalid_within_publisher_duplicate_count", "fresh_count",
                "stale_count", "missing_count", "age_samples_ns",
                "age_samples_root_sha256", "age_seconds_p50",
                "age_seconds_p95", "age_seconds_max", "status",
                "source_degraded", "receipt_sha256",
            ],
            "types_and_nulls": "source is literal VIX|ES|VX|SPX_REFERENCE; session is an authorized YYYY-MM-DD; manifest path/hash must belong to the source's exact integrity partition; requested/exists/source_degraded are exact bool; decoded/count/age_samples_ns are nonnegative exact ints; age summaries are finite float64 or null exactly for no samples; status is FRESH or one closed missing code",
            "ordering": "source-major VIX,ES,VX,SPX_REFERENCE then chronological authorized session; exactly one row per source/session, including NOT_REQUESTED_PRE_VX_BOUNDARY and missing/empty sources",
            "root": "source_receipts_root_sha256=stable_hash of the exact ordered list; VIX/ES/VX bind the post-entry diagnostic inventory and SPX_REFERENCE binds the core integrity receipt",
        },
        "source_selection": {
            "schema": "EntryContextSourceSelectionV1",
            "component_locator_fields_in_order": [
                "publisher_id", "instrument_id", "row_group", "row_index",
                "canonical_row_sha256", "event_time_ns", "available_at_ns",
            ],
            "source_order": ["VIX", "ES", "VX", "SPX_REFERENCE"],
            "numeric_encoding": "all selected close values are canonical Python float.hex strings and round-trip losslessly before formulas/binning; absent values and locators are null/empty only under the closed status law",
        },
        "anchor_context_rows": {
            "schema": "EntryContextAnchorRowV1",
            "anchor_kinds_in_order": ["SESSION_1000", "ENTRY_DECISION"],
            "policy_contract": context_policy_journal_spec(),
            "ordering": "policy order, chronological session, anchor_time_ns, anchor-kind order SESSION_1000 then ENTRY_DECISION, then exact journal transition order; no deduplication",
            "coverage": "one SESSION_1000 row per policy/session and one ENTRY_DECISION row per validated EntryActionObservationV1 including ENTER, WAIT, fill, and no-fill",
            "root": "anchor_context_rows_root_sha256=stable_hash of every lossless ordered row; pooled rows are the exact fold-1-through-fold-5 concatenation",
        },
        "age_samples": {
            "sources_in_order": ["VIX", "ES", "VX", "SPX_REFERENCE"],
            "sample": "current_age_ns from every FRESH source selection in exact anchor-row order; nonnegative exact int, bool forbidden",
            "outer_root": "age_samples_root_sha256=stable_hash(age_samples_ns_by_source)",
            "pooled": "concatenate each source's exact fold-1-through-fold-5 samples before statistics; never average fold quantiles",
            "quantile_method": "Hyndman-Fan type 7: sort x; h=(n-1)*q; j=floor(h); g=h-j; value_ns=(1-g)*x[j]+g*x[min(j+1,n-1)]; seconds=value_ns/1e9; q exact float 0.5 or 0.95; n=0 => null",
            "immutable_function": "v4.research.pathd_entry_exit.context_age_quantile_type7",
        },
        "tables": {
            "keys_in_order": ["coverage_rows", "bin_rows", "sensitivity_rows"],
            "coverage_row_fields_in_order": [
                "scope", "outer_fold", "source", "file_count", "row_count",
                "timestamp_count", "valid_cross_publisher_rows_consolidated",
                "invalid_within_publisher_duplicate_count", "fresh_count",
                "stale_count", "missing_count", "age_seconds_p50",
                "age_seconds_p95", "age_seconds_max", "source_degraded_count",
            ],
            "bin_row_fields_in_order": [
                "scope", "outer_fold", "anchor_kind", "stratifier", "bin_id",
                *leading, *policy_columns,
            ],
            "sensitivity_row_fields_in_order": [
                "scope", "outer_fold", "anchor_kind", "source", "cohort",
                *leading, *policy_columns,
            ],
            "row_types": "scope literal OUTER_FOLD|POOLED_OUTER; outer_fold exact 1..5 or null pooled; anchor_kind is SESSION_1000|ENTRY_DECISION; count/trade/action/fill/PnL/delta cells exact ints; age/calibration cells finite float64 or registered null; bool is never int",
            "coverage_order": "per-fold and pooled coverage source VIX,ES,VX,SPX_REFERENCE",
            "bin_order": "anchor-kind order, registered one-dimensional stratifier order, then every registered bin including zero-count bins",
            "sensitivity_order": "anchor-kind order then source VIX,ES,VX,SPX_REFERENCE and cohort SOURCE_CLEAN,SOURCE_DEGRADED,SOURCE_PRESENT,SOURCE_MISSING",
            "tables_hash": "tables_sha256=stable_hash of the exact three-key table object and is recomputed from anchor rows and exact age samples",
            "pooled_rule": "pooled tables are recomputed from exact concatenated validated rows/samples; no source decode, caller rows, or fold-quantile averaging",
        },
    }


def context_diagnostics_spec() -> dict[str, Any]:
    """Closed post-primary VIX/ES/VX diagnostics with core SPX reference."""

    missing_priority = [
        "NOT_REQUESTED_PRE_VX_BOUNDARY",
        "MISSING_FILE",
        "EMPTY_SOURCE",
        "NO_CAUSAL_BAR",
        "STALE_GT90S",
        "NONFINITE_CLOSE",
        "DUPLICATE_WITHIN_PUBLISHER",
        "SCHEMA_INVALID",
    ]
    level_bins = ["LT15", "15_20", "20_25", "25_30", "GE30"]
    point_change_bins = [
        "LT_NEG1",
        "NEG1_NEG0_25",
        "NEG0_25_POS0_25",
        "POS0_25_POS1",
        "GE1",
    ]
    edge = lambda value: float(value).hex()
    return {
        "schema_version": "pathd.context_diagnostics.v1",
        "role": "diagnostics_and_strata_only",
        "alpha_or_gate": False,
        "execution_authority": {
            "current_preregistration": "AUTHORIZED_POST_ENTRY_PRIMARY_READ_ONLY",
            "blocking_precondition": "all five immutable primary outer result receipts, the immutable pooled entry-acceptance result receipt, the current core integrity receipt, and the post-entry 604-file diagnostic inventory receipt must exist before the dedicated read-only authorization or any VIX/ES/VX/SPX_REFERENCE source decode",
            "authorization_type": "FrozenContextDiagnosticsAuthorizationV1",
            "authorization_scope": "the exact five primary outer session lists and their already frozen entry-policy decisions/journals; no fit, calibration, action, replay, comparator, or gate capability",
            "current_outputs": "five fixed per-fold entry context artifacts and receipts followed by one pooled artifact and receipt",
            "four_box_and_protected_holdout": "UNAUTHORIZED_AND_UNREACHABLE because the exact exit-weight count is below 60 and the run terminates insufficient_evidence before exit fit; a future data-era plan/preregistration must bind their schemas/APIs/writers/receipts",
        },
        "decode_order": "First seal all five primary outer result receipts and pooled entry acceptance. Rehash the exact 604-file diagnostic partition without decoding and freeze its inventory receipt. Then issue exactly one read-only context authorization bound to both integrity receipts, immutable policy-journal roots, upstream receipts, and exact five primary session lists. The runner creates one O_EXCL access receipt before decode, resolves paths internally, writes fold artifact/receipt pairs 1..5, and derives pooled rows/tables only from validated fold artifacts. Context cannot change a primary result or verdict",
        "source_degradation_rule": "source_degraded is a separate Boolean and never replaces the exact missing-status code",
        "source_status_vocabulary": ["FRESH", *missing_priority],
        "sources": {
            "VIX": {
                "path_template": "vendor/thetadata/index/vix_1m/{session}.parquet",
                "required_fields": [
                    "event_time", "symbol", "open", "high", "low", "close",
                    "context_source", "is_derived", "is_proxy",
                    "is_official_index_data",
                ],
                "row_invariants": {
                    "symbol": "VIX",
                    "context_source": "thetadata_index_history_ohlc",
                    "is_derived": False,
                    "is_proxy": False,
                    "is_official_index_data": True,
                },
                "coverage": "251/251 option sessions paired",
                "non_option_extras_never_join": [
                    "2025-09-01", "2025-11-27", "2026-01-19", "2026-02-16"
                ],
                "clock": {
                    "kind": "HISTORICAL_BAR_END_CLOCK_ONLY",
                    "timestamp_semantics": "BAR_OPEN_TIMESTAMP",
                    "available_at": "event_time+60 seconds",
                    "eligible": "latest available_at<=decision_time and age_seconds from available_at in [0,90]",
                    "live_parity_claim": False,
                },
            },
            "ES": {
                "path_template": "raw/databento/glbx_es_ohlcv_1m/{session}.es_c_0.ohlcv-1m.parquet",
                "required_fields": [
                    "ts_event", "symbol", "publisher_id", "instrument_id",
                    "open", "high", "low", "close", "volume",
                ],
                "required_symbol": "ES.c.0",
                "option_session_filenames": 251,
                "empty_source_sessions": [
                    "2025-09-19", "2025-12-19", "2026-03-20", "2026-06-18"
                ],
                "degraded_sessions": list(GLBX_DIAGNOSTIC_DEGRADED_DATES),
                "clock": {
                    "available_at": "ts_event+60 seconds",
                    "eligible": "latest available_at<=decision_time and age_seconds from available_at in [0,90]",
                    "kind": "HISTORICAL_BAR_END_CLOCK_ONLY",
                    "live_parity_claim": False,
                },
            },
            "VX": {
                "path_template": "raw/databento/xcbf_vx_ohlcv_1m/{session}.vx_c_0.ohlcv-1m.parquet",
                "required_fields": [
                    "ts_event", "symbol", "publisher_id", "instrument_id",
                    "open", "high", "low", "close", "volume",
                ],
                "required_symbol": "VX.c.0",
                "availability_boundary_session": "2026-04-01",
                "pre_boundary_status": "NOT_REQUESTED_PRE_VX_BOUNDARY",
                "post_boundary_option_sessions": 84,
                "post_boundary_nonempty_required": 84,
                "non_option_extras_never_join": ["2026-04-03"],
                "known_publishers": [105, 106],
                "clock": {
                    "available_at": "ts_event+60 seconds",
                    "eligible": "latest available_at<=decision_time and age_seconds from available_at in [0,90]",
                    "kind": "HISTORICAL_BAR_END_CLOCK_ONLY",
                    "live_parity_claim": False,
                },
            },
            "SPX_REFERENCE": {
                "path_template": "vendor/thetadata/index/spx_1m/{session}.parquet",
                "role": "ES_MINUS_SPX_BASIS_REFERENCE_ONLY_NONALPHA",
                "integrity_partition": CORE_INTEGRITY_PARTITION,
                "receipt_required": True,
                "required_fields": [
                    "event_time", "symbol", "open", "high", "low", "close", "volume",
                    "context_source", "is_derived", "is_proxy",
                    "is_official_index_data",
                ],
                "row_invariants": {
                    "symbol": "SPX",
                    "context_source": "thetadata_index_history_ohlc",
                    "is_derived": False,
                    "is_proxy": False,
                    "is_official_index_data": True,
                },
                "clock": {
                    "kind": "HISTORICAL_BAR_END_CLOCK_ONLY",
                    "timestamp_semantics": "BAR_OPEN_TIMESTAMP",
                    "available_at": "event_time+60 seconds",
                    "eligible": "latest available_at<=anchor and age_seconds from available_at in [0,90]",
                    "live_parity_claim": False,
                },
            },
        },
        "publisher_consolidation": {
            "identity": ["symbol", "instrument_id", "ts_event"],
            "within_publisher_duplicate": "invalid:DUPLICATE_WITHIN_PUBLISHER",
            "publisher_order": "ascending numeric publisher_id",
            "one_row_per_publisher": True,
            "open": "positive-volume-weighted mean across publishers; equal-weight mean when total positive volume is zero",
            "close": "positive-volume-weighted mean across publishers; equal-weight mean when total positive volume is zero",
            "high": "maximum across publishers",
            "low": "minimum across publishers",
            "volume": "sum across publishers",
            "deterministic": True,
        },
        "lag_15m": {
            "rule": "latest consolidated bar with available_at<=decision_time-15 minutes and age_seconds relative to that lag clock in [0,90]",
            "forward_fill": False,
        },
        "derived_values": {
            "numeric_rule": "IEEE-754 float64 operations in the written order; retain the unrounded value for binning and reporting; no display rounding may affect a bin",
            "vix_15m_point_change": "VIX_close_current - VIX_close_lag15 in index points",
            "vx_15m_point_change": "VX_close_current - VX_close_lag15 in index points",
            "es_15m_bps": "10000.0*(ES_close_current/ES_close_lag15-1.0); nonpositive or nonfinite denominator => SCHEMA_INVALID",
            "es_minus_spx_basis_bps": "10000.0*((ES_close_current/ES_close_lag15-1.0)-(SPX_close_current/SPX_close_lag15-1.0)); SPX_REFERENCE uses receipted core vendor/thetadata official close at the same current and 15m lag clocks with available_at=event_time+60 seconds<=anchor clock and age from available_at<=90s; nonpositive/nonfinite denominator => SCHEMA_INVALID",
            "vx_minus_vix_points": "VX_close_current - VIX_close_current in index points at the same anchor clock",
            "zero_denominator": "SCHEMA_INVALID; retain the primary session and emit no numeric stratum value",
        },
        "missing_status_priority": missing_priority,
        "bins": {
            "interval_rule": "left-closed/right-open except the unbounded edge bins",
            "edge_encoding": "NEG_INF and POS_INF sentinels bound canonical Python float.hex finite edges; binning decodes float64 once and never rounds",
            "vix_level": level_bins,
            "vx_level": level_bins,
            "vix_15m_point_change": point_change_bins,
            "vx_15m_point_change": point_change_bins,
            "es_15m_bps": [
                "LT_NEG25", "NEG25_NEG10", "NEG10_POS10", "POS10_POS25", "GE25"
            ],
            "es_minus_spx_basis_bps": ["LT_NEG5", "NEG5_POS5", "GE5"],
            "vx_minus_vix_points": ["LT_NEG2", "NEG2_0", "0_2", "2_4", "GE4"],
            "vx_era": ["PRE_BOUNDARY_NOT_REQUESTED", "POST_BOUNDARY_EXPECTED"],
            "status": ["FRESH", *missing_priority],
            "edges_float64_hex": {
                "vix_level": ["NEG_INF", *[edge(value) for value in (15, 20, 25, 30)], "POS_INF"],
                "vx_level": ["NEG_INF", *[edge(value) for value in (15, 20, 25, 30)], "POS_INF"],
                "vix_15m_point_change": ["NEG_INF", *[edge(value) for value in (-1, -0.25, 0.25, 1)], "POS_INF"],
                "vx_15m_point_change": ["NEG_INF", *[edge(value) for value in (-1, -0.25, 0.25, 1)], "POS_INF"],
                "es_15m_bps": ["NEG_INF", *[edge(value) for value in (-25, -10, 10, 25)], "POS_INF"],
                "es_minus_spx_basis_bps": ["NEG_INF", edge(-5), edge(5), "POS_INF"],
                "vx_minus_vix_points": ["NEG_INF", *[edge(value) for value in (-2, 0, 2, 4)], "POS_INF"],
            },
            "zero_retention": "every registered one-dimensional bin appears even when all counts and metrics are zero",
        },
        "anchors": {
            "population": "the exact immutable context policy journal bundle only; diagnostics never regenerate, filter, rescore, or accept a policy decision",
            "policy_contract": context_policy_journal_spec(),
            "anchor_kinds_in_order": ["SESSION_1000", "ENTRY_DECISION"],
            "ENTRY_DECISION": "one row per validated journal EntryActionObservationV1 including ENTER, WAIT, fill, and no-fill; bind its exact transition, outcome, optional trade, and economics",
            "SESSION_1000": "one row per policy/session at exact 10:00 America/New_York, bound to that session terminal journal/PnL/trade set",
            "row_schema": "EntryContextAnchorRowV1",
            "lossless_root": "anchor_context_rows_root_sha256=stable_hash of all exact ordered source locators/hashes, float.hex values, statuses, bin ids, policy facts, and upstream transition/evaluation/journal hashes",
            "joint_or_multidimensional_strata": False,
        },
        "artifact_contract": {
            "authorization_path": repo_path_label(
                ENTRY_CONTEXT_DIAGNOSTICS_AUTHORIZATION_PATH
            ),
            "diagnostic_inventory_receipt_path": repo_path_label(
                CONTEXT_DIAGNOSTIC_INVENTORY_RECEIPT_PATH
            ),
            "diagnostic_inventory_receipt": context_diagnostic_inventory_receipt_spec(),
            "access_receipt_path": repo_path_label(
                ENTRY_CONTEXT_DIAGNOSTICS_ACCESS_RECEIPT_PATH
            ),
            "outer_artifact_paths": [
                repo_path_label(
                    _outer_fold_artifact_path(
                        fold, "entry_context_diagnostics.json"
                    )
                )
                for fold in range(1, 6)
            ],
            "outer_receipt_paths": [
                repo_path_label(
                    _outer_fold_artifact_path(
                        fold, "entry_context_diagnostics_receipt.json"
                    )
                )
                for fold in range(1, 6)
            ],
            "pooled_artifact_path": repo_path_label(
                ENTRY_POOLED_CONTEXT_DIAGNOSTICS_PATH
            ),
            "pooled_receipt_path": repo_path_label(
                ENTRY_POOLED_CONTEXT_DIAGNOSTICS_RECEIPT_PATH
            ),
            "abort_receipt_path": repo_path_label(
                ENTRY_CONTEXT_DIAGNOSTICS_ABORT_RECEIPT_PATH
            ),
            "payload_spec": context_diagnostic_payload_spec(),
            "transaction": context_diagnostics_transaction_spec(),
            "outer_payload": "typed EntryOuterContextDiagnosticsV1 with validated policy-journal bindings, four-source receipts, lossless ordered anchor_context_rows/root, exact age samples/root, complete zero-retaining tables/hash, both integrity receipts, access receipt, upstream receipt/session bindings, and semantic self-hash",
            "pooled_payload": "typed EntryPooledContextDiagnosticsV1 contains exact fold-1-through-fold-5 concatenated anchor rows and age samples plus recomputed pooled tables; it is derived only from five revalidated outer artifacts/receipts",
            "writer": "run_fixed_entry_context_diagnostics accepts only the exact typed authorization; it creates one access receipt before decode, resolves every source/journal path internally, writes fold pairs 1..5 then pooled artifact/receipt under the closed transaction spec, and burns any partial/crashed transaction without re-decode",
            "forbidden_caller_inputs": [
                "source path", "source row", "dataset", "policy", "journal",
                "decision", "anchor", "table", "bin", "metric", "status",
                "receipt", "verdict",
            ],
        },
        "mandatory_entry_outputs": {
            "authorized_in_current_preregistration": True,
            "scopes": [
                "each_outer_fold", "pooled_outer", "degradation_sensitivity",
            ],
            "coverage_table_columns": [
                "file_count", "row_count", "timestamp_count",
                "valid_cross_publisher_rows_consolidated",
                "invalid_within_publisher_duplicate_count",
                "fresh_count", "stale_count",
                "missing_count", "age_seconds_p50", "age_seconds_p95",
                "age_seconds_max", "source_degraded_count",
            ],
            "every_one_dimensional_bin_leading_counts": [
                "unique_session_count", "distinct_trajectory_count", "decision_row_count"
            ],
            "policy_columns": [
                "HGB_trades", "HGB_net_pnl", "neural_trades", "neural_net_pnl",
                "P5_trades", "P5_net_pnl", "matched_random_trades",
                "matched_random_net_pnl", "action_count", "fill_count", "no_fill_count",
                "HGB_minus_P5_paired_delta", "HGB_minus_matched_random_paired_delta",
                "descriptive_action_calibration",
            ],
            "degraded_date_identities": {
                "ES": list(GLBX_DIAGNOSTIC_DEGRADED_DATES),
                "VX": [],
            },
            "sensitivity_tables": [
                "source_degraded_vs_clean", "source_missing_vs_present"
            ],
        },
        "failure": "missing or malformed entry context makes only the post-primary diagnostic incomplete; it never removes a primary session, changes an entry result/verdict, or authorizes repair/redecode",
        "four_box_and_protected_holdout": "not authorized by this preregistration. A future data-era preregistration must freeze actual source-receipt/table/schema/API/writer/receipt contracts before any such decode; protected diagnostics must run once after primary reconstruction and before terminal seal with their actual hashes bound into the seal",
        "forbidden_consumers": [
            "model feature", "loss", "action mask", "calibrator", "feature scaler",
            "target scaler", "early stopping", "family selection", "control selection",
            "comparator selection", "threshold",
            "minimum power", "session inclusion", "entry or exit action", "gate",
            "feasibility verdict", "owner-review routing", "best-bin claim",
            "inferential test",
        ],
    }


def calibration_and_statistics_spec() -> dict[str, Any]:
    return {
        "aref_decision_critical_topology": aref_decision_critical_topology_spec(),
        "composite_calibration_terminal_rule": composite_calibration_terminal_rule_spec(),
        "input_validity": "target-validity is frozen from source/label completeness before predictions exist; a nonfinite prediction, residual, or predicted width on any target-valid row invalidates that head/bundle and may never be post-hoc dropped; only preregistered nonfinite-label rows are masked",
        "seed_derivation": {
            "key_object": "canonical compact sorted-key JSON object with exactly {campaign:'pathd.tier_s.v1',plan_sha256:<binding hash>,purpose:<closed purpose>,model_family:<closed token>,fold_scope:<closed token>,statistic:<closed token>}; UTF-8, allow_nan=false, no local aliases or display labels",
            "closed_tokens": {
                "model_family": [
                    "HGB", "NEURAL", "GLOBAL_HGB_ENTRY",
                    "CONSTANT", "SIGN_REVERSED_HGB", "SIGN_REVERSED_NEURAL",
                    "TIME_SHIFTED_HGB", "TIME_SHIFTED_NEURAL",
                    "SHUFFLED_TARGET_HGB_01", "SHUFFLED_TARGET_HGB_02", "SHUFFLED_TARGET_HGB_03", "SHUFFLED_TARGET_HGB_04",
                    "SHUFFLED_TARGET_HGB_05", "SHUFFLED_TARGET_HGB_06", "SHUFFLED_TARGET_HGB_07", "SHUFFLED_TARGET_HGB_08",
                    "SHUFFLED_TARGET_NEURAL_01", "SHUFFLED_TARGET_NEURAL_02", "SHUFFLED_TARGET_NEURAL_03", "SHUFFLED_TARGET_NEURAL_04",
                    "SHUFFLED_TARGET_NEURAL_05", "SHUFFLED_TARGET_NEURAL_06", "SHUFFLED_TARGET_NEURAL_07", "SHUFFLED_TARGET_NEURAL_08",
                    "P5", "MATCHED_RANDOM_AGGREGATE", "GLOBAL_SELECTED_COMPARATOR",
                    "BOX_A", "BOX_B", "BOX_C", "BOX_D", "INTERACTION_D_MINUS_B_MINUS_C_PLUS_A"
                ],
                "fold_scope": [
                    "NESTED_OUTER_1_INNER_1", "NESTED_OUTER_1_INNER_2", "NESTED_OUTER_1_INNER_3", "NESTED_OUTER_1_INNER_4",
                    "NESTED_OUTER_2_INNER_1", "NESTED_OUTER_2_INNER_2", "NESTED_OUTER_2_INNER_3", "NESTED_OUTER_2_INNER_4",
                    "NESTED_OUTER_3_INNER_1", "NESTED_OUTER_3_INNER_2", "NESTED_OUTER_3_INNER_3", "NESTED_OUTER_3_INNER_4",
                    "NESTED_OUTER_4_INNER_1", "NESTED_OUTER_4_INNER_2", "NESTED_OUTER_4_INNER_3", "NESTED_OUTER_4_INNER_4",
                    "NESTED_OUTER_5_INNER_1", "NESTED_OUTER_5_INNER_2", "NESTED_OUTER_5_INNER_3", "NESTED_OUTER_5_INNER_4",
                    "OUTER_1", "OUTER_2", "OUTER_3", "OUTER_4", "OUTER_5", "POOLED_OUTER_1_5", "FULL_PRE_HOLDOUT", "PROTECTED_HOLDOUT_ONCE"
                ],
                "statistic_grammar": ["HEAD::<exact registered head name>", "ACTION::ENTER", "ACTION::WAIT", "ACTION::HOLD", "ACTION::EXIT", "ECON::CANDIDATE_VS_P5", "ECON::CANDIDATE_VS_MATCHED_RANDOM", "ECON::BOX_D_VS_GLOBAL_COMPARATOR", "ECON::BOX_C_MINUS_BOX_A", "ECON::INTERACTION", "ECON::NEURAL_VS_HGB", "HOLDOUT::BOX_D", "CALIBRATION::<exact registered head/action>"],
            },
            "integer": "first four SHA-256 digest bytes interpreted unsigned big-endian",
            "generator": "numpy.random.Generator(numpy.random.PCG64(integer))",
            "purposes": {
                "mean_lcb_bootstrap": 2_000,
                "calibration_uncertainty": 2_000,
                "action_decile_bootstrap": 5_000,
                "pooled_economic_lcb": 10_000,
                "holdout_report_bootstrap": 10_000,
            },
            "binding_key_rules": {
                "mean_lcb_head": "model_family is exactly HGB or NEURAL; fold_scope is the exact NESTED_OUTER_i_INNER_j for a nested bundle, OUTER_i for a final outer bundle, or FULL_PRE_HOLDOUT for full fit; statistic=HEAD::<registered head>; purpose=mean_lcb_bootstrap",
                "calibration_uncertainty": "model_family/fold_scope identify the exact HGB or NEURAL nested/outer/full bundle as above, or GLOBAL_HGB_ENTRY/POOLED_OUTER_1_5 for pooled ENTER/WAIT and BOX_D/POOLED_OUTER_1_5 for HOLD/EXIT; statistic=CALIBRATION::<exact registered head or ACTION::ENTER|ACTION::WAIT|ACTION::HOLD|ACTION::EXIT>; purpose=calibration_uncertainty",
                "entry_action_decile": "model_family=GLOBAL_HGB_ENTRY; fold_scope=POOLED_OUTER_1_5; statistic=ACTION::<ENTER|WAIT>; purpose=action_decile_bootstrap",
                "negative_control_statistics": "for each eligible HGB/NEURAL family, CONSTANT uses model_family=CONSTANT; sign reversal uses SIGN_REVERSED_<family>; time-shift uses TIME_SHIFTED_<family>; shuffle attempt j uses SHUFFLED_TARGET_<family>_<j as two-digit 01..08. Fold scope is the exact nested/outer/full scope for head/calibration work and POOLED_OUTER_1_5 for complete ENTER/WAIT action audit; statistic and purpose otherwise follow the identical mean_lcb_head, calibration_uncertainty, and action_decile rules. Reuse of the positive-family seed key is forbidden.",
                "exit_action_decile": "model_family=BOX_D; fold_scope=POOLED_OUTER_1_5; statistic=ACTION::<HOLD|EXIT>; purpose=action_decile_bootstrap",
                "candidate_vs_p5": "model_family=GLOBAL_HGB_ENTRY; fold_scope=POOLED_OUTER_1_5; statistic=ECON::CANDIDATE_VS_P5; purpose=pooled_economic_lcb",
                "candidate_vs_random": "model_family=GLOBAL_HGB_ENTRY; fold_scope=POOLED_OUTER_1_5; statistic=ECON::CANDIDATE_VS_MATCHED_RANDOM; purpose=pooled_economic_lcb",
                "nested_neural_vs_hgb": "diagnostic attribution only: model_family=NEURAL; fold_scope=OUTER_i for the parent outer fold; statistic=ECON::NEURAL_VS_HGB; purpose=pooled_economic_lcb; never a selection or rescue input",
                "full_family_neural_vs_hgb": "diagnostic attribution only: model_family=NEURAL; fold_scope=POOLED_OUTER_1_5; statistic=ECON::NEURAL_VS_HGB; purpose=pooled_economic_lcb; never a selection or rescue input",
                "global_comparator": "model_family=BOX_D; fold_scope=POOLED_OUTER_1_5; statistic=ECON::BOX_D_VS_GLOBAL_COMPARATOR; purpose=pooled_economic_lcb",
                "box_c_minus_a": "model_family=BOX_C; fold_scope=POOLED_OUTER_1_5; statistic=ECON::BOX_C_MINUS_BOX_A; purpose=pooled_economic_lcb",
                "interaction": "model_family=INTERACTION_D_MINUS_B_MINUS_C_PLUS_A; fold_scope=POOLED_OUTER_1_5; statistic=ECON::INTERACTION; purpose=pooled_economic_lcb",
                "holdout": "model_family=BOX_D; fold_scope=PROTECTED_HOLDOUT_ONCE; statistic=HOLDOUT::BOX_D; purpose=holdout_report_bootstrap",
                "forbidden_ambiguity": "no other allowed-token combination may be used for these operations; an unlisted bootstrap operation is invalid until a new preregistration",
            },
            "receipt": "store the exact key object, canonical bytes SHA-256, integer, library version, and replicate count before use; a token outside the closed table/grammar is invalid_result",
        },
        "calibration_partition": {
            "rule": "last ceil(20% of primary training sessions)",
            "rounding": "ceil",
            "extra_pre_calibration_embargo": True,
            "pre_calibration_embargo": "the single immediately preceding eligible session is excluded from weights and calibration",
            "weights_fit_on_calibration": False,
            "outer_or_future_rows_allowed": False,
        },
        "mean_lcb": {
            "level": 0.90,
            "residual": "observation_minus_prediction",
            "cluster_reduction": "equal mean residual within each calibration session",
            "correction": "empirical q10 of the session-bootstrap distribution of equal-session mean residual; LCB=prediction+correction",
            "order_statistic": "nearest-rank k=ceil((B+1)*0.10), clipped [1,B], one-based, no interpolation",
            "bootstrap_seed": "derived per model/fold/head with seed_derivation purpose mean_lcb_bootstrap",
            "bootstrap_resamples": 2_000,
            "minimum_sessions": 10,
        },
        "q10_conformal": {
            "nominal_quantile": 0.10,
            "residual": "observation_minus_raw_q10_prediction",
            "weighting": "with S sessions and n_s finite rows/session, each row weight=1/(S*n_s); stable-sort by residual then full row identity and combine exact ties",
            "correction": "smallest residual whose cumulative equal-session weight is >=0.10, added to raw q10",
            "order_statistic": "weighted empirical q10, no interpolation",
            "minimum_sessions": 10,
        },
        "monotone_q10_q50_q90": {
            "scope": "every exit target family's already monotone raw q10/q50/q90 outputs",
            "row_identity": ["session", "trajectory_id", "decision_time_ns", "source_neutral_contract_id", "target_family"],
            "weights": "with S calibration sessions and n_s finite target rows in session s, each row has exact weight 1/(S*n_s); require >=10 sessions",
            "location": "let e50=y-raw_q50; stable-sort by (e50,row_identity), combine exact e50 ties, and take c50 as the smallest tied value whose cumulative normalized weight is >=0.50",
            "lower_ratio": "after q50_cal=raw_q50+c50, for width_low=raw_q50-raw_q10>0 define r_low=max(0,(q50_cal-y)/width_low); width_low=0 rows are permanently covered iff y>=q50_cal and otherwise remain uncovered for every finite scale",
            "lower_width": "candidate scales are 0 plus every finite r_low; stable-sort by (scale,row_identity), combine exact scale ties, evaluate all rows including zero-width permanent states, and choose the smallest scale s_low whose equal-session weighted coverage of y>=q50_cal-s_low*width_low is >=0.90",
            "upper_ratio": "for width_high=raw_q90-raw_q50>0 define r_high=max(0,(y-q50_cal)/width_high); width_high=0 rows are permanently covered iff y<=q50_cal and otherwise remain uncovered for every finite scale",
            "upper_width": "candidate scales are 0 plus every finite r_high; use the same stable tie rule and choose the smallest s_high whose equal-session weighted coverage of y<=q50_cal+s_high*width_high is >=0.90",
            "outputs": "q50_cal=raw_q50+c50; q10_cal=q50_cal-s_low*(raw_q50-raw_q10); q90_cal=q50_cal+s_high*(raw_q90-raw_q50)",
            "noncrossing": "s_low and s_high are nonnegative and raw widths are nonnegative by model parameterization, so q10_cal<=q50_cal<=q90_cal by construction; sorting/projection is forbidden",
            "failure": "only rows whose labels were preregistered invalid before prediction are excluded; any nonfinite prediction/width on a target-valid row invalidates the head; if either side cannot attain 0.90 coverage because of permanent zero-width failures or no finite candidate scale, invalidate the head; no epsilon, row dropping, or posthoc rescue",
            "minimum_sessions": 10,
            "aref_role": "the A_ref q10/q50/q90 triplet is one atomic decision-critical calibration node; q50 is the registered location supporting calibrated q10 and q90, and q90 is binding EXIT reliability support",
            "aref_complete_triplet_rule": "all three raw predictions and both side corrections must be finite and valid; target, row, side, or fold deletion and partial/survivor calibration are invalid_result",
        },
        "calibration_bootstrap": {
            "seed": "derived per model/fold/head/action by seed_derivation purpose calibration_uncertainty and statistic CALIBRATION::<exact head/action>",
            "resamples": 2_000,
            "sampling_unit": "whole session with equal session weight",
            "role": "head-calibration uncertainty bands only; not the point correction and not the action-decile monotonicity diagnostic",
            "action_decile_exclusion": "the distinct action-conditioned decile audit uses exactly 5,000 valid session-bootstrap replicates under action_calibration.deciles and seed purpose action_decile_bootstrap",
        },
        "economic_bootstrap": {
            "seed": "derived per comparison by seed_derivation purpose pooled_economic_lcb",
            "resamples": 10_000,
            "sampling_unit": "paired whole session",
            "lower_bound": "one-sided 95% percentile; nearest-rank k=ceil((B+1)*0.05), clipped [1,B], one-based, no interpolation",
            "zero_trade_sessions": "retained with zero PnL",
        },
        "outer_fold_reports": "opened only after every fold artifact/config/calibrator/composer is immutable",
    }


def action_calibration_spec() -> dict[str, Any]:
    return {
        "scope": "pooled five outer folds with fold-specific earlier calibrators; fold tables remain diagnostic",
        "preopen_node_scope": "these post-evidence action-trajectory coverage, decile, and bootstrap gates are not pre-open calibration nodes; B-R pre-open nodes are frozen separately in composite_calibration_terminal_rule_spec",
        "distinct_unit": "unique underlying policy trajectory; repeated seconds/episodes on the same filled position never add power for HOLD or EXIT",
        "actions": {
            "ENTER": {
                "unit": "one legal OOF ENTER intent trajectory, whether filled or failed at next-minute arrival; at most one reliability observation per intent",
                "predicted_mean": "selected contract's calibrated equal-horizon mean-upside dollar composite at intent time",
                "raw_composed_lower": "arithmetic mean of the selected contract's individually calibrated q10 dollar components for exactly the decision-clock-available registered horizons and both MFE/profit-area families after fold-specific seed aggregation/inverse scaling",
                "composite_conformal": "on only that bundle's disjoint chronologically earlier calibration ENTER intents, form residual=realized exact equal-horizon dollar composite minus raw_composed_lower; with S sessions and n_s intents/session use weight 1/(S*n_s), stable-sort by (residual,session,intent_id), combine exact ties, take the smallest residual whose cumulative weight is >=0.10, and add it to every raw composed lower prediction; require >=10 distinct sessions and never refit after seeing outer evidence",
                "predicted_lower": "the selected contract's exact ENTER composed lower statistic after the dedicated disjoint composite conformal correction; individual marginal q10 calibration alone is never treated as composite coverage",
                "realized": "the same MFE/profit-area dollar composite over exactly that intent's decision-clock-available horizon set; failed-fill intents are zero and remain in calibration/power for ENTER reliability",
                "coverage": "realized >= predicted_lower",
            },
            "WAIT": {
                "unit": "one maximal contiguous flat WAIT episode",
                "predicted_mean": "composed max missed-upside statistic frozen in entry.wait_action_calibration_audit",
                "predicted_lower": "WAIT-episode q10 calibrated on disjoint earlier calibration episodes",
                "realized": "nonnegative missed-upside target frozen in entry.wait_action_calibration_audit",
                "coverage": "realized >= predicted_lower",
            },
            "HOLD": {
                "unit": "one filled position trajectory that emits HOLD; anchor is that trajectory's first HOLD, and later HOLD episodes do not add power",
                "predicted_mean": "calibrated conditional mean of A_ref at episode anchor",
                "predicted_lower": "calibrated A_ref q10 at episode anchor",
                "realized": "A_ref(anchor)=H_anchor-E_anchor under the shared fill law",
                "coverage": "realized >= predicted_lower",
            },
            "EXIT": {
                "unit": "one filled position trajectory that emits EXIT; anchor is that trajectory's first EXIT intent, and retries/later EXIT episodes do not add power",
                "predicted_mean": "negative calibrated conditional mean of A_ref at episode anchor",
                "predicted_lower": "negative calibrated A_ref q90 upper-coverage bound at episode anchor",
                "realized": "-A_ref(anchor)=E_anchor-H_anchor under the shared fill law",
                "coverage": "realized >= predicted_lower",
                "coverage_identity": "A_ref(anchor)<=calibrated_q90_upper iff -A_ref(anchor)>=-calibrated_q90_upper",
                "quantile_identity_caveat": "do not claim literal q10(-A_ref)=-q90(A_ref) without a tie-compatible empirical-quantile convention",
            },
        },
        "minimum_distinct_trajectories_per_action": 30,
        "nominal_one_sided_coverage": 0.90,
        "minimum_empirical_coverage": 0.85,
        "coverage_counting": "one Boolean per distinct underlying trajectory/action; repeated second rows never count",
        "deciles": {
            "construction": "stable-sort distinct episodes by predicted_mean then identity; ten equal-count bins with remainder assigned to lower-numbered bins",
            "minimum_per_decile": 3,
            "minimum_distinct_sessions_per_decile": 3,
            "realized_statistic": "equal-session mean realized action value within decile",
            "bootstrap": "for each ACTION initialize exactly one registered PCG64 generator once; process adjacent pairs strictly (decile_1,decile_2), (decile_2,decile_3), ..., (decile_9,decile_10) on that single continuing generator without reset or substream; for each pair draw S session identities with replacement from the S pooled evidence sessions, compute both equal-session decile means only when the draw contains >=1 observation from each decile, and consume discarded draws from the same stream until exactly 5,000 valid replicates or 50,000 total draws for that pair before continuing to the next pair",
            "empty_resample": "record and discard a draw missing either adjacent decile; never impute or silently reduce B; fewer than 5,000 valid by the 50,000-draw cap returns insufficient_evidence",
            "pass": "for every adjacent pair, the upper bound is nearest-rank k=ceil((B+1)*0.90), clipped [1,B], one-based, no interpolation, of lower-decile mean minus higher-decile mean; require <=0; ties are allowed",
            "insufficient": "any empty/underpowered decile returns insufficient_evidence",
        },
        "calibration_source": "only each fold's disjoint chronologically earlier sessions; no outer residual refit",
    }


def entry_composer_spec() -> dict[str, Any]:
    return {
        "prediction_order": [
            "take median across the three seed predictions separately for every mean and q10 head",
            "invert the frozen per-target affine scaler",
            "apply that fold/head's disjoint-calibration mean-LCB or q10 correction",
            "apply complete-ladder, exact-identity, physical, D48, D49, affordability, pending, occupancy, cutoff, and daily-state masks",
            "compose and select exactly once",
        ],
        "required_gate_components": {
            "dollars": [
                f"{horizon}:{family}:dollars"
                for horizon in (10, 20, 45, 90, "remaining_session")
                for family in ("MFE", "profit_area")
            ],
            "returns": [
                f"{horizon}:{family}:return"
                for horizon in (10, 20, 45, 90, "remaining_session")
                for family in ("MFE", "profit_area")
            ],
        },
        "causal_horizon_availability": "derive only from decision clock before reading labels/predictions: expected BUY arrival is the next registered completed minute; finite hN is available iff arrival+N minutes<=exact 15:55, and remaining_session is available for every legal t<15:30; h3/h5 remain diagnostic; record the ordered available set from [h10,h20,h45,h90,remaining_session] and never infer availability from future marks or label-validity",
        "mean_composites": "LCB_mean_upside_$ and LCB_mean_upside_return each arithmetic-mean the two MFE/profit-area calibrated conditional-mean LCB components for every horizon in that row's causal available set; this is 2*H components with equal weight, no premium/horizon/family weighting, and remaining_session guarantees H>=1 for a legal entry",
        "gate": "both composites must be finite and strictly >0; any missing/nonfinite component required by that row's causal available set fails that contract closed; unavailable finite-horizon heads are ignored rather than extrapolated or zero-imputed",
        "q10_ranking": "among gate-passers, arithmetic-mean the corresponding available-horizon calibrated q10 dollar components and available-horizon return components under the identical set; maximize dollar composite first, then return composite",
        "tie_break": [
            "smaller absolute strike_offset_points",
            "smaller signed strike_offset_points",
            "C before P",
            "smaller expiry_yyyymmdd",
            "smaller strike_milli_points",
            "lexicographically smaller source_neutral_contract_id",
        ],
        "WAIT": "choose WAIT when the ladder is globally invalid/stale, no contract is legally eligible, or no eligible contract passes both strict mean-LCB gates",
        "forbidden": ["h3/h5 in gate/rank", "q10 as gate", "q10 inside mean LCB", "post-selection recalibration", "P5 timing/direction/filter"],
    }


def entry_matched_random_spec() -> dict[str, Any]:
    """Freeze the FT2-10 random-key law plus the Path-D shared-exit overlay."""

    authority = (
        REPO_ROOT
        / "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/"
        "matched_random_generator_spec.json"
    )
    return {
        "authority_path": repo_path_label(authority),
        "authority_sha256": sha256_path(authority),
        "attempt_seeds_in_order": list(MATCHED_RANDOM_SEEDS),
        "binding_global_overrides": {
            "authority": "binding Path-D plan plus owner-confirmed fill law and action mask; these overrides apply identically to candidate and every control",
            "fill": "submitted BUY hard limit and no-fill-as-zero under fill_law_hash replace the FT2 arrival-ask price-improvement rule",
            "mask": "the frozen Path-D mask replaces older FT2 wording: all 42 slots need finite bid>=0 and ask>0 with bid<=ask for global completeness, while a selected BUY proposal separately needs finite bid>0, ask>0, strict bid<ask, and every per-action/account guard; a locked slot may preserve global schema completeness but is never an executable selected action",
            "unchanged_authority_scope": "attempt seeds/order, matching cells/bands, canonical random key, no-redraw rules, post-match tolerances, multiplicity, and aggregate; the cross-cell collision resolution is the explicit Path-D global-schedule override below",
            "whole_artifact_byte_identity_claimed": False,
        },
        "hold_to_flat_channel": {
            "schema_version": "PathDMatchedRandomHoldToFlatV1",
            "acceptance_control": "use the authority random key/cells with the Path-D global collision schedule, exact hold-to-flat lifecycle, and at most one successful fill per session subject to binding_global_overrides; no shared-exit overlay applies",
            "candidate_budget": "exact causal candidate BUY-intent counts by fold/session/call_or_put/ATM_NEAR_WING/decision_time_premium_band under hold-to-flat and the active fee path",
            "proposal_population": "the same time-t physical/future-blind population law as shared_exit_overlay",
            "schedule": "annotate every proposal with its canonical cell, globally stable-sort by (unsigned rank64, full 32-byte digest, canonical cell tuple, decision_time_ns, exact-contract identity tuple), traverse once, accept only while that cell quota remains and that decision minute is unused, and stop only when all quotas fill or population exhausts; this single global traversal is the sole winner when multiple cells propose the same minute",
            "replay_and_failure": "chronological on its independent one-position account; structural skips/no-fills never redraw, substitute, or backfill; any unrealized cell quota makes that seed INSUFFICIENT_MATCHED_CONTROL",
            "literal_legacy_authority_replay": "optional diagnostic only because favorable arrival-ask pricing and legacy masks violate the binding Path-D comparison game",
        },
        "shared_exit_overlay": {
            "schema_version": "PathDMatchedRandomSharedExitV1",
            "scope": "only fold_specific_shared_transparent_exit lifecycle/re-entry; hold-to-flat excludes this lifecycle overlay but still uses the binding Path-D global cross-cell collision schedule",
            "inherited_unchanged": [
                "eight seeds and their order",
                "matching cells and composer-defined bands",
                "canonical SHA-256 random key and candidate-hash exclusion",
                "label/future blindness",
                "no redraw, replacement, borrowing, or best-seed selection",
                "post-replay tolerances",
                "equal-weight aggregate only after all eight pass",
            ],
            "scoped_override": "replace hold-to-flat/one-fill lifecycle with the already selected single shared transparent exit, permit re-entry only after a filled SELL, and separate proposal selection from replay occupancy",
            "candidate_budget": "exact causal candidate BUY-intent counts by fold/session/call_or_put/ATM_NEAR_WING/decision_time_premium_band under the same shared exit and fee path",
            "proposal_population": "time-t physical identity, quote, context, cutoff, globally complete 42-slot ladder, selected-pair bid>0/ask>0, ask-floor, and source-neutral contract eligibility only; exclude labels, t+1 arrival, fill success, future path, and the random attempt's future ledger state",
            "schedule": "annotate every proposal with its canonical matching-cell tuple; globally stable-sort all proposals by (unsigned rank64, full 32-byte digest, canonical cell tuple in registered field order, decision_time_ns, exact-contract identity tuple); traverse once, accepting only while that cell quota remains and that decision minute is unused; stop when every quota fills or the global population exhausts",
            "freeze_time": "freeze proposal-population, budget, and ordered schedule hashes before comparator PnL or summary statistics are computed",
            "replay": "chronological under each attempt's own account, D48, D49, pending, daily-stop, active fee path, and the fold's shared transparent exit; affordability is recomputed at every scheduled completed minute",
            "structural_skip": "when a frozen selected pair is occupied, pending, daily-stopped, unaffordable at that completed minute, or otherwise ledger-ineligible, record the exact reason, issue no intent, substitute nothing, and never backfill",
            "fill_failure": "preserve the selected exact identity, substitute nothing, and permit the next decision no earlier than t+2",
            "achieved_matching": "realized BUY-intent counts must still equal every frozen candidate cell budget; any structural skip or no-fill timing interaction that prevents equality makes the attempt INSUFFICIENT_MATCHED_CONTROL",
            "receipts": [
                "proposal_population_hash",
                "candidate_budget_hash",
                "ordered_schedule_hash",
                "structural_skip_trace_hash",
                "realized_intents_and_fills_hash",
                "post_match_metrics",
                "authority_sha256",
                "overlay_semantic_hash",
            ],
        },
        "channels_built_separately": [
            "PathDMatchedRandomHoldToFlatV1",
            "PathDMatchedRandomSharedExitV1",
        ],
        "fee_paths_built_separately": [3, 4],
        "policy_specific_ownership": "construct a separate causal budget, proposal/schedule receipt, eight replays, and aggregate for every evaluated learned, constant, sign-reversed, time-shifted-features, and individual strong-shuffle policy/no-screen-arm identity in each fold/channel/fee path; never reuse another policy's budget or schedule",
        "policy_identity_in_random_key": False,
        "common_ordering": "select/freeze lifecycle first; freeze candidate budget and schedules next; replay all attempts; run post-match checks; aggregate last",
        "matching_cells": [
            "outer_fold",
            "session",
            "call_or_put",
            "ATM_NEAR_WING",
            "decision_time_premium_band",
        ],
        "premium_and_moneyness_definitions": "exact composer_spec.json and matched_random_generator_spec.json definitions and hashes; no local rebucketing",
        "selected_pair_budget": "exact in every cell before replay; an underfilled proposal cell makes that attempt incomplete",
        "post_replay_matching": {
            "executed_trade_count_relative_drift_max": 0.05,
            "call_put_total_variation_max": 0.05,
            "moneyness_total_variation_max": 0.06,
            "mean_executed_premium_relative_drift_max": 0.10,
            "mean_realized_holding_minutes_relative_drift_max": 0.10,
            "total_realized_occupancy_relative_drift_max": 0.10,
        },
        "all_eight_required": True,
        "failure": "INSUFFICIENT_MATCHED_CONTROL and therefore insufficient_evidence at the entry gate",
        "aggregate": "after replay and only when all eight pass, equal-weight the eight attempt-specific session PnL series; retain every attempt separately",
        "forbidden": [
            "redraw",
            "replacement",
            "borrow across cells",
            "best-seed selection",
            "outcome matching",
            "candidate-specific random-key namespace",
            "dropping structural skips or zero-trade sessions",
        ],
    }


def entry_negative_control_spec() -> dict[str, Any]:
    return {
        "required_control_ids_in_order": list(ENTRY_NEGATIVE_CONTROL_IDS),
        "required_per_base_family": ["HGB", "NEURAL"],
        "bundle_api": "fit_entry_negative_control_bundle accepts only a current weights authorization and internally loaded sealed dataset; control_id/base_family must be in the closed table, and the artifact binds its dedicated seed key/data permutation or transform. No generic payload can stand in for a control fit",
        "constant_output": {
            "source": "the same model-fit role only; calibration/outer/holdout targets are forbidden",
            "weights": "with S model-fit sessions and n_s finite rows per target, every finite row has weight 1/(S*n_s)",
            "mean_head": "weighted arithmetic marginal mean in original target units, independently for each of the exact 20 target axes",
            "q10_head": "weighted empirical q10 in original target units, stable-sort by value then full row identity, combine exact ties, and take the smallest value with cumulative weight >=0.10",
            "prediction": "broadcast those 40 fixed values to every action; no feature, time, contract, side, geometry, or state dependence",
            "consumer": "fit an independent disjoint-session calibrator for every head, then use the identical masks, composer, shared lifecycle, and ledgers",
            "failed_fills": "retain registered zero target bundles",
            "seeds": "one seed-independent control; duplicating it across ensemble seeds is forbidden",
        },
        "sign_reversed": {
            "source": "the frozen evaluated-family ensemble after median aggregation, inverse scaling, and its ordinary disjoint-session calibration",
            "operation": "immediately before composition, multiply every calibrated mean-LCB component and every calibrated q10 ranking component by -1",
            "q10_interpretation": "synthetic orientation-bug probe, not a mathematically relabeled quantile; no unavailable q90 is invented",
            "recalibration_or_refit": "none",
            "preserved": "row/action identities, complete-ladder and legal masks, target observations, composer order, shared lifecycle, and causal ledgers",
        },
        "strong_shuffled_targets": {
            "attempt_seeds_in_order": list(SHUFFLED_TARGET_SEEDS),
            "population": "the complete model-fit row population in canonical fold/session/decision_time_ns/source_neutral_contract_id order; calibration and outer rows are untouched",
            "bundle": "move each row's complete entry target bundle and all target-validity bits together; never shuffle heads or missingness independently",
            "permutation": "for each fixed attempt seed, sort target-row identities by the exact FT2-10 SHA-256 random-key law and assign those ordered bundles to the canonical feature-row identities one-for-one",
            "row_count": "source and destination populations must be identical; duplicates, truncation, or unmatched rows fail",
            "fit": "refit the same registered family, architecture, hyperparameters, ensemble seeds, sampling budget, calibrators, composer, shared lifecycle, and ledgers",
            "all_eight_required": True,
            "no_redraw_no_selection": True,
        },
        "time_shifted_features": {
            "operation": "for every decision frame replace only the signed-17 value/finite history tensor used by the model with that same session's tensor ending exactly 30 completed minutes earlier; preserve current exact-contract geometry, current physical/safety/action masks, ledger state, labels, and row identity",
            "early_history": "when the shifted endpoint predates the session's registered context, use the ordinary all-missing history representation and finite flags false; never borrow another session, wrap, or read forward",
            "fit": "fit the same registered HGB or neural family, hyperparameters, weights roles, disjoint calibrators, composer, shared lifecycle, and causal ledgers under its dedicated TIME_SHIFTED_<family> seed namespace",
            "selection": "reported only and never selectable",
        },
        "acceptance": "constant, sign-reversed, time-shifted-features, and every one of the eight shuffled attempts must fail at least one complete entry economic-plus-ENTER/WAIT action gate for any family eligible to freeze; one accepted negative invalidates that family",
    }


def exit_matched_random_spec() -> dict[str, Any]:
    return {
        "schema_version": "PathDMatchedRateRandomExitV1",
        "seeds_in_order": list(EXIT_MATCHED_RANDOM_SEEDS),
        "episode_unit": "one anchor per held-position trajectory: the first actionable second at which the learned composer requests EXIT while no exit intent is pending; all subsequent no-fill retries through fill/terminal belong to that same episode and never add count",
        "matching_cells": [
            "outer_fold",
            "session",
            "held_right",
            "entry_premium_band:[1,2),[2,3),[3,5),[5,inf)",
            "time_held_band:[0,60),[60,300),[300,900),[900,inf)",
        ],
        "candidate_budget": "exact learned EXIT episode-anchor count in every cell, frozen separately for each entry trace/box, fold, fee path, delay rung, and floor setting before comparator PnL/statistics",
        "proposal_population": "every causally actionable completed second on the corresponding frozen held-position traces before 15:55, excluding seconds with pending exit; uses identities/state through that second only and no A_ref, future fill, future quote, PnL, or oracle field",
        "random_key": "SHA-256 compact sorted-key JSON {seed,outer_fold,session,trajectory_id,decision_time_ns,source_neutral_contract_id}; rank64 first 8 bytes unsigned big-endian, then full digest",
        "schedule": "globally stable-sort by (rank64,full_digest,canonical_cell_tuple,session,trajectory_id,decision_time_ns,contract_id); greedily accept only while the cell quota remains and trajectory_id has not been reserved by any previously accepted proposal, then reserve that trajectory permanently on acceptance; traversal rank, not clock order, determines its sole anchor; stop at filled quotas or population exhaustion",
        "replay": "chronological on an independent causal account using the same frozen entry policy and shared fill law; at a selected anchor issue EXIT, and after no-fill retry that same episode at every next actionable second until fill or exact 15:55; never consume a second random draw for retries",
        "structural_skip": "if the random account does not hold the scheduled trajectory/contract at its anchor because earlier exits/re-entry changed occupancy, record reason, issue nothing, substitute nothing, and do not backfill",
        "achieved_matching": "realized random EXIT episode-anchor counts must equal every candidate cell budget and inherited exposure-band tolerances must pass; retries are reported separately; any mismatch makes that attempt INSUFFICIENT_MATCHED_CONTROL",
        "all_eight_required": True,
        "aggregate": "only after all eight complete, equal-weight the eight attempt-specific session PnL series; report every attempt and zero-trade session",
        "forbidden": [
            "redraw or replacement",
            "best-seed selection",
            "counting retry intents as episodes",
            "outcome/PnL/oracle matching",
            "borrowing across cells",
            "policy/artifact identity in random-key namespace",
        ],
    }


def comparator_replay_spec() -> dict[str, Any]:
    return {
        "common_rules": {
            "shared_fill_law": True,
            "independent_ledger_per_policy": True,
            "headline_floor": "fixed catastrophic floor applies to candidate and every comparator; repeat every policy floor-off as mandatory ablation",
            "priority": ["15:55 forced flat", "feed-loss safety", "catastrophic floor", "fixed stop", "fixed target", "fixed time", "hold"],
            "ordinary_quote_gap": "before the frozen feed-loss trigger, an absent, stale, locked, crossed, or zero-bid ordinary completed-second state emits no SELL intent and remains HOLD; never call legacy lifecycle_exit_intent and never substitute a stale safety price",
            "exit_no_fill": "remain open and retry from the next completed second; charge no fee until a fill",
            "trigger_mark": "current valid SELL hard limit derived from reference bid under the shared tick law",
            "return_basis": "((trigger_mark-entry_fill)*100-active_round_trip_fee)/max(entry_fill*100,one cent)",
        },
        "panel": {
            "exit_immediate": "request exit at the first actionable completed second strictly after entry fill",
            "hold_to_flat": "no ordinary exit before exact 15:55 forced flat",
            "time_seconds": [60, 300, 900],
            "stop_returns": [-0.25, -0.50],
            "target_returns": [0.25, 0.50, 1.00],
            "stop_target_time_grid": "all 2x3x3 combinations, with stop before target before time when simultaneous",
            "legacy_p5_lifecycle": "include only when every required legacy input is causally present; it is selectable only with >=25 outer sessions and >=50 completed trades, otherwise diagnostic/ineligible",
            "matched_rate_random_8": exit_matched_random_spec(),
        },
        "best_honest_comparator": {
            "candidate_set": "the exact Cartesian set {P5-under-cap entry, frozen global-HGB learned entry} x {exit_immediate, hold_to_flat, time_60s, time_300s, time_900s, all 18 fixed stop-target-time policies, eligible legacy_P5_lifecycle, matched_rate_random_8_aggregate}, plus P5-under-cap entry x frozen HGB learned exit (Box C); exclude only global-HGB learned entry x HGB learned exit (Box D). The eight random attempts are reported but never individually selectable: only their preregistered equal-weight aggregate is one candidate",
            "canonical_policy_id": "entry_id+'::'+exit_id using entry ids P5_UNDER_CAP_V2 or GLOBAL_HGB_LEARNED_ENTRY and the canonical panel ids in listed order; Box C id is P5_UNDER_CAP_V2::GLOBAL_HGB_LEARNED_EXIT; the selected exact comparator id and its per-fold artifact hashes are frozen in the complete pre-holdout packet",
            "selection": "single canonical policy id with maximum pooled five-fold $3 floor-on net PnL across the exact candidate set on identical sessions; select once globally with lexicographic canonical-id tie break, never a different maximum per fold",
            "fixed_across_stress_and_holdout": "carry that exact entry/exit policy identity unchanged into every fold delta, $4 replay, SELL-delay rung, floor-off ablation, and the one-shot holdout; stress results never reselect. On holdout, GLOBAL_HGB_LEARNED_ENTRY/GLOBAL_HGB_LEARNED_EXIT resolve to the already frozen full-fit HGB bundles while P5 and transparent policy semantics remain identical",
            "candidate_delta": "paired session PnL candidate minus that single global comparator",
            "bootstrap": "one-sided 95% paired whole-session lower bound",
            "fold_consistency": "candidate minus the same globally selected comparator in each fold",
            "random_aggregate": "eligible only if all eight schedules complete; equal-weight mean of attempt-specific session PnL, with every attempt separately reported",
            "incomplete_random_cell": "INSUFFICIENT_MATCHED_CONTROL; no borrow, redraw, replacement, or best-seed selection",
        },
        "exit_random_bands": {
            "entry_premium_dollars": ["[1,2)", "[2,3)", "[3,5)", "[5,inf)"],
            "boundary_rule": "left inclusive and right exclusive for every finite band; the final band is unbounded, so every action-eligible entry premium >=$1 has exactly one band and an out-of-band episode invalidates the attempt",
            "time_held_seconds": ["[0,60)", "[60,300)", "[300,900)", "[900,inf)"],
            "matching": "exact learned EXIT episode-anchor count under PathDMatchedRateRandomExitV1; no-fill retries never add count",
        },
    }


def account_replay_spec() -> dict[str, Any]:
    return {
        "outer_fold_reset": "each policy/comparator/box starts each outer fold with an independent $10,000 account",
        "within_fold_carry": "cash/equity carry chronologically across sessions inside that fold",
        "pooled_result": "sum paired session/fold PnL across five independent fold accounts; never chain capital from one outer fold into another",
        "holdout_reset": "one new independent $10,000 account per fixed policy at protected scoring",
        "session_start_equity": "prior session closing equity within the fold; first session is $10,000",
        "D48": "floor(5% of policy session-start equity) maximum entry hard-limit premium*100 plus active round-trip fee reserve",
        "D49": "reset each session; block when realized session loss plus proposed entry cost exceeds floor(5% of session-start equity); daily stop at realized PnL<=negative budget",
        "quantity": 1,
        "max_open_positions": 1,
        "pending_counts_as_occupied": True,
        "premium_plus_fee_affordability": True,
        "last_entry": "strictly before 15:30 ET",
        "forced_flat": "exactly 15:55 ET",
        "headline_fee": {"per_filled_side_dollars": 1.50, "round_trip_reserve_dollars": 3.00},
        "fee4_sensitivity": {"per_filled_side_dollars": 2.00, "round_trip_reserve_dollars": 4.00, "separate_causal_action_masks_and_ledgers": True},
        "floor_fee_constant": "owner-locked floor always uses +$3/100 even in the $4 sensitivity; actual $4 fees still enter net PnL and D48/D49",
    }


def holdout_open_spec() -> dict[str, Any]:
    return {
        "open_count": 0,
        "public_api": "execute_protected_holdout_once() takes no arguments and synchronously performs access-receipt creation, decode, fixed evaluation, result write, and terminal seal; no public begin/load/seal/abort exists and no live authorization is returned. inspect/recover are read-only with respect to corpus decode, and recovery can only burn an interrupted transaction",
        "threat_model": "this is an accidental-misuse and researcher-self-fooling firewall enforced by a clean hash-verified registered Python process, not a security sandbox against a hostile same-process debugger/monkeypatcher or an operator with direct corpus filesystem access. The supported holdout command must run from the frozen dependency closure with no injected code, debugger, plugin, or interactive hook",
        "prerequisites": [
            "refined B-R has an exact five-VALID entry and exit manifest; an entry-only PASS is insufficient",
            "all five outer-fold entry and exit bundles/composers/thresholds frozen",
            "complete four-box packet and guard panel frozen with immutable root hash",
            "every outer acceptance gate passed",
            "full-fit entry and exit bundles frozen from pre-holdout roles only",
            "no survival, source, lineage, calibration, fill-law, claim, or minimum-power violation",
            "immutable-source hashes and the exact fit/scoring environment are current and packet-bound",
            "holdout_access_receipt absent and holdout_open_count zero",
        ],
        "full_preholdout_order": "FULL_PRE_HOLDOUT entry exists only after immutable outer entry PASS; FULL_PRE_HOLDOUT exit and the complete packet exist only after immutable combined PASS; every required full-scope calibration node is VALID before open",
        "transaction": [
            "validate and hash the complete pre-open packet",
            "atomically write OPENING receipt with count=1, exact protected-session hash, packet hash, and timestamp before model/economic reads",
            "read and score exactly the frozen 30 sessions once",
            "write exactly one typed ProtectedHoldoutTraceRecordV1 per protected session; reconstruct every PnL/trade/guard/survival/result field from the two full policy journals and sealed dataset, never from caller summary flags",
            "atomically seal receipt with result/output hashes; never rewrite artifacts or reopen",
            "if any exception or process interruption occurs after the count-one access receipt but before the normal live seal, permanently abort/burn; never recovery-seal an unsealed durable result",
        ],
        "pass": {
            "primary_scope": "29 non-OPRA-degraded protected sessions only; 2026-07-31 is degradation sensitivity and cannot help or hurt pass",
            "Box_D_net_pnl_dollars_on_primary_29": ">0",
            "Box_D_minus_same_frozen_best_comparator_paired_net_pnl_on_primary_29": ">0",
            "primary_non_degraded_sessions": "29 and >=25",
            "completed_Box_D_trades_on_primary_29": ">=50",
            "guards": "all pass",
            "survival": "no overlap, unaffordable trade, daily-risk breach, or nonflat close",
        },
        "uncertainty": "session-bootstrap interval is reported but no new holdout threshold is invented",
        "failure": "record failure or interruption permanently; no repair, recovery seal, threshold change, candidate swap, or second open; a new data-era plan is required",
    }


def fill_law() -> dict[str, Any]:
    law = {
        "version": "pathd.tier_s.provisional_fill.v1",
        "numeric_units": {
            "option_price": "USD_OPTION_PRICE_MICROS",
            "cash": "USD_CASH_MICROS",
            "contract_multiplier": 100,
        },
        "raw_cbbo_1s_clock": {
            "source_fact": "ts_event is NaT and the parquet index/field ts_recv is the causal availability timestamp",
            "represented_interval": "[ts_recv-1 second, ts_recv); canonical represented_second=ts_recv-1 second",
            "available_at": "ts_recv; never treat represented_second as the availability clock",
            "exit_second_quote": "for an exact contract at exit decision availability d, choose the latest represented interval with ts_recv<=d and d-ts_recv<=2,000ms; do not cross contract/session and do not carry beyond 2 seconds",
            "minute_exit_mark": "a registered completed-minute descriptive SELL mark u uses the exact-contract raw 1s state available by boundary u under exit_second_quote; it is not an arbitrary earlier-minute forward fill",
            "sell_delayed_arrival": "at the exact registered SELL arrival boundary a=d+L, use exit_second_quote with d replaced by a; if no fresh actionable exact-contract state is available by a, that intent is no-fill/cancelled and may not rest until a later quote; a retry is a new intent on the next permitted second",
            "forced_flat_boundary": "zero-delay 15:55 uses the latest exact-contract actionable BBO with ts_recv<=15:55 and age<=2,000ms as the boundary state; no older carry is allowed; absent, stale, locked, or crossed boundary state resolves under the frozen zero terminal law",
        },
        "entry_minute_clock": {
            "source": "aligned/processed/minute_entry.contract_quote_metadata at the registered completed-minute boundary",
            "decision_quote": "same exact source-neutral identity, quote source/receipt timestamp<=decision boundary, age<=90,000ms, finite selected bid>0/ask>0 and strict bid<ask",
            "arrival": "the next registered completed-minute row t+1 for only the selected exact identity under the same 90,000ms selector; never use the raw 1s 2-second carry rule for BUY arrival",
            "missing_or_stale_arrival": "failed fill with the complete registered target bundle exactly zero; no later resting or alternate contract",
            "selected_identity_fill_rechecks": "apply identically to candidate, P5, nearest-actionable, and every matched-random/negative-control policy: at t+1 require the same exact source-neutral identity, a finite fresh actionable bid/ask with ask>=1.00 and bid>0 and strict bid<ask, arrival ask<=the frozen BUY hard limit, actual hard-limit premium plus the active fee affordable under current funds, D48, and that policy ledger's D49; any failure is one failed fill with zero targets, no fee, no position, no substitute, and no next action before t+2",
        },
        "tick": {
            "threshold_option_price_micros": 3_000_000,
            "below_threshold_micros": 50_000,
            "at_or_above_threshold_micros": 100_000,
            "basis": "reference ask for BUY and reference bid for SELL before adding/subtracting the tick; never the submitted limit or arrival quote",
        },
        "entry": "completed-minute BUY limit at reference ask plus one tick; evaluate at next completed minute; fill only when arrival ask <= limit; fill at limit",
        "exit": "completed-second SELL limit at max(0, reference bid minus one tick); evaluate after frozen delay; fill only when arrival bid >= limit; fill at limit",
        "fill_price_mode": "SUBMITTED_HARD_LIMIT",
        "headline_delay_ms": 1_000,
        "delay_sensitivity_ms": [0, 1_000, 2_000, 5_000],
        "paired_quote_latency_bounds_ms": [100, 250, 500, 1000],
        "delay_scope": "SELL/exit intents only: the entry BUY always uses the owner-frozen next completed-minute arrival and is never shifted by a delay or paired-latency rung; 0/1/2/5-second SELL rungs are acceptance stresses and 100/250/500/1000ms paired SELL bounds are diagnostic-only",
        "fee_per_filled_side_cash_micros": 1_500_000,
        "completed_round_trip_fee_cash_micros": 3_000_000,
        "fee_4_sensitivity_cash_micros": 4_000_000,
        "fee_4_per_filled_side_cash_micros": 2_000_000,
        "fee_paths": "primary and $4 are separate causal ledgers/action masks/reserves; no metrics-only subtraction",
        "fee_sensitivity_forecast_law": "fit the 20 mean and 20 q10 entry heads, all scalers/calibrators, ENTER/WAIT composite calibrators, and composer thresholds exactly once on the primary $3 fee-adjusted targets; the $4 stress reuses those exact frozen predictions/calibrators/gates with no target refit and no deterministic $1 forecast subtraction, then reapplies the $4 affordability/D48/D49 masks and recomposes among the surviving exact actions on its own causal ledger; entry choices may differ only through those masks/ledger state/tie population, and all realized PnL/fees use $4",
        "exit_no_fill": "position remains open and redecides next completed second",
        "locked_crossed_stale": "non-actionable and counted",
        "no_bid": "zero option value for adverse labels; an ordinary or terminal bid=0 is non-actionable and never instantiates an ExecutionIntent",
        "freshness": {
            "entry_completed_minute_max_age_ms": 90_000,
            "exit_completed_second_max_age_ms": 2_000,
        },
        "forced_flat": {
            "clock": "exact 15:55 America/New_York",
            "quote": "latest exact-contract actionable BBO causally available by 15:55 with age<=2,000ms; never carry an older quote or use a future quote; absent, stale, locked, or crossed state is the explicit zero terminal law",
            "zero_terminal_law": "when the boundary has no fresh actionable BBO, emit no ExecutionIntent and make no simulated-fill claim; instead emit one TERMINAL_ZERO_WRITE_DOWN accounting event with provenance DERIVED_NO_ACTIONABLE_BBO, the last actual option-feed watermark (or null when none), zero proceeds, and one active modeled terminal penalty equal to the configured close-side fee ($1.50 primary or $2 sensitivity), then mark flat and count absent/stale/locked/crossed reason; a pending ordinary exit is consumed/cancelled once and may not survive the boundary",
            "delay_ms": 0,
            "reason": "terminal safety exception required to make exact-boundary flat compatible with the one-second ordinary-exit delay",
            "pending_exit": "resolve once; boundary quote and fee may not be double-consumed",
            "actionable_reference_vendor": "DATABENTO_OPRA",
            "invalid_boundary_provenance": "DERIVED_NO_ACTIONABLE_BBO",
            "zero_event_counting": "no order, intent, SELL fill, or no-fill count; it is one completed simulated position trajectory and one completed trade for serial PnL, equity, distinct-trajectory power, and Pickles assignment; the modeled terminal penalty enters net PnL exactly once but is not described as a broker commission or observed fill fee",
        },
        "feed_loss_safety": {
            "priority": "after exact 15:55 handling and before floor/learned/comparator exits",
            "trigger": "while held before 15:55, only an explicit source-health record with option_feed_available=false triggers feed-loss safety; exact-contract quote age, sparse updates, no update, locked/crossed, or zero bid never imply global feed loss and remain ordinary non-actionable HOLD states",
            "historical_availability_mapping": "the current raw quote corpus has no connection-status channel, so an intact verified session file maps option_feed_available=true and mere absence of an exact-contract quote never becomes global feed-down; only an explicit future source-health record may set false, and this corpus is expected to have zero such records",
            "quote_gap_behavior": "until a fresh raw-1s exact-contract actionable BBO arrives, no ordinary floor/learned/control SELL intent is emitted and occupancy remains LONG_ONE; there is no age-based capacity release or write-down",
            "outcome": "on explicit global feed-down, because no truthful executable price exists, emit no ExecutionIntent and never use a stale DATABENTO or invented IBKR price; emit one FEED_LOSS_ZERO_WRITE_DOWN accounting event with provenance DERIVED_FEED_LOSS, the last actual option watermark or null, zero proceeds, and one active modeled close-side penalty ($1.50/$2), then transition LONG_ONE or EXIT_PENDING to FLAT_TERMINAL and stop that policy for the session",
            "pending_exit": "headline replay evaluates feed-loss only before constructing an ordinary SELL or immediately after its synchronous fill/no-fill transcript, so feed loss never races a pending headline intent; the EXIT_PENDING transition is retained for diagnostic partial/cancel/disconnect scenarios and, if a safety event encounters it there, consumes the exact pending intent once with no later event",
            "counting": "same as TERMINAL_ZERO_WRITE_DOWN: no order/SELL-fill/no-fill count, but one completed simulated trade/trajectory included in serial PnL, equity, power, and Pickles; reason is reported separately",
            "entry_recovery": "none within that session after a feed-loss write-down; FLAT_TERMINAL forbids re-entry even if a later source-health record recovers",
            "legacy_factory_forbidden": "never call DeterministicGovernor.lifecycle_exit_intent; RISK_GOVERNOR/FORCED_FLAT is used only when a truthful actionable safety/terminal price exists",
        },
        "non_order_safety_event_schema": {
            "schema_version": "pathd.research.non_order_safety_event.v1",
            "not_execution_event": "research-ledger accounting event only; never encode as ExecutionEventV1, broker transcript, order, intent, fill, or no-fill",
            "required_fields": [
                "schema_version",
                "event_id",
                "event_type",
                "session",
                "policy_id",
                "fee_path",
                "authorization_sha256",
                "event_time_utc",
                "event_time_ns",
                "source_neutral_contract_id",
                "osi_symbol",
                "state_from",
                "state_to",
                "invalid_reason",
                "provenance",
                "last_actual_option_watermark_utc",
                "zero_proceeds_cash_micros",
                "modeled_close_penalty_cash_micros",
                "cash_before_micros",
                "cash_after_micros",
                "equity_before_micros",
                "equity_after_micros",
                "pending_intent_id_consumed",
                "fill_law_hash",
                "prior_ledger_sha256",
                "result_ledger",
                "event_sha256",
            ],
            "event_types": ["FEED_LOSS_ZERO_WRITE_DOWN", "TERMINAL_ZERO_WRITE_DOWN"],
            "state_transitions_by_event_type": {
                "FEED_LOSS_ZERO_WRITE_DOWN": ["LONG_ONE->FLAT_TERMINAL", "EXIT_PENDING->FLAT_TERMINAL"],
                "TERMINAL_ZERO_WRITE_DOWN": ["LONG_ONE->FLAT_TERMINAL", "EXIT_PENDING->FLAT_TERMINAL"],
            },
            "numeric_law": "zero_proceeds_cash_micros=0; modeled close penalty is the active $1.50/$2 side amount implied by fee_path; cash_after=cash_before-penalty; equity_after=cash_after with no position value; event_time_utc and event_time_ns identify the same exact instant; event_id is SHA-256 of the immutable causal identity fields; event_sha256 seals every field except itself, including prior_ledger_sha256 and the full result_ledger; pending_intent_id_consumed is non-null and consumed exactly once iff state_from=EXIT_PENDING, otherwise it is null",
        },
        "slippage": "arrival quote plus through-limit only; no second ad-hoc term",
        "simulator": {
            "base_seam": "v4.path_d.execution.simulated::SimulatedExecutor with only the preregistered default-preserving ExecutionScenario.fill_price_mode extension",
            "fill_modes": "ARRIVAL_QUOTE remains the default and byte-behavior-compatible; SUBMITTED_HARD_LIMIT is explicit research mode and returns intent.price_budget.hard_limit_micros only after ordinary arrival-BBO marketability passes",
            "missing_arrival": "an empty/no-fresh arrival tape emits auditable cancel/no-fill rather than raising or substituting a stale quote; no sentinel is recorded as market data or label value",
            "scenario_mapping": {
                "outcome": "FULL for every research intent; actual marketability still decides fill versus no-fill/cancel",
                "submit_latency_ms": "exact nonnegative registered arrival offset: next-completed-minute boundary minus BUY decision boundary, frozen SELL delay rung for ordinary exits, and 0 at terminal",
                "acknowledge_latency_ms": 0,
                "cancel_after_ms": 0,
                "cancel_confirm_latency_ms": 0,
                "quote_tape": "exact selected identity has one quote at the registered arrival offset, or an empty tape when arrival is missing/non-actionable; never include a later quote",
                "synchronous_priority_mapping": "at each ordinary SELL decision boundary, evaluate feed-loss before constructing the intent; if fresh, construct, authorize, and submit the intent exactly once and treat its synchronous registered-arrival transcript as atomic before the next safety decision; if the scheduled arrival is at or after exact 15:55, construct no ordinary SELL and advance to the higher-priority terminal branch instead; after a no-fill at an earlier arrival, re-evaluate feed-loss before any retry; thus every constructed headline BUY/SELL reaches SimulatedExecutor and no ordinary fill/event can occur at or after 15:55",
                "transcript": "SUBMIT_AUTHORIZED occurs at decision; SUBMITTED, ACKNOWLEDGED, and WORKING occur at registered arrival; a marketable quote yields one FILLED event, otherwise CANCEL_REQUESTED then CANCEL_CONFIRMED at the same arrival clock; no-fill opens no BUY position and leaves a SELL position held for next-second redecision",
            },
            "governor": "DeterministicGovernor.evaluate authorizes every manually constructed entry/ordinary-exit research intent; never call its legacy lifecycle_exit_intent/upward-floor policy; an actionable 15:55 close is created only by the new governor-owned explicit forced_flat_intent factory",
            "terminal_intent": "with a fresh actionable 15:55 BBO, DeterministicGovernor.forced_flat_intent truthfully creates RISK_GOVERNOR/FORCED_FLAT using DATABENTO_OPRA, the actual quote/watermark, and its own producer, then evaluate and simulate; with an invalid boundary, use the non-order TERMINAL_ZERO_WRITE_DOWN event and never label derived zero as a vendor quote",
            "one_executor_per_intent": True,
            "quote_offsets": "rebased to that intent's decision time",
            "transcript_quantity": "maximum cumulative filled_quantity, never sum",
        },
        "execution_intent_and_governor_mapping": {
            "clock_units": "all semantic clocks are exact UTC RFC3339 derived from registered nanosecond boundaries; never wall-clock now",
            "buy_clocks": "for BUY decision boundary d and registered next-minute arrival a: event_interval_end=the selected processed minute quote's exact source represented-interval end (never d unless equal); option watermark=that quote's exact received timestamp; SPX watermark=the official causal context watermark; decision_available_at=model_started=model_finished=intent_emitted=d; valid_until=a; require event end and both watermarks<=d<=a; governor evaluate now=d, so the 2s intent-age guard applies at decision and valid_until explicitly covers arrival",
            "ordinary_sell_clocks": "for SELL decision boundary d and arrival a=d+delay: event_interval_end=the selected raw CBBO represented-interval end, exactly ts_recv under [ts_recv-1s,ts_recv); option watermark=that exact actionable BBO ts_recv; SPX watermark=latest official context watermark; decision_available_at=model_started=model_finished=intent_emitted=d; valid_until=a; require watermarks<=d<=a; evaluate governor at d",
            "forced_flat_clocks": "for actionable forced-flat boundary d=a: event_interval_end=the actual selected raw BBO ts_recv (represented interval end), option watermark=that same actual ts_recv<=d, SPX watermark=latest official context<=d, decision_available_at=model_started=model_finished=intent_emitted=valid_until=d; evaluate at d",
            "intent_identity": "trace ids are audit-only; parent id is null for entry and first exit, or the exact pending predecessor when applicable; every producer/version/reason/action/contract/price/clock/precondition field is deterministic and receipted before submit",
            "broker_snapshot": {
                "captured_at": "exact decision boundary d",
                "schema_version": "pathd.broker_state_snapshot.v1",
                "account_id_redacted": "PATHD-TIER-S-OFFLINE-***",
                "snapshot_version": "lowercase SHA-256 of canonical compact sorted-key JSON {policy_id,fee_path,session,decision_time_ns,cash_micros,realized_session_pnl_micros,position_or_null,pending_intent_id_or_null}",
                "available_funds_micros": "max(0, causal policy-ledger cash_micros) before the proposed intent; never starting-equity/default fake funds",
                "daily_pnl_micros": "signed causal realized session PnL including every charged fee/penalty through d",
                "open_positions": "empty iff flat; otherwise exactly one PositionV1 for the held OSI, quantity=1, average_cost_micros=actual submitted-hard-limit BUY fill, opened_at=actual fill boundary",
                "connectivity_and_source": "connectivity=CONNECTED and source=FAKE_GATEWAY for this offline simulation; this is not broker evidence",
                "precondition": "position_snapshot_version equals the deterministic snapshot version; OPEN expects FLAT/null symbol and CLOSE expects LONG_ONE/exact held OSI",
                "account_checks": "upstream active-fee affordability/D48/D49 masks and DeterministicGovernor evaluate must both pass; the snapshot may not be replaced by fake_broker_state defaults",
            },
        },
    }
    return {**law, "fill_law_hash": stable_hash(law)}


def floor_law() -> dict[str, Any]:
    law = {
        "version": "pathd.tier_s.catastrophic_floor.v1",
        "formula": "floor_bid=max(0,0.50*P_entry+3/100)",
        "trigger": "current executable bid <= floor_bid",
        "equivalent": "net PnL after $3 fee <= -50% of entry premium dollars",
        "trailing": False,
        "learned": False,
        "priority": [
            "15:55_forced_flat",
            "feed_loss_safety",
            "catastrophic_floor",
            "learned_exit",
            "hold",
        ],
        "intent_origin": "DETERMINISTIC_EXIT",
        "intent_urgency": "PROTECTIVE_EXIT",
        "ordinary_quote_gap": "before the frozen feed-loss trigger, no floor/learned intent on absent/stale/locked/crossed/zero-bid seconds; remain held until a fresh actionable quote, feed-loss zero write-down, or exact 15:55 handling",
    }
    return {**law, "floor_law_hash": stable_hash(law)}


def foundation_correction_spec() -> dict[str, Any]:
    """Return the exact pre-fit P1/P2/P3 correction authority and boundaries."""

    corrected_v3_authorization = validated_corrected_v3_owner_authorization()
    authority_paths = (
        CORRECTION_PROPOSAL_PATH,
        FIELD_PARITY_FINDING_PATH,
        ENTRY_MICROSTRUCTURE_FINDING_PATH,
        TERMINAL_RULE_RECONCILIATION_PATH,
        CORRECTED_V3_OWNER_AUTHORIZATION_PATH,
    )
    authority = [
        {
            "path": repo_path_label(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in authority_paths
    ]
    return {
        "schema_version": "pathd.entry_exit.foundation_correction.v3",
        "status": "CORRECTED_V3_BUILD_ONLY_PENDING_CLAUDE_VERIFICATION",
        "authority": authority,
        "superseded_audit_root": repo_path_label(SUPERSEDED_AUDIT_ROOT),
        "intermediate_corrected_audit_root": repo_path_label(
            INTERMEDIATE_CORRECTED_AUDIT_ROOT
        ),
        "intermediate_corrected_foundation_files": [
            dict(row) for row in INTERMEDIATE_CORRECTED_FOUNDATION_FILE_BINDINGS
        ],
        "previous_corrected_v2_audit_root": repo_path_label(
            PREVIOUS_CORRECTED_V2_AUDIT_ROOT
        ),
        "previous_corrected_v2_foundation_files": [
            dict(row) for row in PREVIOUS_CORRECTED_V2_FOUNDATION_FILE_BINDINGS
        ],
        "previous_corrected_v2_foundation_generation_sha256": (
            PREVIOUS_CORRECTED_V2_FOUNDATION_GENERATION_SHA256
        ),
        "corrected_audit_root": repo_path_label(AUDIT_ROOT),
        "p1_foundation_restoration": {
            "classification": "BENIGN_TEST_CONTAMINATION",
            "scientific_data_changed": False,
            "root_cause": (
                "an exact typed synthetic test authorization reached the real evidence "
                "validator before process-local capability validation; its fixed fixture "
                "transaction id wrote a zero-access burn into the then-active fold-1 path"
            ),
            "superseded_foundation_files": [
                dict(row) for row in SUPERSEDED_FOUNDATION_FILE_BINDINGS
            ],
            "quarantined_burn": dict(QUARANTINED_TEST_BURN_BINDING),
            "restored_outer_folds": [1, 2, 3, 4, 5],
            "restoration_rule": (
                "the corrected preregistration and restoration receipt define a distinct "
                "generation; the quarantined v1 burn remains immutable history"
            ),
            "acceptance_rule_unchanged": "pooled AND at least 4 of 5 chronological outer folds",
        },
        "p2_live_twin_correction": {
            "last_causal_open_interest": "DROP_FROM_INTRADAY_EXIT_FEATURE_SET",
            "last_causal_minute_volume": (
                "DROP_UNTIL_A_SHARED_HISTORICAL_LIVE_OHLCV_OR_TRADES_ADAPTER_PROVES_"
                "EXACT_COMPLETED_BAR_SPARSE_MINUTE_CONTRACT_AND_CARRY_SEMANTICS"
            ),
            "entry_option_ladder_historical_source": "DATABENTO_OPRA_CBBO_1M",
            "entry_option_ladder_live_twin": (
                "DIRECT_COMPLETED_CBBO_1M_OR_EXACT_CBBO_1S_OR_CMBP_1_TO_CBBO_1M"
            ),
            "corrected_exit_feature_count": 47,
            "field_parity_inventory_is_ground_truth": True,
            "feature_without_live_twin_must_fail_closed": True,
            "integration_owner": "exit_feature_spec_and_feature_lineage_live_twin_correction",
            "stability_seal_requires_corrected_lineage": True,
        },
        "p3_forward_only_entry_enrichment": {
            "current_entry_baseline": "EXACTLY_SIGNED_17_UNCHANGED",
            "candidate": "size_imbalance",
            "optional_companions": ["bid_size", "ask_size"],
            "current_run_alpha_allowed": False,
            "future_test_rule": "fresh separately preregistered widen-entry experiment only",
        },
        "adopted_owner_decisions": {
            "authorization_schema_version": corrected_v3_authorization[
                "schema_version"
            ],
            "authorization_sha256": CORRECTED_V3_OWNER_AUTHORIZATION_SHA256,
            "AREF_ACTION_CALIBRATION_ROLE": {
                "decision": corrected_v3_authorization[
                    "aref_action_calibration_role"
                ]["decision"],
                "topology": aref_decision_critical_topology_spec(),
            },
            "COMPOSITE_CALIBRATION_TERMINAL_RULE": {
                "decision": corrected_v3_authorization[
                    "composite_calibration_terminal_rule"
                ]["decision"],
                "rule": composite_calibration_terminal_rule_spec(),
            },
        },
        "prefit_pause": {
            "terminal_marker": "STOP_FOR_CLAUDE_VERIFICATION",
            "foundation_stability_gate_implemented": True,
            "foundation_stability_receipt_sealed": False,
            "machinery_seal_authorized": False,
            "foundation_stability_seal_authorized": False,
            "model_fit_authorized": False,
            "corpus_decode_authorized": False,
            "nested_or_outer_evidence_open_authorized": False,
            "protected_holdout_open_authorized": False,
            "unresolved_owner_decision": None,
            "resolved_owner_decisions": [
                "AREF_ACTION_CALIBRATION_ROLE",
                "COMPOSITE_CALIBRATION_TERMINAL_RULE",
            ],
            "claude_verification_pending": True,
            "separate_post_verification_release_required": True,
        },
        "holdout_open_count": 0,
        "model_fit_executed": False,
    }


def assert_correction_prefit_release(payload: Mapping[str, Any]) -> None:
    """Require a superseding owner-resolved correction before machinery/fit."""

    pause = payload.get("foundation_correction", {}).get("prefit_pause", {})
    if (
        pause.get("foundation_stability_gate_implemented") is not True
        or pause.get("unresolved_owner_decision") is not None
        or pause.get("machinery_seal_authorized") is not True
        or pause.get("foundation_stability_seal_authorized") is not True
        or pause.get("model_fit_authorized") is not True
        or pause.get("claude_verification_pending") is not False
        or pause.get("separate_post_verification_release_required") is not False
    ):
        raise RuntimeError(
            "STOP_FOR_CLAUDE_VERIFICATION: corrected-v3 decisions are resolved, but "
            "the build-only authorization does not release machinery, stability, "
            "corpus decode, evidence opening, or model fitting"
        )


def _source_hashes() -> dict[str, str]:
    # Immutable roots are mandatory at freeze.  Predeclared implementation roots are
    # included iff already present; missing future roots are bound by their owner receipt.
    labels = tuple(
        dict.fromkeys(
            (
                *IMMUTABLE_SOURCE_PATHS_AT_FIT,
                *REQUIRED_ENTRY_FIT_IMPLEMENTATION_PATHS,
                "v4/scripts/run_pathd_exit_label_smoke.py",
                "v4/scripts/run_pathd_exit_backtest_smoke.py",
            )
        )
    )
    hashes: dict[str, str] = {}
    for label in labels:
        path = REPO_ROOT / label
        if path.exists():
            hashes[label] = sha256_path(path)
        elif label in IMMUTABLE_SOURCE_PATHS_AT_FIT:
            raise FileNotFoundError(f"immutable preregistration source is absent: {label}")
    return hashes


def preregistration_payload() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    assignments = session_assignments()
    lineage = feature_lineage()
    owner_authorization = validated_owner_authorization()
    corrected_v3_authorization = validated_corrected_v3_owner_authorization()
    source_hashes = _source_hashes()
    integrity_contract = integrity_manifest_contract()
    hgb = HGBConfig(
        seed=101,
        max_iter=100,
        learning_rate=0.05,
        max_depth=3,
        l2_regularization=1.0,
        max_examples=250_000,
        min_samples_leaf=30,
    )
    exit_hgb = HGBConfig(
        seed=301,
        max_iter=100,
        learning_rate=0.05,
        max_depth=3,
        l2_regularization=1.0,
        max_examples=500_000,
        min_samples_leaf=50,
    )
    calibration_statistics = calibration_and_statistics_spec()
    payload = {
        "schema_version": "pathd.entry_exit.preregistration.v2",
        "status": "FROZEN_BEFORE_ANY_MODEL_FIT",
        "semantic_contract_excludes_wall_clock": True,
        "quarantine_labels": list(QUARANTINE_LABELS),
        "claim_boundary": CLAIM_BOUNDARY,
        "holdout_caveat": HOLDOUT_CAVEAT,
        "result_envelope": {
            "scope": "standalone or enveloped scientific findings only, under the exhaustive future-schema and fixed-path classification; control/provenance artifacts are not falsely called research results",
            "artifact_classification": result_artifact_classification(),
            "required_fields_every_result": [
                "quarantine_labels",
                "claim_boundary",
                "holdout_caveat",
                "plan_sha256",
                "preregistration_sha256",
                "fill_law_hash",
                "holdout_open_count",
            ],
            "validator": "v4.research.pathd_entry_exit::validate_research_result_envelope",
            "writer_rule": "construct and validate a full envelope for every enveloped scientific result; standalone scientific schemas carry the exact caveat and are written only through their fixed fully metadata-bound terminal writer/receipt; embedded schemas never appear bare",
            "missing_field": "invalid_result",
        },
        "verdict_vocabulary": {
            "allowed_feasibility_verdicts": [
                "PASS",
                "no_genuine_signal",
                "insufficient_evidence",
                "owner_decision_required",
            ],
            "out_of_band_terminal_statuses": ["invalid_result"],
            "status_precedence": [
                "invalid_result",
                "insufficient_evidence",
                "owner_decision_required",
                "no_genuine_signal",
                "PASS",
            ],
            "entry_failure_mapping": {
                "adequate_power_but_signal_or_action_gate_fails": {
                    "feasibility_verdict": "no_genuine_signal",
                    "stop_reason": "no_genuine_entry_signal",
                },
                "minimum_power_misses": {
                    "feasibility_verdict": "insufficient_evidence",
                    "stop_reason": "insufficient_entry_evidence",
                    "reason": "distinguishes an underpowered result from an adequately powered genuine no-signal result",
                },
            },
            "exit_and_composite_failure_mapping": {
                "INVALID_TARGET_COVERAGE": {
                    "terminal_status": "invalid_result",
                    "success_receipt_allowed": False,
                },
                "INSUFFICIENT_EVIDENCE": {
                    "feasibility_verdict": "insufficient_evidence",
                    "stop_reason": "insufficient_exit_evidence",
                },
                "ADEQUATELY_POWERED_SIGNAL_OR_ACTION_GATE_FAILURE": {
                    "feasibility_verdict": "no_genuine_signal",
                    "stop_reason": "no_genuine_exit_or_combined_signal",
                },
            },
            "terminal_marker": "STOP_FOR_CLAUDE_VERIFICATION",
        },
        "binding_plan": {
            "path": str(PLAN_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_path(PLAN_PATH),
        },
        "owner_authorization": owner_authorization,
        "corrected_v3_owner_authorization": corrected_v3_authorization,
        "foundation_correction": foundation_correction_spec(),
        "composite_calibration_terminal_rule": composite_calibration_terminal_rule_spec(),
        "corpus": {
            "root": str(CORPUS_ROOT),
            "entry_substrate": "aligned/processed/minute_entry",
            "entry_label_substrate": [
                "aligned/normalized/databento_spxw_0dte_YYYY-MM-DD.parquet",
                "aligned/normalized/*_official_context.parquet",
                "raw/databento/opra_spxw_cbbo_1s for shared-law exit-until-filled marks",
            ],
            "exit_substrate": "raw/databento/opra_spxw_cbbo_1s",
            "existing_exit_label_audit": "aligned/exit_labels/pathd_exit_labels_12m.parquet",
            "integrity_manifest": "v4/audit/autoresearch/protocol101_pathd_data_acquisition/integrity_sha256.jsonl",
            "integrity_manifest_sha256": integrity_contract["manifest_sha256"],
            "integrity_contract": integrity_contract,
            "integrity_audit": "the manifest has exactly 3,708 files and 21,484,792,678 bytes partitioned into 2,766 authoritative core files, 604 post-entry diagnostic files, and 338 unconsumed raw/index/vix cache files; the historical acquisition audit reported zero mismatches",
            "pre_fit_reverification": "after entry machinery is frozen and immediately before the first fit authorization, rehash only the exact 2,766-file CORE_AUTHORITATIVE partition and freeze the fixed core corpus-integrity receipt; every fit/evidence/holdout loader rehashes each requested core file immediately before read. The 604-file diagnostic inventory is separately rehashed only after all five outer results and pooled acceptance are frozen; the 338-file raw/index/vix cache never receives loader authority",
            "loader_path_rule": "manifest membership is insufficient. A core loader may open only an exact CORE_AUTHORITATIVE grammar path under a current core receipt; the post-entry context loader may open only an exact POST_ENTRY_CONTEXT_DIAGNOSTICS grammar path under its dedicated immutable authorization and inventory receipt; raw/index/vix cache is forbidden under every authority. Reject traversal, symlinks, size/hash drift, aliases, and paths outside CORPUS_ROOT",
            "scheduled_short_session_rule": "2025-11-28 and 2025-12-24 cannot satisfy the binding 15:55 target/survival clock and are therefore schema/hash/diagnostic-only: exclude both from every fit, calibration, model-family selection, primary outer economic/action evidence, exit trajectory power, full-fit role, and comparator selection; do not invent a 13:00 target, do not shift fold indices, and retain them in their original outer partitions only as explicitly labeled shortened-session diagnostics",
        },
        "owner_locked_decisions": dict(owner_authorization["decisions"]),
        "sessions_hash": stable_hash(assignments),
        "feature_lineage_hash": stable_hash(lineage),
        "calibration_and_statistics": calibration_statistics,
        "context_diagnostics": context_diagnostics_spec(),
        "build_time_overrides_to_legacy_ft2_10": {
            "authority": "binding Path-D plan plus owner locks",
            "calibration_count": "ceil(0.20*N), not legacy max(20,ceil(0.20*N))",
            "gate_horizons": "h10/h20/h45/h90/remaining_session; h3/h5 diagnostic only",
            "entry_fill": "shared Path-D hard-limit fill law, not legacy actual-arrival-ask improvement",
            "composer": "A6 mean-LCB gate with q10 ranking only; legacy q10 gate is forbidden",
            "legacy_specs_pinned_only_for": [
                "P5-under-cap causal selection semantics",
                "eight SHA-ranked matched-random schedules",
                "constant/sign-reversed/strong-shuffle/ablation identities",
            ],
        },
        "entry": {
            "fit_environment": entry_fit_environment_spec(),
            "global_family_freeze": entry_global_family_spec(),
            "action_space": "WAIT plus exact eligible contract from complete 42-slot ladder",
            "action_mask": {
                "decision_population": "completed minute t, frozen 30-minute context_ready mask true, t < 15:30 ET, flat, not pending, and not daily-stopped; contract affordability is recomputed from the current ladder and current ledger at every such minute",
                "complete_ladder": "all 42 canonical offset/right identities exist and every current quote has finite nonnegative bid, finite positive ask, bid<=ask, source/receipt time<=decision time, and age<=90 seconds; any missing/stale/invalid slot makes every BUY false",
                "per_action_required": [
                    "persisted source-neutral candidate_mask is true",
                    "finite positive current bid and ask with strict bid<ask",
                    "received/source quote time <= decision time and age<=90 seconds",
                    "current ask >= $1.00",
                    "exact SPXW same-day expiry identity",
                    "BUY hard limit ask+valid tick plus the active fee-path reserve is affordable ($3 primary, $4 sensitivity)",
                    "BUY hard limit*100 plus the active $3/$4 fee-path reserve <= floor(5% of this policy's session-start equity)",
                    "realized session loss plus that active-fee-path cost <= floor(5% of this policy's session-start equity)",
                ],
                "future_forbidden": ["arrival quote", "fill success", "label availability", "future path"],
                "feature_missingness": "D-family NaN is retained for native missing handling and does not become an undeclared action filter",
            },
            "serial_state_machine": {
                "states": ["FLAT", "ENTRY_PENDING", "LONG_ONE", "EXIT_PENDING", "DAILY_STOPPED", "FLAT_TERMINAL"],
                "entry_commit": "a legal BUY choice at completed minute t reserves the sole position slot immediately and permits no other action while pending",
                "entry_resolution": "at the next completed minute t+1 inspect only the selected exact identity; a marketable arrival fills one contract at submitted hard limit and charges one entry-side fee",
                "entry_no_fill": "open no position, charge no premium/fee/PnL, substitute no contract, release pending after resolution, and permit the next entry decision no earlier than t+2",
                "exit_no_fill": "remain LONG_ONE/EXIT_PENDING without fee or occupancy release and redecide at the next completed second",
                "reentry_after_sell": "after a filled SELL at second s, next entry may occur only at the first completed minute strictly greater than s, if t<15:30 and every mask passes",
                "temporary_unaffordability": "when no current otherwise physically eligible contract with ask>=1 passes active-fee affordability/D48/D49, emit WAIT/structural-skip for only that completed minute; never permanently close the session because later premium decay may restore affordability",
                "daily_stop": "permanently enter DAILY_STOPPED when signed realized session PnL <= -floor(5% of session-start equity)",
                "terminal": "15:55 boundary consumes any pending exit exactly once and ends flat",
            },
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "gate_horizons_minutes": list(GATE_HORIZONS),
            "diagnostic_only_horizons_minutes": list(DIAGNOSTIC_HORIZONS),
            "target_axes": list(TARGET_AXES),
            "head_targets": list(ENTRY_HEAD_TARGETS),
            "mean_heads": list(ENTRY_HEAD_TARGETS),
            "q10_heads": list(ENTRY_HEAD_TARGETS),
            "label_builder": {
                "artifact_status_at_preregistration": "NOT_BUILT",
                "source_population": "all causally governed exact action pairs in pre-holdout sessions",
                "entry_intent": "BUY ask_t plus valid tick at completed minute t",
                "entry_arrival": "next completed minute for the selected exact identity",
                "entry_fill": "arrival ask <= submitted limit fills at the submitted limit; failed next-minute fill remains a valid time-t intent example with the entire target bundle set to zero",
                "estimand": "expected utility of a legal time-t entry intent including execution risk, never conditional on future fill success",
                "mark_value": "at each registered completed-minute mark u, deterministic SELL-until-filled starts from bid_u-minus-tick, resolves on subsequent completed 1-second quotes at submitted hard limits, and uses the actual zero-delay 15:55 bid-or-zero boundary",
                "finite_window_endpoints": "if BUY fills at completed-minute boundary e, hN uses exactly the N registered one-minute marks e+1 minute through e+N minutes inclusive; e itself is excluded, deadline=e+N minutes, and profit-area has exactly N one-minute terms",
                "remaining_session_endpoints": "use every registered completed-minute mark strictly after fill boundary e through exact 15:55 inclusive; e is excluded and 15:55 is included once; profit-area has exactly floor((15:55-e)/one minute) terms",
                "mfe": "maximum fee-adjusted PnL/return over marks through the horizon deadline",
                "profit_area": "sum max(fee-adjusted PnL/return,0) times one minute through the horizon deadline",
                "full_window": "finite horizon requires every registered minute through deadline; remaining_session requires every minute through exact 15:55",
                "late_entry_horizon_mask": "build finite-horizon labels only when the decision-clock-only composer availability rule says their full nominal window can end by 15:55; otherwise mark that head unavailable before any future read, exclude it from loss/calibration/composition/action reliability, and never shorten/extrapolate it; remaining_session stays required",
                "missing_future_mark": "full-window target invalid when the registered minute decision quote is absent; intervening locked/crossed/stale seconds are non-actionable and redecide without forward fill; exact no-bid boundary value is zero",
                "diagnostic_h3_h5": "built and reported but excluded from composer and selection",
                "legacy_labels": "forbidden; seven stored stop/target/hold policies are not Path-D A6 targets",
                "shared_fill_law_hash": fill_law()["fill_law_hash"],
            },
            "wait_action_calibration_audit": {
                "role": "required action-conditioned diagnostic only; never enters the entry composer, threshold, family selection, or loss",
                "trajectory": "one maximal contiguous flat WAIT episode; anchor is its first WAIT decision; an episode ending in pending/entry/session-close is one distinct trajectory",
                "realized_target": "at the anchor, max(0, maximum over time-t legal exact actions of the realized MFE/profit-area dollar composite using each action's identical decision-clock-available horizon set under the shared no-fill-as-zero label law); if the legal-action set is empty, define the maximum as 0 and retain the WAIT episode",
                "predicted_mean": "max(0, maximum over time-t legal exact actions of the raw calibrated conditional-mean dollar composite using the identical causal horizon set); empty legal-action set is exactly 0",
                "raw_lower_statistic": "at the WAIT anchor, for each legal exact action compose its equal-horizon dollar q10 statistic from the already head-calibrated q10 components using exactly the action's decision-clock-available horizon set; raw_wait_lower=max(0,max over those action statistics), and an empty legal-action set is exactly 0",
                "predicted_q10": "fit a second-stage one-sided conformal correction to raw_wait_lower itself on only that bundle's disjoint calibration WAIT episodes; never treat the uncorrected maximum of per-action q10 values as coverage evidence",
                "conformal_residual": "realized_target minus raw_wait_lower; with S distinct calibration sessions and n_s retained WAIT episodes in session s, each episode weight is 1/(S*n_s); stable-sort by (residual,session,episode_id), combine exact residual ties, and choose the smallest residual whose cumulative weight is >=0.10",
                "calibrated_lower": "max(0,raw_wait_lower+the weighted residual q10 correction); coverage is the indicator realized_target>=calibrated_lower, including empty-legal-set episodes",
                "calibration_power": "require at least 10 distinct calibration sessions and at least 30 retained calibration WAIT episodes; otherwise the WAIT calibrator is invalid and outer action evidence returns insufficient_evidence",
                "episode_identity": "(outer_fold,session,anchor_decision_time_ns,policy_id); duplicates fail and ordering is chronological session, anchor_decision_time_ns, policy_id",
                "interpretation": "missed-upside/opportunity-cost reliability for choosing WAIT, not a claim that the immediate cash value of WAIT differs from zero",
                "minimum_distinct_episodes": 30,
                "one_sided_coverage_minimum": 0.85,
                "monotone_deciles": True,
            },
            "representation": entry_representation_spec(),
            "gate": "both calibrated session-clustered mean-LCB composites in dollars and return strictly >0 after fees; q10 ranks gate-passers only",
            "composer": entry_composer_spec(),
            "hgb": {
                **hgb.to_dict(),
                "family": "sklearn_hist_gradient_boosting",
                "ensemble_seeds": [101, 102, 103],
                "ensemble_aggregation": "median prediction per head",
                "mean_loss": "squared_error",
                "q10_loss": "quantile_tau_0.10",
                "loss_weighting": "for each head after deterministic cap selection, let S be contributing sessions and n_s the selected target-valid rows in session s; assign every row weight 1/(S*n_s), rescale all weights by selected_row_count so their mean is one, and pass that exact sample_weight to sklearn; weights are recomputed per head from model-fit sessions only and never depend on labels beyond preregistered validity",
                "hyperparameter_candidates": 1,
                "seed_count": 3,
                "max_bins": 255,
                "early_stopping": False,
                "sampling": {
                    "canonical_identity": ["session", "decision_time_ns", "source_neutral_contract_id", "target_axis"],
                    "key": "SHA-256 of compact sorted-key JSON {seed,session,decision_time_ns,source_neutral_contract_id,target_axis}; finite integers/strings only",
                    "algorithm": "rank within each session by full digest then canonical identity; take rows round-robin over chronological sessions until max_examples or exhaustion, skipping exhausted sessions",
                    "duplicates": "exact identity duplicates fail",
                    "receipt": "ordered selected identity list hash, per-session counts, population hash, seed, and head",
                },
                "fits_per_weight_bundle": 120,
                "no_outer_feedback": True,
            },
            "neural": {
                "family": "small_gru_set_attention_multitask",
                "ensemble_seeds": [211, 212, 213],
                "ensemble_aggregation": "median prediction per head",
                "input_history": "full 90x42x17 tensor plus masks and current non-alpha geometry",
                "token_projection": "Linear(34,32)+GELU",
                "temporal_encoder": "one-layer GRU hidden=32 shared over actions",
                "set_encoder": "one self-attention block d_model=64 heads=4 feedforward=128",
                "dropout": 0.10,
                "epochs": 20,
                "batch_size_decision_frames": 64,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "optimizer": "AdamW",
                "optimizer_kwargs": {
                    "lr": 0.001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-08,
                    "weight_decay": 0.0001,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": False,
                    "capturable": False,
                    "differentiable": False,
                    "fused": False,
                },
                "gradient_clip_norm": 1.0,
                "gradient_clip_call": "torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,norm_type=2.0,error_if_nonfinite=True,foreach=False)",
                "scheduler": "none",
                "data_loader_workers": 0,
                "mean_loss": "squared_error",
                "q10_loss": "pinball_tau_0.10",
                "loss_weighting": "for each of the 40 heads independently, let S be contributing model-fit sessions and n_s the target-valid frame-action rows in session s; row weight is 1/(S*n_s), fixed before epoch 1; each head loss is the weighted sum divided by that head's fixed total weight, and the multitask loss is the equal mean of all 40 mandatory finite head losses; no action, session, head, or label-magnitude reweighting",
                "early_stopping": False,
                "training_frames": "all model-fit decision frames in canonical order with seed-deterministic epoch shuffle",
                "batch_order": "canonical frame identity is (session,decision_time_ns); at epoch e apply torch.randperm(N, generator=CPU torch.Generator seeded with ensemble_seed+e) to the canonical frame array, without replacement; keep each frame's 42 actions in canonical identity order, retain the final short batch, and use data_loader_workers=0",
                "hyperparameter_candidates": 1,
                "determinism": "fresh process satisfies entry.fit_environment exactly; before module construction call random.seed(seed), numpy.random.seed(seed), torch.manual_seed(seed), torch.set_num_threads(1), torch.set_num_interop_threads(1), torch.use_deterministic_algorithms(True,warn_only=False), torch.set_deterministic_debug_mode(2); CPU tensors/modules only and no MPS/CUDA operation",
                "training_loop": "construct model then AdamW once; call model.train() once before epoch 0 and again at each epoch start; for every canonical batch run optimizer.zero_grad(set_to_none=True), forward, exact masked multitask loss, require finite scalar loss, loss.backward(), exact clip_grad_norm_ call, optimizer.step(); no gradient accumulation, autocast, scaler, closure, compile, scheduler, validation-mode interruption, checkpoint selection, or early stop; after epoch 19 call model.eval(), and every prediction runs under torch.inference_mode() without returning to train mode",
                "exact_module_graph": {
                    "input": "float32 [batch,90,42,34] = 17 standardized values (NaN->0 after mask) concatenated with 17 finite flags",
                    "token": "for each action/time Linear(34,32,bias=True) then GELU(approximate='none'); no token dropout",
                    "temporal": "reshape actions into batch; GRU(input_size=32,hidden_size=32,num_layers=1,bias=True,batch_first=True,dropout=0,bidirectional=False) with an explicit all-zero float32 h0; take the final t=89 output after left-padding/mask values were zeroed",
                    "action_projection": "concatenate final GRU state with [strike_offset_points/50,right_is_call], apply Linear(34,64,bias=True) then GELU(approximate='none')",
                    "attention": "pre-norm block: qkv=LayerNorm(64,eps=1e-5,elementwise_affine=True,bias=True)(x); MultiheadAttention(embed_dim=64,num_heads=4,dropout=0.10,bias=True,add_bias_kv=False,add_zero_attn=False,kdim=None,vdim=None,batch_first=True) with need_weights=False and key_padding_mask=true only for absent current actions; x=x+Dropout(0.10)(attention_output)",
                    "feedforward": "pre-norm y=LayerNorm(64,eps=1e-5,elementwise_affine=True,bias=True)(x); ff=Linear(64,128,bias=True)->GELU(approximate='none')->Dropout(0.10)->Linear(128,64,bias=True); x=x+Dropout(0.10)(ff); zero every absent-action output after the residual block",
                    "output": "one per-action Linear(64,40,bias=True); columns 0..19 are ENTRY_HEAD_TARGETS conditional means in frozen order and 20..39 are their raw q10 heads in the same order; no sigmoid, sorting, projection, pooling, or extra head",
                    "initialization": "construct modules exactly in token, GRU, action projection, norm1, attention, norm2, FFN linear1, FFN linear2, output order after seeding; retain PyTorch 2.5.1 CPU default reset_parameters for every module; no manual reinitialization",
                    "mask_semantics": "historical token finite flags carry missingness; missing values are zero only after fold-fit standardization; key padding uses current exact-contract presence, never target validity/action eligibility; eligibility is applied only after all predictions",
                },
            },
            "family_evaluation": {
                "global_freeze": entry_global_family_spec(),
                "HGB_must_run_and_seal_first_for_exact_scope": True,
                "neural_challenger_is_mandatory_when_machinery_and_power_allow": True,
                "nested_and_outer_scoring": "score and report HGB plus the mandatory neural challenger on their registered evidence scopes for attribution, machinery evidence, and multiplicity only",
                "candidate_use": "only HGB supplies every outer candidate, acceptance result, exit OOF trajectory, four-box learned-entry leg, full-fit entry artifact, and protected-holdout learned-entry leg",
                "neural_can_select_replace_rescue_or_reweight": False,
                "pooled_or_postouter_reselection": False,
                "outer_multiplicity": "both prespecified families and all eight negative-control attempts remain reported",
                "hyperparameter_tuning": "none; one fixed configuration per family",
            },
            "pooled_acceptance": entry_pooled_acceptance_spec(),
            "multitask_loss": {
                "target_transform": "for each of the 20 target axes use every target-valid frame-action row in model-fit sessions; with S contributing sessions and n_s valid rows/session, weight each row 1/(S*n_s); compute the float64 mean by math.fsum in canonical (session,decision_time_ns,contract_id) order and population variance by a second same-order math.fsum of w*(y-mean)^2; sqrt gives std. Any nonfinite input/result, no contributing session, or std<=0 invalidates the complete bundle; subtract/divide in float64 then cast finite transformed targets to float32",
                "shared_transform": "the same axis transform is used for its mean and q10 head and inverted before calibration/composition",
                "branch_weights": {"conditional_means_total": 0.5, "q10_total": 0.5},
                "within_branch": "equal 1/20 per head",
                "masked_minibatch_formula": "before epoch 1 compute each head's target-valid row weights and fixed full-dataset denominator under neural.loss_weighting; in a minibatch, an absent head contributes numerator 0 and is never renormalized away, while a present head contributes sum(w_i*loss_i)/that head's fixed full-dataset denominator; total batch loss is 0.5*(sum 20 mean-head contributions/20)+0.5*(sum 20 q10-head contributions/20); a mandatory head with zero valid rows over the complete fit role invalidates the bundle, but a head absent only in one batch is allowed",
                "missing_targets": "mask target-invalid rows before each head numerator; no per-batch finite-head, action, branch, or session renormalization",
            },
            "controls": {
                "common_control_replay": {
                    "channels": ["hold_to_flat", "fold_specific_shared_transparent_exit"],
                    "control_exit": "the shared channel uses exactly the one fold-specific transparent exit selected on earlier P5 model-fit sessions; no policy-specific exit choice",
                    "ledgers": "independent per policy/control/attempt/fee path/outer fold",
                    "fee_paths": "separate causal $3 and $4 replays",
                    "masks_fill_state": "identical action masks, hard-limit fill law, pending/occupancy/re-entry/daily-stop rules, per-minute affordability recomputation, and account law as the candidate",
                    "missing_or_incomplete": "insufficient_evidence; never silently drop a control or permit candidate acceptance",
                },
                "p5_under_cap": "objective_spec.json P5_VWAP_SIDE_NEAREST_ELIGIBLE_UNDER_CAP_V2, with Path-D fill-law override, on an independent causal ledger",
                "nearest_atm": "at the first action-eligible completed minute at or after 10:00 ET in each session, select P5 VWAP side then the lexicographically nearest legal exact identity; one attempt per session; independent ledger",
                "matched_random": entry_matched_random_spec(),
                "negative_controls": entry_negative_control_spec(),
                "matched_random_seeds": list(MATCHED_RANDOM_SEEDS),
                "strong_shuffled_target_seeds": list(SHUFFLED_TARGET_SEEDS),
                "ablations": [
                    "remove_context_12_keep_D_E",
                    "remove_D_keep_context_E",
                    "remove_E_keep_context_D",
                    "geometry_and_masks_only",
                ],
                "no_screen_arm": {
                    "alpha": 0.0,
                    "semantics": "skip every learned quality-threshold comparison while retaining physical/safety masks, decision-clock horizon availability, both positive mean-LCB gates, q10 ranking, uncertainty/action gates, D48/D49, WAIT, fill law, and serial ledgers",
                    "first_run_status": "mandatory primary arm because this run's exact 20 A6 upside targets do not add the legacy quality-head namespace",
                    "preservation": "evaluate this identical arm for HGB, neural, every feature-family ablation, and every negative control; it may never be dropped, replaced, or inferred from a screened arm",
                },
            },
            "power_and_trajectory_population": {
                "hold_to_flat_replay": "mandatory first entry attribution; at most one fill per session and never used alone to assert the 40/fold threshold",
                "entry_power_lifecycle": "the fold-specific single shared transparent control exit selected from earlier P5 model-fit sessions by entry.control_exit_selection; it is identical for learned, P5, nearest-ATM, and all matched entry schedules",
                "trajectory_generation_lifecycle": "fixed transparent time_300s, frozen before fitting, with re-entry only after an actual filled exit; this supplies potential exit-research OOF positions but not entry acceptance power",
                "reason": "the 28-session outer folds make the hold-to-flat maximum 28 trades; the threshold remains unchanged and the shared-control replay either earns it or returns insufficient_evidence",
                "entry_trade_identity": "one unique filled BUY intent paired with at most one realized SELL under that policy's independent ledger and the shared control exit; intents/no-fills/counterfactual rows do not count",
                "exit_trajectory_identity": "one unique filled OOF BUY intent from the fixed time_300s serial entry-policy replay, whose quote path is then extended for labels through 15:55; rows do not count",
                "required": {"entry_trades_per_fold": 40, "entry_trades_pooled": 200, "oof_trajectories_per_fold": 50, "oof_trajectories_pooled": 300},
                "miss": "insufficient_evidence; thresholds are never lowered",
            },
            "acceptance": {
                "candidate_identity": "the globally preregistered HGB family in all five outer folds; fold-specific HGB weights/calibrators use only earlier roles, but neither nested nor outer outcomes can select a family",
                "hold_to_flat_channel": "candidate paired session PnL must be >P5 and >the fixed eight-schedule matched-random aggregate pooled and in >=4/5 folds versus each; required even though its one-trade/session structure is not the power denominator",
                "shared_transparent_channel": "within each fold use exactly the one earlier-selected shared control exit for candidate, P5, nearest-ATM, constant/sign/time-shifted/shuffle controls, and every policy-specific matched-entry schedule; candidate paired PnL must exceed P5 and the aggregate matched-random control pooled and in >=4/5 folds versus each",
                "power_denominator": "candidate completed trades under the fold-specific shared transparent exit; if it misses 40/fold, return insufficient_evidence without lifecycle substitution",
                "trajectory_generator": "time_300s remains a separate prefit-frozen neutral OOF-entry trajectory generator for potential exit research; it does not replace the acceptance power denominator",
                "action_gate": "complete ENTER/WAIT calibration gate from metrics_and_gates.action_calibration",
                "negative_controls": "constant, sign-reversed, time-shifted-features, and each of eight strong-shuffle policies must not satisfy the complete economic+action acceptance gate; any accepted negative invalidates entry signal",
                "minimum_power": {"per_fold_completed_trades": 40, "pooled_completed_trades": 200},
                "statistical_point_rule": "strict paired pooled and fold deltas are binding per section 4.4; one-sided 95% session-bootstrap bounds are reported but do not add an unreviewed entry-only threshold",
                "$4_sensitivity": "repeat all policies on separate $4 ledgers under the same selected exit; candidate-minus-P5 and candidate-minus-random aggregate must remain pooled >0, otherwise owner_decision_required",
                "failure": "stop before exit; apply verdict_vocabulary.entry_failure_mapping: insufficient_entry_evidence for minimum-power misses, otherwise no_genuine_entry_signal for adequately powered signal/action failure",
            },
            "control_exit_selection": {
                "eligible_panel": ["exit_immediate", "hold_to_flat", "time_60s", "time_300s", "time_900s", "all_18_fixed_stop_target_time_combinations", "legacy_P5_lifecycle_when_valid"],
                "excluded": ["learned_exit", "matched-rate-random exit schedules", "oracle/future-best exits"],
                "per_outer_fold_data": "P5-under-cap replay on that fold's model_fit sessions only; no calibration/outer/future/holdout results",
                "account_law": "for each eligible comparator start an independent $10,000 account on the first chronological model_fit session, carry equity across every model_fit session, retain zero-trade sessions, require flat close daily, and never reset between sessions; all panel members receive identical initialization but mutate only their own ledger",
                "selection_metric": "maximum total $3-path net PnL over the eligible panel with the common fixed floor on; require >=25 valid model-fit sessions, otherwise insufficient_evidence",
                "tie_break": "lexicographically smallest canonical comparator id",
                "freeze_time": "before scoring that fold's outer sessions",
                "application": "the selected one exit is applied unchanged to every entry policy/control on independent ledgers in that fold",
                "full_fit_holdout_choice": "after all outer work is frozen, select one exit by maximum pooled P5 OOF $3 net PnL across the same panel; tie lexicographically; freeze before holdout open",
                "lifecycle_interaction": "selection is a preregistered earlier-session control choice, not a candidate-specific ex-post lifecycle; Box A and Box B therefore share the exact exit within fold",
            },
        },
        "exit": {
            "cadence_seconds": 1,
            "feature_contract": exit_feature_spec(),
            "reference": "frozen_hold_to_15:55",
            "target": "A_ref(t)=H_t-E_t",
            "oracle_fields": "audit_only_forbidden_from_features_targets_actions_thresholds_selection",
            "local_horizon_seconds": 300,
            "downside_penalty": 0.25,
            "composer": exit_action_composer_spec(),
            "aref_decision_critical_topology": aref_decision_critical_topology_spec(),
            "hgb": {
                **exit_hgb.to_dict(),
                "family": "sklearn_hgb_monotone_latent_stack",
                "library": "scikit-learn 1.5.2 HistGradientBoostingRegressor",
                "ensemble_seeds": [301, 302, 303],
                "ensemble_aggregation": "median prediction per head",
                "mean_loss": "squared_error",
                "quantile_loss": "independent frozen-order A_ref q10, q50-location, and q90-location HGB estimators use sklearn quantile loss at 0.10/0.50/0.90 respectively",
                "monotone_parameterization": "q10=r10_frozen; gap50=r50-q10; q50=q10+softplus(gap50); gap90=r90-q50; q90=q50+softplus(gap90), with stable softplus(x)=log1p(exp(-abs(x)))+max(x,0); support estimators never mutate the frozen q10 estimator",
                "posthoc_sort_or_projection": False,
                "hyperparameter_candidates": 1,
                "seed_count": 3,
                "early_stopping": False,
                "max_bins": 255,
                "warm_start": False,
                "exact_topology": aref_decision_critical_topology_spec()["hgb"],
                "implementation_gate": "corrected-v3 exact constructor, frozen-anchor formula, monotonicity, scaler identity, serialization, seed, and library-version tests must pass before exit fit",
            },
            "neural": {
                "family": "small_causal_temporal_conv_mlp",
                "ensemble_seeds": [311, 312, 313],
                "ensemble_aggregation": "median prediction per head",
                "input_history": "full 301-second causal feature tensor t-300 through t plus masks",
                "preprocessing": "per exit feature, fit session/trajectory-weighted mean and std on exit model-weight rows only; zero/nonfinite std fails; create finite mask, standardize finite values, map NaN to zero, concatenate masks",
                "early_history": "left-pad before 300 seconds with NaN/value-zero-after-transform and mask false; never cross entry time, trajectory, session, or contract identity",
                "temporal_conv_channels": [32, 32, 32],
                "temporal_conv_kernel": 3,
                "temporal_conv_dilations": [1, 4, 16],
                "temporal_padding": "left_causal_only",
                "time_pool": ["masked_mean", "masked_max", "current"],
                "hidden_sizes": [128, 64],
                "dropout": 0.10,
                "epochs": 20,
                "batch_size": 4096,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "optimizer": "AdamW(beta1=0.9,beta2=0.999,eps=1e-8)",
                "losses": "A_REF_ACTION_CORE uses exactly 0.5*MSE(mean)+0.5*pinball_tau_0.10(q10); independent A_REF_Q50_SUPPORT uses pinball_tau_0.50(q50) with q10 detached; independent A_REF_Q90_SUPPORT uses pinball_tau_0.90(q90) with q10/q50 detached; four independent local-path triplet modules use equal one-third pinball losses",
                "gradient_clip_norm": 1.0,
                "scheduler": "none",
                "data_loader_workers": 0,
                "batch_order": "start each epoch from canonical selected-row identity order, then apply torch.randperm with a per-model torch.Generator seeded by ensemble_seed+epoch; no replacement; final short batch retained",
                "early_stopping": False,
                "max_session_balanced_training_rows": 500_000,
                "maximum_gradient_updates": 4_000,
                "stopping_rule": "stop after the earlier of completing epoch 20 or applying exactly update 4,000; if update 4,000 occurs mid-epoch, stop immediately after that optimizer step; no validation/metric stopping or checkpoint choice",
                "hyperparameter_candidates": 1,
                "determinism": "seed Python/NumPy/Torch; deterministic algorithms; record device and library versions",
                "monotone_parameterization": "per target family emit z10,gap50,gap90; q10=z10; q50=q10+softplus(gap50); q90=q50+softplus(gap90) before loss and inference",
                "posthoc_sort_or_projection": False,
                "exact_topology": aref_decision_critical_topology_spec()["neural"],
            },
            "distributional_targets": {
                "A_ref_action_inputs": ["mean", "q10"],
                "A_ref_decision_critical_calibration_support": ["q10", "q50", "q90"],
                "A_ref_complete_required_outputs": ["mean", "q10", "q50", "q90"],
                "diagnostic_only_families": [
                    "downside_300",
                    "recovery_300",
                    "giveback_300",
                    "remaining_tail_300",
                ],
                "downside_300": ["q10", "q50", "q90"],
                "recovery_300": ["q10", "q50", "q90"],
                "giveback_300": ["q10", "q50", "q90"],
                "remaining_tail_300": ["q10", "q50", "q90"],
            },
            "diagnostic_target_isolation": exit_diagnostic_model_isolation_spec(),
            "matched_rate_random": exit_matched_random_spec(),
            "training_sampling": {
                "unit_weights": "each model-fit session total weight 1/S; each eligible OOF entry trajectory within session 1/T_s; each finite second row within trajectory 1/n_trajectory",
                "HGB_cap_rows": 500_000,
                "canonical_row_identity": ["session", "trajectory_id", "decision_time_ns", "source_neutral_contract_id", "target_family"],
                "key": "SHA-256 compact sorted-key JSON over {seed,session,trajectory_id,decision_time_ns,source_neutral_contract_id,target_family}",
                "cap_algorithm": "rank rows within trajectory by full digest/identity; round-robin first over chronological sessions, then sorted trajectory ids within each session, taking one next row per live trajectory until 500,000 or exhaustion; record population/ordered-selection/per-stratum hashes",
                "neural_rows": "same selected identities and weights as HGB; canonical order before seed-deterministic shuffling",
                "future_or_action_selection": False,
            },
            "negative_controls": {
                "constant": "model-fit weighted marginal A_ref/distributional outputs through the same calibrator/composer/replay",
                "strong_shuffle_seeds": list(EXIT_MATCHED_RANDOM_SEEDS),
                "strong_shuffle": "within the complete exit model-fit row population, stable-sort feature-row identities canonically; independently sort the same target-row identities by SHA-256 compact JSON {seed,session,trajectory_id,decision_time_ns,contract_id}; assign each row's complete 16-head target bundle in that permuted order; row count is identical so no truncation/alignment rule; calibration/test untouched; all eight reported",
                "sign_reversed": "multiply calibrated mean A_ref and every signed A_ref quantile by -1 with q10/q90 role exchange before the frozen composer; masks/labels unchanged",
                "time_shifted_features": "pair the same trajectory's causal features from t-60 seconds (when present) with targets at t; missing early rows remain missing; no future shift",
                "acceptance": "every negative control must fail at least one required signal/economic gate; a control clearing the full gate invalidates the learned-signal claim",
            },
            "oof_skill": {
                "R2": "with S sessions, assign each row weight 1/(S*n_s); ybar=sum(w*y); R2=1-sum(w*(y-pred)^2)/sum(w*(y-ybar)^2)",
                "population": "all finite A_ref_mean OOF rows from eligible OOF-entry trajectories before policy action filtering",
                "tripwire": "R2<=0 or zero/nonfinite denominator is no skill and blocks learned-signal language",
                "MAE": "same session-balanced weights, sum(w*abs(y-pred))",
                "policy_economics": "must also beat exit-immediate, full honest comparator panel, and matched-rate random; R2 alone never passes",
            },
            "action_loss": {
                "target_transform": "one frozen session/trajectory-balanced A_ref affine mean/std payload is independently serialized byte-identically for action, q50-support, and q90-support bundles; no mutable scaler object is shared",
                "weights": {"A_ref_mean": 0.5, "A_ref_q10": 0.5},
                "support_losses": {
                    "A_ref_q50": "pinball_tau_0.50; action q10 anchor detached and frozen",
                    "A_ref_q90": "pinball_tau_0.90; q10/q50 anchors detached and frozen",
                },
                "local_diagnostic_loss_or_gradient_contribution_to_aref": 0.0,
                "missing_targets": "mask only preregistered invalid labels before loss; any missing/nonfinite required A_ref mean/q10/q50/q90 prediction or calibration support invalidates the atomic decision-critical bundle",
                "fit_and_serialization_order": [
                    "A_REF_ACTION_CORE",
                    "A_REF_Q50_SUPPORT",
                    "A_REF_Q90_SUPPORT",
                    "A_REF_JOINT_Q10_Q50_Q90_CALIBRATOR",
                ],
            },
            "family_evaluation": {
                "selectable_candidate": "HGB",
                "selectable_candidates": ["HGB"],
                "mandatory_challenger": "NEURAL",
                "challenger_role": "nonselectable_attribution_multiplicity_only_if_exit_power_gate_allows_any_fit",
                "build_order": "HGB labels, calibration, replay, and machinery must pass for the exact scope before neural exit fitting may begin",
                "candidate_use": "only HGB supplies every learned-exit leg in Boxes C/D, full fit, and protected holdout",
                "neural_can_select_replace_rescue_or_reweight": False,
                "pooled_postouter_or_holdout_reselection": False,
                "no_best_seed_fold_or_head_selection": True,
            },
            "power_count_semantics": {
                "model_fit_sessions": "distinct nested-entry-OOF trajectory sessions assigned to exit weights only; exit calibration sessions and raw entry weight-fit history do not count",
                "known_exit_weight_session_counts_by_fold": [0, 0, 19, 48, 56],
                "known_exit_calibration_session_counts_by_fold": [0, 15, 19, 23, 28],
                "minimum_model_fit_sessions": 60,
                "consequence": "if entry clears, exit fitting must stop insufficient_evidence before weights because every fold has fewer than 60 valid nested-entry-OOF exit-weight sessions; invalid inner bundles are never counted or used",
                "entry_only_verdict": "the immutable entry pooled verdict and receipt remain authoritative and are referenced by hash; the exit-power stop cannot overwrite, relabel, or erase them",
                "combined_terminal": {
                    "status": "insufficient_evidence",
                    "stop_reason": "insufficient_exit_evidence",
                    "exit_fit_executed": False,
                    "four_box_instantiated": False,
                    "holdout_open_count": 0,
                },
                "eligible_trajectory": "distinct filled OOF entry intent with complete exact-contract 1-second path and valid reference labels; seconds are not samples for power",
                "minimum_eligible_trajectories_per_fit_fold": 500,
            },
        },
        "fill_law": fill_law(),
        "floor_law": floor_law(),
        "comparators": {
            "entry": ["P5-under-cap", "nearest-ATM", "matched-random-8"],
            "exit": [
                "exit-immediate",
                "hold-to-flat",
                "time-60s",
                "time-300s",
                "time-900s",
                "stop-target-time-grid:-25/-50:+25/+50/+100:60/300/900",
                "legacy-P5-when-valid",
                "matched-rate-random-8",
            ],
            "execution_and_selection": comparator_replay_spec(),
        },
        "combined_evaluation": {
            "composite_calibration_terminal_rule": composite_calibration_terminal_rule_spec(),
            "four_boxes": {
                "A": "P5-under-cap entry plus the fold-specific shared transparent control exit",
                "B": "frozen learned entry plus that exact same fold-specific shared transparent control exit",
                "C": "P5-under-cap entry plus frozen learned exit",
                "D": "frozen learned entry plus frozen learned exit",
            },
            "identity": "within fold, A/B share the identical control exit; C/D share the identical learned-exit bundle; A/C share P5 entry; B/D share the frozen learned-entry bundle; each box owns an independent causal account",
            "replay_boundary": "every BUY/SELL becomes ExecutionIntentV1, passes immutable DeterministicGovernor.evaluate, and executes via the SimulatedExecutor-derived research adapter; dataframe-direct PnL is diagnostic only",
            "acceptance": {
                "primary": "Box D minus the one globally selected best honest comparator has pooled paired net PnL >0 and a strictly positive one-sided 95% registered session-bootstrap lower bound",
                "folds": "the same paired delta is >0 in >=4/5 chronological outer folds",
                "survival": "every box/comparator/fold has zero overlap, unaffordable fill, D48/D49 breach, quantity breach, duplicate terminal consumption, and nonflat close; any violation invalidates the run",
                "entry_attribution": "Box B must have already passed the complete frozen entry acceptance gate",
                "exit_or_interaction": "either Box C-A has pooled paired delta >0 and is >0 in >=4/5 folds, or interaction I=(D-B)-(C-A)=D-B-C+A has a strictly positive pooled one-sided 95% session-bootstrap lower bound and is >0 in >=4/5 folds; report both regardless",
                "action_gate": "ENTER, WAIT, HOLD, and EXIT each pass the frozen action-conditioned coverage/decile/minimum-trajectory gate",
                "fee4": "on separate causal $4 ledgers Box D net PnL and Box-D-minus-same-comparator paired pooled delta must both remain >0; failure routes owner_decision_required",
                "latency": "at each frozen SELL/exit delay rung 0/1/2/5 seconds, keep every BUY fixed to its next-completed-minute arrival, replay candidate and comparator on separate causal ledgers, and require Box D net PnL and paired delta >0; any sign failure routes owner_decision_required; paired SELL bounds 100/250/500/1000ms are diagnostic-only",
                "floor": "pass the frozen floor-on/off large-win-capture and harvest-ratio rule",
                "pr_ap": "diagnostic only and never a gate/selector",
            },
            "known_run_consequence": "after preserving any immutable entry-only verdict, the frozen exit model-fit session counts [0,0,19,48,56] are all <60, so this run stops insufficient_evidence before exit weights and cannot instantiate C/D, full-exit, pre-holdout, or holdout artifacts",
        },
        "economics": {
            "starting_equity_dollars": 10_000,
            "quantity": 1,
            "max_open_positions": 1,
            "d48_fraction_session_start_equity": 0.05,
            "d49_daily_loss_fraction": 0.05,
            "last_entry_et": "15:29",
            "forced_flat_et": "15:55",
            "fee_primary_dollars": 3.00,
            "fee_sensitivity_dollars": 4.00,
            "serial_account_contract": account_replay_spec(),
        },
        "metrics_and_gates": {
            "primary": "paired one-account net-PnL delta versus best honest comparator",
            "pooled": "strictly positive one-sided 95% session-block-bootstrap lower bound",
            "fold_consistency": "positive paired delta in at least 4 of 5 outer folds",
            "action_calibration": action_calibration_spec(),
            "composite_calibration_terminal_rule": composite_calibration_terminal_rule_spec(),
            "pr_ap": "diagnostic_only",
            "floor_ablation_harvest_collapse_threshold": 0.20,
            "floor_ablation_metrics": {
                "pairing": "run floor-on and floor-off from identical initial accounts, entry/composer artifacts, sessions, fee/delay rung, and random seeds; preserve each ledger's causal divergence",
                "floor_off_big_win_reference_set": "floor-off filled trajectory identities whose final fee-adjusted return on entry premium is >=+25%",
                "large_win_capture": "sum max(0, realized_net_pnl) under the tested floor setting over the floor-off big-win reference identities divided by sum max(0, floor-off realized_net_pnl) on those identities; missing identities after causal divergence contribute zero",
                "harvest_ratio": "sum max(0, realized_net_pnl) over all filled trajectories divided by sum max(0, pathwise executable MFE_net_dollars) over those same trajectory identities",
                "relative_decline": "1-(floor_on_metric/floor_off_metric); >0.20 for either metric routes owner_decision_required",
                "zero_denominator": "owner_decision_required; never impute one or drop the metric",
            },
            "pickles_buckets_return_on_premium": {
                "big_win": ">=0.25",
                "small_win_scratch": "(-0.05,0.25)",
                "small_loss": "(-0.30,-0.05]",
                "big_loss": "<=-0.30",
                "share_denominator": "all completed round-trip trades, one trajectory identity once",
                "big_loss_owner_review": "big_loss_count/completed_trade_count >0.04 routes owner_decision_required",
                "near_zero_big_win": "big_win_count <= max(2,floor(0.01*completed_trade_count)) routes owner_decision_required",
                "zero_trades": "insufficient_evidence",
            },
            "large_improvement_audit": {
                "ratio_trigger": "candidate pooled net PnL > 2*max(best honest comparator pooled net PnL,0); if comparator PnL<=0, any candidate pooled PnL>0 triggers",
                "bootstrap_trigger": "candidate-minus-best-comparator pooled paired-session total delta > 2 times the sample standard deviation (ddof=1) of the 10,000 registered paired whole-session bootstrap total deltas",
                "trigger_logic": "either trigger; evaluate on the frozen $3 headline path before claim language",
                "required_checks": [
                    "entry and exit fill rates by fold/policy/delay rung",
                    "ENTER WAIT HOLD EXIT action rates and retry counts",
                    "fee reserve, side-fee, slippage, and hard-limit reconciliation",
                    "duplicate intent, trade, and trajectory identity audit",
                    "feature/label clock and temporal leakage audit",
                    "protected-holdout access count and receipt audit",
                    "side/moneyness/time/premium/regime/session concentration",
                    "candidate versus comparator exposure/occupancy matching",
                ],
                "missing_or_failed_check": "owner_decision_required and no learned-signal/feasibility claim",
                "not_triggered": "record NOT_TRIGGERED with both trigger inputs; never omit the panel",
            },
            "required_owner_facing_output_schema": {
                "leading_columns_in_order": [
                    "unique_session_count",
                    "distinct_trajectory_count",
                    "one_second_row_count",
                ],
                "required_metrics": [
                    "paired_session_pnl",
                    "total_pnl",
                    "mean_session_pnl",
                    "median_session_pnl",
                    "profit_factor",
                    "maximum_drawdown",
                    "worst_session_pnl",
                    "daily_positive_fraction",
                    "exposure_seconds",
                    "occupancy_fraction",
                    "entry_and_exit_churn",
                    "entry_and_exit_no_fill_rate",
                    "side_concentration",
                    "moneyness_concentration",
                    "time_bucket_concentration",
                    "premium_band_concentration",
                    "regime_concentration",
                    "skipped_opportunity_count_and_value",
                    "action_conditioned_calibration",
                    "equity_curve",
                    "four_bucket_pickles_counts_shares_and_pnl",
                    "minimum_power_ledger",
                    "guard_panel",
                ],
                "missing_metric": "invalid_result; selective reporting is forbidden",
            },
            "minimum_power": {
                "outer_fold_non_degraded_sessions": 25,
                "model_fit_sessions": 60,
                "eligible_exit_trajectories_per_model_fit_fold": 500,
                "filled_oof_entry_trajectories_per_outer_fold": 50,
                "filled_oof_entry_trajectories_pooled": 300,
                "entry_executed_trades_per_fold": 40,
                "entry_executed_trades_pooled": 200,
                "protected_non_degraded_sessions": 25,
                "protected_trades": 50,
            },
            "self_fooling_guards": [
                "exit-immediate and matched-rate-random",
                "session-weighted OOF R2 tripwire",
                "constant/shuffled/sign-reversed/time-shifted controls",
                "mutate-future feature and prediction invariance",
                "large-win automatic audit",
            ],
        },
        "source_hashes_at_freeze": source_hashes,
        "protected_holdout_open": holdout_open_spec(),
        "source_hash_policy": {
            "immutable_and_rehashed_at_every_fit": list(IMMUTABLE_SOURCE_PATHS_AT_FIT),
            "lineage_required_paths": list(LINEAGE_IMPLEMENTATION_PATHS),
            "machinery_required_paths": list(MACHINERY_IMPLEMENTATION_PATHS),
            "authorized_in_repo_dependency_paths": list(
                AUTHORIZED_IN_REPO_DEPENDENCY_PATHS
            ),
            "authorized_runtime_dependency_paths": list(
                AUTHORIZED_RUNTIME_DEPENDENCY_PATHS
            ),
            "dependency_closure_rule": "Every statically resolved v4 Python import, including executed package __init__.py files, must be registered and rehashed through either the immutable freeze or its exact owner receipt. Production roots may import only authorized_runtime_dependency_paths and may never import v4/tests; test artifacts are hash/evidence roots, not runtime dependencies. Production roots use only the exact positive offline import-root allowlist and forbid dynamic/reflection/import escapes, remote schemes/clients, process/network/broker launch, joblib Parallel/delayed, and any n_jobs other than literal None or 1; joblib dump/load remain local-only. Torch loaders use workers=0 and the frozen fit-environment thread/runtime caps. No network, broker, subprocess, parallel worker, dynamic code, or unregistered helper escape is permitted.",
            "extensible_path_owners": {
                **{path: "lineage" for path in LINEAGE_IMPLEMENTATION_PATHS},
                **{path: "machinery" for path in MACHINERY_IMPLEMENTATION_PATHS},
            },
            "required_post_build": [
                {
                    "path": path,
                    "owner": (
                        "lineage" if path in LINEAGE_IMPLEMENTATION_PATHS else "machinery"
                    ),
                    "existed_at_freeze": path in source_hashes,
                    "freeze_sha256": source_hashes.get(path),
                }
                for path in REQUIRED_ENTRY_FIT_IMPLEMENTATION_PATHS
            ],
            "extensible_at_freeze": [
                {
                    "path": path,
                    "owner": (
                        "lineage" if path in LINEAGE_IMPLEMENTATION_PATHS else "machinery"
                    ),
                    "freeze_sha256": source_hashes[path],
                }
                for path in REQUIRED_ENTRY_FIT_IMPLEMENTATION_PATHS
                if path in source_hashes
            ],
            "fit_session_roles": fit_session_role_spec(),
            "evidence_session_roles": evidence_session_role_spec(),
            "entry_future_api_contract": entry_future_api_contract(),
            "entry_dataset_seal_contract": entry_dataset_seal_spec(),
            "outer_evidence_open_gate": outer_evidence_open_gate_spec(),
            "fit_authorization_contract": {
                "type": "v4.research.pathd_entry_exit::FrozenFitAuthorization",
                "gate": "v4.research.pathd_entry_exit::assert_entry_fit_ready",
                "caller_sessions_argument": False,
                "dataset_rule": "fit/calibration wrappers accept no caller dataset; they internally call the verified loader with only a FrozenFitAuthorization, consume exactly authorization.sessions, validate every example session plus per-session source receipt, seal a canonical dataset_sha256, and call assert_fit_authorization_current both immediately before the first corpus read and again immediately before estimator/calibrator implementation",
                "authorization_fields": [
                    "role",
                    "outer_fold",
                    "inner_fold",
                    "sessions",
                    "sessions_sha256_newline",
                    "preregistration_sha256",
                    "session_assignments_sha256",
                    "source_hash_policy_sha256",
                    "corpus_integrity_receipt_sha256",
                    "lineage_receipt_sha256",
                    "machinery_receipt_sha256",
                    "fit_environment_sha256",
                    "foundation_generation_sha256",
                    "foundation_stability_receipt_sha256",
                ],
            },
            "evidence_authorization_contract": {
                "type": "v4.research.pathd_entry_exit::FrozenEvidenceAuthorization",
                "gate": "v4.research.pathd_entry_exit::assert_entry_evidence_ready",
                "roles": evidence_session_role_spec(),
                "primary_rule": "nested_validation and outer_test_primary each use one fixed durable O_EXCL access receipt with count exactly one before the sole dataset decode; authorization embeds that receipt hash and cannot be issued twice. Pooled statistics use only FrozenResultAggregationAuthorization over five immutable result/trace receipts and never load/replay a pooled raw dataset",
                "diagnostic_rule": "outer_test_shortened_diagnostic may open exactly once only after its fold's primary result receipt, through a dedicated preopen/access/dataset/result chain, and is forbidden from every primary result, fit, calibration, power denominator, family/comparator choice, or exit trajectory",
                "foundation_fields": [
                    "foundation_generation_sha256",
                    "foundation_stability_receipt_sha256",
                ],
                "fit_forbidden": "every fit/model/calibrator wrapper requires exact FrozenFitAuthorization type and rejects FrozenEvidenceAuthorization",
                "crash_rule": "any lease loss after an access receipt burns the block unless the normal live terminal receipt is already complete. Recovery never seals durable-but-unsealed dataset/result bytes and can never decode, reopen, recreate authorization, or reset access count",
                "holdout": "protected holdout uses a separate future atomic one-shot authorization only after the complete pre-holdout packet is frozen; no evidence role resolves firewall/holdout dates",
            },
            "entry_fit_api_rule": "every positive-family weight fit must enter only fit_hgb_entry_bundle or fit_neural_entry_bundle with one current outer/nested/full weights authorization; wrappers internally load and seal EntryFitDatasetV1 and accept no caller dataset. Every positive calibrator fit enters only fit_entry_calibrators with its paired earlier weight bundle and exact disjoint calibration authorization. CONSTANT, shuffled-target, and time-shifted controls use their dedicated weight plus disjoint-control-calibration plus control-composer APIs; SIGN_REVERSED alone reuses the positive calibrated outputs and transforms immediately before composition with no fit/recalibration. Every artifact binds authorization, sessions, dataset, family/control, scope, and a closed payload schema",
            "research_execution_api_rule": "headline construction must enter only prepare_research_execution from an exact dataset/current-authorization-validated verified context, an internally composed EntryActionDecisionV1, and a full hash-valid ResearchLedgerJournalV1. PreparedExecution embeds the complete context and journal-root binding; replay revalidates them and internally selects the sole source-backed arrival. Hand-authored intents/contexts/decisions, caller governors/configs, raw bid/ask, fake_broker_state defaults, unsealed tapes, raw replay arguments, and ledger/journal resets are forbidden",
            "research_execution_provenance_contract": {
                "verified_source_row": "read_verified_research_quote_row/read_verified_official_spx_row accept only a current fit/evidence authorization, its exact sealed dataset, session, causal at_or_before clock, and registered age/actionability requirements (plus exact contract for options). They rehash every session-receipted regular non-symlink source file before decode, enumerate all eligible rows, and deterministically select the unique latest by max (available_at_ns,ts_recv_ns,represented_interval_end_ns,source_relative_path,row_group,row_index); the receipt includes proof counts and the selected locator, and any tie on the complete key rejects. No caller path, locator, price, or watermark exists",
                "quote_seal": "seal_research_quote accepts only VerifiedResearchQuoteRowV1 and no caller price/clock/contract arguments; quote_sha256=stable_hash of every ResearchQuoteV1 field except quote_sha256 and preserves source file hash, row locator/hash, source record hash, exact ContractIdentityV1, and receipt. source_vendor=DATABENTO_OPRA; represented_interval_end_ns<=ts_recv_ns<=available_at_ns; actionable iff bid>0, ask>0, strict bid<ask and invalid_reason is null. Invalid-reason priority is MISSING_BID, MISSING_ASK, ZERO_BID, ZERO_ASK, CROSSED_BBO, LOCKED_BBO, STALE_QUOTE, SOURCE_CLOCK_INVALID, and no caller may override it",
                "context_seal": "context_sha256=stable_hash of every ResearchDecisionContextV1 field except context_sha256 and embeds the sealed selected option quote and official-SPX row. The BUY factory accepts only an example contained by the exact sealed dataset/current authorization plus the exact validated EntryActionDecisionV1; contract/quote cannot be caller-selected, decision_time is the example completed minute, and canonical selectors rerun. The SELL factory accepts only the current full validated long journal, fixes decision_time to exactly last_decision_time_ns+1 second (terminal fixes 15:55), and invokes selectors for the held contract. It binds authorization,dataset,example-or-null,both receipts and rows; represented interval<=ts_recv<=available_at<=decision",
                "decision_quote_match": "session, exact contract, source receipt, represented interval end, ts_recv option watermark, and available_at<=decision must match the sealed context; any price/clock/contract/source mismatch rejects before intent construction",
                "composer_dynamic_mask": "EntryModelInputV1 contains signed-17 alpha, geometry, and only the immutable physical/source mask. EntryPredictionV1 self-hashes and binds family, exact model_artifact_sha256, fit authorization, fit dataset, and input. compose_entry_action accepts the exact model_bundle named by the composer (not a caller prediction), revalidates it, constructs the exact model input from the sealed evidence example, and internally invokes predict_entry_bundle before calibration/composition; it also requires the exact evidence dataset/current authorization/full journal/fill law. It internally derives current affordability, D48, D49, pending, occupancy, cutoff, daily-stop and active-fee masks, intersects them, and emits a self-hashed EntryActionDecisionV1. Affordability is recomputed every completed minute and cannot create a permanent soft-close. No caller-authored prediction, mask, contract, price, or altered example is accepted",
                "governor_factory": "prepare_research_execution accepts the exact dataset/current authorization/context/journal but no governor/config. It revalidates the full context and journal chain, then BUY internally constructs research_entry_governor_config; ORDINARY_SELL constructs research_exit_governor_config; TERMINAL_SELL constructs research_terminal_governor_config. Prepared embeds the complete prior journal plus its root, context, and governor-config hash; replay is stateless, revalidates all bindings from Prepared, and may not recover a journal from hidden mutable state",
                "ledger_genesis": "genesis_research_ledger_journal alone creates one account from the exact current evidence authorization. The first transition embeds a derived genesis ledger: authorization/session/account scope and indices are derived, fixed cash=session-start-equity=$10,000, sequence=0, null clock, zero realized PnL, no position/pending, zero prior hash. Journal root commits the full transition list and tip; no caller session, scope, ledger, or starting equity exists",
                "ledger_session_advance": "advance_research_ledger_session alone crosses sessions inside one authorization: require the same current authorization, exact authorization/session/account-scope hashes, flat/no pending, a terminal prior session, and session_index+1 within authorization.sessions; derive the next exact authorization session, increment both session indices, carry exact closing cash as new session_start_equity, reset only realized_session_pnl to zero and last_decision_time to null, increment sequence, and hash-chain to the prior ledger. Starting a second genesis, skipping/reordering a session (including a zero-trade session), or resetting equity is invalid",
                "ledger_authorization_handoff": "handoff_research_ledger_authorization is the only cross-authorization transition and is permitted only for the same outer-fold nested HGB-or-neural chronological account from one calibration-valid inner block to the next calibration-valid inner block. It validates the current block's final session/terminal ledger, every intervening invalid-block skip receipt, the already durably opened next authorization, identical derived account_scope_sha256, and the exact next global account_session_index; it carries cash, resets only session PnL/clock, changes the bound authorization/session hash and local session_index to zero, and hash-chains. No outer-fold, family, fee, policy, or hidden-block skip is allowed",
                "ledger_transition": "every genesis, dataset frame coverage, action decision (including WAIT/HOLD), post-outcome EntryActionObservationV1, execution result, non-order event, session terminal, session advance, and nested authorization handoff appends one ResearchLedgerTransitionV1 containing the complete typed event payload, its schema/hash, the full next ledger, and exact prior/next hashes. Exactly one EntryFrameCoverageV1 is derived from the sealed dataset and current ledger for every ordered dataset example: ACTION_DECISION requires exactly one matching subsequent EntryActionDecisionV1 and observation; STATE_INELIGIBLE requires a causal state-derived reason and forbids a decision. Coverage example hashes must equal the complete dataset receipt order, so omitting an unfavorable eligible frame is invalid. Event payload hashes are recomputed before state replay; a hash-only event is invalid. EntryActionObservationV1 is constructed only from the exact sealed dataset/decision/execution outcome after the causal action is complete and records the exact ENTER/WAIT reliability unit without entering model input. validate_research_ledger_journal replays the complete chain from genesis and rejects any altered cash/PnL/position/pending/session/index/time/authorization/policy/fee/frame-coverage/action/observation/execution/transition. EntryPolicyEvaluationV1 retains each session-terminal journal root including zero-trade sessions, the full terminal journal, a transition-derived execution-trace root, and a coverage hash over exact sessions, PnL, trades, trace and journal roots; its economics/actions/metrics must equal the registered reconstruction from that journal and sealed dataset",
                "arrival_quote_match": "replay accepts no quote rows/tape. It revalidates the prepared dataset+authorization and internally invokes the canonical option selector at exactly arrival_base_time_ns+the registered delay, for the same session/exact contract/receipt/vendor; the selected row must be strictly later than the decision represented interval. Omission, substitution, earlier/later clock, duplicate-key tie, or caller price is impossible/rejected; only that selected row may drive fill",
                "execution_result": "result_sha256=stable_hash of every ResearchExecutionResultV1 field except result_sha256; it binds prepared/full decision context/arrival/source hashes, exact ExecutionEvent transcript, fee/cash/position delta, prior ledger hash, and full resulting ledger but contains no journal (avoiding a hash cycle). ResearchExecutionOutcomeV1 separately seals that result plus the complete result journal whose final transition embeds the cycle-free result payload",
            },
            "receipt_contracts": receipt_contract_spec(),
            "future_exit_only_frozen_seams": [
                "v4/scripts/run_pathd_exit_label_smoke.py",
                "v4/scripts/run_pathd_exit_backtest_smoke.py",
            ],
            "future_exit_seam_rule": "these two frozen source hashes are not consumed or authorized by the entry build/fit gate; if entry clears, a separate preregistered exit machinery gate must rehash them and bind every exit-only extension before any exit fit",
            "foundation_extension_constraints": {
                "simulated.py": "only add ExecutionScenario.fill_price_mode with ARRIVAL_QUOTE default and SUBMITTED_HARD_LIMIT research behavior, plus empty-arrival cancel/no-fill; all existing default transcripts/state transitions must remain byte-equal in tests",
                "governor.py": "only add default-preserving allowed_forced_flat_reference_vendors (default IBKR_SAFETY only), daily_loss_blocks_close (default true), and DeterministicGovernor.forced_flat_intent as an explicit governor-owned factory; the factory requires a fresh actionable supplied BBO, actual reference vendor/watermark, reason_code exactly PATHD_RESEARCH_TERMINAL, and creates its own pathd.deterministic-governor.v1 producer; it uses max_adverse_move_micros=the valid tick at actual reference bid and hard_limit_micros=max(0,reference_bid_micros-tick); it rejects absent/stale/locked/crossed/zero-bid inputs; all existing default decisions/lifecycle behavior remain byte-equal",
                "research_entry_governor_config": "max_quantity=1, max_open_positions=1, max_daily_loss_micros=floor(0.05*session_start_equity_micros), max_feed_age_ms=90000, max_intent_age_ms=2000, daily_loss_blocks_close=true, default forced-flat vendors; D48/D49/active-fee affordability also pass upstream masks",
                "research_exit_governor_config": "same quantity/position/dynamic daily-loss value, max_feed_age_ms=2000, max_intent_age_ms=2000, daily_loss_blocks_close=false so risk exits cannot be blocked by the entry stop; default forced-flat vendors for ordinary exits",
                "research_terminal_governor_config": "exit config plus allowed_forced_flat_reference_vendors=(DATABENTO_OPRA,); only a latest <=2s fresh actionable actual BBO may enter forced_flat_intent/evaluate/simulator, with reference bid/ask from that BBO, max adverse move equal to the bid-based tick, and hard limit max(0,bid-tick); an invalid boundary or feed-loss trigger uses its named non-order zero-write-down accounting event instead",
                "research_forced_flat_reason_codes": list(RESEARCH_FORCED_FLAT_REASON_CODES),
                "research_session_terminal_reason_codes": list(RESEARCH_SESSION_TERMINAL_REASON_CODES),
                "feed_health_mapping": "for an actual quote, option_received_timestamp is its exact ts_recv; for a non-order zero-write-down, retain the last actual option watermark or null in the accounting event and do not construct FeedHealth/Intent; SPX received timestamp is the exact official-context watermark; availability Booleans reflect source truth; never rewrite timestamps to now",
            },
            "entry_fit_rule": "immutable drift always blocks; the lineage and machinery receipt implementation path sets must exactly equal their frozen owner-specific required sets, with canonical repo-relative paths and current hashes; missing, extra, duplicate, moved, symlinked, or unrelated substitutions block",
            "future_path_rule": "a post-preregistration entry implementation file is admissible only when its exact future canonical path and receipt owner are frozen here",
        },
        "prohibited": [
            "live",
            "broker",
            "IBKR",
            "paper",
            "runtime",
            "registry",
            "launchd",
            "Protocol158",
            "Protocol160",
            "parent-authority-edit",
            "governance-freeze",
            "cmbp-1",
            "Tier-T",
            "v4/.env",
        ],
        "holdout_open_count": 0,
    }
    return payload, assignments, lineage


def validate_result_artifact_classification(payload: dict[str, Any]) -> None:
    observed = payload.get("result_envelope", {}).get("artifact_classification")
    expected = result_artifact_classification()
    if observed != expected:
        raise ValueError("result artifact classification drift")
    classes = observed["classes"]
    named_sets = [set(classes[name]) for name in (
        "standalone_scientific_result",
        "embedded_scientific_result",
        "non_result_contract_control_or_evidence_artifact",
    )]
    if any(named_sets[left] & named_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("result artifact classification overlaps")
    if set().union(*named_sets) != set(entry_future_api_contract()["schema_versions"]):
        raise ValueError("result artifact classification is not exhaustive")
    fields = entry_future_api_contract()["dataclass_fields"]
    for name in classes["standalone_scientific_result"]:
        if "holdout_caveat" not in fields.get(name, []):
            raise ValueError(f"standalone scientific result lacks caveat: {name}")


def validate_owner_locked_operational_contract(
    payload: dict[str, Any], lineage: dict[str, Any]
) -> None:
    """Cross-bind every owner lock to the exact executable prereg structures."""

    owner = validated_owner_authorization()
    corrected_v3_owner = validated_corrected_v3_owner_authorization()
    if payload.get("owner_authorization") != owner:
        raise ValueError("owner authorization payload drift")
    if payload.get("corrected_v3_owner_authorization") != corrected_v3_owner:
        raise ValueError("corrected-v3 owner authorization payload drift")
    if payload.get("owner_locked_decisions") != owner["decisions"]:
        raise ValueError("owner decision projection drift")
    if owner["required_holdout_caveat"] != HOLDOUT_CAVEAT:
        raise ValueError("owner holdout caveat drift")
    if owner["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("owner claim boundary drift")
    if owner["terminal_marker"] != payload.get("verdict_vocabulary", {}).get(
        "terminal_marker"
    ):
        raise ValueError("owner terminal marker drift")

    expected_lineage = feature_lineage()
    if lineage != expected_lineage:
        raise ValueError("canonical feature lineage drift")
    entry = payload.get("entry", {})
    if (
        entry.get("feature_names") != list(FEATURE_NAMES)
        or entry.get("feature_count") != 17
        or lineage.get("entry_feature_names_in_exact_order") != list(FEATURE_NAMES)
        or lineage.get("entry_feature_count") != 17
    ):
        raise ValueError("owner signed-17 operational binding drift")
    if entry.get("representation") != entry_representation_spec():
        raise ValueError("entry representation operational binding drift")
    if entry.get("composer") != entry_composer_spec():
        raise ValueError("entry composer operational binding drift")
    if entry.get("global_family_freeze") != entry_global_family_spec():
        raise ValueError("entry global-family freeze drift")
    family_eval = entry.get("family_evaluation", {})
    if family_eval.get("global_freeze") != entry_global_family_spec():
        raise ValueError("entry family evaluation binding drift")
    if entry.get("pooled_acceptance") != entry_pooled_acceptance_spec():
        raise ValueError("entry pooled-acceptance operational binding drift")

    exit_spec = payload.get("exit", {})
    if exit_spec.get("feature_contract") != exit_feature_spec():
        raise ValueError("exit feature operational binding drift")
    horizon = owner["decisions"]["exit_local_horizon_seconds"]
    expected_targets = {
        "A_ref_action_inputs": ["mean", "q10"],
        "A_ref_decision_critical_calibration_support": ["q10", "q50", "q90"],
        "A_ref_complete_required_outputs": ["mean", "q10", "q50", "q90"],
        "diagnostic_only_families": [
            "downside_300",
            "recovery_300",
            "giveback_300",
            "remaining_tail_300",
        ],
        "downside_300": ["q10", "q50", "q90"],
        "recovery_300": ["q10", "q50", "q90"],
        "giveback_300": ["q10", "q50", "q90"],
        "remaining_tail_300": ["q10", "q50", "q90"],
    }
    if (
        horizon != 300
        or exit_spec.get("local_horizon_seconds") != horizon
        or exit_spec.get("feature_contract", {}).get("history_seconds") != horizon
        or exit_spec.get("distributional_targets") != expected_targets
    ):
        raise ValueError("exit H=300 target/history/name binding drift")
    if exit_spec.get("composer") != exit_action_composer_spec():
        raise ValueError("structured exit composer drift")
    aref_topology = aref_decision_critical_topology_spec()
    if (
        exit_spec.get("aref_decision_critical_topology") != aref_topology
        or payload.get("calibration_and_statistics", {}).get(
            "aref_decision_critical_topology"
        )
        != aref_topology
        or any(
            value.startswith("A_ref")
            for value in aref_topology["diagnostic_only_families"]
        )
        or exit_spec["composer"].get("action_inputs")
        != ["A_ref_mean", "A_ref_q10"]
        or exit_spec["composer"].get("transitive_composer_support")
        != ["A_ref_q50"]
        or exit_spec["composer"].get("exit_reliability_only_support")
        != ["A_ref_q90"]
    ):
        raise ValueError("A_ref decision-critical dependency closure drift")
    if (
        exit_spec["composer"]["mean_bound"]["level"]
        != owner["decisions"]["calibration_level"]
        or exit_spec["composer"]["downside_term"]["penalty"]
        != owner["decisions"]["exit_downside_penalty"]
    ):
        raise ValueError("exit calibration/penalty owner lock drift")
    if exit_spec.get("diagnostic_target_isolation") != (
        exit_diagnostic_model_isolation_spec()
    ):
        raise ValueError("exit diagnostic isolation drift")
    exit_family = exit_spec.get("family_evaluation", {})
    if (
        exit_family.get("selectable_candidate") != "HGB"
        or exit_family.get("selectable_candidates") != ["HGB"]
        or exit_family.get("neural_can_select_replace_rescue_or_reweight") is not False
    ):
        raise ValueError("exit global HGB family freeze drift")

    if payload.get("calibration_and_statistics") != calibration_and_statistics_spec():
        raise ValueError("calibration/statistics operational binding drift")
    terminal_rule = composite_calibration_terminal_rule_spec()
    if (
        payload.get("composite_calibration_terminal_rule") != terminal_rule
        or payload.get("metrics_and_gates", {}).get(
            "composite_calibration_terminal_rule"
        )
        != terminal_rule
        or payload.get("combined_evaluation", {}).get(
            "composite_calibration_terminal_rule"
        )
        != terminal_rule
    ):
        raise ValueError("composite calibration B-R cross-binding drift")
    if payload.get("metrics_and_gates", {}).get("action_calibration") != (
        action_calibration_spec()
    ):
        raise ValueError("action calibration operational binding drift")
    if payload.get("floor_law") != floor_law():
        raise ValueError("catastrophic floor operational binding drift")
    replay = payload.get("economics", {}).get("serial_account_contract")
    if replay != account_replay_spec():
        raise ValueError("serial account replay operational binding drift")
    if replay.get("floor_fee_constant") != account_replay_spec()["floor_fee_constant"]:
        raise ValueError("floor fee cross-binding drift")
    if payload.get("context_diagnostics") != context_diagnostics_spec():
        raise ValueError("VIX/ES/VX diagnostics operational binding drift")
    if lineage.get("vix_es_vx_model_alpha") is not False:
        raise ValueError("VIX/ES/VX entered model alpha")


def validate_preregistration_payload(
    payload: dict[str, Any], assignments: dict[str, Any], lineage: dict[str, Any]
) -> None:
    """Fail closed on semantic drift before writing the immutable preregistration."""

    validate_immutable_gate_source()
    validate_fixed_science_gate_source()
    expected_payload, expected_assignments, expected_lineage = preregistration_payload()
    if assignments != expected_assignments:
        raise ValueError("canonical session assignment reconstruction drift")
    if lineage != expected_lineage:
        raise ValueError("canonical feature lineage reconstruction drift")
    # Future implementation files may lawfully appear after preregistration under
    # exact owner receipts; these three freeze-time byte/existence projections are
    # validated separately below. Every scientific/operational field is otherwise
    # required to deep-equal a fresh canonical reconstruction.
    observed_canonical = strict_json_loads(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    expected_canonical = strict_json_loads(
        json.dumps(expected_payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    for key in ("source_hashes_at_freeze",):
        expected_canonical[key] = observed_canonical.get(key)
    for key in ("required_post_build", "extensible_at_freeze"):
        expected_canonical["source_hash_policy"][key] = observed_canonical.get(
            "source_hash_policy", {}
        ).get(key)
    validate_owner_locked_operational_contract(payload, lineage)
    validate_result_artifact_classification(payload)
    if (
        type(payload.get("holdout_open_count")) is not int
        or payload.get("holdout_open_count") != 0
        or type(assignments.get("holdout_open_count")) is not int
        or assignments.get("holdout_open_count") != 0
    ):
        raise ValueError("holdout must be unopened at preregistration")
    if payload.get("holdout_caveat") != HOLDOUT_CAVEAT:
        raise ValueError("mandatory holdout caveat drift")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("claim boundary drift")
    if payload.get("sessions_hash") != stable_hash(assignments):
        raise ValueError("session manifest hash mismatch")
    if payload.get("feature_lineage_hash") != stable_hash(lineage):
        raise ValueError("feature lineage hash mismatch")
    correction = payload.get("foundation_correction")
    if correction != foundation_correction_spec():
        raise ValueError("corrected-generation authority binding drift")
    _assert_superseded_foundation_history()
    if (
        correction.get("previous_corrected_v2_audit_root")
        != repo_path_label(PREVIOUS_CORRECTED_V2_AUDIT_ROOT)
        or correction.get("previous_corrected_v2_foundation_files")
        != [dict(row) for row in PREVIOUS_CORRECTED_V2_FOUNDATION_FILE_BINDINGS]
        or correction.get("previous_corrected_v2_foundation_generation_sha256")
        != PREVIOUS_CORRECTED_V2_FOUNDATION_GENERATION_SHA256
    ):
        raise ValueError("corrected-v2 immutable predecessor binding drift")
    pause = correction.get("prefit_pause", {})
    if (
        pause.get("unresolved_owner_decision") is not None
        or pause.get("resolved_owner_decisions")
        != [
            "AREF_ACTION_CALIBRATION_ROLE",
            "COMPOSITE_CALIBRATION_TERMINAL_RULE",
        ]
        or pause.get("claude_verification_pending") is not True
        or pause.get("machinery_seal_authorized") is not False
        or pause.get("foundation_stability_seal_authorized") is not False
        or pause.get("model_fit_authorized") is not False
        or pause.get("corpus_decode_authorized") is not False
        or pause.get("nested_or_outer_evidence_open_authorized") is not False
        or pause.get("protected_holdout_open_authorized") is not False
    ):
        raise ValueError("corrected-v3 build-only prefit pause drift")
    terminal_rule = payload.get("composite_calibration_terminal_rule")
    if terminal_rule != composite_calibration_terminal_rule_spec():
        raise ValueError("refined B-R terminal rule drift")
    current_exit_stop = terminal_rule["current_run_exit_power_stop"]
    if (
        current_exit_stop["known_exit_weight_session_counts_by_fold"]
        != [0, 0, 19, 48, 56]
        or current_exit_stop["minimum_model_fit_sessions"] != 60
        or current_exit_stop["terminal_status_after_entry_pass"]
        != "insufficient_evidence"
        or current_exit_stop["entry_only_verdict_preserved_by_hash"] is not True
        or current_exit_stop["holdout_open_count"] != 0
    ):
        raise ValueError("refined B-R exit-power stop drift")
    p1 = correction["p1_foundation_restoration"]
    if (
        p1.get("classification") != "BENIGN_TEST_CONTAMINATION"
        or p1.get("scientific_data_changed") is not False
        or p1.get("restored_outer_folds") != [1, 2, 3, 4, 5]
        or p1.get("acceptance_rule_unchanged")
        != "pooled AND at least 4 of 5 chronological outer folds"
    ):
        raise ValueError("P1 corrected-generation restoration drift")
    acceptance = payload.get("entry", {}).get("acceptance", {})
    if not all(
        ">=4/5" in str(acceptance.get(key, ""))
        for key in ("hold_to_flat_channel", "shared_transparent_channel")
    ):
        raise ValueError("corrected generation changed the frozen >=4/5 acceptance")
    exit_features = payload.get("exit", {}).get("feature_contract", {}).get(
        "feature_names_in_exact_order", []
    )
    if (
        exit_features != list(EXIT47_CORRECTED_FEATURE_NAMES)
        or any(
            name in exit_features
            for name in (
                "last_causal_minute_volume",
                "last_causal_open_interest",
            )
        )
    ):
        raise ValueError(
            "P2 correction is not integrated: unproved intraday fields remain in exit alpha"
        )
    if lineage.get("entry_feature_names_in_exact_order") != list(FEATURE_NAMES):
        raise ValueError("signed-17 order drift")
    if lineage.get("training_allowed_at_preregistration") is not False:
        raise ValueError("lineage implementation must not be pre-attested")
    if lineage.get("implementation_receipt_required_before_fit") is not True:
        raise ValueError("lineage implementation receipt requirement missing")
    if len(lineage.get("features", [])) != 17:
        raise ValueError("lineage feature count drift")
    observed_integrity = payload.get("corpus", {}).get("integrity_contract")
    expected_integrity = integrity_manifest_contract()
    if observed_integrity != expected_integrity:
        raise ValueError("corpus integrity contract drift")
    if (
        expected_integrity["file_count"] != 3_708
        or expected_integrity["total_bytes"] != 21_484_792_678
    ):
        raise ValueError("corpus integrity count/byte arithmetic drift")
    expected_partitions = expected_integrity["partitions"]
    exact_partition_totals = {
        CORE_INTEGRITY_PARTITION: (
            2_766,
            21_471_285_394,
            "31e1abb88b9a5539f9445a63c9af650385f8ded207a94021422a53a364a29497",
        ),
        CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION: (
            604,
            8_843_589,
            "11605b5f8cd88957b5b0227ef45ec8380d80cfbfae16ff0f0111c55857d8ae1c",
        ),
        UNCONSUMED_VIX_CACHE_PARTITION: (
            338,
            4_663_695,
            "61bfb0daf39de84879ee31df9d9023d0563b04c41e36196c91580c35155df56f",
        ),
    }
    if list(expected_integrity["partitions_in_order"]) != list(exact_partition_totals):
        raise ValueError("corpus integrity partition order drift")
    for partition_id, expected_values in exact_partition_totals.items():
        partition = expected_partitions[partition_id]
        if (
            partition["file_count"],
            partition["total_bytes"],
            partition["ordered_entries_semantic_sha256"],
        ) != expected_values:
            raise ValueError(f"corpus integrity partition arithmetic drift: {partition_id}")

    fill = dict(payload["fill_law"])
    fill_hash = fill.pop("fill_law_hash")
    if fill_hash != stable_hash(fill):
        raise ValueError("fill-law self hash mismatch")
    floor = dict(payload["floor_law"])
    floor_hash = floor.pop("floor_law_hash")
    if floor_hash != stable_hash(floor):
        raise ValueError("floor-law self hash mismatch")

    protected = set(assignments["protected_holdout_30"])
    burned = set(assignments["burned_smoke_dates"])
    degraded = set(assignments["opra_degraded_diagnostic_only"])
    shortened = set(assignments["scheduled_short_sessions_diagnostic_only"])
    evidence_sessions: set[str] = set()
    exit_weight_counts: list[int] = []
    exit_calibration_counts: list[int] = []
    for fold in assignments["folds"]:
        roles = [
            set(fold["model_fit"]),
            set(fold["calibration_embargo"]),
            set(fold["calibration_last_20_percent"]),
            set(fold["embargo"]),
            set(fold["outer_test"]),
        ]
        if any(protected.intersection(role) or burned.intersection(role) for role in roles):
            raise ValueError("firewall/smoke session entered evidence role")
        if any(degraded.intersection(role) for role in (roles[0], roles[2], roles[4])):
            raise ValueError("OPRA-degraded session entered primary fit/calibration/test")
        if any(shortened.intersection(role) for role in (roles[0], roles[2])):
            raise ValueError("scheduled-short session entered fit/calibration")
        expected_primary_outer = [
            session for session in fold["outer_test"] if session not in shortened
        ]
        if fold.get("outer_test_primary_1555_complete") != expected_primary_outer:
            raise ValueError("scheduled-short primary outer mapping drift")
        if fold.get("outer_test_shortened_diagnostic_only") != [
            session for session in fold["outer_test"] if session in shortened
        ]:
            raise ValueError("scheduled-short diagnostic mapping drift")
        if not (
            max(fold["model_fit"])
            < fold["calibration_embargo"][0]
            < min(fold["calibration_last_20_percent"])
            < fold["embargo"][0]
            < min(fold["outer_test"])
        ):
            raise ValueError("outer/calibration chronology drift")
        for inner in fold["inner_forward_folds"]["scored_forward_folds"]:
            if not (
                max(inner["model_fit"])
                < inner["calibration_embargo"][0]
                < min(inner["calibration"])
                < inner["outer_validation_embargo"][0]
                < min(inner["validation"])
            ):
                raise ValueError("inner chronology drift")
            if inner.get("calibration_minimum_sessions") != 10:
                raise ValueError("inner calibration minimum drift")
            if inner.get("calibration_valid") is not (len(inner["calibration"]) >= 10):
                raise ValueError("inner calibration validity drift")
        inner_rows = fold["inner_forward_folds"]["scored_forward_folds"]
        expected_exit_weights = [
            session
            for inner in inner_rows[:3]
            if inner["calibration_valid"]
            for session in inner["validation"]
        ]
        expected_exit_calibration = (
            list(inner_rows[3]["validation"])
            if inner_rows[3]["calibration_valid"]
            else []
        )
        mapping = fold["exit_training_oof_mapping"]
        if mapping.get("exit_model_weight_sessions") != expected_exit_weights:
            raise ValueError("invalid inner bundle entered exit weights")
        if mapping.get("exit_calibration_sessions") != expected_exit_calibration:
            raise ValueError("invalid inner bundle entered exit calibration")
        if mapping.get("exit_outer_evidence_sessions") != expected_primary_outer:
            raise ValueError("scheduled-short session entered exit outer evidence")
        if mapping.get("weight_session_count") != len(expected_exit_weights):
            raise ValueError("exit weight-session count drift")
        if mapping.get("calibration_session_count") != len(expected_exit_calibration):
            raise ValueError("exit calibration-session count drift")
        exit_weight_counts.append(len(expected_exit_weights))
        exit_calibration_counts.append(len(expected_exit_calibration))
        evidence_sessions.update(fold["outer_test_primary_1555_complete"])
    if len(evidence_sessions) != 138:
        raise ValueError("outer-test session identity drift")

    locked = payload["owner_locked_decisions"]
    expected = {
        "entry_alpha": "exactly_signed_17_exit_microstructure_only",
        "exit_local_horizon_seconds": 300,
        "exit_downside_penalty": 0.25,
        "calibration_level": 0.90,
        "catastrophic_floor": "max(0,0.50*P_entry+3/100)",
        "vix_es_vx": "diagnostics_and_strata_only",
        "holdout_not_pristine_caveat_accepted": True,
    }
    for key, value in expected.items():
        if locked.get(key) != value:
            raise ValueError(f"owner lock drift: {key}")
    if payload["entry"]["hgb"]["ensemble_seeds"] != [101, 102, 103]:
        raise ValueError("entry HGB seed drift")
    if payload["entry"]["neural"]["ensemble_seeds"] != [211, 212, 213]:
        raise ValueError("entry neural seed drift")
    if payload["entry"].get("fit_environment") != entry_fit_environment_spec():
        raise ValueError("entry fit environment contract drift")
    if payload["entry"]["controls"]["strong_shuffled_target_seeds"] != list(
        MATCHED_RANDOM_SEEDS
    ):
        raise ValueError("strong-shuffle seed namespace drift")
    power = payload["exit"]["power_count_semantics"]
    if power.get("known_exit_weight_session_counts_by_fold") != exit_weight_counts:
        raise ValueError("preregistered exit weight counts do not match frozen roles")
    if power.get("known_exit_calibration_session_counts_by_fold") != exit_calibration_counts:
        raise ValueError("preregistered exit calibration counts do not match frozen roles")
    minimum_exit_sessions = int(power.get("minimum_model_fit_sessions", 0))
    if minimum_exit_sessions != 60 or any(
        count >= minimum_exit_sessions for count in exit_weight_counts
    ):
        raise ValueError("exit minimum-power consequence drift")

    source_policy = payload["source_hash_policy"]
    immutable = source_policy.get("immutable_and_rehashed_at_every_fit")
    lineage_paths = source_policy.get("lineage_required_paths")
    machinery_paths = source_policy.get("machinery_required_paths")
    authorized_dependency_paths = source_policy.get("authorized_in_repo_dependency_paths")
    authorized_runtime_paths = source_policy.get("authorized_runtime_dependency_paths")
    owners = source_policy.get("extensible_path_owners")
    if immutable != list(IMMUTABLE_SOURCE_PATHS_AT_FIT):
        raise ValueError("immutable source policy drift")
    if lineage_paths != list(LINEAGE_IMPLEMENTATION_PATHS):
        raise ValueError("lineage implementation path policy drift")
    if machinery_paths != list(MACHINERY_IMPLEMENTATION_PATHS):
        raise ValueError("machinery implementation path policy drift")
    if authorized_dependency_paths != list(AUTHORIZED_IN_REPO_DEPENDENCY_PATHS):
        raise ValueError("authorized dependency path policy drift")
    if authorized_runtime_paths != list(AUTHORIZED_RUNTIME_DEPENDENCY_PATHS):
        raise ValueError("authorized runtime dependency path policy drift")
    if len(authorized_dependency_paths) != len(set(authorized_dependency_paths)):
        raise ValueError("authorized dependency paths are not unique")
    if len(set(lineage_paths + machinery_paths)) != len(lineage_paths + machinery_paths):
        raise ValueError("implementation path ownership is not unique")
    expected_owners = {
        **{path: "lineage" for path in lineage_paths},
        **{path: "machinery" for path in machinery_paths},
    }
    if owners != expected_owners:
        raise ValueError("implementation path owner drift")
    frozen_sources = payload.get("source_hashes_at_freeze", {})
    for relative in immutable:
        if relative not in frozen_sources:
            raise ValueError(f"immutable source hash absent: {relative}")
    required_post_build = source_policy.get("required_post_build")
    if not isinstance(required_post_build, list):
        raise ValueError("required post-build path policy absent")
    expected_required_rows = [
        {
            "path": path,
            "owner": expected_owners[path],
            "existed_at_freeze": path in frozen_sources,
            "freeze_sha256": frozen_sources.get(path),
        }
        for path in lineage_paths + machinery_paths
    ]
    if required_post_build != expected_required_rows:
        raise ValueError("required post-build path policy drift")
    expected_extensible_rows = [
        {
            "path": path,
            "owner": expected_owners[path],
            "freeze_sha256": frozen_sources[path],
        }
        for path in lineage_paths + machinery_paths
        if path in frozen_sources
    ]
    if source_policy.get("extensible_at_freeze") != expected_extensible_rows:
        raise ValueError("extensible-at-freeze policy drift")
    if source_policy.get("fit_session_roles") != fit_session_role_spec():
        raise ValueError("closed fit-session role policy drift")
    if source_policy.get("evidence_session_roles") != evidence_session_role_spec():
        raise ValueError("closed evidence-session role policy drift")
    evidence_contract = source_policy.get("evidence_authorization_contract", {})
    if evidence_contract.get("roles") != evidence_session_role_spec():
        raise ValueError("evidence authorization contract drift")
    if source_policy.get("entry_future_api_contract") != entry_future_api_contract():
        raise ValueError("entry future API contract drift")
    if source_policy.get("entry_dataset_seal_contract") != entry_dataset_seal_spec():
        raise ValueError("entry dataset seal contract drift")
    if source_policy.get("outer_evidence_open_gate") != outer_evidence_open_gate_spec():
        raise ValueError("outer evidence open-gate contract drift")
    if source_policy.get("receipt_contracts") != receipt_contract_spec():
        raise ValueError("receipt/evidence contract drift")
    _verify_authorized_dependency_closure(payload, allow_missing_future=True)
    if observed_canonical != expected_canonical:
        raise ValueError("canonical preregistration semantic reconstruction drift")
    if "frozen_at_utc" in payload:
        raise ValueError("wall-clock timestamp entered semantic preregistration")


def _assert_strict_json_value(value: Any, *, path: str = "$") -> None:
    """Reject NaN, type coercions, non-string keys, and non-JSON containers."""

    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"invalid_result: nonfinite number at {path}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _assert_strict_json_value(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"invalid_result: non-string key at {path}")
            _assert_strict_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"invalid_result: non-JSON type at {path}: {type(value).__name__}")


def _validate_result_envelope_against(
    result: dict[str, Any],
    *,
    prereg_payload: dict[str, Any],
    preregistration_sha256: str,
    expected_holdout_open_count: int,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("invalid_result: result must be an object")
    _assert_strict_json_value(result)
    json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    required = prereg_payload["result_envelope"]["required_fields_every_result"]
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(f"invalid_result: missing envelope fields: {missing}")
    expected = {
        "quarantine_labels": list(QUARANTINE_LABELS),
        "claim_boundary": CLAIM_BOUNDARY,
        "holdout_caveat": HOLDOUT_CAVEAT,
        "plan_sha256": prereg_payload["binding_plan"]["sha256"],
        "preregistration_sha256": preregistration_sha256,
        "fill_law_hash": prereg_payload["fill_law"]["fill_law_hash"],
        "holdout_open_count": expected_holdout_open_count,
    }
    drift = [name for name, value in expected.items() if result.get(name) != value]
    if (
        type(result.get("quarantine_labels")) is not list
        or type(result.get("claim_boundary")) is not str
        or type(result.get("holdout_caveat")) is not str
        or type(result.get("plan_sha256")) is not str
        or type(result.get("preregistration_sha256")) is not str
        or type(result.get("fill_law_hash")) is not str
        or type(result.get("holdout_open_count")) is not int
    ):
        raise ValueError("invalid_result: envelope field type drift")
    if drift:
        raise ValueError(f"invalid_result: envelope field drift: {drift}")
    return result


def validate_research_result_envelope(
    result: dict[str, Any], *, expected_holdout_open_count: int = 0
) -> dict[str, Any]:
    """Fail closed before every Path-D research result write or CLI print."""

    if type(expected_holdout_open_count) is not int or expected_holdout_open_count != 0:
        raise ValueError(
            "invalid_result: count=1 is reserved for the future atomic holdout transaction"
        )
    from v4.research.pathd_holdout_gate import _assert_protected_holdout_unopened

    _assert_protected_holdout_unopened()
    receipt = assert_preregistration_frozen()
    return _validate_result_envelope_against(
        result,
        prereg_payload=read_json(PREREG_PATH),
        preregistration_sha256=receipt["preregistration_sha256"],
        expected_holdout_open_count=expected_holdout_open_count,
    )


def enveloped_research_result(
    fields: dict[str, Any], *, expected_holdout_open_count: int = 0
) -> dict[str, Any]:
    """Construct, then validate, one canonical result envelope."""

    if type(expected_holdout_open_count) is not int or expected_holdout_open_count != 0:
        raise ValueError(
            "invalid_result: count=1 is reserved for the future atomic holdout transaction"
        )
    from v4.research.pathd_holdout_gate import _assert_protected_holdout_unopened

    _assert_protected_holdout_unopened()
    receipt = assert_preregistration_frozen()
    payload = read_json(PREREG_PATH)
    required = set(payload["result_envelope"]["required_fields_every_result"])
    if required.intersection(fields):
        raise ValueError("invalid_result: caller may not override envelope fields")
    result = {
        **fields,
        "quarantine_labels": list(QUARANTINE_LABELS),
        "claim_boundary": CLAIM_BOUNDARY,
        "holdout_caveat": HOLDOUT_CAVEAT,
        "plan_sha256": payload["binding_plan"]["sha256"],
        "preregistration_sha256": receipt["preregistration_sha256"],
        "fill_law_hash": payload["fill_law"]["fill_law_hash"],
        "holdout_open_count": expected_holdout_open_count,
    }
    return validate_research_result_envelope(
        result, expected_holdout_open_count=expected_holdout_open_count
    )


def freeze_preregistration() -> dict[str, Any]:
    """Freeze the complete research preregistration, refusing overwrite/drift."""

    payload, assignments, lineage = preregistration_payload()
    validate_preregistration_payload(payload, assignments, lineage)
    if PREREG_PATH.exists():
        observed = sha256_path(PREREG_PATH)
        recorded = PREREG_HASH_PATH.read_text(encoding="utf-8").strip().split()[0]
        if observed != recorded:
            raise RuntimeError("existing Path-D preregistration hash drift")
        receipt = read_json(FREEZE_RECEIPT_PATH)
        if receipt.get("preregistration_sha256") != observed:
            raise RuntimeError("existing Path-D preregistration receipt mismatch")
        return assert_preregistration_frozen()

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_canonical_json_exclusive(SESSION_PATH, assignments)
    _write_canonical_json_exclusive(LINEAGE_PATH, lineage)
    _write_canonical_json_exclusive(PREREG_PATH, payload)
    prereg_hash = sha256_path(PREREG_PATH)
    _write_text_exclusive(
        PREREG_HASH_PATH, f"{prereg_hash}  {PREREG_PATH.name}\n"
    )
    receipt_contracts = payload["source_hash_policy"]["receipt_contracts"]
    receipt = {
        "schema_version": "pathd.entry_exit.preregistration_freeze_receipt.v4",
        "status": "FROZEN_BEFORE_ANY_MODEL_FIT",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "preregistration_path": receipt_contracts["preregistration_path"],
        "preregistration_sha256": prereg_hash,
        "session_assignments_path": receipt_contracts["session_assignments_path"],
        "session_assignments_sha256": sha256_path(SESSION_PATH),
        "feature_lineage_path": receipt_contracts["feature_lineage_path"],
        "feature_lineage_sha256": sha256_path(LINEAGE_PATH),
        "plan_sha256": payload["binding_plan"]["sha256"],
        "foundation_correction_sha256": stable_hash(
            payload["foundation_correction"]
        ),
        "correction_proposal_path": repo_path_label(CORRECTION_PROPOSAL_PATH),
        "correction_proposal_sha256": sha256_path(CORRECTION_PROPOSAL_PATH),
        "terminal_rule_reconciliation_path": repo_path_label(
            TERMINAL_RULE_RECONCILIATION_PATH
        ),
        "terminal_rule_reconciliation_sha256": (
            TERMINAL_RULE_RECONCILIATION_SHA256
        ),
        "corrected_v3_owner_authorization_path": repo_path_label(
            CORRECTED_V3_OWNER_AUTHORIZATION_PATH
        ),
        "corrected_v3_owner_authorization_sha256": (
            CORRECTED_V3_OWNER_AUTHORIZATION_SHA256
        ),
        "superseded_audit_root": repo_path_label(SUPERSEDED_AUDIT_ROOT),
        "intermediate_corrected_audit_root": repo_path_label(
            INTERMEDIATE_CORRECTED_AUDIT_ROOT
        ),
        "previous_corrected_v2_audit_root": repo_path_label(
            PREVIOUS_CORRECTED_V2_AUDIT_ROOT
        ),
        "previous_corrected_v2_foundation_generation_sha256": (
            PREVIOUS_CORRECTED_V2_FOUNDATION_GENERATION_SHA256
        ),
        "holdout_open_count": 0,
        "holdout_caveat": HOLDOUT_CAVEAT,
        "claim_boundary": CLAIM_BOUNDARY,
        "quarantine_labels": list(QUARANTINE_LABELS),
        "fill_law_hash": payload["fill_law"]["fill_law_hash"],
        "model_fit_executed": False,
        "fit_gate": {
            "function": "v4.research.pathd_entry_exit::assert_entry_fit_ready",
            "lineage_implementation_receipt_required": receipt_contracts[
                "lineage_receipt_path"
            ],
            "entry_machinery_receipt_required": receipt_contracts[
                "machinery_receipt_path"
            ],
            "foundation_restoration_receipt_required": receipt_contracts[
                "foundation_restoration_receipt_path"
            ],
            "foundation_stability_receipt_required": receipt_contracts[
                "foundation_stability_receipt_path"
            ],
        },
    }
    _write_canonical_json_exclusive(FREEZE_RECEIPT_PATH, receipt)
    return receipt


def assert_preregistration_frozen() -> dict[str, Any]:
    try:
        prereg_path = _canonical_repo_regular_file(repo_path_label(PREREG_PATH))
    except RuntimeError as exc:
        raise RuntimeError(
            "Path-D research fitting is blocked until preregistration freeze"
        ) from exc
    payload = read_json(prereg_path)
    try:
        prereg_hash_path = _frozen_contract_file(
            payload,
            contract_key="preregistration_hash_path",
            expected_path=PREREG_HASH_PATH,
        )
        session_path = _frozen_contract_file(
            payload,
            contract_key="session_assignments_path",
            expected_path=SESSION_PATH,
        )
        lineage_path = _frozen_contract_file(
            payload,
            contract_key="feature_lineage_path",
            expected_path=LINEAGE_PATH,
        )
        freeze_receipt_path = _frozen_contract_file(
            payload,
            contract_key="freeze_receipt_path",
            expected_path=FREEZE_RECEIPT_PATH,
        )
        if payload.get("source_hash_policy", {}).get("receipt_contracts", {}).get(
            "preregistration_path"
        ) != repo_path_label(PREREG_PATH):
            raise RuntimeError("fixed preregistration path drift")
    except RuntimeError as exc:
        raise RuntimeError("Path-D research fitting is blocked until preregistration freeze")
    observed = sha256_path(prereg_path)
    recorded = prereg_hash_path.read_text(encoding="utf-8").strip().split()[0]
    receipt = read_json(freeze_receipt_path)
    contracts = payload["source_hash_policy"]["receipt_contracts"]
    expected_receipt_keys = {
        "schema_version",
        "status",
        "frozen_at_utc",
        "preregistration_path",
        "preregistration_sha256",
        "session_assignments_path",
        "session_assignments_sha256",
        "feature_lineage_path",
        "feature_lineage_sha256",
        "plan_sha256",
        "foundation_correction_sha256",
        "correction_proposal_path",
        "correction_proposal_sha256",
        "terminal_rule_reconciliation_path",
        "terminal_rule_reconciliation_sha256",
        "corrected_v3_owner_authorization_path",
        "corrected_v3_owner_authorization_sha256",
        "superseded_audit_root",
        "intermediate_corrected_audit_root",
        "previous_corrected_v2_audit_root",
        "previous_corrected_v2_foundation_generation_sha256",
        "holdout_open_count",
        "holdout_caveat",
        "claim_boundary",
        "quarantine_labels",
        "fill_law_hash",
        "model_fit_executed",
        "fit_gate",
    }
    if (
        set(receipt) != expected_receipt_keys
        or receipt.get("schema_version")
        != "pathd.entry_exit.preregistration_freeze_receipt.v4"
        or receipt.get("status") != "FROZEN_BEFORE_ANY_MODEL_FIT"
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T[^\s]+Z", str(receipt.get("frozen_at_utc", "")))
        is None
        or receipt.get("holdout_open_count") != 0
        or receipt.get("holdout_caveat") != HOLDOUT_CAVEAT
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
        or receipt.get("model_fit_executed") is not False
        or receipt.get("plan_sha256") != payload["binding_plan"]["sha256"]
        or receipt.get("foundation_correction_sha256")
        != stable_hash(payload["foundation_correction"])
        or receipt.get("correction_proposal_path")
        != repo_path_label(CORRECTION_PROPOSAL_PATH)
        or receipt.get("correction_proposal_sha256")
        != sha256_path(CORRECTION_PROPOSAL_PATH)
        or receipt.get("terminal_rule_reconciliation_path")
        != repo_path_label(TERMINAL_RULE_RECONCILIATION_PATH)
        or receipt.get("terminal_rule_reconciliation_sha256")
        != TERMINAL_RULE_RECONCILIATION_SHA256
        or receipt.get("corrected_v3_owner_authorization_path")
        != repo_path_label(CORRECTED_V3_OWNER_AUTHORIZATION_PATH)
        or receipt.get("corrected_v3_owner_authorization_sha256")
        != CORRECTED_V3_OWNER_AUTHORIZATION_SHA256
        or receipt.get("superseded_audit_root")
        != repo_path_label(SUPERSEDED_AUDIT_ROOT)
        or receipt.get("intermediate_corrected_audit_root")
        != repo_path_label(INTERMEDIATE_CORRECTED_AUDIT_ROOT)
        or receipt.get("previous_corrected_v2_audit_root")
        != repo_path_label(PREVIOUS_CORRECTED_V2_AUDIT_ROOT)
        or receipt.get("previous_corrected_v2_foundation_generation_sha256")
        != PREVIOUS_CORRECTED_V2_FOUNDATION_GENERATION_SHA256
    ):
        raise RuntimeError("Path-D preregistration freeze receipt semantic drift")
    try:
        _validate_result_envelope_against(
            receipt,
            prereg_payload=payload,
            preregistration_sha256=observed,
            expected_holdout_open_count=0,
        )
    except ValueError as exc:
        raise RuntimeError("Path-D preregistration result envelope drift") from exc
    if (
        receipt.get("preregistration_path") != contracts["preregistration_path"]
        or receipt.get("session_assignments_path")
        != contracts["session_assignments_path"]
        or receipt.get("feature_lineage_path") != contracts["feature_lineage_path"]
        or receipt.get("fit_gate", {}).get("lineage_implementation_receipt_required")
        != contracts["lineage_receipt_path"]
        or receipt.get("fit_gate", {}).get("entry_machinery_receipt_required")
        != contracts["machinery_receipt_path"]
        or receipt.get("fit_gate", {}).get(
            "foundation_restoration_receipt_required"
        )
        != contracts["foundation_restoration_receipt_path"]
        or receipt.get("fit_gate", {}).get("foundation_stability_receipt_required")
        != contracts["foundation_stability_receipt_path"]
    ):
        raise RuntimeError("Path-D preregistration fixed-artifact identity drift")
    if observed != recorded or receipt.get("preregistration_sha256") != observed:
        raise RuntimeError("Path-D preregistration freeze integrity failed")
    if receipt.get("session_assignments_sha256") != sha256_path(session_path):
        raise RuntimeError("Path-D session-assignment freeze integrity failed")
    if receipt.get("feature_lineage_sha256") != sha256_path(lineage_path):
        raise RuntimeError("Path-D feature-lineage freeze integrity failed")
    assignments = read_json(session_path)
    lineage = read_json(lineage_path)
    validate_preregistration_payload(payload, assignments, lineage)
    if payload["binding_plan"]["sha256"] != sha256_path(PLAN_PATH):
        raise RuntimeError("binding Path-D plan drifted after preregistration")
    if receipt.get("holdout_open_count") != 0:
        raise RuntimeError("protected holdout has already been opened")
    return receipt


@dataclass(frozen=True)
class FrozenFitAuthorization:
    """Exact hash-bound session role supplied to a fit/calibration dataset loader."""

    role: str
    outer_fold: int | None
    inner_fold: int | None
    sessions: tuple[str, ...]
    sessions_sha256_newline: str
    preregistration_sha256: str
    session_assignments_sha256: str
    source_hash_policy_sha256: str
    corpus_integrity_receipt_sha256: str
    lineage_receipt_sha256: str
    machinery_receipt_sha256: str
    fit_environment_sha256: str
    entry_pooled_acceptance_receipt_sha256: str | None
    # Defaults keep synthetic pre-correction fixtures constructible; every real
    # authorization minted by assert_entry_fit_ready requires exact hex digests.
    foundation_generation_sha256: str | None = None
    foundation_stability_receipt_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sessions"] = list(self.sessions)
        return payload


@dataclass(frozen=True)
class FrozenEvidenceAuthorization:
    """Hash-bound non-fit session role for outer evidence or isolated diagnostics."""

    role: str
    outer_fold: int | None
    inner_fold: int | None
    sessions: tuple[str, ...]
    sessions_sha256_newline: str
    preregistration_sha256: str
    session_assignments_sha256: str
    source_hash_policy_sha256: str
    corpus_integrity_receipt_sha256: str
    lineage_receipt_sha256: str
    machinery_receipt_sha256: str
    fit_environment_sha256: str
    open_gate_receipts_sha256: tuple[str, ...]
    access_scope_id: str
    access_receipt_path: str
    access_receipt_sha256: str
    transaction_id: str
    # Evidence-gate v2 propagates both fields from the pre-open claim.  Optional
    # defaults are transitional only and never satisfy corrected-generation gates.
    foundation_generation_sha256: str | None = None
    foundation_stability_receipt_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sessions"] = list(self.sessions)
        payload["open_gate_receipts_sha256"] = list(self.open_gate_receipts_sha256)
        return payload


@dataclass(frozen=True)
class FrozenResultAggregationAuthorization:
    """Artifact-only pooled authority; it can never authorize a corpus loader."""

    role: str
    sessions: tuple[str, ...]
    sessions_sha256_newline: str
    preregistration_sha256: str
    session_assignments_sha256: str
    source_hash_policy_sha256: str
    machinery_receipt_sha256: str
    outer_result_receipts_sha256: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sessions"] = list(self.sessions)
        payload["outer_result_receipts_sha256"] = list(
            self.outer_result_receipts_sha256
        )
        return payload


@dataclass(frozen=True)
class FrozenContextDiagnosticsAuthorizationV1:
    """Opaque, immutable, non-fit authority for post-entry context reporting."""

    SCHEMA_VERSION: ClassVar[str] = "pathd.frozen_context_diagnostics_authorization.v1"

    schema_version: str
    status: str
    preregistration_sha256: str
    plan_sha256: str
    session_assignments_sha256: str
    source_hash_policy_sha256: str
    corpus_integrity_receipt_sha256: str
    machinery_receipt_sha256: str
    holdout_open_count: int
    diagnostic_inventory_receipt_path: str
    diagnostic_inventory_receipt_sha256: str
    context_access_receipt_path: str
    context_access_count: int
    pooled_acceptance_receipt_path: str
    pooled_acceptance_receipt_sha256: str
    outer_result_receipt_paths: tuple[str, ...]
    outer_result_receipt_sha256s: tuple[str, ...]
    outer_sessions_by_fold: tuple[tuple[str, ...], ...]
    outer_sessions_root_sha256: str
    policy_journal_binding_roots_by_fold: tuple[str, ...]
    policy_journal_bindings_root_sha256: str
    diagnostic_spec_sha256: str
    authorization_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outer_result_receipt_paths"] = list(
            self.outer_result_receipt_paths
        )
        payload["outer_result_receipt_sha256s"] = list(
            self.outer_result_receipt_sha256s
        )
        payload["outer_sessions_by_fold"] = [
            list(sessions) for sessions in self.outer_sessions_by_fold
        ]
        payload["policy_journal_binding_roots_by_fold"] = list(
            self.policy_journal_binding_roots_by_fold
        )
        return payload


def _canonical_repo_regular_file(label: Any) -> Path:
    """Resolve one canonical repo-relative regular file without following symlinks."""

    if type(label) is not str or not label:
        raise RuntimeError("entry fitting blocked: implementation path is not a string")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", label) or "\\" in label or "\x00" in label:
        raise RuntimeError(f"entry fitting blocked: noncanonical implementation path: {label!r}")
    relative = PurePosixPath(label)
    if (
        relative.is_absolute()
        or relative.as_posix() != label
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise RuntimeError(f"entry fitting blocked: noncanonical implementation path: {label!r}")
    root = REPO_ROOT.resolve(strict=True)
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise RuntimeError(f"entry fitting blocked: symlinked implementation path: {label}")
    try:
        resolved = candidate.resolve(strict=True)
        canonical = resolved.relative_to(root).as_posix()
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(
            f"entry fitting blocked: missing/out-of-repo implementation path: {label}"
        ) from exc
    if canonical != label or not resolved.is_file():
        raise RuntimeError(f"entry fitting blocked: implementation is not a regular file: {label}")
    return resolved


def _frozen_contract_file(
    payload: dict[str, Any], *, contract_key: str, expected_path: Path
) -> Path:
    contracts = payload.get("source_hash_policy", {}).get("receipt_contracts", {})
    label = contracts.get(contract_key)
    if label != repo_path_label(expected_path):
        raise RuntimeError(
            f"entry fitting blocked: fixed artifact path drift: {contract_key}"
        )
    resolved = _canonical_repo_regular_file(label)
    try:
        expected_resolved = expected_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"entry fitting blocked: fixed artifact is absent: {contract_key}"
        ) from exc
    if resolved != expected_resolved:
        raise RuntimeError(
            f"entry fitting blocked: fixed artifact target drift: {contract_key}"
        )
    return resolved


def _raw_regular_file_binding(label: str) -> dict[str, Any]:
    """Hash one canonical regular file through a no-follow descriptor."""

    path = _canonical_repo_regular_file(label)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"foundation file is not regular: {label}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or size != after.st_size:
            raise RuntimeError(f"foundation file changed while hashing: {label}")
    finally:
        os.close(descriptor)
    return {"path": label, "bytes": size, "sha256": digest.hexdigest()}


def _assert_superseded_foundation_history() -> dict[str, Any]:
    for expected in SUPERSEDED_FOUNDATION_FILE_BINDINGS:
        if _raw_regular_file_binding(expected["path"]) != expected:
            raise RuntimeError(
                f"superseded foundation history drift: {expected['path']}"
            )
    for expected in INTERMEDIATE_CORRECTED_FOUNDATION_FILE_BINDINGS:
        if _raw_regular_file_binding(expected["path"]) != expected:
            raise RuntimeError(
                f"intermediate corrected foundation history drift: {expected['path']}"
            )
    for expected in PREVIOUS_CORRECTED_V2_FOUNDATION_FILE_BINDINGS:
        if _raw_regular_file_binding(expected["path"]) != expected:
            raise RuntimeError(
                f"corrected-v2 foundation history drift: {expected['path']}"
            )
    expected_burn_file = {
        key: QUARANTINED_TEST_BURN_BINDING[key]
        for key in ("path", "bytes", "sha256")
    }
    if _raw_regular_file_binding(expected_burn_file["path"]) != expected_burn_file:
        raise RuntimeError("quarantined test-contamination burn bytes drifted")
    burn = read_json(REPO_ROOT / expected_burn_file["path"])
    _validate_self_hashed_receipt(burn, field="receipt_sha256")
    for key in (
        "receipt_sha256",
        "transaction_id",
        "status",
        "reason",
        "role",
        "outer_fold",
        "access_count",
        "access_receipt_sha256",
        "dataset_receipt_sha256",
        "result_sha256",
        "reopen_permitted",
        "holdout_open_count",
    ):
        if burn.get(key) != QUARANTINED_TEST_BURN_BINDING[key]:
            raise RuntimeError(f"quarantined burn semantic drift: {key}")
    return burn


def _corrected_foundation_base_bindings() -> list[dict[str, Any]]:
    return [
        _raw_regular_file_binding(repo_path_label(path))
        for path in (
            PREREG_PATH,
            PREREG_HASH_PATH,
            SESSION_PATH,
            LINEAGE_PATH,
            FREEZE_RECEIPT_PATH,
        )
    ]


def _pristine_corrected_fold_namespaces() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer_fold in range(1, 6):
        path = ENTRY_FOLD_ARTIFACT_ROOT / f"fold_{outer_fold}"
        if path.is_symlink():
            raise RuntimeError("corrected fold namespace is symlinked")
        if path.exists():
            if not path.is_dir() or any(path.iterdir()):
                raise RuntimeError(
                    f"corrected fold {outer_fold} is not pristine before restoration"
                )
            state = "PRISTINE_EMPTY"
        else:
            state = "PRISTINE_ABSENT"
        rows.append(
            {
                "outer_fold": outer_fold,
                "path": repo_path_label(path),
                "state": state,
            }
        )
    return rows


def _foundation_generation_sha256(
    *, corrected_files: list[dict[str, Any]], correction_sha256: str
) -> str:
    return stable_hash(
        {
            "schema_version": "pathd.entry_exit.corrected_foundation_generation.v3",
            "corrected_audit_root": repo_path_label(AUDIT_ROOT),
            "corrected_files": corrected_files,
            "correction_sha256": correction_sha256,
            "superseded_foundation_files": [
                dict(row) for row in SUPERSEDED_FOUNDATION_FILE_BINDINGS
            ],
            "intermediate_corrected_foundation_files": [
                dict(row)
                for row in INTERMEDIATE_CORRECTED_FOUNDATION_FILE_BINDINGS
            ],
            "previous_corrected_v2_audit_root": repo_path_label(
                PREVIOUS_CORRECTED_V2_AUDIT_ROOT
            ),
            "previous_corrected_v2_foundation_files": [
                dict(row)
                for row in PREVIOUS_CORRECTED_V2_FOUNDATION_FILE_BINDINGS
            ],
            "previous_corrected_v2_foundation_generation_sha256": (
                PREVIOUS_CORRECTED_V2_FOUNDATION_GENERATION_SHA256
            ),
            "quarantined_burn_file_sha256": QUARANTINED_TEST_BURN_BINDING[
                "sha256"
            ],
            "restored_outer_folds": [1, 2, 3, 4, 5],
        }
    )


def seal_foundation_restoration_receipt() -> dict[str, Any]:
    """Seal the benign P1 restoration in the distinct corrected generation."""

    if os.path.lexists(FOUNDATION_RESTORATION_RECEIPT_PATH):
        raise RuntimeError("foundation restoration receipt already exists")
    prereg_receipt = assert_preregistration_frozen()
    payload = read_json(PREREG_PATH)
    _assert_superseded_foundation_history()
    corrected = _corrected_foundation_base_bindings()
    folds = _pristine_corrected_fold_namespaces()
    correction_sha = stable_hash(payload["foundation_correction"])
    generation_sha = _foundation_generation_sha256(
        corrected_files=corrected, correction_sha256=correction_sha
    )
    semantic = enveloped_research_result(
        {
            "schema_version": payload["source_hash_policy"]["receipt_contracts"][
                "foundation_restoration_receipt_schema_version"
            ],
            "status": "RESTORED_FIVE_FOLDS_BENIGN_TEST_CONTAMINATION",
            "restored_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "foundation_generation_sha256": generation_sha,
            "foundation_correction_sha256": correction_sha,
            "corrected_audit_root": repo_path_label(AUDIT_ROOT),
            "corrected_foundation_files": corrected,
            "superseded_audit_root": repo_path_label(SUPERSEDED_AUDIT_ROOT),
            "superseded_foundation_files": [
                dict(row) for row in SUPERSEDED_FOUNDATION_FILE_BINDINGS
            ],
            "intermediate_corrected_audit_root": repo_path_label(
                INTERMEDIATE_CORRECTED_AUDIT_ROOT
            ),
            "intermediate_corrected_foundation_files": [
                dict(row)
                for row in INTERMEDIATE_CORRECTED_FOUNDATION_FILE_BINDINGS
            ],
            "previous_corrected_v2_audit_root": repo_path_label(
                PREVIOUS_CORRECTED_V2_AUDIT_ROOT
            ),
            "previous_corrected_v2_foundation_files": [
                dict(row)
                for row in PREVIOUS_CORRECTED_V2_FOUNDATION_FILE_BINDINGS
            ],
            "previous_corrected_v2_foundation_generation_sha256": (
                PREVIOUS_CORRECTED_V2_FOUNDATION_GENERATION_SHA256
            ),
            "quarantined_burn": dict(QUARANTINED_TEST_BURN_BINDING),
            "fold_namespaces": folds,
            "restored_outer_folds": [1, 2, 3, 4, 5],
            "acceptance_rule": "pooled AND at least 4 of 5 chronological outer folds",
            "model_fit_executed": False,
        }
    )
    receipt = {**semantic, "receipt_sha256": stable_hash(semantic)}
    _write_canonical_json_exclusive(FOUNDATION_RESTORATION_RECEIPT_PATH, receipt)
    return assert_foundation_restoration_frozen()


def assert_foundation_restoration_frozen() -> dict[str, Any]:
    prereg_receipt = assert_preregistration_frozen()
    payload = read_json(PREREG_PATH)
    path = _frozen_contract_file(
        payload,
        contract_key="foundation_restoration_receipt_path",
        expected_path=FOUNDATION_RESTORATION_RECEIPT_PATH,
    )
    receipt = read_json(path)
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    try:
        _validate_result_envelope_against(
            receipt,
            prereg_payload=payload,
            preregistration_sha256=prereg_receipt["preregistration_sha256"],
            expected_holdout_open_count=0,
        )
    except ValueError as exc:
        raise RuntimeError("foundation restoration envelope drift") from exc
    _assert_superseded_foundation_history()
    corrected = _corrected_foundation_base_bindings()
    correction_sha = stable_hash(payload["foundation_correction"])
    expected_keys = {
        "schema_version",
        "status",
        "restored_at_utc",
        "foundation_generation_sha256",
        "foundation_correction_sha256",
        "corrected_audit_root",
        "corrected_foundation_files",
        "superseded_audit_root",
        "superseded_foundation_files",
        "intermediate_corrected_audit_root",
        "intermediate_corrected_foundation_files",
        "previous_corrected_v2_audit_root",
        "previous_corrected_v2_foundation_files",
        "previous_corrected_v2_foundation_generation_sha256",
        "quarantined_burn",
        "fold_namespaces",
        "restored_outer_folds",
        "acceptance_rule",
        "model_fit_executed",
        "quarantine_labels",
        "claim_boundary",
        "holdout_caveat",
        "plan_sha256",
        "preregistration_sha256",
        "fill_law_hash",
        "holdout_open_count",
        "receipt_sha256",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version")
        != payload["source_hash_policy"]["receipt_contracts"][
            "foundation_restoration_receipt_schema_version"
        ]
        or receipt.get("status")
        != "RESTORED_FIVE_FOLDS_BENIGN_TEST_CONTAMINATION"
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T[^\s]+Z",
            str(receipt.get("restored_at_utc", "")),
        )
        is None
        or receipt.get("foundation_correction_sha256") != correction_sha
        or receipt.get("corrected_audit_root") != repo_path_label(AUDIT_ROOT)
        or receipt.get("corrected_foundation_files") != corrected
        or receipt.get("superseded_foundation_files")
        != [dict(row) for row in SUPERSEDED_FOUNDATION_FILE_BINDINGS]
        or receipt.get("intermediate_corrected_audit_root")
        != repo_path_label(INTERMEDIATE_CORRECTED_AUDIT_ROOT)
        or receipt.get("intermediate_corrected_foundation_files")
        != [
            dict(row) for row in INTERMEDIATE_CORRECTED_FOUNDATION_FILE_BINDINGS
        ]
        or receipt.get("previous_corrected_v2_audit_root")
        != repo_path_label(PREVIOUS_CORRECTED_V2_AUDIT_ROOT)
        or receipt.get("previous_corrected_v2_foundation_files")
        != [dict(row) for row in PREVIOUS_CORRECTED_V2_FOUNDATION_FILE_BINDINGS]
        or receipt.get("previous_corrected_v2_foundation_generation_sha256")
        != PREVIOUS_CORRECTED_V2_FOUNDATION_GENERATION_SHA256
        or receipt.get("superseded_audit_root")
        != repo_path_label(SUPERSEDED_AUDIT_ROOT)
        or receipt.get("quarantined_burn") != dict(QUARANTINED_TEST_BURN_BINDING)
        or receipt.get("restored_outer_folds") != [1, 2, 3, 4, 5]
        or receipt.get("acceptance_rule")
        != "pooled AND at least 4 of 5 chronological outer folds"
        or receipt.get("model_fit_executed") is not False
        or receipt.get("foundation_generation_sha256")
        != _foundation_generation_sha256(
            corrected_files=corrected, correction_sha256=correction_sha
        )
    ):
        raise RuntimeError("foundation restoration receipt drift")
    folds = receipt.get("fold_namespaces")
    if (
        type(folds) is not list
        or [row.get("outer_fold") for row in folds if type(row) is dict]
        != [1, 2, 3, 4, 5]
        or any(
            set(row) != {"outer_fold", "path", "state"}
            or row.get("path")
            != repo_path_label(
                ENTRY_FOLD_ARTIFACT_ROOT / f"fold_{row['outer_fold']}"
            )
            or row.get("state") not in {"PRISTINE_ABSENT", "PRISTINE_EMPTY"}
            for row in folds
        )
    ):
        raise RuntimeError("foundation restoration fold ledger drift")
    return receipt


def _foundation_stability_file_bindings(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    contracts = payload["source_hash_policy"]["receipt_contracts"]
    labels = [
        repo_path_label(path)
        for path in (
            PREREG_PATH,
            PREREG_HASH_PATH,
            SESSION_PATH,
            LINEAGE_PATH,
            FREEZE_RECEIPT_PATH,
            FOUNDATION_RESTORATION_RECEIPT_PATH,
            ENTRY_MACHINERY_TEST_EVIDENCE_PATH,
            LINEAGE_IMPLEMENTATION_RECEIPT_PATH,
            ENTRY_MACHINERY_RECEIPT_PATH,
            CORPUS_INTEGRITY_RECEIPT_PATH,
        )
    ]
    labels.extend(payload["source_hash_policy"]["authorized_in_repo_dependency_paths"])
    labels.extend(
        command["junit_path"] for command in contracts["required_test_commands"]
    )
    labels.extend(row["path"] for row in SUPERSEDED_FOUNDATION_FILE_BINDINGS)
    labels.extend(
        row["path"] for row in INTERMEDIATE_CORRECTED_FOUNDATION_FILE_BINDINGS
    )
    labels.extend(
        row["path"] for row in PREVIOUS_CORRECTED_V2_FOUNDATION_FILE_BINDINGS
    )
    labels.append(QUARANTINED_TEST_BURN_BINDING["path"])
    ordered = list(dict.fromkeys(labels))
    if len(ordered) != len(set(ordered)):
        raise RuntimeError("foundation stability allowlist contains duplicates")
    return [_raw_regular_file_binding(label) for label in ordered]


def _foundation_root_sha256(
    *,
    generation_sha256: str,
    files: list[dict[str, Any]],
    fit_environment_sha256: str,
) -> str:
    return stable_hash(
        {
            "schema_version": "pathd.entry_exit.foundation_byte_snapshot.v3",
            "foundation_generation_sha256": generation_sha256,
            "files": files,
            "fit_environment_sha256": fit_environment_sha256,
        }
    )


def _assert_corrected_live_twin_contract(payload: dict[str, Any]) -> None:
    validate_inventory_contracts()
    lineage = read_json(LINEAGE_PATH)
    inventory = lineage.get("live_twin_inventory", {})
    exit_contract = payload.get("exit", {}).get("feature_contract", {})
    if (
        exit_contract.get("feature_names_in_exact_order")
        != list(EXIT47_CORRECTED_FEATURE_NAMES)
        or exit_contract.get("feature_count") != 47
        or any(
            name in exit_contract.get("feature_names_in_exact_order", [])
            for name in (
                "last_causal_minute_volume",
                "last_causal_open_interest",
            )
        )
        or inventory.get("corrected_exit47_feature_names")
        != list(EXIT47_CORRECTED_FEATURE_NAMES)
        or inventory.get("original_exit49_feature_names")
        != list(EXIT49_FEATURE_NAMES)
        or inventory.get("corrected_exit47_records_sha256")
        != stable_hash(
            [asdict(record) for record in EXIT47_CORRECTED_LIVE_TWIN_INVENTORY]
        )
    ):
        raise RuntimeError("corrected live-twin inventory/exit-47 contract drift")
    minute_volume = live_twin_record("exit", "last_causal_minute_volume")
    if (
        minute_volume.recommended_action != DROP_UNTIL_EXACT_ADAPTER_RECEIPT
        or exit_contract.get("last_causal_minute_volume", {}).get(
            "intraday_alpha"
        )
        != "DROPPED"
    ):
        raise RuntimeError("minute-volume dropped disposition drift")


def seal_foundation_stability_receipt() -> dict[str, Any]:
    """Seal all fixed pre-fit bytes after machinery and corpus verification."""

    if os.path.lexists(FOUNDATION_STABILITY_RECEIPT_PATH):
        raise RuntimeError("foundation stability receipt already exists")
    from v4.research.pathd_holdout_gate import _assert_protected_holdout_unopened

    _assert_protected_holdout_unopened()
    prereg_receipt = assert_preregistration_frozen()
    payload = read_json(PREREG_PATH)
    assert_correction_prefit_release(payload)
    restoration = assert_foundation_restoration_frozen()
    _assert_corrected_live_twin_contract(payload)
    _verify_immutable_sources(payload)
    lineage = assert_lineage_implementation_frozen()
    machinery = _assert_entry_machinery_frozen(prereg_receipt, payload, lineage)
    corpus = _validated_corpus_integrity_receipt(payload, prereg_receipt, machinery)
    _verify_authorized_dependency_closure(payload)
    fit_environment_sha = stable_hash(assert_entry_fit_environment_current())
    files = _foundation_stability_file_bindings(payload)
    generation_sha = restoration["foundation_generation_sha256"]
    pristine_folds = _pristine_corrected_fold_namespaces()
    root_sha = _foundation_root_sha256(
        generation_sha256=generation_sha,
        files=files,
        fit_environment_sha256=fit_environment_sha,
    )
    semantic = enveloped_research_result(
        {
            "schema_version": payload["source_hash_policy"]["receipt_contracts"][
                "foundation_stability_receipt_schema_version"
            ],
            "status": "BYTE_IDENTICAL_FOUNDATION_STABLE_BEFORE_FIT_OR_EVIDENCE",
            "sealed_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "foundation_generation_sha256": generation_sha,
            "foundation_root_sha256": root_sha,
            "foundation_file_count": len(files),
            "foundation_files": files,
            "foundation_correction_sha256": stable_hash(
                payload["foundation_correction"]
            ),
            "restoration_receipt_sha256": sha256_path(
                FOUNDATION_RESTORATION_RECEIPT_PATH
            ),
            "session_assignments_sha256": sha256_path(SESSION_PATH),
            "feature_lineage_sha256": sha256_path(LINEAGE_PATH),
            "source_hash_policy_sha256": stable_hash(payload["source_hash_policy"]),
            "lineage_receipt_sha256": sha256_path(
                LINEAGE_IMPLEMENTATION_RECEIPT_PATH
            ),
            "machinery_receipt_sha256": sha256_path(ENTRY_MACHINERY_RECEIPT_PATH),
            "corpus_integrity_receipt_sha256": sha256_path(
                CORPUS_INTEGRITY_RECEIPT_PATH
            ),
            "fit_environment_sha256": fit_environment_sha,
            "fold_namespaces_at_stability": pristine_folds,
            "restored_outer_folds": [1, 2, 3, 4, 5],
            "model_fit_executed": False,
        }
    )
    receipt = {**semantic, "receipt_sha256": stable_hash(semantic)}
    _write_canonical_json_exclusive(FOUNDATION_STABILITY_RECEIPT_PATH, receipt)
    return assert_research_foundation_stable()


def assert_research_foundation_stable() -> dict[str, Any]:
    """Reproduce the sealed foundation byte root before fit or evidence decode."""

    prereg_receipt = assert_preregistration_frozen()
    payload = read_json(PREREG_PATH)
    path = _frozen_contract_file(
        payload,
        contract_key="foundation_stability_receipt_path",
        expected_path=FOUNDATION_STABILITY_RECEIPT_PATH,
    )
    receipt = read_json(path)
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    restoration = assert_foundation_restoration_frozen()
    _assert_corrected_live_twin_contract(payload)
    _verify_immutable_sources(payload)
    lineage = assert_lineage_implementation_frozen()
    machinery = _assert_entry_machinery_frozen(prereg_receipt, payload, lineage)
    _validated_corpus_integrity_receipt(payload, prereg_receipt, machinery)
    _verify_authorized_dependency_closure(payload)
    fit_environment_sha = stable_hash(assert_entry_fit_environment_current())
    files = _foundation_stability_file_bindings(payload)
    generation_sha = restoration["foundation_generation_sha256"]
    expected_root = _foundation_root_sha256(
        generation_sha256=generation_sha,
        files=files,
        fit_environment_sha256=fit_environment_sha,
    )
    sealed_fold_namespaces = receipt.get("fold_namespaces_at_stability")
    expected_keys = {
        "schema_version",
        "status",
        "sealed_at_utc",
        "foundation_generation_sha256",
        "foundation_root_sha256",
        "foundation_file_count",
        "foundation_files",
        "foundation_correction_sha256",
        "restoration_receipt_sha256",
        "session_assignments_sha256",
        "feature_lineage_sha256",
        "source_hash_policy_sha256",
        "lineage_receipt_sha256",
        "machinery_receipt_sha256",
        "corpus_integrity_receipt_sha256",
        "fit_environment_sha256",
        "fold_namespaces_at_stability",
        "restored_outer_folds",
        "model_fit_executed",
        "quarantine_labels",
        "claim_boundary",
        "holdout_caveat",
        "plan_sha256",
        "preregistration_sha256",
        "fill_law_hash",
        "holdout_open_count",
        "receipt_sha256",
    }
    try:
        _validate_result_envelope_against(
            receipt,
            prereg_payload=payload,
            preregistration_sha256=prereg_receipt["preregistration_sha256"],
            expected_holdout_open_count=0,
        )
    except ValueError as exc:
        raise RuntimeError("foundation stability envelope drift") from exc
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version")
        != payload["source_hash_policy"]["receipt_contracts"][
            "foundation_stability_receipt_schema_version"
        ]
        or receipt.get("status")
        != "BYTE_IDENTICAL_FOUNDATION_STABLE_BEFORE_FIT_OR_EVIDENCE"
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T[^\s]+Z",
            str(receipt.get("sealed_at_utc", "")),
        )
        is None
        or receipt.get("foundation_generation_sha256") != generation_sha
        or receipt.get("foundation_root_sha256") != expected_root
        or receipt.get("foundation_files") != files
        or receipt.get("foundation_file_count") != len(files)
        or receipt.get("foundation_correction_sha256")
        != stable_hash(payload["foundation_correction"])
        or receipt.get("restoration_receipt_sha256")
        != sha256_path(FOUNDATION_RESTORATION_RECEIPT_PATH)
        or receipt.get("session_assignments_sha256") != sha256_path(SESSION_PATH)
        or receipt.get("feature_lineage_sha256") != sha256_path(LINEAGE_PATH)
        or receipt.get("source_hash_policy_sha256")
        != stable_hash(payload["source_hash_policy"])
        or receipt.get("lineage_receipt_sha256")
        != sha256_path(LINEAGE_IMPLEMENTATION_RECEIPT_PATH)
        or receipt.get("machinery_receipt_sha256")
        != sha256_path(ENTRY_MACHINERY_RECEIPT_PATH)
        or receipt.get("corpus_integrity_receipt_sha256")
        != sha256_path(CORPUS_INTEGRITY_RECEIPT_PATH)
        or receipt.get("fit_environment_sha256") != fit_environment_sha
        or type(sealed_fold_namespaces) is not list
        or [
            row.get("outer_fold")
            for row in sealed_fold_namespaces
            if type(row) is dict
        ]
        != [1, 2, 3, 4, 5]
        or any(
            row.get("path")
            != repo_path_label(ENTRY_FOLD_ARTIFACT_ROOT / f"fold_{index}")
            or set(row) != {"outer_fold", "path", "state"}
            or row.get("state") not in {"PRISTINE_ABSENT", "PRISTINE_EMPTY"}
            for index, row in enumerate(sealed_fold_namespaces, start=1)
        )
        or receipt.get("restored_outer_folds") != [1, 2, 3, 4, 5]
        or receipt.get("model_fit_executed") is not False
    ):
        raise RuntimeError("foundation stability receipt/byte root drift")
    return receipt


def _implementation_receipt_map(
    receipt: dict[str, Any], *, expected_paths: list[str], owner: str
) -> dict[str, str]:
    rows = receipt.get("implementations")
    if not isinstance(rows, list):
        raise RuntimeError(f"entry fitting blocked: {owner} implementation list is absent")
    observed: dict[str, str] = {}
    for item in rows:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise RuntimeError(f"entry fitting blocked: malformed {owner} implementation row")
        label = item["path"]
        digest = item["sha256"]
        path = _canonical_repo_regular_file(label)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"entry fitting blocked: malformed {owner} SHA-256: {label}")
        if label in observed:
            raise RuntimeError(f"entry fitting blocked: duplicate {owner} implementation: {label}")
        if sha256_path(path) != digest:
            raise RuntimeError(f"entry fitting blocked: {owner} implementation hash drift: {label}")
        observed[label] = digest
    if set(observed) != set(expected_paths) or len(observed) != len(expected_paths):
        missing = sorted(set(expected_paths) - set(observed))
        extra = sorted(set(observed) - set(expected_paths))
        raise RuntimeError(
            f"entry fitting blocked: {owner} implementation coverage mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return observed


def _source_policy_and_hash(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    policy = payload.get("source_hash_policy")
    if not isinstance(policy, dict):
        raise RuntimeError("entry fitting blocked: frozen source policy is absent")
    return policy, stable_hash(policy)


def _verify_immutable_sources(payload: dict[str, Any]) -> None:
    policy, _ = _source_policy_and_hash(payload)
    frozen_sources = payload.get("source_hashes_at_freeze", {})
    immutable = policy.get("immutable_and_rehashed_at_every_fit")
    if not isinstance(immutable, list) or not immutable:
        raise RuntimeError("entry fitting blocked: frozen immutable source list is absent")
    for relative in immutable:
        path = _canonical_repo_regular_file(relative)
        if sha256_path(path) != frozen_sources.get(relative):
            raise RuntimeError(f"entry fitting blocked: immutable source drift: {relative}")


def _module_execution_paths(module_name: str) -> tuple[str, ...]:
    """Return every existing repo-local Python file executed by a static import."""

    if not module_name or not all(part.isidentifier() for part in module_name.split(".")):
        return ()
    parts = module_name.split(".")
    if not (REPO_ROOT / parts[0]).exists() and not (
        REPO_ROOT / f"{parts[0]}.py"
    ).exists():
        return ()
    rows: list[str] = []
    for size in range(1, len(parts)):
        package_init = REPO_ROOT.joinpath(*parts[:size], "__init__.py")
        if package_init.is_file():
            rows.append(package_init.relative_to(REPO_ROOT).as_posix())
    module_file = REPO_ROOT.joinpath(*parts).with_suffix(".py")
    package_file = REPO_ROOT.joinpath(*parts, "__init__.py")
    if module_file.is_file():
        rows.append(module_file.relative_to(REPO_ROOT).as_posix())
    elif package_file.is_file():
        rows.append(package_file.relative_to(REPO_ROOT).as_posix())
    return tuple(dict.fromkeys(rows))


def _resolved_v4_import_paths(label: str, tree: ast.AST) -> set[str]:
    """Resolve static absolute/relative imports, including package initializers."""

    source_path = PurePosixPath(label)
    module_parts = list(source_path.with_suffix("").parts)
    is_package = module_parts[-1] == "__init__"
    if is_package:
        module_parts.pop()
    package_parts = module_parts if is_package else module_parts[:-1]
    resolved: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        child_candidates: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package_parts) + 1:
                    raise RuntimeError(
                        f"entry fitting blocked: invalid relative import in {label}"
                    )
                base_parts = package_parts[: len(package_parts) - (node.level - 1)]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                names.append(base)
                child_candidates.extend(
                    f"{base}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
        for name in (*names, *child_candidates):
            resolved.update(_module_execution_paths(name))
    return resolved


def _reject_dynamic_dependency_escape(label: str, tree: ast.AST) -> None:
    """Reject direct import, code, process, and reflection escapes in frozen source."""

    # These exact test bytes are already inside the immutable source set and are
    # rehashed at every gate.  Only the one post-preregistration fixed-science
    # test remains extensible, so it receives the narrow test allowlist and the
    # same process/remote/dynamic analysis as production machinery.
    if (
        label.startswith("v4/tests/")
        and label in IMMUTABLE_SOURCE_PATHS_AT_FIT
        and label != "v4/tests/test_pathd_fixed_science_contract.py"
    ):
        return

    allowed_roots = frozenset(FROZEN_OFFLINE_IMPORT_ROOTS)
    if label.startswith("v4/tests/"):
        allowed_roots |= frozenset(FROZEN_EXTENSIBLE_TEST_IMPORT_ROOTS)
    forbidden_dynamic_calls = {"__import__", "compile", "eval", "exec"}
    forbidden_reflection_calls = {"globals", "locals", "vars"}
    forbidden_os_attributes = {
        "fork",
        "forkpty",
        "kill",
        "killpg",
        "popen",
        "posix_spawn",
        "posix_spawnp",
        "startfile",
        "system",
    }
    forbidden_os_reflection = {"__dict__", "__getattribute__"}
    forbidden_sys_attributes = {
        "__dict__",
        "__getattribute__",
        "meta_path",
        "modules",
        "path",
        "path_hooks",
        "path_importer_cache",
    }
    safe_reflective_os_attributes = {"O_DIRECTORY", "O_NOFOLLOW"}
    forbidden_remote_attributes = {
        "AzureFileSystem",
        "FlightClient",
        "GcsFileSystem",
        "HadoopFileSystem",
        "S3FileSystem",
        "download_url_to_file",
        "load_state_dict_from_url",
    }
    forbidden_parallel_attributes = {"Parallel", "delayed"}
    remote_scheme = re.compile(
        r"(?i)^(?:abfs|azure|ftp|gcs|gs|https?|s3)://"
    )

    def is_forbidden_os_attribute(name: str) -> bool:
        return (
            name in forbidden_os_attributes
            or name.startswith("spawn")
            or name.startswith("exec")
        )

    os_aliases = {"os"}
    sys_aliases = {"sys"}
    getattr_aliases = {"getattr"}
    setattr_aliases = {"setattr", "delattr"}
    dynamic_call_aliases = set(forbidden_dynamic_calls)
    reflection_call_aliases = set(forbidden_reflection_calls)
    remote_literal_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.partition(".")[0]
                if root not in allowed_roots:
                    raise RuntimeError(
                        "entry fitting blocked: prohibited external dependency "
                        f"import in {label}: {root}"
                    )
                if root == "os" and (alias.name == "os" or alias.asname is None):
                    os_aliases.add(alias.asname or "os")
                if root == "sys" and (alias.name == "sys" or alias.asname is None):
                    sys_aliases.add(alias.asname or "sys")
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                raise RuntimeError(
                    f"entry fitting blocked: wildcard import dependency escape in {label}"
                )
            if node.level == 0 and node.module:
                root = node.module.partition(".")[0]
                if root not in allowed_roots:
                    raise RuntimeError(
                        "entry fitting blocked: prohibited external dependency "
                        f"import in {label}: {root}"
                    )
                if root == "os":
                    for alias in node.names:
                        if (
                            is_forbidden_os_attribute(alias.name)
                            or alias.name in forbidden_os_reflection
                        ):
                            raise RuntimeError(
                                "entry fitting blocked: process-launch dependency "
                                f"escape in {label}"
                            )
                if root == "sys":
                    for alias in node.names:
                        if alias.name in forbidden_sys_attributes:
                            raise RuntimeError(
                                f"entry fitting blocked: sys state dependency escape in {label}"
                            )
                if root == "joblib" and any(
                    alias.name in forbidden_parallel_attributes for alias in node.names
                ):
                    raise RuntimeError(
                        f"entry fitting blocked: parallel dependency escape in {label}"
                    )

    def expression_categories(node: ast.AST | None) -> set[str]:
        if node is None:
            return set()
        if isinstance(node, ast.Name):
            categories: set[str] = set()
            if node.id in os_aliases:
                categories.add("os")
            if node.id in sys_aliases:
                categories.add("sys")
            if node.id in getattr_aliases:
                categories.add("getattr")
            if node.id in setattr_aliases:
                categories.add("setattr")
            if node.id in dynamic_call_aliases:
                categories.add("dynamic")
            if node.id in reflection_call_aliases:
                categories.add("reflection")
            if node.id in remote_literal_aliases:
                categories.add("remote")
            return categories
        if (
            isinstance(node, ast.Constant)
            and type(node.value) is str
            and remote_scheme.match(node.value) is not None
        ):
            return {"remote"}
        if isinstance(node, ast.Subscript):
            return expression_categories(node.value)
        if isinstance(node, ast.Attribute):
            return expression_categories(node.value)
        if isinstance(node, ast.IfExp):
            return expression_categories(node.body) | expression_categories(
                node.orelse
            )
        if isinstance(node, ast.BinOp):
            return expression_categories(node.left) | expression_categories(
                node.right
            )
        if isinstance(node, ast.JoinedStr):
            return set().union(
                *(expression_categories(item) for item in node.values)
            )
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return set().union(*(expression_categories(item) for item in node.elts))
        if isinstance(node, ast.Dict):
            return set().union(
                *(expression_categories(item) for item in (*node.keys, *node.values))
            )
        return set()

    def target_names(node: ast.AST) -> set[str]:
        if isinstance(node, ast.Name):
            return {node.id}
        if isinstance(node, (ast.Tuple, ast.List)):
            return set().union(*(target_names(item) for item in node.elts))
        return set()

    assignments: list[tuple[set[str], ast.AST | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = set().union(*(target_names(target) for target in node.targets))
            assignments.append((names, node.value))
        elif isinstance(node, ast.AnnAssign):
            assignments.append((target_names(node.target), node.value))
        elif isinstance(node, ast.NamedExpr):
            assignments.append((target_names(node.target), node.value))

    changed = True
    while changed:
        changed = False
        for names, value in assignments:
            categories = expression_categories(value)
            for category, aliases in (
                ("os", os_aliases),
                ("sys", sys_aliases),
                ("getattr", getattr_aliases),
                ("setattr", setattr_aliases),
                ("dynamic", dynamic_call_aliases),
                ("reflection", reflection_call_aliases),
                ("remote", remote_literal_aliases),
            ):
                if category in categories:
                    before = len(aliases)
                    aliases.update(names)
                    changed = changed or len(aliases) != before

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and (node.id in dynamic_call_aliases or node.id == "__builtins__")
        ):
            raise RuntimeError(
                f"entry fitting blocked: dynamic code dependency escape in {label}"
            )
        if isinstance(node, ast.Attribute):
            categories = expression_categories(node.value)
            if "os" in categories and (
                is_forbidden_os_attribute(node.attr)
                or node.attr in forbidden_os_reflection
            ):
                raise RuntimeError(
                    f"entry fitting blocked: process-launch dependency escape in {label}"
                )
            if "sys" in categories and node.attr in forbidden_sys_attributes:
                raise RuntimeError(
                    f"entry fitting blocked: sys state dependency escape in {label}"
                )
            if node.attr == "import_module":
                raise RuntimeError(
                    f"entry fitting blocked: dynamic import dependency escape in {label}"
                )
            if node.attr in forbidden_remote_attributes:
                raise RuntimeError(
                    f"entry fitting blocked: remote I/O dependency escape in {label}"
                )
            if node.attr in forbidden_parallel_attributes:
                raise RuntimeError(
                    f"entry fitting blocked: parallel dependency escape in {label}"
                )
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "n_jobs":
                continue
            value = keyword.value
            valid = (
                isinstance(value, ast.Constant)
                and (
                    value.value is None
                    or (type(value.value) is int and value.value == 1)
                )
            )
            if not valid:
                raise RuntimeError(
                    f"entry fitting blocked: parallel n_jobs dependency escape in {label}"
                )
        if any("remote" in expression_categories(argument) for argument in node.args):
            raise RuntimeError(
                f"entry fitting blocked: remote I/O dependency escape in {label}"
            )
        if isinstance(node.func, ast.Name):
            if node.func.id in dynamic_call_aliases:
                raise RuntimeError(
                    f"entry fitting blocked: dynamic code dependency escape in {label}"
                )
            if node.func.id in reflection_call_aliases:
                raise RuntimeError(
                    f"entry fitting blocked: reflective dependency escape in {label}"
                )
            if node.func.id in getattr_aliases and node.args:
                target_categories = expression_categories(node.args[0])
                attribute = (
                    node.args[1].value
                    if len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and type(node.args[1].value) is str
                    else None
                )
                if "os" in target_categories and (
                    attribute not in safe_reflective_os_attributes
                ):
                    raise RuntimeError(
                        "entry fitting blocked: reflective os dependency escape "
                        f"in {label}"
                    )
                if "sys" in target_categories:
                    raise RuntimeError(
                        f"entry fitting blocked: reflective sys dependency escape in {label}"
                    )
            if node.func.id in setattr_aliases and node.args:
                if expression_categories(node.args[0]) & {"os", "sys"}:
                    raise RuntimeError(
                        "entry fitting blocked: reflective dependency escape "
                        f"in {label}"
                    )


def _verify_authorized_dependency_closure(
    payload: dict[str, Any], *, allow_missing_future: bool = False
) -> dict[str, str]:
    """Prove every executable in-repo helper is registered and hash-covered."""

    policy, _ = _source_policy_and_hash(payload)
    labels = policy.get("authorized_in_repo_dependency_paths")
    expected = list(AUTHORIZED_IN_REPO_DEPENDENCY_PATHS)
    if labels != expected or len(labels) != len(set(labels)):
        raise RuntimeError("entry fitting blocked: authorized dependency roots drifted")
    allowed = set(labels)
    runtime_labels = policy.get("authorized_runtime_dependency_paths")
    if runtime_labels != list(AUTHORIZED_RUNTIME_DEPENDENCY_PATHS):
        raise RuntimeError("entry fitting blocked: authorized runtime roots drifted")
    runtime_allowed = set(runtime_labels)
    observed_hashes: dict[str, str] = {}
    for label in labels:
        candidate = REPO_ROOT / label
        if not candidate.exists():
            if allow_missing_future and label in REQUIRED_ENTRY_FIT_IMPLEMENTATION_PATHS:
                continue
            raise RuntimeError(
                f"entry fitting blocked: authorized dependency is absent: {label}"
            )
        path = _canonical_repo_regular_file(label)
        observed_hashes[label] = sha256_path(path)
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise RuntimeError(
                f"entry fitting blocked: authorized dependency does not parse: {label}"
            ) from exc
        _reject_dynamic_dependency_escape(label, tree)
        imports = _resolved_v4_import_paths(label, tree)
        permitted = allowed if label.startswith("v4/tests/") else runtime_allowed
        escaped = sorted(imports - permitted)
        if escaped:
            raise RuntimeError(
                f"entry fitting blocked: unregistered in-repo dependency from {label}: {escaped}"
            )
    return observed_hashes


def _validated_machinery_test_evidence(
    payload: dict[str, Any], prereg_receipt: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    policy, policy_hash = _source_policy_and_hash(payload)
    contracts = policy.get("receipt_contracts", {})
    evidence_path = _frozen_contract_file(
        payload,
        contract_key="machinery_test_evidence_path",
        expected_path=ENTRY_MACHINERY_TEST_EVIDENCE_PATH,
    )
    evidence = read_json(evidence_path)
    try:
        _validate_result_envelope_against(
            evidence,
            prereg_payload=payload,
            preregistration_sha256=prereg_receipt["preregistration_sha256"],
            expected_holdout_open_count=0,
        )
    except ValueError as exc:
        raise RuntimeError("entry fitting blocked: machinery result envelope is invalid") from exc
    if (
        evidence.get("schema_version")
        != contracts.get("machinery_test_evidence_schema_version")
        or evidence.get("status") != "PASS"
        or evidence.get("preregistration_sha256")
        != prereg_receipt["preregistration_sha256"]
        or evidence.get("source_hash_policy_sha256") != policy_hash
    ):
        raise RuntimeError("entry fitting blocked: machinery test evidence is invalid")
    expected_commands = contracts.get("required_test_commands")
    commands = evidence.get("commands")
    if not isinstance(expected_commands, list) or not isinstance(commands, list):
        raise RuntimeError("entry fitting blocked: machinery test command contract absent")
    observed_testcases = _validate_registered_test_commands(commands, expected_commands)
    expected_check_names = list(contracts.get("lineage_evidence_checks", [])) + list(
        contracts.get("machinery_evidence_checks", [])
    )
    expected_checks = {name: "PASS" for name in expected_check_names}
    if evidence.get("checks") != expected_checks:
        raise RuntimeError("entry fitting blocked: machinery test evidence checks drifted")
    mapping = contracts.get("evidence_check_to_testcase")
    if (
        type(mapping) is not dict
        or set(mapping) != set(expected_check_names)
        or evidence.get("check_to_testcase") != mapping
        or any(type(value) is not str for value in mapping.values())
        or tuple(observed_testcases) != REGISTERED_MACHINERY_TEST_NAMES
        or not set(mapping.values()).issubset(set(observed_testcases))
        or any(observed_testcases.count(value) != 1 for value in mapping.values())
    ):
        raise RuntimeError(
            "entry fitting blocked: semantic check-to-JUnit attestation drift"
        )
    return evidence_path, evidence


def _validate_registered_test_commands(
    commands: list[Any], expected_commands: list[Any]
) -> list[str]:
    if len(commands) != len(expected_commands):
        raise RuntimeError("entry fitting blocked: machinery test command coverage mismatch")
    all_names: list[str] = []
    for observed, expected in zip(commands, expected_commands, strict=True):
        if not isinstance(observed, dict) or set(observed) != {
            "argv",
            "cwd",
            "env",
            "exit_code",
            "stdout_sha256",
            "stderr_sha256",
            "junit_path",
            "junit_sha256",
            "junit_summary",
        }:
            raise RuntimeError("entry fitting blocked: malformed machinery test command row")
        if any(observed.get(key) != expected.get(key) for key in ("argv", "cwd", "env")):
            raise RuntimeError("entry fitting blocked: unregistered machinery test command")
        if observed.get("exit_code") != 0 or any(
            type(observed.get(key)) is not str
            or re.fullmatch(r"[0-9a-f]{64}", observed[key]) is None
            for key in ("stdout_sha256", "stderr_sha256")
        ):
            raise RuntimeError("entry fitting blocked: machinery test command did not pass")
        if observed.get("junit_path") != expected.get("junit_path"):
            raise RuntimeError("entry fitting blocked: machinery JUnit path drift")
        junit_path = _canonical_repo_regular_file(observed["junit_path"])
        if (
            type(observed.get("junit_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", observed["junit_sha256"])
            is None
            or observed["junit_sha256"] != sha256_path(junit_path)
        ):
            raise RuntimeError("entry fitting blocked: machinery JUnit hash drift")
        try:
            root = ElementTree.parse(junit_path).getroot()
        except (ElementTree.ParseError, OSError) as exc:
            raise RuntimeError("entry fitting blocked: malformed machinery JUnit") from exc
        cases = root.findall(".//testcase")
        summary = {
            "tests": len(cases),
            "failures": sum(case.find("failure") is not None for case in cases),
            "errors": sum(case.find("error") is not None for case in cases),
            "skipped": sum(case.find("skipped") is not None for case in cases),
        }
        summary["passed"] = (
            summary["tests"]
            - summary["failures"]
            - summary["errors"]
            - summary["skipped"]
        )
        names = [str(case.attrib.get("name", "")) for case in cases]
        if (
            observed.get("junit_summary") != summary
            or summary != expected.get("expected_junit_summary")
            or names != expected.get("required_testcase_names")
            or len(names) != len(set(names))
        ):
            raise RuntimeError("entry fitting blocked: machinery JUnit evidence drift")
        all_names.extend(names)
    if len(all_names) != len(set(all_names)):
        raise RuntimeError("entry fitting blocked: duplicate testcase across JUnit commands")
    return all_names


def _validated_corpus_integrity_receipt(
    payload: dict[str, Any],
    prereg_receipt: dict[str, Any],
    machinery_receipt: dict[str, Any],
) -> dict[str, Any]:
    policy, policy_hash = _source_policy_and_hash(payload)
    contracts = policy["receipt_contracts"]
    receipt_spec = core_corpus_integrity_receipt_spec()
    try:
        receipt_path = _frozen_contract_file(
            payload,
            contract_key="corpus_integrity_receipt_path",
            expected_path=CORPUS_INTEGRITY_RECEIPT_PATH,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "entry fitting blocked: current core-corpus integrity receipt is absent"
        ) from exc
    receipt = read_json(receipt_path)
    if set(receipt) != set(receipt_spec["fields_in_order"]):
        raise RuntimeError(
            "entry fitting blocked: core-corpus integrity receipt field drift"
        )
    try:
        _validate_result_envelope_against(
            receipt,
            prereg_payload=payload,
            preregistration_sha256=prereg_receipt["preregistration_sha256"],
            expected_holdout_open_count=0,
        )
    except ValueError as exc:
        raise RuntimeError("entry fitting blocked: corpus receipt envelope is invalid") from exc
    manifest_contract = payload["corpus"]["integrity_contract"]
    frozen = manifest_contract["partitions"][CORE_INTEGRITY_PARTITION]
    if (
        receipt.get("schema_version")
        != contracts.get("corpus_integrity_receipt_schema_version")
        or receipt.get("status") != receipt_spec["status"]
        or receipt.get("partition_id") != CORE_INTEGRITY_PARTITION
        or receipt.get("source_hash_policy_sha256") != policy_hash
        or receipt.get("machinery_receipt_sha256")
        != sha256_path(ENTRY_MACHINERY_RECEIPT_PATH)
        or receipt.get("command") != contracts.get("required_integrity_command")
        or receipt.get("corpus_root") != str(CORPUS_ROOT)
        or receipt.get("integrity_contract") != frozen
        or receipt.get("manifest_contract_sha256")
        != stable_hash(manifest_contract)
        or receipt.get("verified_file_count") != frozen["file_count"]
        or receipt.get("verified_total_bytes") != frozen["total_bytes"]
        or receipt.get("verified_ordered_entries_semantic_sha256")
        != frozen["ordered_entries_semantic_sha256"]
        or receipt.get("diagnostic_files_verified") != 0
        or receipt.get("cache_files_verified") != 0
        or receipt.get("mismatches") != []
        or receipt.get("unreceipted_paths") != []
        or receipt.get("loader_rehash_required") is not True
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T[^\s]+Z",
            str(receipt.get("verification_completed_at_utc", "")),
        )
        is None
    ):
        raise RuntimeError("entry fitting blocked: corpus integrity receipt is invalid")
    verified_at = datetime.fromisoformat(
        str(receipt["verification_completed_at_utc"]).replace("Z", "+00:00")
    )
    machinery_at = datetime.fromisoformat(
        str(machinery_receipt["completed_at_utc"]).replace("Z", "+00:00")
    )
    prereg_at = datetime.fromisoformat(
        str(prereg_receipt["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if verified_at < machinery_at or verified_at < prereg_at:
        raise RuntimeError(
            "entry fitting blocked: corpus verification predates frozen machinery"
        )
    if receipt.get("machinery_receipt_sha256") != sha256_path(
        ENTRY_MACHINERY_RECEIPT_PATH
    ) or machinery_receipt.get("status") != "PASS":
        raise RuntimeError("entry fitting blocked: corpus receipt machinery binding drift")
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    return receipt


def assert_lineage_implementation_frozen() -> dict[str, Any]:
    """Block fitting until the exact frozen signed-17 implementation is receipted."""

    prereg_receipt = assert_preregistration_frozen()
    payload = read_json(PREREG_PATH)
    policy, policy_hash = _source_policy_and_hash(payload)
    contracts = policy.get("receipt_contracts", {})
    try:
        receipt_path = _frozen_contract_file(
            payload,
            contract_key="lineage_receipt_path",
            expected_path=LINEAGE_IMPLEMENTATION_RECEIPT_PATH,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "entry fitting blocked: lineage implementation receipt is absent"
        ) from exc
    receipt = read_json(receipt_path)
    try:
        _validate_result_envelope_against(
            receipt,
            prereg_payload=payload,
            preregistration_sha256=prereg_receipt["preregistration_sha256"],
            expected_holdout_open_count=0,
        )
    except ValueError as exc:
        raise RuntimeError("entry fitting blocked: lineage result envelope is invalid") from exc
    if receipt.get("schema_version") != contracts.get("lineage_schema_version"):
        raise RuntimeError("entry fitting blocked: lineage receipt schema drift")
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T[^\s]+Z", str(receipt.get("completed_at_utc", ""))
    ) is None:
        raise RuntimeError("entry fitting blocked: lineage receipt timestamp drift")
    if receipt.get("status") != "PASS" or receipt.get("training_allowed") is not True:
        raise RuntimeError("entry fitting blocked: lineage implementation did not pass")
    if receipt.get("preregistration_sha256") != prereg_receipt["preregistration_sha256"]:
        raise RuntimeError("entry fitting blocked: lineage receipt targets another preregistration")
    if receipt.get("feature_lineage_sha256") != sha256_path(LINEAGE_PATH):
        raise RuntimeError("entry fitting blocked: lineage receipt hash mismatch")
    if receipt.get("source_hash_policy_sha256") != policy_hash:
        raise RuntimeError("entry fitting blocked: lineage receipt source-policy mismatch")
    expected = policy.get("lineage_required_paths")
    if not isinstance(expected, list):
        raise RuntimeError("entry fitting blocked: frozen lineage path list is absent")
    _implementation_receipt_map(receipt, expected_paths=expected, owner="lineage")
    evidence_path, evidence = _validated_machinery_test_evidence(payload, prereg_receipt)
    if (
        receipt.get("test_evidence_path") != repo_path_label(evidence_path)
        or receipt.get("test_evidence_sha256") != sha256_path(evidence_path)
    ):
        raise RuntimeError("entry fitting blocked: lineage test-evidence binding drift")
    lineage_check_names = contracts.get("lineage_evidence_checks", [])
    expected_checks = {name: "PASS" for name in lineage_check_names}
    if receipt.get("checks") != expected_checks or any(
        evidence.get("checks", {}).get(name) != "PASS" for name in lineage_check_names
    ):
        raise RuntimeError("entry fitting blocked: lineage checks did not pass")
    return receipt


def _assert_entry_machinery_frozen(
    prereg_receipt: dict[str, Any], payload: dict[str, Any], lineage_receipt: dict[str, Any]
) -> dict[str, Any]:
    policy, policy_hash = _source_policy_and_hash(payload)
    contracts = policy.get("receipt_contracts", {})
    try:
        receipt_path = _frozen_contract_file(
            payload,
            contract_key="machinery_receipt_path",
            expected_path=ENTRY_MACHINERY_RECEIPT_PATH,
        )
    except RuntimeError as exc:
        raise RuntimeError("entry fitting blocked: entry machinery receipt is absent") from exc
    receipt = read_json(receipt_path)
    try:
        _validate_result_envelope_against(
            receipt,
            prereg_payload=payload,
            preregistration_sha256=prereg_receipt["preregistration_sha256"],
            expected_holdout_open_count=0,
        )
    except ValueError as exc:
        raise RuntimeError("entry fitting blocked: machinery result envelope is invalid") from exc
    if receipt.get("schema_version") != contracts.get("machinery_schema_version"):
        raise RuntimeError("entry fitting blocked: machinery receipt schema drift")
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T[^\s]+Z", str(receipt.get("completed_at_utc", ""))
    ) is None:
        raise RuntimeError("entry fitting blocked: machinery receipt timestamp drift")
    if receipt.get("status") != "PASS" or receipt.get("fit_allowed") is not True:
        raise RuntimeError("entry fitting blocked: entry machinery did not pass")
    if receipt.get("preregistration_sha256") != prereg_receipt["preregistration_sha256"]:
        raise RuntimeError("entry fitting blocked: machinery receipt targets another preregistration")
    if receipt.get("source_hash_policy_sha256") != policy_hash:
        raise RuntimeError("entry fitting blocked: machinery receipt source-policy mismatch")
    if receipt.get("lineage_receipt_sha256") != sha256_path(
        LINEAGE_IMPLEMENTATION_RECEIPT_PATH
    ):
        raise RuntimeError("entry fitting blocked: machinery lineage receipt mismatch")
    expected = policy.get("machinery_required_paths")
    if not isinstance(expected, list):
        raise RuntimeError("entry fitting blocked: frozen machinery path list is absent")
    _implementation_receipt_map(receipt, expected_paths=expected, owner="machinery")

    evidence_path, evidence = _validated_machinery_test_evidence(payload, prereg_receipt)
    if receipt.get("test_evidence_path") != repo_path_label(evidence_path):
        raise RuntimeError("entry fitting blocked: machinery test-evidence path mismatch")
    if receipt.get("test_evidence_sha256") != sha256_path(evidence_path):
        raise RuntimeError("entry fitting blocked: machinery test-evidence hash drift")
    machinery_check_names = contracts.get("machinery_evidence_checks", [])
    expected_checks = {name: "PASS" for name in machinery_check_names}
    if receipt.get("checks") != expected_checks or any(
        evidence.get("checks", {}).get(name) != "PASS"
        for name in machinery_check_names
    ):
        raise RuntimeError("entry fitting blocked: machinery checks did not pass")
    return receipt


def _resolve_frozen_fit_sessions(
    assignments: dict[str, Any],
    *,
    role: str,
    outer_fold: int | None,
    inner_fold: int | None,
) -> tuple[str, ...]:
    roles = {
        "outer_weights",
        "outer_calibration",
        "nested_weights",
        "nested_calibration",
        "full_weights",
        "full_calibration",
    }
    if role not in roles:
        raise RuntimeError(f"entry fitting blocked: unsupported frozen fit role: {role}")
    if role.startswith("full_"):
        if outer_fold is not None or inner_fold is not None:
            raise RuntimeError("entry fitting blocked: full role cannot name a fold")
        full = assignments["full_fit_pre_holdout_bundle"]
        key = "entry_model_fit" if role == "full_weights" else "entry_calibration"
        values = full[key]
    else:
        if type(outer_fold) is not int or not 1 <= outer_fold <= 5:
            raise RuntimeError("entry fitting blocked: outer fold must be an integer 1..5")
        outer = assignments["folds"][outer_fold - 1]
        if outer.get("fold") != outer_fold:
            raise RuntimeError("entry fitting blocked: frozen outer-fold identity drift")
        if role.startswith("outer_"):
            if inner_fold is not None:
                raise RuntimeError("entry fitting blocked: outer role cannot name an inner fold")
            key = "model_fit" if role == "outer_weights" else "calibration_last_20_percent"
            values = outer[key]
        else:
            if type(inner_fold) is not int or not 1 <= inner_fold <= 4:
                raise RuntimeError("entry fitting blocked: inner fold must be an integer 1..4")
            rows = outer["inner_forward_folds"]["scored_forward_folds"]
            matches = [row for row in rows if row.get("inner_fold") == inner_fold]
            if len(matches) != 1:
                raise RuntimeError("entry fitting blocked: frozen inner-fold identity drift")
            inner = matches[0]
            if not inner.get("calibration_valid"):
                raise RuntimeError(
                    "entry fitting blocked: nested bundle has fewer than 10 calibration sessions"
                )
            key = "model_fit" if role == "nested_weights" else "calibration"
            values = inner[key]

    sessions = tuple(str(value) for value in values)
    if (
        not sessions
        or len(sessions) != len(set(sessions))
        or any(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None for value in sessions)
    ):
        raise RuntimeError("entry fitting blocked: frozen fit sessions are malformed")
    forbidden_groups = {
        "firewall": set(assignments["firewall_raw_indices_216_251"]),
        "burned": set(assignments["burned_smoke_dates"]),
        "protected": set(assignments["protected_holdout_30"]),
    }
    for name, forbidden in forbidden_groups.items():
        overlap = sorted(set(sessions).intersection(forbidden))
        if overlap:
            raise RuntimeError(
                f"entry fitting blocked: {name} session entered frozen fit role: {overlap}"
            )
    if list(sessions) != sorted(sessions):
        raise RuntimeError("entry fitting blocked: frozen fit sessions are malformed")
    pre_holdout = set(assignments["pre_holdout_session_indices_1_215"])
    if not set(sessions).issubset(pre_holdout):
        raise RuntimeError("entry fitting blocked: fit role left the frozen pre-holdout era")
    return sessions


def _resolve_frozen_evidence_sessions(
    assignments: dict[str, Any], *, role: str, outer_fold: int | None,
    inner_fold: int | None,
) -> tuple[str, ...]:
    if role == "nested_validation":
        if type(outer_fold) is not int or not 1 <= outer_fold <= 5:
            raise RuntimeError("entry evidence blocked: nested role needs outer fold 1..5")
        if type(inner_fold) is not int or not 1 <= inner_fold <= 4:
            raise RuntimeError("entry evidence blocked: nested role needs inner fold 1..4")
        rows = assignments["folds"][outer_fold - 1]["inner_forward_folds"]["scored_forward_folds"]
        matches = [row for row in rows if row.get("inner_fold") == inner_fold]
        if len(matches) != 1 or not matches[0].get("calibration_valid"):
            raise RuntimeError("entry evidence blocked: nested validation identity invalid")
        values = matches[0]["validation"]
    elif role in {"outer_test_primary", "outer_test_shortened_diagnostic"}:
        if inner_fold is not None:
            raise RuntimeError("entry evidence blocked: outer role cannot name inner fold")
        if type(outer_fold) is not int or not 1 <= outer_fold <= 5:
            raise RuntimeError("entry evidence blocked: outer fold must be integer 1..5")
        fold = assignments["folds"][outer_fold - 1]
        if fold.get("fold") != outer_fold:
            raise RuntimeError("entry evidence blocked: frozen outer-fold identity drift")
        key = (
            "outer_test_primary_1555_complete"
            if role == "outer_test_primary"
            else "outer_test_shortened_diagnostic_only"
        )
        values = fold[key]
        if role == "outer_test_shortened_diagnostic" and not values:
            raise RuntimeError("entry evidence blocked: requested shortened diagnostic is empty")
    else:
        raise RuntimeError(f"entry evidence blocked: unsupported evidence role: {role}")
    sessions = tuple(str(value) for value in values)
    if (
        not sessions
        or len(sessions) != len(set(sessions))
        or list(sessions) != sorted(sessions)
        or any(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None for value in sessions)
    ):
        raise RuntimeError("entry evidence blocked: frozen evidence sessions are malformed")
    firewall = set(assignments["firewall_raw_indices_216_251"])
    if firewall.intersection(sessions):
        raise RuntimeError("entry evidence blocked: firewall session entered evidence role")
    shortened = set(assignments["scheduled_short_sessions_diagnostic_only"])
    overlap = shortened.intersection(sessions)
    if role == "outer_test_shortened_diagnostic":
        if overlap != set(sessions):
            raise RuntimeError("entry evidence blocked: diagnostic role contains primary session")
    elif role != "nested_validation" and overlap:
        raise RuntimeError("entry evidence blocked: shortened session entered primary evidence")
    return sessions


def _validate_self_hashed_receipt(receipt: dict[str, Any], *, field: str) -> None:
    digest = receipt.get(field)
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError("entry evidence blocked: malformed self-hashed receipt")
    semantic = dict(receipt)
    semantic.pop(field, None)
    if stable_hash(semantic) != digest:
        raise RuntimeError("entry evidence blocked: receipt semantic hash mismatch")


def _nested_artifact_names(inner_fold: int) -> tuple[str, ...]:
    prefix = f"nested_inner_{inner_fold}"
    return (
        f"{prefix}_hgb_model_bundle.json",
        f"{prefix}_hgb_calibration_bundle.json",
        f"{prefix}_hgb_composer.json",
        f"{prefix}_neural_model_bundle.json",
        f"{prefix}_neural_calibration_bundle.json",
        f"{prefix}_neural_composer.json",
        f"{prefix}_replay_config.json",
    )


def _nested_role_record(
    assignments: dict[str, Any], outer_fold: int, inner_fold: int
) -> dict[str, Any]:
    if type(outer_fold) is not int or not 1 <= outer_fold <= 5:
        raise RuntimeError("entry evidence blocked: invalid outer fold")
    if type(inner_fold) is not int or not 1 <= inner_fold <= 4:
        raise RuntimeError("entry evidence blocked: invalid inner fold")
    rows = assignments["folds"][outer_fold - 1]["inner_forward_folds"][
        "scored_forward_folds"
    ]
    matches = [row for row in rows if row.get("inner_fold") == inner_fold]
    if len(matches) != 1:
        raise RuntimeError("entry evidence blocked: nested role identity drift")
    return matches[0]


def _validate_nested_skip_receipt_one(
    payload: dict[str, Any], assignments: dict[str, Any], outer_fold: int,
    inner_fold: int,
) -> dict[str, Any]:
    role = _nested_role_record(assignments, outer_fold, inner_fold)
    if role.get("calibration_valid") is not False:
        raise RuntimeError("entry evidence blocked: valid nested block was skipped")
    skip_path = _canonical_repo_regular_file(
        repo_path_label(
            _outer_fold_artifact_path(
                outer_fold, f"nested_inner_{inner_fold}_skip_receipt.json"
            )
        )
    )
    receipt = read_json(skip_path)
    expected_keys = {
        "schema_version", "status", "frozen_at_utc", "outer_fold", "inner_fold",
        "preregistration_sha256", "session_assignments_sha256",
        "source_hash_policy_sha256", "weights_sessions_sha256_newline",
        "calibration_sessions_sha256_newline", "validation_sessions_sha256_newline",
        "calibration_session_count", "calibration_minimum_sessions", "reason",
        "validation_access_count", "dataset_opened", "result_generated",
        "quarantine_labels", "claim_boundary", "holdout_caveat",
        "plan_sha256", "fill_law_hash", "holdout_open_count", "receipt_sha256",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version")
        != "pathd.entry_nested_insufficient_calibration_skip.v1"
        or receipt.get("status") != "SKIPPED_INSUFFICIENT_CALIBRATION"
        or type(receipt.get("outer_fold")) is not int
        or receipt.get("outer_fold") != outer_fold
        or type(receipt.get("inner_fold")) is not int
        or receipt.get("inner_fold") != inner_fold
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T[^\s]+Z", str(receipt.get("frozen_at_utc", ""))
        )
        is None
        or receipt.get("preregistration_sha256") != sha256_path(PREREG_PATH)
        or receipt.get("session_assignments_sha256") != sha256_path(SESSION_PATH)
        or receipt.get("source_hash_policy_sha256")
        != stable_hash(payload["source_hash_policy"])
        or receipt.get("weights_sessions_sha256_newline")
        != role["model_fit_sha256_newline"]
        or receipt.get("calibration_sessions_sha256_newline")
        != role["calibration_sha256_newline"]
        or receipt.get("validation_sessions_sha256_newline")
        != role["validation_sha256_newline"]
        or type(receipt.get("calibration_session_count")) is not int
        or receipt.get("calibration_session_count") != len(role["calibration"])
        or type(receipt.get("calibration_minimum_sessions")) is not int
        or receipt.get("calibration_minimum_sessions")
        != role["calibration_minimum_sessions"]
        or receipt.get("reason") != "CALIBRATION_SESSIONS_LT_10"
        or type(receipt.get("validation_access_count")) is not int
        or receipt.get("validation_access_count") != 0
        or receipt.get("dataset_opened") is not False
        or receipt.get("result_generated") is not False
        or receipt.get("quarantine_labels") != list(QUARANTINE_LABELS)
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
        or receipt.get("holdout_caveat") != HOLDOUT_CAVEAT
        or receipt.get("plan_sha256") != payload["binding_plan"]["sha256"]
        or receipt.get("fill_law_hash") != payload["fill_law"]["fill_law_hash"]
        or type(receipt.get("holdout_open_count")) is not int
        or receipt.get("holdout_open_count") != 0
    ):
        raise RuntimeError("entry evidence blocked: invalid nested skip receipt")
    prefix = f"nested_inner_{inner_fold}_"
    allowed = {f"{prefix}skip_receipt.json"}
    fold_dir = _outer_fold_artifact_path(outer_fold, "placeholder").parent
    if fold_dir.exists():
        unexpected = [
            path.name
            for path in fold_dir.iterdir()
            if path.name.startswith(prefix) and path.name not in allowed
        ]
        if unexpected:
            raise RuntimeError(
                "entry evidence blocked: skipped nested block has data/result artifacts"
            )
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    return receipt


def _validate_nested_block_receipt_one(
    payload: dict[str, Any], assignments: dict[str, Any], outer_fold: int,
    inner_fold: int,
) -> tuple[Path, dict[str, Any]]:
    role = _nested_role_record(assignments, outer_fold, inner_fold)
    if role.get("calibration_valid") is True:
        value = _validate_nested_result_receipt_one(
            payload, assignments, outer_fold, inner_fold
        )
        path = _outer_fold_artifact_path(
            outer_fold, f"nested_inner_{inner_fold}_result_receipt.json"
        )
    else:
        value = _validate_nested_skip_receipt_one(
            payload, assignments, outer_fold, inner_fold
        )
        path = _outer_fold_artifact_path(
            outer_fold, f"nested_inner_{inner_fold}_skip_receipt.json"
        )
    return path, value


def _validate_nested_preopen_receipt_one(
    payload: dict[str, Any], assignments: dict[str, Any], outer_fold: int, inner_fold: int
) -> dict[str, Any]:
    path = _canonical_repo_regular_file(
        repo_path_label(
            _outer_fold_artifact_path(
                outer_fold, f"nested_inner_{inner_fold}_preopen_receipt.json"
            )
        )
    )
    receipt = read_json(path)
    expected_keys = {
        "schema_version", "status", "frozen_at_utc", "outer_fold", "inner_fold",
        "preregistration_sha256", "session_assignments_sha256",
        "source_hash_policy_sha256", "weights_sessions_sha256_newline",
        "calibration_sessions_sha256_newline", "validation_sessions_sha256_newline",
        "artifacts", "fit_environment_sha256", "validation_access_count",
        "quarantine_labels", "claim_boundary", "holdout_caveat",
        "plan_sha256", "fill_law_hash", "holdout_open_count", "receipt_sha256",
    }
    rows = assignments["folds"][outer_fold - 1]["inner_forward_folds"]["scored_forward_folds"]
    matches = [row for row in rows if row.get("inner_fold") == inner_fold]
    if len(matches) != 1:
        raise RuntimeError("entry evidence blocked: nested role identity drift")
    role = matches[0]
    if role.get("calibration_valid") is not True:
        raise RuntimeError("entry evidence blocked: invalid nested block has preopen artifacts")
    for prior in range(1, inner_fold):
        _validate_nested_block_receipt_one(
            payload, assignments, outer_fold, prior
        )
    artifacts: dict[str, dict[str, str]] = {}
    for name in _nested_artifact_names(inner_fold):
        artifact_path = _outer_fold_artifact_path(outer_fold, name)
        resolved = _canonical_repo_regular_file(repo_path_label(artifact_path))
        artifacts[name] = {
            "path": repo_path_label(artifact_path),
            "sha256": sha256_path(resolved),
        }
    from v4.research import pathd_entry_models as entry_models
    prefix = f"nested_inner_{inner_fold}"
    for family_prefix in ("hgb", "neural"):
        model = read_json(
            _outer_fold_artifact_path(
                outer_fold, f"{prefix}_{family_prefix}_model_bundle.json"
            )
        )
        calibration = read_json(
            _outer_fold_artifact_path(
                outer_fold, f"{prefix}_{family_prefix}_calibration_bundle.json"
            )
        )
        composer = read_json(
            _outer_fold_artifact_path(
                outer_fold, f"{prefix}_{family_prefix}_composer.json"
            )
        )
        entry_models.validate_entry_model_bundle(model)
        entry_models.validate_entry_calibration_bundle(
            calibration, model_bundle=model
        )
        entry_models.validate_entry_composer_bundle(
            composer, model_bundle=model, calibration_bundle=calibration
        )
    nested_replay = read_json(
        _outer_fold_artifact_path(outer_fold, f"{prefix}_replay_config.json")
    )
    entry_models.validate_entry_nested_replay_config(nested_replay)
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != "pathd.entry_nested_preopen_receipt.v1"
        or receipt.get("status") != "FROZEN_BEFORE_NESTED_OPEN"
        or receipt.get("outer_fold") != outer_fold
        or receipt.get("inner_fold") != inner_fold
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T[^\s]+Z", str(receipt.get("frozen_at_utc", ""))) is None
        or receipt.get("preregistration_sha256") != sha256_path(PREREG_PATH)
        or receipt.get("session_assignments_sha256") != sha256_path(SESSION_PATH)
        or receipt.get("source_hash_policy_sha256") != stable_hash(payload["source_hash_policy"])
        or receipt.get("weights_sessions_sha256_newline") != canonical_session_hash(role["model_fit"])
        or receipt.get("calibration_sessions_sha256_newline") != canonical_session_hash(role["calibration"])
        or receipt.get("validation_sessions_sha256_newline") != canonical_session_hash(role["validation"])
        or receipt.get("artifacts") != artifacts
        or receipt.get("fit_environment_sha256") != stable_hash(assert_entry_fit_environment_current())
        or receipt.get("validation_access_count") != 0
        or type(receipt.get("validation_access_count")) is not int
        or receipt.get("quarantine_labels") != list(QUARANTINE_LABELS)
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
        or receipt.get("holdout_caveat") != HOLDOUT_CAVEAT
        or receipt.get("plan_sha256") != payload["binding_plan"]["sha256"]
        or receipt.get("fill_law_hash") != payload["fill_law"]["fill_law_hash"]
        or receipt.get("holdout_open_count") != 0
        or type(receipt.get("holdout_open_count")) is not int
    ):
        raise RuntimeError("entry evidence blocked: invalid nested preopen receipt")
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    return receipt


def _validate_policy_evaluation_dict(
    value: dict[str, Any], *, authorization: FrozenEvidenceAuthorization,
    dataset_sha256: str,
    sessions: tuple[str, ...], role: str, outer_fold: int, inner_fold: int | None,
) -> None:
    expected_fields = set(
        entry_future_api_contract()["dataclass_fields"]["EntryPolicyEvaluationV1"]
    )
    authorization_sha256 = stable_hash(authorization.to_dict())
    _assert_strict_json_value(value)
    if set(value) != expected_fields:
        raise RuntimeError("entry evidence blocked: policy evaluation schema drift")
    pnl = value.get("session_pnl_micros")
    terminal_journals = value.get("session_terminal_journal_sha256s")
    completed_ids = value.get("completed_trade_ids")
    completed_sessions = value.get("completed_trade_sessions")
    zero = value.get("zero_trade_sessions")
    terminal_journal = value.get("terminal_journal")
    metrics = value.get("metrics")
    semantic = dict(value)
    digest = semantic.pop("result_sha256", None)
    if (
        value.get("schema_version") != "pathd.entry_policy_evaluation.v1"
        or value.get("holdout_caveat") != HOLDOUT_CAVEAT
        or value.get("evidence_role") != role
        or value.get("outer_fold") != outer_fold
        or value.get("inner_fold") != inner_fold
        or value.get("authorization_sha256") != authorization_sha256
        or value.get("dataset_sha256") != dataset_sha256
        or value.get("sessions") != list(sessions)
        or value.get("sessions_sha256_newline") != canonical_session_hash(sessions)
        or type(pnl) is not list
        or len(pnl) != len(sessions)
        or any(type(item) is not int for item in pnl)
        or type(terminal_journals) is not list
        or len(terminal_journals) != len(sessions)
        or any(
            type(item) is not str or re.fullmatch(r"[0-9a-f]{64}", item) is None
            for item in terminal_journals
        )
        or type(completed_ids) is not list
        or any(type(item) is not str or not item for item in completed_ids)
        or len(completed_ids) != len(set(completed_ids))
        or type(completed_sessions) is not list
        or len(completed_sessions) != len(completed_ids)
        or any(session not in sessions for session in completed_sessions)
        or any(
            sessions.index(left) > sessions.index(right)
            for left, right in zip(completed_sessions, completed_sessions[1:])
        )
        or type(zero) is not list
        or zero
        != [session for session in sessions if session not in set(completed_sessions)]
        or type(terminal_journal) is not dict
        or terminal_journal.get("journal_root_sha256") != terminal_journals[-1]
        or value.get("terminal_journal_sha256") != terminal_journals[-1]
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("execution_trace_root_sha256", ""))
        )
        is None
        or type(metrics) is not dict
        or value.get("session_coverage_sha256")
        != stable_hash(
            {
                "sessions": list(sessions),
                "session_pnl_micros": pnl,
                "session_terminal_journal_sha256s": terminal_journals,
                "completed_trade_ids": completed_ids,
                "completed_trade_sessions": completed_sessions,
                "zero_trade_sessions": zero,
                "execution_trace_root_sha256": value.get(
                    "execution_trace_root_sha256"
                ),
            }
        )
        or digest != stable_hash(semantic)
    ):
        raise RuntimeError("entry evidence blocked: invalid policy evaluation")
    from v4.path_d.execution.research_replay import (
        reconstruct_entry_policy_evaluation_from_journal,
        validate_research_ledger_journal,
    )
    validate_research_ledger_journal(
        terminal_journal, authorization=authorization
    )
    reconstructed = reconstruct_entry_policy_evaluation_from_journal(
        terminal_journal,
        authorization=authorization,
        dataset_sha256=dataset_sha256,
        sessions=sessions,
        evidence_role=role,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        policy_id=value["policy_id"],
    )
    if type(reconstructed) is not dict or reconstructed != value:
        raise RuntimeError(
            "entry evidence blocked: policy evaluation does not reconstruct from journal"
        )


def _validate_policy_evaluation_dataset_membership(
    value: dict[str, Any], dataset_receipt: dict[str, Any]
) -> None:
    """Bind the current authorization segment to its sealed dataset order.

    Nested journals intentionally retain prior validated blocks so one account's
    cash history is continuous.  Dataset membership is therefore checked only
    after the unique GENESIS/AUTHORIZATION_HANDOFF boundary for the authorization
    at the journal tip; earlier transitions remain protected by full journal and
    frozen prior-result validation.
    """

    ordered = dataset_receipt.get("ordered_example_hashes")
    session_rows = dataset_receipt.get("session_example_hashes")
    terminal_journal = value.get("terminal_journal")
    transitions = (
        terminal_journal.get("transitions")
        if type(terminal_journal) is dict
        else None
    )
    authorization_sha256 = value.get("authorization_sha256")
    if (
        type(ordered) is not list
        or not ordered
        or len(ordered) != len(set(ordered))
        or any(
            type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in ordered
        )
        or type(session_rows) is not list
        or type(transitions) is not list
        or not transitions
        or type(authorization_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", authorization_sha256) is None
    ):
        raise RuntimeError(
            "entry evidence blocked: policy journal dataset membership inputs drifted"
        )
    control_schema = "pathd.research_ledger_control_event.v1"
    boundaries: list[int] = []
    for index, transition in enumerate(transitions):
        if (
            type(transition) is not dict
            or transition.get("event_schema_version") != control_schema
            or transition.get("next_authorization_sha256")
            != authorization_sha256
        ):
            continue
        event_payload = transition.get("event_payload")
        kind = event_payload.get("event_kind") if type(event_payload) is dict else None
        if kind == "GENESIS":
            if index != 0 or transition.get("prior_authorization_sha256") is not None:
                raise RuntimeError(
                    "entry evidence blocked: policy journal genesis boundary drifted"
                )
            boundaries.append(index)
        elif kind == "AUTHORIZATION_HANDOFF":
            if transition.get("prior_authorization_sha256") == authorization_sha256:
                raise RuntimeError(
                    "entry evidence blocked: policy journal handoff did not change authority"
                )
            boundaries.append(index)
    if len(boundaries) != 1:
        raise RuntimeError(
            "entry evidence blocked: policy journal lacks one current authorization boundary"
        )
    current_transitions = transitions[boundaries[0] + 1 :]
    if any(
        transition.get("prior_authorization_sha256") != authorization_sha256
        or transition.get("next_authorization_sha256") != authorization_sha256
        for transition in current_transitions
        if type(transition) is dict
    ):
        raise RuntimeError(
            "entry evidence blocked: policy journal authorization segment is not contiguous"
        )

    position = {digest: index for index, digest in enumerate(ordered)}
    example_session: dict[str, str] = {}
    expected_sessions: list[str] = []
    rebuilt: list[str] = []
    for row in session_rows:
        if (
            type(row) is not dict
            or set(row) != {"session", "ordered_example_hashes"}
            or type(row.get("session")) is not str
            or type(row.get("ordered_example_hashes")) is not list
            or not row["ordered_example_hashes"]
        ):
            raise RuntimeError(
                "entry evidence blocked: policy journal session membership drifted"
            )
        session = row["session"]
        expected_sessions.append(session)
        for digest in row["ordered_example_hashes"]:
            if digest in example_session or digest not in position:
                raise RuntimeError(
                    "entry evidence blocked: duplicate or unknown session example hash"
                )
            example_session[digest] = session
            rebuilt.append(digest)
    if rebuilt != ordered or expected_sessions != dataset_receipt.get("sessions"):
        raise RuntimeError(
            "entry evidence blocked: policy journal dataset order does not reconstruct"
        )

    coverage_schema = "pathd.entry_frame_coverage.v1"
    decision_schema = "pathd.entry_action_decision.v1"
    observation_schema = "pathd.entry_action_observation.v1"
    coverage: list[tuple[str, str]] = []
    decisions: list[str] = []
    observations: list[str] = []
    coverage_by_digest: dict[str, str] = {}
    decision_seen: set[str] = set()
    observation_seen: set[str] = set()
    for transition in current_transitions:
        if type(transition) is not dict:
            raise RuntimeError(
                "entry evidence blocked: policy journal transition is not typed"
            )
        schema = transition.get("event_schema_version")
        event_payload = transition.get("event_payload")
        if type(event_payload) is not dict:
            raise RuntimeError(
                "entry evidence blocked: policy journal event payload is absent"
            )
        has_example_field = (
            "example_sha256" in event_payload
            or "entry_example_sha256" in event_payload
        )
        if schema in {coverage_schema, decision_schema, observation_schema}:
            digest = event_payload.get("example_sha256")
            if type(digest) is not str or digest not in position:
                raise RuntimeError(
                    "entry evidence blocked: journal references an unsealed example"
                )
            if schema == coverage_schema:
                disposition = event_payload.get("disposition")
                if disposition not in {"ACTION_DECISION", "STATE_INELIGIBLE"}:
                    raise RuntimeError(
                        "entry evidence blocked: invalid frame coverage disposition"
                    )
                if digest in coverage_by_digest:
                    raise RuntimeError(
                        "entry evidence blocked: duplicate frame coverage"
                    )
                coverage_by_digest[digest] = disposition
                coverage.append((digest, disposition))
            elif schema == decision_schema:
                if (
                    coverage_by_digest.get(digest) != "ACTION_DECISION"
                    or digest in decision_seen
                ):
                    raise RuntimeError(
                        "entry evidence blocked: action decision precedes or disagrees with coverage"
                    )
                decision_seen.add(digest)
                decisions.append(digest)
            else:
                if digest not in decision_seen or digest in observation_seen:
                    raise RuntimeError(
                        "entry evidence blocked: action observation precedes or duplicates its decision"
                    )
                observation_seen.add(digest)
                observations.append(digest)
        elif has_example_field:
            raise RuntimeError(
                "entry evidence blocked: example reference used by an unexpected event"
            )
    if (
        [digest for digest, _disposition in coverage] != ordered
        or decisions
        != [
            digest
            for digest, disposition in coverage
            if disposition == "ACTION_DECISION"
        ]
        or observations != decisions
    ):
        raise RuntimeError(
            "entry evidence blocked: policy decisions do not match sealed dataset order"
        )


def _validate_entry_evidence_dataset_receipt(
    path: Path, *, authorization: FrozenEvidenceAuthorization,
) -> dict[str, Any]:
    receipt = read_json(_canonical_repo_regular_file(repo_path_label(path)))
    _assert_strict_json_value(receipt)
    expected_keys = {
        "schema_version", "status", "created_at_utc", "role", "outer_fold",
        "inner_fold", "authorization_sha256", "sessions",
        "sessions_sha256_newline", "source_receipts", "source_receipts_root_sha256",
        "session_example_hashes", "ordered_example_hashes", "dataset_sha256",
        "quarantine_labels", "claim_boundary", "holdout_caveat", "plan_sha256",
        "preregistration_sha256", "fill_law_hash", "holdout_open_count",
        "receipt_sha256",
    }
    authorization_sha256 = stable_hash(authorization.to_dict())
    source_receipts = receipt.get("source_receipts")
    session_example_hashes = receipt.get("session_example_hashes")
    ordered_example_hashes = receipt.get("ordered_example_hashes")
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version")
        != "pathd.entry_evidence_dataset_receipt.v1"
        or receipt.get("status") != "SEALED_BEFORE_RESULT"
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T[^\s]+Z", str(receipt.get("created_at_utc", ""))
        )
        is None
        or receipt.get("role") != authorization.role
        or receipt.get("outer_fold") != authorization.outer_fold
        or receipt.get("inner_fold") != authorization.inner_fold
        or receipt.get("authorization_sha256") != authorization_sha256
        or receipt.get("sessions") != list(authorization.sessions)
        or receipt.get("sessions_sha256_newline")
        != authorization.sessions_sha256_newline
        or type(source_receipts) is not list
        or type(session_example_hashes) is not list
        or any(type(row) is not dict for row in source_receipts)
        or any(type(row) is not dict for row in session_example_hashes)
        or type(ordered_example_hashes) is not list
        or any(
            type(value) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in ordered_example_hashes
        )
        or receipt.get("quarantine_labels") != list(QUARANTINE_LABELS)
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
        or receipt.get("holdout_caveat") != HOLDOUT_CAVEAT
        or receipt.get("plan_sha256") != sha256_path(PLAN_PATH)
        or receipt.get("preregistration_sha256") != sha256_path(PREREG_PATH)
        or receipt.get("fill_law_hash") != read_json(PREREG_PATH)["fill_law"]["fill_law_hash"]
        or type(receipt.get("holdout_open_count")) is not int
        or receipt.get("holdout_open_count") != 0
    ):
        raise RuntimeError("entry evidence blocked: evidence dataset receipt drift")
    if [row.get("session") for row in source_receipts] != list(
        authorization.sessions
    ) or [row.get("session") for row in session_example_hashes] != list(
        authorization.sessions
    ):
        raise RuntimeError("entry evidence blocked: dataset session receipt coverage drift")
    rebuilt_ordered: list[str] = []
    for source_receipt, example_row in zip(
        source_receipts, session_example_hashes, strict=True
    ):
        hashes = (
            example_row.get("ordered_example_hashes")
            if type(example_row) is dict
            else None
        )
        source_files = (
            source_receipt.get("source_files")
            if type(source_receipt) is dict
            else None
        )
        if (
            type(example_row) is not dict
            or set(example_row) != {"session", "ordered_example_hashes"}
            or type(source_receipt) is not dict
            or set(source_receipt)
            != {
                "session", "source_files", "source_files_sha256",
                "example_count", "ordered_example_list_sha256",
                "session_content_sha256", "receipt_sha256",
            }
            or type(hashes) is not list
            or not hashes
            or any(
                type(value) is not str
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in hashes
            )
            or type(source_files) is not list
            or any(type(row) is not dict for row in source_files)
            or source_files != sorted(source_files, key=lambda row: row.get("relative_path"))
            or len(source_files)
            != len({row.get("relative_path") for row in source_files})
            or any(
                set(row) != {"relative_path", "bytes", "sha256"}
                or type(row.get("relative_path")) is not str
                or type(row.get("bytes")) is not int
                or row["bytes"] < 0
                or type(row.get("sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
                for row in source_files
            )
            or source_receipt.get("source_files_sha256") != stable_hash(source_files)
            or source_receipt.get("example_count") != len(hashes)
            or source_receipt.get("ordered_example_list_sha256") != stable_hash(hashes)
        ):
            raise RuntimeError("entry evidence blocked: source receipt seal drift")
        expected_session_content = stable_hash(
            {
                "schema_version": "pathd.entry_session.v1",
                "session": source_receipt["session"],
                "source_files": source_files,
                "source_files_sha256": source_receipt["source_files_sha256"],
                "ordered_example_hashes": hashes,
            }
        )
        if source_receipt.get("session_content_sha256") != expected_session_content:
            raise RuntimeError("entry evidence blocked: session content seal drift")
        _validate_self_hashed_receipt(source_receipt, field="receipt_sha256")
        rebuilt_ordered.extend(hashes)
    if rebuilt_ordered != ordered_example_hashes:
        raise RuntimeError("entry evidence blocked: ordered example coverage drift")
    if receipt.get("source_receipts_root_sha256") != stable_hash(source_receipts):
        raise RuntimeError("entry evidence blocked: source receipt root drift")
    expected_dataset_sha = stable_hash(
        {
            "schema_version": "pathd.entry_evidence_dataset.v1",
            "authorization_sha256": authorization_sha256,
            "role": authorization.role,
            "sessions": list(authorization.sessions),
            "sessions_sha256_newline": authorization.sessions_sha256_newline,
            "source_receipts": source_receipts,
            "ordered_example_hashes": ordered_example_hashes,
        }
    )
    if receipt.get("dataset_sha256") != expected_dataset_sha:
        raise RuntimeError("entry evidence blocked: evidence dataset digest drift")
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    return receipt


def _validate_nested_result_receipt_one(
    payload: dict[str, Any], assignments: dict[str, Any], outer_fold: int, inner_fold: int
) -> dict[str, Any]:
    role = _nested_role_record(assignments, outer_fold, inner_fold)
    if role.get("calibration_valid") is not True:
        raise RuntimeError("entry evidence blocked: invalid nested block has a result")
    for prior in range(1, inner_fold):
        _validate_nested_block_receipt_one(
            payload, assignments, outer_fold, prior
        )
    _validate_nested_preopen_receipt_one(payload, assignments, outer_fold, inner_fold)
    receipt_path = _canonical_repo_regular_file(
        repo_path_label(
            _outer_fold_artifact_path(
                outer_fold, f"nested_inner_{inner_fold}_result_receipt.json"
            )
        )
    )
    result_path = _canonical_repo_regular_file(
        repo_path_label(
            _outer_fold_artifact_path(outer_fold, f"nested_inner_{inner_fold}_result.json")
        )
    )
    dataset_receipt_path = _canonical_repo_regular_file(
        repo_path_label(
            _outer_fold_artifact_path(
                outer_fold, f"nested_inner_{inner_fold}_dataset_receipt.json"
            )
        )
    )
    receipt = read_json(receipt_path)
    result = read_json(result_path)
    sessions = tuple(role["validation"])
    from v4.research.pathd_evidence_gate import (
        read_frozen_entry_evidence_authorization,
    )
    expected_authorization = read_frozen_entry_evidence_authorization(
        role="nested_validation", outer_fold=outer_fold, inner_fold=inner_fold
    )
    expected_authorization_sha256 = stable_hash(expected_authorization.to_dict())
    dataset_receipt = _validate_entry_evidence_dataset_receipt(
        dataset_receipt_path, authorization=expected_authorization
    )
    expected_keys = {
        "schema_version", "status", "frozen_at_utc", "outer_fold", "inner_fold",
        "preopen_receipt_sha256", "access_receipt_path", "access_receipt_sha256",
        "validation_access_count", "authorization_sha256", "dataset_receipt_path",
        "dataset_receipt_sha256", "dataset_sha256", "result_path", "result_sha256",
        "source_receipts_root_sha256", "hgb_evaluation_sha256",
        "neural_evaluation_sha256", "quarantine_labels", "claim_boundary",
        "holdout_caveat", "plan_sha256", "fill_law_hash", "holdout_open_count",
        "receipt_sha256",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != "pathd.entry_nested_result_receipt.v1"
        or receipt.get("status") != "FROZEN_NESTED_VALIDATION_RESULT"
        or receipt.get("outer_fold") != outer_fold
        or receipt.get("inner_fold") != inner_fold
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T[^\s]+Z", str(receipt.get("frozen_at_utc", ""))) is None
        or receipt.get("preopen_receipt_sha256") != sha256_path(_outer_fold_artifact_path(outer_fold, f"nested_inner_{inner_fold}_preopen_receipt.json"))
        or receipt.get("access_receipt_path")
        != expected_authorization.access_receipt_path
        or receipt.get("access_receipt_sha256")
        != expected_authorization.access_receipt_sha256
        or type(receipt.get("validation_access_count")) is not int
        or receipt.get("validation_access_count") != 1
        or receipt.get("authorization_sha256") != expected_authorization_sha256
        or receipt.get("dataset_receipt_path") != repo_path_label(dataset_receipt_path)
        or receipt.get("dataset_receipt_sha256") != sha256_path(dataset_receipt_path)
        or dataset_receipt.get("authorization_sha256") != receipt.get("authorization_sha256")
        or dataset_receipt.get("dataset_sha256") != receipt.get("dataset_sha256")
        or dataset_receipt.get("sessions") != list(sessions)
        or dataset_receipt.get("sessions_sha256_newline") != canonical_session_hash(sessions)
        or receipt.get("source_receipts_root_sha256")
        != dataset_receipt.get("source_receipts_root_sha256")
        or receipt.get("result_path") != repo_path_label(result_path)
        or receipt.get("result_sha256") != sha256_path(result_path)
        or receipt.get("holdout_open_count") != 0
        or type(receipt.get("holdout_open_count")) is not int
        or receipt.get("quarantine_labels") != list(QUARANTINE_LABELS)
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
        or receipt.get("holdout_caveat") != HOLDOUT_CAVEAT
        or receipt.get("plan_sha256") != payload["binding_plan"]["sha256"]
        or receipt.get("fill_law_hash") != payload["fill_law"]["fill_law_hash"]
        or result.get("schema_version") != "pathd.entry_nested_family_evaluation.v1"
        or result.get("holdout_caveat") != HOLDOUT_CAVEAT
        or result.get("outer_fold") != outer_fold
        or result.get("inner_fold") != inner_fold
        or result.get("authorization_sha256") != receipt.get("authorization_sha256")
        or result.get("dataset_sha256") != receipt.get("dataset_sha256")
        or set(result) != {
            "schema_version", "holdout_caveat", "outer_fold", "inner_fold",
            "authorization_sha256",
            "dataset_sha256", "source_receipts_root_sha256",
            "access_receipt_sha256", "hgb", "neural", "result_sha256",
        }
        or result.get("source_receipts_root_sha256")
        != dataset_receipt.get("source_receipts_root_sha256")
        or result.get("access_receipt_sha256")
        != expected_authorization.access_receipt_sha256
    ):
        raise RuntimeError("entry evidence blocked: invalid nested result receipt")
    semantic_result = dict(result)
    result_digest = semantic_result.pop("result_sha256", None)
    if result_digest != stable_hash(semantic_result):
        raise RuntimeError("entry evidence blocked: nested result hash mismatch")
    for policy_id in ("HGB", "NEURAL"):
        evaluation = result.get(policy_id.lower())
        if type(evaluation) is not dict or evaluation.get("policy_id") != policy_id:
            raise RuntimeError("entry evidence blocked: nested policy identity drift")
        _validate_policy_evaluation_dict(
            evaluation,
            authorization=expected_authorization,
            dataset_sha256=receipt["dataset_sha256"],
            sessions=sessions,
            role="nested_validation",
            outer_fold=outer_fold,
            inner_fold=inner_fold,
        )
        _validate_policy_evaluation_dataset_membership(
            evaluation, dataset_receipt
        )
        expected_field = f"{policy_id.lower()}_evaluation_sha256"
        if receipt.get(expected_field) != evaluation.get("result_sha256"):
            raise RuntimeError("entry evidence blocked: nested evaluation hash drift")
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    return receipt


def _validate_outer_preopen_receipt_one(
    payload: dict[str, Any], assignments: dict[str, Any], outer_fold: int
) -> dict[str, Any]:
    for prior in range(1, outer_fold):
        _validate_outer_result_receipt_one(payload, assignments, prior)
    path = _canonical_repo_regular_file(
        repo_path_label(_outer_fold_artifact_path(outer_fold, "preopen_receipt.json"))
    )
    receipt = read_json(path)
    expected_keys = {
        "schema_version", "status", "frozen_at_utc", "outer_fold",
        "preregistration_sha256", "session_assignments_sha256",
        "source_hash_policy_sha256", "model_fit_sessions_sha256_newline",
        "calibration_sessions_sha256_newline", "primary_outer_sessions_sha256_newline",
        "shortened_diagnostic_sessions_sha256_newline", "nested_block_receipts",
        "artifacts", "selected_family",
        "selected_model_bundle_sha256", "selected_calibration_bundle_sha256",
        "selected_composer_sha256", "fit_environment_sha256",
        "outer_evidence_access_count", "quarantine_labels", "claim_boundary",
        "holdout_caveat", "plan_sha256", "fill_law_hash",
        "holdout_open_count", "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise RuntimeError("entry evidence blocked: preopen receipt schema drift")
    prereg_sha = sha256_path(PREREG_PATH)
    policy_hash = stable_hash(payload["source_hash_policy"])
    fold = assignments["folds"][outer_fold - 1]
    nested_rows: list[dict[str, Any]] = []
    for inner_fold in range(1, 5):
        nested_path, nested_receipt = _validate_nested_block_receipt_one(
            payload, assignments, outer_fold, inner_fold
        )
        digest = sha256_path(nested_path)
        nested_rows.append(
            {
                "inner_fold": inner_fold,
                "status": nested_receipt["status"],
                "path": repo_path_label(nested_path),
                "sha256": digest,
            }
        )
    expected_artifacts: dict[str, dict[str, str]] = {}
    for name in outer_evidence_open_gate_spec()["artifact_names"]:
        artifact_path = _outer_fold_artifact_path(outer_fold, name)
        resolved = _canonical_repo_regular_file(repo_path_label(artifact_path))
        expected_artifacts[name] = {
            "path": repo_path_label(artifact_path),
            "sha256": sha256_path(resolved),
        }
    from v4.research import pathd_entry_models as entry_models
    typed: dict[str, dict[str, Any]] = {
        name: read_json(_outer_fold_artifact_path(outer_fold, name))
        for name in outer_evidence_open_gate_spec()["artifact_names"]
        if name.endswith(".json")
    }
    for prefix in ("hgb", "neural"):
        model = typed[f"{prefix}_model_bundle.json"]
        calibration = typed[f"{prefix}_calibration_bundle.json"]
        composer = typed[f"{prefix}_composer.json"]
        entry_models.validate_entry_model_bundle(model)
        entry_models.validate_entry_calibration_bundle(
            calibration, model_bundle=model
        )
        entry_models.validate_entry_composer_bundle(
            composer, model_bundle=model, calibration_bundle=calibration
        )
    manifest = typed["negative_control_manifest.json"]
    entry_models.validate_entry_negative_control_manifest(manifest)
    replay_config = typed["control_replay_config.json"]
    entry_models.validate_entry_control_replay_config(replay_config)
    control_exit = typed["control_exit.json"]
    entry_models.validate_entry_control_exit_selection(control_exit)
    if (
        receipt.get("schema_version") != "pathd.entry_outer_preopen_receipt.v1"
        or receipt.get("status") != "FROZEN_BEFORE_OUTER_OPEN"
        or receipt.get("outer_fold") != outer_fold
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T[^\s]+Z", str(receipt.get("frozen_at_utc", ""))) is None
        or receipt.get("preregistration_sha256") != prereg_sha
        or receipt.get("session_assignments_sha256") != sha256_path(SESSION_PATH)
        or receipt.get("source_hash_policy_sha256") != policy_hash
        or receipt.get("model_fit_sessions_sha256_newline") != canonical_session_hash(fold["model_fit"])
        or receipt.get("calibration_sessions_sha256_newline") != canonical_session_hash(fold["calibration_last_20_percent"])
        or receipt.get("primary_outer_sessions_sha256_newline") != canonical_session_hash(fold["outer_test_primary_1555_complete"])
        or receipt.get("shortened_diagnostic_sessions_sha256_newline") != canonical_session_hash(fold["outer_test_shortened_diagnostic_only"])
        or receipt.get("nested_block_receipts") != nested_rows
        or receipt.get("artifacts") != expected_artifacts
        or receipt.get("selected_family")
        != payload["entry"]["global_family_freeze"]["selectable_candidate"]
        or manifest.get("outer_fold") != outer_fold
        or replay_config.get("outer_fold") != outer_fold
        or control_exit.get("outer_fold") != outer_fold
        or replay_config.get("control_exit_sha256")
        != expected_artifacts["control_exit.json"]["sha256"]
        or receipt.get("outer_evidence_access_count") != 0
        or type(receipt.get("outer_evidence_access_count")) is not int
        or receipt.get("quarantine_labels") != list(QUARANTINE_LABELS)
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
        or receipt.get("holdout_caveat") != HOLDOUT_CAVEAT
        or receipt.get("plan_sha256") != payload["binding_plan"]["sha256"]
        or receipt.get("fill_law_hash") != payload["fill_law"]["fill_law_hash"]
        or receipt.get("holdout_open_count") != 0
        or type(receipt.get("holdout_open_count")) is not int
    ):
        raise RuntimeError("entry evidence blocked: invalid preopen receipt")
    selected_prefix = "hgb"
    if (
        receipt.get("selected_model_bundle_sha256")
        != expected_artifacts[f"{selected_prefix}_model_bundle.json"]["sha256"]
        or receipt.get("selected_calibration_bundle_sha256")
        != expected_artifacts[f"{selected_prefix}_calibration_bundle.json"]["sha256"]
        or receipt.get("selected_composer_sha256")
        != expected_artifacts[f"{selected_prefix}_composer.json"]["sha256"]
        or receipt.get("fit_environment_sha256")
        != stable_hash(assert_entry_fit_environment_current())
    ):
        raise RuntimeError("entry evidence blocked: selected preopen artifact drift")
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    return receipt


def _validate_outer_result_receipt_one(
    payload: dict[str, Any], assignments: dict[str, Any], outer_fold: int
) -> dict[str, Any]:
    for prior in range(1, outer_fold):
        _validate_outer_result_receipt_one(payload, assignments, prior)
    preopen = _validate_outer_preopen_receipt_one(payload, assignments, outer_fold)
    receipt_path = _canonical_repo_regular_file(
        repo_path_label(_outer_fold_artifact_path(outer_fold, "outer_result_receipt.json"))
    )
    result_path = _canonical_repo_regular_file(
        repo_path_label(_outer_fold_artifact_path(outer_fold, "outer_primary_result.json"))
    )
    dataset_receipt_path = _canonical_repo_regular_file(
        repo_path_label(
            _outer_fold_artifact_path(outer_fold, "outer_primary_dataset_receipt.json")
        )
    )
    negative_panel_path = _canonical_repo_regular_file(
        repo_path_label(
            _outer_fold_artifact_path(outer_fold, "negative_control_panel.json")
        )
    )
    replay_result_path = _canonical_repo_regular_file(
        repo_path_label(
            _outer_fold_artifact_path(outer_fold, "control_replay_result.json")
        )
    )
    receipt = read_json(receipt_path)
    expected_keys = {
        "schema_version", "status", "frozen_at_utc", "outer_fold",
        "preregistration_sha256", "preopen_receipt_path", "preopen_receipt_sha256",
        "access_receipt_path", "access_receipt_sha256", "outer_evidence_access_count",
        "primary_sessions_sha256_newline", "primary_authorization_sha256",
        "dataset_receipt_path", "dataset_receipt_sha256", "primary_dataset_sha256",
        "source_receipts_root_sha256", "result_path", "result_sha256",
        "hgb_evaluation_sha256", "neural_evaluation_sha256", "selected_family",
        "negative_control_panel_path", "negative_control_panel_sha256",
        "control_replay_result_path", "control_replay_result_sha256",
        "quarantine_labels", "holdout_caveat", "claim_boundary", "plan_sha256",
        "fill_law_hash", "holdout_open_count", "receipt_sha256",
    }
    result = read_json(result_path)
    _assert_strict_json_value(result)
    fold = assignments["folds"][outer_fold - 1]
    open_gate_hashes = tuple(
        [
            sha256_path(_outer_fold_artifact_path(prior, "outer_result_receipt.json"))
            for prior in range(1, outer_fold)
        ]
        + [sha256_path(_outer_fold_artifact_path(outer_fold, "preopen_receipt.json"))]
    )
    sessions = tuple(fold["outer_test_primary_1555_complete"])
    from v4.research.pathd_evidence_gate import (
        read_frozen_entry_evidence_authorization,
    )
    expected_authorization = read_frozen_entry_evidence_authorization(
        role="outer_test_primary", outer_fold=outer_fold, inner_fold=None
    )
    if expected_authorization.open_gate_receipts_sha256 != open_gate_hashes:
        raise RuntimeError("entry evidence blocked: outer access gate receipt drift")
    expected_authorization_sha = stable_hash(expected_authorization.to_dict())
    dataset_receipt = _validate_entry_evidence_dataset_receipt(
        dataset_receipt_path, authorization=expected_authorization
    )
    negative_panel = read_json(negative_panel_path)
    negative_manifest = read_json(
        _outer_fold_artifact_path(outer_fold, "negative_control_manifest.json")
    )
    replay_config = read_json(
        _outer_fold_artifact_path(outer_fold, "control_replay_config.json")
    )
    replay_result = read_json(replay_result_path)
    _validate_entry_control_replay_result_immutable(
        replay_result,
        config=replay_config,
        authorization=expected_authorization,
        dataset_receipt=dataset_receipt,
    )
    _validate_entry_negative_control_panel_immutable(
        negative_panel,
        manifest=negative_manifest,
        authorization=expected_authorization,
        dataset_receipt=dataset_receipt,
    )
    envelope_fields = entry_future_api_contract()["dataclass_fields"][
        "ResearchResultEnvelopeV1"
    ]
    if set(result) != set(envelope_fields):
        raise RuntimeError("entry evidence blocked: outer result envelope schema drift")
    payload_value = result.get("payload")
    if type(payload_value) is not dict:
        raise RuntimeError("entry evidence blocked: outer result payload is malformed")
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != "pathd.entry_outer_result_receipt.v1"
        or receipt.get("status") != "FROZEN_PRIMARY_OUTER_RESULT"
        or receipt.get("outer_fold") != outer_fold
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T[^\s]+Z", str(receipt.get("frozen_at_utc", ""))) is None
        or receipt.get("preregistration_sha256") != sha256_path(PREREG_PATH)
        or receipt.get("preopen_receipt_path") != repo_path_label(_outer_fold_artifact_path(outer_fold, "preopen_receipt.json"))
        or receipt.get("preopen_receipt_sha256") != sha256_path(_outer_fold_artifact_path(outer_fold, "preopen_receipt.json"))
        or receipt.get("access_receipt_path")
        != expected_authorization.access_receipt_path
        or receipt.get("access_receipt_sha256")
        != expected_authorization.access_receipt_sha256
        or type(receipt.get("outer_evidence_access_count")) is not int
        or receipt.get("outer_evidence_access_count") != 1
        or receipt.get("primary_sessions_sha256_newline") != canonical_session_hash(fold["outer_test_primary_1555_complete"])
        or receipt.get("primary_authorization_sha256") != expected_authorization_sha
        or receipt.get("dataset_receipt_path") != repo_path_label(dataset_receipt_path)
        or receipt.get("dataset_receipt_sha256") != sha256_path(dataset_receipt_path)
        or receipt.get("primary_dataset_sha256")
        != dataset_receipt.get("dataset_sha256")
        or receipt.get("source_receipts_root_sha256")
        != dataset_receipt.get("source_receipts_root_sha256")
        or receipt.get("result_path") != repo_path_label(_outer_fold_artifact_path(outer_fold, "outer_primary_result.json"))
        or receipt.get("result_sha256") != sha256_path(result_path)
        or receipt.get("negative_control_panel_path")
        != repo_path_label(negative_panel_path)
        or receipt.get("negative_control_panel_sha256")
        != sha256_path(negative_panel_path)
        or receipt.get("control_replay_result_path")
        != repo_path_label(replay_result_path)
        or receipt.get("control_replay_result_sha256")
        != sha256_path(replay_result_path)
        or receipt.get("selected_family") != preopen["selected_family"]
        or receipt.get("quarantine_labels") != list(QUARANTINE_LABELS)
        or receipt.get("holdout_caveat") != HOLDOUT_CAVEAT
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
        or receipt.get("plan_sha256") != payload["binding_plan"]["sha256"]
        or receipt.get("fill_law_hash") != payload["fill_law"]["fill_law_hash"]
        or receipt.get("holdout_open_count") != 0
        or type(receipt.get("holdout_open_count")) is not int
        or result.get("dataset_sha256") != receipt.get("primary_dataset_sha256")
        or result.get("authorization_sha256") != expected_authorization_sha
        or result.get("sessions_sha256_newline") != canonical_session_hash(sessions)
        or result.get("preregistration_sha256") != sha256_path(PREREG_PATH)
        or result.get("plan_sha256") != payload["binding_plan"]["sha256"]
        or result.get("fill_law_hash") != payload["fill_law"]["fill_law_hash"]
        or result.get("quarantine_labels") != list(QUARANTINE_LABELS)
        or result.get("holdout_caveat") != HOLDOUT_CAVEAT
        or result.get("claim_boundary") != CLAIM_BOUNDARY
        or result.get("holdout_open_count") != 0
        or type(result.get("holdout_open_count")) is not int
        or result.get("payload_sha256") != stable_hash(payload_value)
        or set(payload_value)
        != set(
            entry_future_api_contract()["dataclass_fields"][
                "EntryOuterPrimaryResultV1"
            ]
        )
        or payload_value.get("schema_version")
        != "pathd.entry_outer_primary_result.v1"
        or payload_value.get("selected_family") != preopen["selected_family"]
        or payload_value.get("outer_fold") != outer_fold
        or payload_value.get("evidence_role") != "outer_test_primary"
        or payload_value.get("authorization_sha256") != expected_authorization_sha
        or payload_value.get("dataset_sha256")
        != dataset_receipt.get("dataset_sha256")
        or payload_value.get("source_receipts_root_sha256")
        != dataset_receipt.get("source_receipts_root_sha256")
        or payload_value.get("access_receipt_sha256")
        != expected_authorization.access_receipt_sha256
        or payload_value.get("negative_control_panel_sha256")
        != sha256_path(negative_panel_path)
        or payload_value.get("control_replay_result_sha256")
        != sha256_path(replay_result_path)
        or replay_result.get("candidate_outer_evaluation_sha256")
        != payload_value.get("hgb", {}).get("result_sha256")
        or replay_result.get("neural_outer_evaluation_sha256")
        != payload_value.get("neural", {}).get("result_sha256")
    ):
        raise RuntimeError("entry evidence blocked: invalid outer result receipt")
    semantic_payload = dict(payload_value)
    payload_result_sha = semantic_payload.pop("result_sha256", None)
    if payload_result_sha != stable_hash(semantic_payload):
        raise RuntimeError("entry evidence blocked: outer typed result hash drift")
    for policy_id in ("HGB", "NEURAL"):
        evaluation = payload_value.get(policy_id.lower())
        if type(evaluation) is not dict or evaluation.get("policy_id") != policy_id:
            raise RuntimeError("entry evidence blocked: outer policy identity drift")
        _validate_policy_evaluation_dict(
            evaluation,
            authorization=expected_authorization,
            dataset_sha256=dataset_receipt["dataset_sha256"],
            sessions=sessions,
            role="outer_test_primary",
            outer_fold=outer_fold,
            inner_fold=None,
        )
        _validate_policy_evaluation_dataset_membership(
            evaluation, dataset_receipt
        )
        if receipt.get(f"{policy_id.lower()}_evaluation_sha256") != evaluation.get(
            "result_sha256"
        ):
            raise RuntimeError("entry evidence blocked: outer evaluation hash drift")
    selected_key = preopen["selected_family"].lower()
    if payload_value.get("selected_evaluation_sha256") != payload_value[
        selected_key
    ].get("result_sha256"):
        raise RuntimeError("entry evidence blocked: selected evaluation identity drift")
    coverage_root = stable_hash(
        [
            payload_value[family]["session_coverage_sha256"]
            for family in ("hgb", "neural")
        ]
    )
    if payload_value.get("session_coverage_root_sha256") != coverage_root:
        raise RuntimeError("entry evidence blocked: outer session coverage root drift")
    if (
        negative_panel.get("manifest_sha256")
        != negative_manifest.get("artifact_sha256")
        or replay_result.get("replay_config_sha256")
        != replay_config.get("artifact_sha256")
    ):
        raise RuntimeError("entry evidence blocked: post-open control binding drift")
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    return receipt


def _validate_shortened_preopen_receipt_one(
    payload: dict[str, Any], assignments: dict[str, Any], outer_fold: int
) -> dict[str, Any]:
    fold = assignments["folds"][outer_fold - 1]
    sessions = tuple(fold["outer_test_shortened_diagnostic_only"])
    if not sessions:
        raise RuntimeError("entry evidence blocked: shortened diagnostic is empty")
    path = _canonical_repo_regular_file(
        repo_path_label(
            _outer_fold_artifact_path(outer_fold, "shortened_preopen_receipt.json")
        )
    )
    receipt = read_json(path)
    expected_keys = {
        "schema_version", "status", "frozen_at_utc", "outer_fold",
        "preregistration_sha256", "session_assignments_sha256",
        "source_hash_policy_sha256", "outer_result_receipt_path",
        "outer_result_receipt_sha256", "outer_preopen_receipt_sha256",
        "shortened_sessions_sha256_newline", "diagnostic_access_count",
        "quarantine_labels", "claim_boundary", "holdout_caveat",
        "plan_sha256", "fill_law_hash", "holdout_open_count", "receipt_sha256",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version")
        != "pathd.entry_shortened_diagnostic_preopen_receipt.v1"
        or receipt.get("status") != "FROZEN_AFTER_PRIMARY_BEFORE_DIAGNOSTIC_OPEN"
        or type(receipt.get("outer_fold")) is not int
        or receipt.get("outer_fold") != outer_fold
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T[^\s]+Z", str(receipt.get("frozen_at_utc", ""))
        )
        is None
        or receipt.get("preregistration_sha256") != sha256_path(PREREG_PATH)
        or receipt.get("session_assignments_sha256") != sha256_path(SESSION_PATH)
        or receipt.get("source_hash_policy_sha256")
        != stable_hash(payload["source_hash_policy"])
        or receipt.get("outer_result_receipt_path")
        != repo_path_label(
            _outer_fold_artifact_path(outer_fold, "outer_result_receipt.json")
        )
        or receipt.get("outer_result_receipt_sha256")
        != sha256_path(
            _outer_fold_artifact_path(outer_fold, "outer_result_receipt.json")
        )
        or receipt.get("outer_preopen_receipt_sha256")
        != sha256_path(_outer_fold_artifact_path(outer_fold, "preopen_receipt.json"))
        or receipt.get("shortened_sessions_sha256_newline")
        != canonical_session_hash(sessions)
        or type(receipt.get("diagnostic_access_count")) is not int
        or receipt.get("diagnostic_access_count") != 0
        or receipt.get("quarantine_labels") != list(QUARANTINE_LABELS)
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
        or receipt.get("holdout_caveat") != HOLDOUT_CAVEAT
        or receipt.get("plan_sha256") != payload["binding_plan"]["sha256"]
        or receipt.get("fill_law_hash") != payload["fill_law"]["fill_law_hash"]
        or type(receipt.get("holdout_open_count")) is not int
        or receipt.get("holdout_open_count") != 0
    ):
        raise RuntimeError("entry evidence blocked: invalid shortened preopen receipt")
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    return receipt


def _assert_outer_evidence_open_gate(
    payload: dict[str, Any], assignments: dict[str, Any], *, role: str,
    outer_fold: int | None, inner_fold: int | None,
) -> tuple[str, ...]:
    hashes: list[str] = []
    if role == "nested_validation":
        if type(outer_fold) is not int or type(inner_fold) is not int:
            raise RuntimeError("entry evidence blocked: nested fold identity absent")
        for prior in range(1, inner_fold):
            receipt_path, _ = _validate_nested_block_receipt_one(
                payload, assignments, outer_fold, prior
            )
            hashes.append(sha256_path(receipt_path))
        _validate_nested_preopen_receipt_one(
            payload, assignments, outer_fold, inner_fold
        )
        hashes.append(
            sha256_path(
                _outer_fold_artifact_path(
                    outer_fold, f"nested_inner_{inner_fold}_preopen_receipt.json"
                )
            )
        )
        for later in range(inner_fold + 1, 5):
            prefix = f"nested_inner_{later}_"
            later_dir = _outer_fold_artifact_path(
                outer_fold, "placeholder"
            ).parent
            if later_dir.exists() and any(
                path.name.startswith(prefix) for path in later_dir.iterdir()
            ):
                raise RuntimeError(
                    "entry evidence blocked: later nested artifact exists before current open"
                )
        return tuple(hashes)
    if type(outer_fold) is not int:
        raise RuntimeError("entry evidence blocked: missing outer fold")
    for later in range(outer_fold + 1, 6):
        later_dir = _outer_fold_artifact_path(later, "placeholder").parent
        if later_dir.exists() and any(path.is_file() for path in later_dir.iterdir()):
            raise RuntimeError(
                "entry evidence blocked: later-fold artifact exists before current open"
            )
    if role == "outer_test_shortened_diagnostic":
        _validate_outer_result_receipt_one(payload, assignments, outer_fold)
        hashes.append(
            sha256_path(
                _outer_fold_artifact_path(outer_fold, "outer_result_receipt.json")
            )
        )
        _validate_shortened_preopen_receipt_one(
            payload, assignments, outer_fold
        )
        hashes.append(
            sha256_path(
                _outer_fold_artifact_path(outer_fold, "shortened_preopen_receipt.json")
            )
        )
        return tuple(hashes)
    if role != "outer_test_primary":
        raise RuntimeError("entry evidence blocked: unsupported raw evidence role")
    for prior in range(1, outer_fold):
        _validate_outer_result_receipt_one(payload, assignments, prior)
        hashes.append(
            sha256_path(_outer_fold_artifact_path(prior, "outer_result_receipt.json"))
        )
    _validate_outer_preopen_receipt_one(payload, assignments, outer_fold)
    hashes.append(
        sha256_path(_outer_fold_artifact_path(outer_fold, "preopen_receipt.json"))
    )
    return tuple(hashes)


def assert_entry_fit_ready(
    *,
    role: str,
    outer_fold: int | None = None,
    inner_fold: int | None = None,
) -> FrozenFitAuthorization:
    """Authorize one exact frozen role; callers cannot supply arbitrary sessions."""

    from v4.research.pathd_holdout_gate import _assert_protected_holdout_unopened

    _assert_protected_holdout_unopened()

    prereg_receipt = assert_preregistration_frozen()
    payload = read_json(PREREG_PATH)
    assignments = read_json(SESSION_PATH)
    _verify_immutable_sources(payload)
    lineage_receipt = assert_lineage_implementation_frozen()
    machinery_receipt = _assert_entry_machinery_frozen(
        prereg_receipt, payload, lineage_receipt
    )
    _verify_authorized_dependency_closure(payload)
    fit_environment = assert_entry_fit_environment_current()
    _validated_corpus_integrity_receipt(
        payload, prereg_receipt, machinery_receipt
    )
    foundation_stability = assert_research_foundation_stable()
    policy, policy_hash = _source_policy_and_hash(payload)
    if policy.get("fit_session_roles") != fit_session_role_spec():
        raise RuntimeError("entry fitting blocked: frozen fit-role policy drift")
    sessions = _resolve_frozen_fit_sessions(
        assignments,
        role=role,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    pooled_acceptance_receipt_sha256: str | None = None
    if role in {"full_weights", "full_calibration"}:
        read_frozen_entry_pooled_acceptance(require_pass=True)
        pooled_acceptance_receipt_sha256 = sha256_path(
            _canonical_repo_regular_file(
                repo_path_label(ENTRY_POOLED_ACCEPTANCE_RECEIPT_PATH)
            )
        )
    return FrozenFitAuthorization(
        role=role,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        sessions=sessions,
        sessions_sha256_newline=canonical_session_hash(sessions),
        preregistration_sha256=prereg_receipt["preregistration_sha256"],
        session_assignments_sha256=sha256_path(SESSION_PATH),
        source_hash_policy_sha256=policy_hash,
        corpus_integrity_receipt_sha256=sha256_path(CORPUS_INTEGRITY_RECEIPT_PATH),
        lineage_receipt_sha256=sha256_path(LINEAGE_IMPLEMENTATION_RECEIPT_PATH),
        machinery_receipt_sha256=sha256_path(ENTRY_MACHINERY_RECEIPT_PATH),
        fit_environment_sha256=stable_hash(fit_environment),
        entry_pooled_acceptance_receipt_sha256=(
            pooled_acceptance_receipt_sha256
        ),
        foundation_generation_sha256=foundation_stability[
            "foundation_generation_sha256"
        ],
        foundation_stability_receipt_sha256=sha256_path(
            FOUNDATION_STABILITY_RECEIPT_PATH
        ),
    )


def _prepare_entry_evidence_authorization_claim(
    *, role: str, outer_fold: int | None = None, inner_fold: int | None = None
) -> dict[str, Any]:
    """Validate pre-open prerequisites without opening or decoding evidence."""

    from v4.research.pathd_holdout_gate import _assert_protected_holdout_unopened

    _assert_protected_holdout_unopened()

    prereg_receipt = assert_preregistration_frozen()
    payload = read_json(PREREG_PATH)
    assignments = read_json(SESSION_PATH)
    _verify_immutable_sources(payload)
    lineage_receipt = assert_lineage_implementation_frozen()
    machinery_receipt = _assert_entry_machinery_frozen(
        prereg_receipt, payload, lineage_receipt
    )
    _verify_authorized_dependency_closure(payload)
    fit_environment = assert_entry_fit_environment_current()
    _validated_corpus_integrity_receipt(payload, prereg_receipt, machinery_receipt)
    foundation_stability = assert_research_foundation_stable()
    policy, policy_hash = _source_policy_and_hash(payload)
    if policy.get("evidence_session_roles") != evidence_session_role_spec():
        raise RuntimeError("entry evidence blocked: frozen evidence-role policy drift")
    open_gate_hashes = _assert_outer_evidence_open_gate(
        payload, assignments, role=role, outer_fold=outer_fold, inner_fold=inner_fold
    )
    sessions = _resolve_frozen_evidence_sessions(
        assignments, role=role, outer_fold=outer_fold, inner_fold=inner_fold
    )
    return {
        "role": role,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "sessions": sessions,
        "sessions_sha256_newline": canonical_session_hash(sessions),
        "preregistration_sha256": prereg_receipt["preregistration_sha256"],
        "session_assignments_sha256": sha256_path(SESSION_PATH),
        "source_hash_policy_sha256": policy_hash,
        "corpus_integrity_receipt_sha256": sha256_path(
            CORPUS_INTEGRITY_RECEIPT_PATH
        ),
        "lineage_receipt_sha256": sha256_path(
            LINEAGE_IMPLEMENTATION_RECEIPT_PATH
        ),
        "machinery_receipt_sha256": sha256_path(ENTRY_MACHINERY_RECEIPT_PATH),
        "fit_environment_sha256": stable_hash(fit_environment),
        "foundation_generation_sha256": foundation_stability[
            "foundation_generation_sha256"
        ],
        "foundation_stability_receipt_sha256": sha256_path(
            FOUNDATION_STABILITY_RECEIPT_PATH
        ),
        "open_gate_receipts_sha256": open_gate_hashes,
    }


def assert_entry_evidence_ready(
    *, role: str, outer_fold: int | None = None, inner_fold: int | None = None
) -> FrozenEvidenceAuthorization:
    """Durably open one exact hidden evidence block exactly once before decode."""

    from v4.research.pathd_evidence_gate import begin_entry_evidence_once

    return begin_entry_evidence_once(
        role=role, outer_fold=outer_fold, inner_fold=inner_fold
    )


def assert_entry_evidence_authorization_current(
    authorization: FrozenEvidenceAuthorization,
) -> FrozenEvidenceAuthorization:
    from v4.research.pathd_evidence_gate import validate_entry_evidence_access

    return validate_entry_evidence_access(authorization)


def assert_entry_result_aggregation_ready(
    *, role: str,
) -> FrozenResultAggregationAuthorization:
    """Authorize pooled analysis of immutable receipts without reopening raw evidence."""

    from v4.research.pathd_holdout_gate import _assert_protected_holdout_unopened

    _assert_protected_holdout_unopened()
    if role not in result_aggregation_role_spec():
        raise RuntimeError("entry aggregation blocked: unsupported role")
    prereg_receipt = assert_preregistration_frozen()
    payload = read_json(PREREG_PATH)
    assignments = read_json(SESSION_PATH)
    _verify_immutable_sources(payload)
    lineage_receipt = assert_lineage_implementation_frozen()
    _assert_entry_machinery_frozen(prereg_receipt, payload, lineage_receipt)
    assert_research_foundation_stable()
    receipts: list[str] = []
    sessions: list[str] = []
    for outer_fold in range(1, 6):
        _validate_outer_result_receipt_one(payload, assignments, outer_fold)
        receipt_path = _outer_fold_artifact_path(
            outer_fold, "outer_result_receipt.json"
        )
        receipts.append(sha256_path(receipt_path))
        sessions.extend(
            assignments["folds"][outer_fold - 1][
                "outer_test_primary_1555_complete"
            ]
        )
    values = tuple(sessions)
    return FrozenResultAggregationAuthorization(
        role=role,
        sessions=values,
        sessions_sha256_newline=canonical_session_hash(values),
        preregistration_sha256=prereg_receipt["preregistration_sha256"],
        session_assignments_sha256=sha256_path(SESSION_PATH),
        source_hash_policy_sha256=stable_hash(payload["source_hash_policy"]),
        machinery_receipt_sha256=sha256_path(ENTRY_MACHINERY_RECEIPT_PATH),
        outer_result_receipts_sha256=tuple(receipts),
    )


def _entry_pooled_gate_spec_sha256(payload: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "entry_acceptance": payload["entry"]["acceptance"],
            "entry_controls": payload["entry"]["controls"],
            "entry_power": payload["entry"]["power_and_trajectory_population"],
            "action_calibration": payload["metrics_and_gates"]["action_calibration"],
            "minimum_power": payload["metrics_and_gates"]["minimum_power"],
            "large_improvement_audit": payload["metrics_and_gates"][
                "large_improvement_audit"
            ],
            "fill_law": payload["fill_law"],
            "account_law": payload["economics"]["serial_account_contract"],
            "pooled_input_spec": entry_pooled_gate_input_spec(),
        }
    )


def _is_hex64(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_semantic_self_hash(
    value: Mapping[str, Any], *, field: str, owner: str
) -> None:
    semantic = dict(value)
    digest = semantic.pop(field, None)
    if not _is_hex64(digest) or digest != stable_hash(semantic):
        raise RuntimeError(f"entry pooled reconstruction {owner} self-hash drift")


def _entry_replay_policy_id(
    *, owner_policy_id: str, channel: str, matched_random_seed: int | None
) -> str:
    if matched_random_seed is None:
        return f"{owner_policy_id}::{channel}"
    return (
        f"MATCHED_RANDOM::{owner_policy_id}::SEED_{matched_random_seed}::{channel}"
    )


def _journal_transition_index(evaluation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    journal = evaluation.get("terminal_journal")
    transitions = journal.get("transitions") if type(journal) is dict else None
    if type(transitions) is not list:
        raise RuntimeError("entry pooled reconstruction journal transition drift")
    index: dict[str, dict[str, Any]] = {}
    for transition in transitions:
        if type(transition) is not dict:
            raise RuntimeError("entry pooled reconstruction non-object transition")
        digest = transition.get("transition_sha256")
        if not _is_hex64(digest) or digest in index:
            raise RuntimeError("entry pooled reconstruction transition identity drift")
        index[digest] = transition
    return index


def _validate_entry_replay_evaluation_cell(
    value: Any,
    *,
    expected_outer_fold: int,
    expected_owner_policy_id: str,
    expected_channel: str,
    expected_seed: int | None,
    authorization: FrozenEvidenceAuthorization,
    dataset_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one embedded complete replay from its full journal, never a summary."""

    fields = entry_future_api_contract()["dataclass_fields"][
        "EntryReplayEvaluationCellV1"
    ]
    if type(value) is not dict or set(value) != set(fields):
        raise RuntimeError("entry pooled reconstruction replay-cell schema drift")
    _assert_strict_json_value(value)
    authorization_sha256 = stable_hash(authorization.to_dict())
    replay_policy_id = _entry_replay_policy_id(
        owner_policy_id=expected_owner_policy_id,
        channel=expected_channel,
        matched_random_seed=expected_seed,
    )
    evaluation = value.get("evaluation")
    if (
        value.get("schema_version") != "pathd.entry_replay_evaluation_cell.v1"
        or value.get("outer_fold") != expected_outer_fold
        or value.get("owner_policy_id") != expected_owner_policy_id
        or value.get("replay_policy_id") != replay_policy_id
        or value.get("channel") != expected_channel
        or value.get("matched_random_seed") != expected_seed
        or value.get("authorization_sha256") != authorization_sha256
        or value.get("dataset_sha256") != dataset_receipt.get("dataset_sha256")
        or type(evaluation) is not dict
        or evaluation.get("policy_id") != replay_policy_id
        or value.get("evaluation_sha256") != evaluation.get("result_sha256")
        or value.get("terminal_journal_sha256")
        != evaluation.get("terminal_journal_sha256")
    ):
        raise RuntimeError("entry pooled reconstruction replay-cell identity drift")
    receipt_fields = (
        "candidate_budget_sha256",
        "ordered_schedule_sha256",
        "structural_skip_trace_sha256",
        "realized_intents_and_fills_sha256",
    )
    if expected_seed is None:
        if any(value.get(name) is not None for name in receipt_fields):
            raise RuntimeError("entry pooled reconstruction nonrandom receipt drift")
    elif any(not _is_hex64(value.get(name)) for name in receipt_fields):
        raise RuntimeError("entry pooled reconstruction random receipt drift")
    _validate_policy_evaluation_dict(
        evaluation,
        authorization=authorization,
        dataset_sha256=str(dataset_receipt["dataset_sha256"]),
        sessions=authorization.sessions,
        role="outer_test_primary",
        outer_fold=expected_outer_fold,
        inner_fold=None,
    )
    _validate_policy_evaluation_dataset_membership(evaluation, dict(dataset_receipt))
    _validate_semantic_self_hash(value, field="cell_sha256", owner="replay cell")
    return dict(value)


def _validate_observation_rows(
    rows: Any,
    *,
    outer_fold: int,
    source_cell: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if type(rows) is not list:
        raise RuntimeError("entry pooled reconstruction action observations drift")
    evaluation = source_cell["evaluation"]
    transition_index = _journal_transition_index(evaluation)
    fields = entry_future_api_contract()["dataclass_fields"][
        "EntryActionCalibrationObservationV1"
    ]
    sessions = set(evaluation["sessions"])
    identities: set[tuple[str, str]] = set()
    observed: list[dict[str, Any]] = []
    event_payloads = [
        transition.get("event_payload")
        for transition in transition_index.values()
        if transition.get("event_schema_version")
        == "pathd.entry_action_calibration_observation.v1"
    ]
    for row in rows:
        if type(row) is not dict or set(row) != set(fields):
            raise RuntimeError("entry pooled reconstruction action row schema drift")
        _assert_strict_json_value(row)
        identity = (str(row.get("action")), str(row.get("trajectory_or_episode_id")))
        if (
            row.get("schema_version")
            != "pathd.entry_action_calibration_observation.v1"
            or row.get("outer_fold") != outer_fold
            or row.get("action") not in {"ENTER", "WAIT"}
            or row.get("session") not in sessions
            or type(row.get("trajectory_or_episode_id")) is not str
            or not row["trajectory_or_episode_id"]
            or identity in identities
            or any(
                type(row.get(name)) is not int
                for name in (
                    "predicted_mean_micros",
                    "predicted_lower_micros",
                    "realized_micros",
                )
            )
            or row.get("source_policy_evaluation_sha256")
            != source_cell["evaluation_sha256"]
            or row.get("source_transition_sha256") not in transition_index
        ):
            raise RuntimeError("entry pooled reconstruction action row identity drift")
        _validate_semantic_self_hash(
            row, field="observation_sha256", owner="action observation"
        )
        if sum(payload == row for payload in event_payloads) != 1:
            raise RuntimeError(
                "entry pooled reconstruction action row is not journal-derived"
            )
        identities.add(identity)
        observed.append(dict(row))
    if event_payloads != rows:
        raise RuntimeError(
            "entry pooled reconstruction omitted or reordered action journal evidence"
        )
    return observed


def _validate_time300_rows(
    rows: Any,
    *,
    outer_fold: int,
    source_cell: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if type(rows) is not list:
        raise RuntimeError("entry pooled reconstruction time300 evidence drift")
    evaluation = source_cell["evaluation"]
    transition_index = _journal_transition_index(evaluation)
    fields = entry_future_api_contract()["dataclass_fields"][
        "EntryTime300TrajectoryEvidenceV1"
    ]
    completed = dict(
        zip(
            evaluation["completed_trade_ids"],
            evaluation["completed_trade_sessions"],
            strict=True,
        )
    )
    seen: set[str] = set()
    observed: list[dict[str, Any]] = []
    event_payloads = [
        transition.get("event_payload")
        for transition in transition_index.values()
        if transition.get("event_schema_version")
        == "pathd.entry_time300_trajectory_evidence.v1"
    ]
    for row in rows:
        if type(row) is not dict or set(row) != set(fields):
            raise RuntimeError("entry pooled reconstruction time300 row schema drift")
        _assert_strict_json_value(row)
        trajectory_id = row.get("trajectory_id")
        if (
            row.get("schema_version") != "pathd.entry_time300_trajectory_evidence.v1"
            or row.get("outer_fold") != outer_fold
            or type(trajectory_id) is not str
            or not trajectory_id
            or trajectory_id in seen
            or completed.get(trajectory_id) != row.get("session")
            or row.get("source_policy_evaluation_sha256")
            != source_cell["evaluation_sha256"]
            or row.get("entry_fill_transition_sha256") not in transition_index
            or not _is_hex64(row.get("h300_mark_receipt_sha256"))
        ):
            raise RuntimeError("entry pooled reconstruction time300 row identity drift")
        _validate_semantic_self_hash(
            row, field="evidence_sha256", owner="time300 evidence"
        )
        if sum(payload == row for payload in event_payloads) != 1:
            raise RuntimeError(
                "entry pooled reconstruction time300 row is not journal-derived"
            )
        seen.add(trajectory_id)
        observed.append(dict(row))
    if event_payloads != rows:
        raise RuntimeError(
            "entry pooled reconstruction omitted or reordered time300 evidence"
        )
    return observed


def _validate_survival_records(
    cell: Mapping[str, Any], *, outer_fold: int
) -> dict[str, int]:
    evaluation = cell["evaluation"]
    transition_index = _journal_transition_index(evaluation)
    fields = entry_future_api_contract()["dataclass_fields"][
        "EntrySurvivalAuditRecordV1"
    ]
    keys = entry_pooled_gate_input_spec()["closed_shapes"][
        "survival_violation_counts"
    ]["keys_in_order"]
    rows = [
        transition.get("event_payload")
        for transition in transition_index.values()
        if transition.get("event_schema_version")
        == "pathd.entry_survival_audit_record.v1"
    ]
    if len(rows) != len(evaluation["sessions"]):
        raise RuntimeError("entry pooled reconstruction survival coverage drift")
    counts = {key: 0 for key in keys}
    if [row.get("session") if type(row) is dict else None for row in rows] != evaluation[
        "sessions"
    ]:
        raise RuntimeError("entry pooled reconstruction survival session order drift")
    for row in rows:
        if type(row) is not dict or set(row) != set(fields):
            raise RuntimeError("entry pooled reconstruction survival row schema drift")
        _assert_strict_json_value(row)
        sources = row.get("source_transition_sha256s")
        if (
            row.get("schema_version") != "pathd.entry_survival_audit_record.v1"
            or row.get("outer_fold") != outer_fold
            or row.get("policy_id") != evaluation["policy_id"]
            or any(type(row.get(key)) is not bool for key in keys)
            or type(sources) is not list
            or any(not _is_hex64(item) or item not in transition_index for item in sources)
            or len(sources) != len(set(sources))
        ):
            raise RuntimeError("entry pooled reconstruction survival row identity drift")
        _validate_semantic_self_hash(
            row, field="record_sha256", owner="survival record"
        )
        for key in keys:
            counts[key] += int(row[key])
    return counts


def _validate_entry_control_replay_result_immutable(
    value: Any,
    *,
    config: Mapping[str, Any],
    authorization: FrozenEvidenceAuthorization,
    dataset_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Immutable-main validator for candidate/P5/random full-journal replays."""

    fields = entry_future_api_contract()["dataclass_fields"][
        "EntryControlReplayResultV1"
    ]
    if type(value) is not dict or set(value) != set(fields):
        raise RuntimeError("entry pooled reconstruction replay-result schema drift")
    _assert_strict_json_value(value)
    outer_fold = authorization.outer_fold
    if type(outer_fold) is not int:
        raise RuntimeError("entry pooled reconstruction replay fold absent")
    config_fields = entry_future_api_contract()["dataclass_fields"][
        "EntryControlReplayConfigV1"
    ]
    if type(config) is not dict or set(config) != set(config_fields):
        raise RuntimeError("entry pooled reconstruction replay config schema drift")
    _validate_semantic_self_hash(config, field="artifact_sha256", owner="replay config")
    if (
        config.get("schema_version") != "pathd.entry_control_replay_config.v1"
        or config.get("outer_fold") != outer_fold
        or config.get("fill_law_hash") != read_json(PREREG_PATH)["fill_law"]["fill_law_hash"]
        or config.get("matched_random_seeds") != list(MATCHED_RANDOM_SEEDS)
        or config.get("fee_paths") != [3, 4]
        or config.get("sell_delay_rungs_ms") != [0, 1_000, 2_000, 5_000]
        or not _is_hex64(config.get("control_exit_sha256"))
        or not _is_hex64(config.get("headline_config_sha256"))
    ):
        raise RuntimeError("entry pooled reconstruction replay config drift")
    if (
        value.get("schema_version") != "pathd.entry_control_replay_result.v1"
        or value.get("holdout_caveat") != HOLDOUT_CAVEAT
        or value.get("outer_fold") != outer_fold
        or value.get("replay_config_sha256") != config.get("artifact_sha256")
        or not _is_hex64(value.get("candidate_outer_evaluation_sha256"))
        or not _is_hex64(value.get("neural_outer_evaluation_sha256"))
        or value.get("channels") != list(ENTRY_REPLAY_CHANNELS)
        or value.get("matched_random_owner_policy_ids")
        != list(entry_matched_random_owner_policy_ids())
        or value.get("matched_random_seeds") != list(MATCHED_RANDOM_SEEDS)
    ):
        raise RuntimeError("entry pooled reconstruction replay-result identity drift")
    candidate_rows = value.get("candidate_channel_evaluations")
    p5_rows = value.get("p5_channel_evaluations")
    if type(candidate_rows) is not list or type(p5_rows) is not list:
        raise RuntimeError("entry pooled reconstruction candidate/P5 array drift")
    if len(candidate_rows) != 4 or len(p5_rows) != 4:
        raise RuntimeError("entry pooled reconstruction candidate/P5 channel gap")
    candidate = [
        _validate_entry_replay_evaluation_cell(
            row,
            expected_outer_fold=outer_fold,
            expected_owner_policy_id="HGB",
            expected_channel=channel,
            expected_seed=None,
            authorization=authorization,
            dataset_receipt=dataset_receipt,
        )
        for channel, row in zip(ENTRY_REPLAY_CHANNELS, candidate_rows, strict=True)
    ]
    p5 = [
        _validate_entry_replay_evaluation_cell(
            row,
            expected_outer_fold=outer_fold,
            expected_owner_policy_id="P5",
            expected_channel=channel,
            expected_seed=None,
            authorization=authorization,
            dataset_receipt=dataset_receipt,
        )
        for channel, row in zip(ENTRY_REPLAY_CHANNELS, p5_rows, strict=True)
    ]
    random_rows = value.get("matched_random_channel_evaluations")
    dimensions = (len(entry_matched_random_owner_policy_ids()), 8, 4)
    _validate_nested_tensor(
        random_rows,
        dimensions,
        lambda item: item is None or type(item) is dict,
        "replay random-cell",
    )
    random_validated: list[list[list[dict[str, Any] | None]]] = []
    receipt_names = (
        "candidate_budget_sha256s",
        "ordered_schedule_sha256s",
        "structural_skip_trace_sha256s",
        "realized_intents_and_fills_sha256s",
    )
    receipt_matrices = {name: value.get(name) for name in receipt_names}
    for name, matrix in receipt_matrices.items():
        _validate_nested_tensor(
            matrix,
            dimensions,
            lambda item: item is None or _is_hex64(item),
            name,
        )
    for policy_index, owner in enumerate(entry_matched_random_owner_policy_ids()):
        policy_rows: list[list[dict[str, Any] | None]] = []
        for seed_index, seed in enumerate(MATCHED_RANDOM_SEEDS):
            seed_rows: list[dict[str, Any] | None] = []
            for channel_index, channel in enumerate(ENTRY_REPLAY_CHANNELS):
                raw = random_rows[policy_index][seed_index][channel_index]
                if raw is None:
                    cell = None
                else:
                    cell = _validate_entry_replay_evaluation_cell(
                        raw,
                        expected_outer_fold=outer_fold,
                        expected_owner_policy_id=owner,
                        expected_channel=channel,
                        expected_seed=seed,
                        authorization=authorization,
                        dataset_receipt=dataset_receipt,
                    )
                for matrix_name, cell_field in zip(
                    receipt_names,
                    (
                        "candidate_budget_sha256",
                        "ordered_schedule_sha256",
                        "structural_skip_trace_sha256",
                        "realized_intents_and_fills_sha256",
                    ),
                    strict=True,
                ):
                    expected = None if cell is None else cell[cell_field]
                    if receipt_matrices[matrix_name][policy_index][seed_index][channel_index] != expected:
                        raise RuntimeError(
                            "entry pooled reconstruction random receipt matrix drift"
                        )
                seed_rows.append(cell)
            policy_rows.append(seed_rows)
        random_validated.append(policy_rows)
    action = _validate_observation_rows(
        value.get("candidate_action_calibration_observations"),
        outer_fold=outer_fold,
        source_cell=candidate[2],
    )
    time300 = _validate_time300_rows(
        value.get("candidate_time300_trajectory_evidence"),
        outer_fold=outer_fold,
        source_cell=candidate[2],
    )
    hashes = [row["cell_sha256"] for row in (*candidate, *p5)]
    hashes.extend(
        cell["cell_sha256"]
        for policy_rows in random_validated
        for seed_rows in policy_rows
        for cell in seed_rows
        if cell is not None
    )
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("entry pooled reconstruction replay cell reuse")
    _validate_semantic_self_hash(value, field="artifact_sha256", owner="replay result")
    return {
        "raw": dict(value),
        "candidate": candidate,
        "p5": p5,
        "random": random_validated,
        "action": action,
        "time300": time300,
    }


def _validate_entry_negative_control_panel_immutable(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    authorization: FrozenEvidenceAuthorization,
    dataset_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Immutable-main validator for every negative-control channel journal."""

    fields = entry_future_api_contract()["dataclass_fields"][
        "EntryNegativeControlPanelV1"
    ]
    if type(value) is not dict or set(value) != set(fields):
        raise RuntimeError("entry pooled reconstruction negative-panel schema drift")
    _assert_strict_json_value(value)
    outer_fold = authorization.outer_fold
    controls = entry_negative_control_policy_ids()
    if (
        type(outer_fold) is not int
        or value.get("schema_version") != "pathd.entry_negative_control_panel.v1"
        or value.get("holdout_caveat") != HOLDOUT_CAVEAT
        or value.get("outer_fold") != outer_fold
        or value.get("manifest_sha256") != manifest.get("artifact_sha256")
        or value.get("required_control_ids") != list(controls)
        or value.get("channels") != list(ENTRY_REPLAY_CHANNELS)
        or value.get("control_bundle_sha256s")
        != manifest.get("control_bundle_sha256s")
    ):
        raise RuntimeError("entry pooled reconstruction negative-panel identity drift")
    matrix = value.get("control_channel_evaluations")
    _validate_nested_tensor(
        matrix,
        (22, 4),
        lambda item: item is None or type(item) is dict,
        "negative control-cell",
    )
    action_rows = value.get("control_action_calibration_observations")
    time_rows = value.get("control_time300_trajectory_evidence")
    if (
        type(action_rows) is not list
        or len(action_rows) != 22
        or type(time_rows) is not list
        or len(time_rows) != 22
    ):
        raise RuntimeError("entry pooled reconstruction negative evidence shape drift")
    validated: list[list[dict[str, Any] | None]] = []
    actions: list[list[dict[str, Any]]] = []
    times: list[list[dict[str, Any]]] = []
    hashes: list[str] = []
    for control_index, control in enumerate(controls):
        channel_rows: list[dict[str, Any] | None] = []
        for channel_index, channel in enumerate(ENTRY_REPLAY_CHANNELS):
            raw = matrix[control_index][channel_index]
            if raw is None:
                cell = None
            else:
                cell = _validate_entry_replay_evaluation_cell(
                    raw,
                    expected_outer_fold=outer_fold,
                    expected_owner_policy_id=control,
                    expected_channel=channel,
                    expected_seed=None,
                    authorization=authorization,
                    dataset_receipt=dataset_receipt,
                )
                hashes.append(cell["cell_sha256"])
            channel_rows.append(cell)
        source = channel_rows[2]
        if source is None:
            if action_rows[control_index] != [] or time_rows[control_index] != []:
                raise RuntimeError(
                    "entry pooled reconstruction orphan negative-control evidence"
                )
            control_actions: list[dict[str, Any]] = []
            control_times: list[dict[str, Any]] = []
        else:
            control_actions = _validate_observation_rows(
                action_rows[control_index], outer_fold=outer_fold, source_cell=source
            )
            control_times = _validate_time300_rows(
                time_rows[control_index], outer_fold=outer_fold, source_cell=source
            )
        validated.append(channel_rows)
        actions.append(control_actions)
        times.append(control_times)
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("entry pooled reconstruction negative replay reuse")
    _validate_semantic_self_hash(value, field="artifact_sha256", owner="negative panel")
    return {"raw": dict(value), "cells": validated, "action": actions, "time300": times}


def _seed_derivation_integer(
    *, purpose: str, model_family: str, statistic: str
) -> int:
    key = {
        "campaign": "pathd.tier_s.v1",
        "plan_sha256": sha256_path(PLAN_PATH),
        "purpose": purpose,
        "model_family": model_family,
        "fold_scope": "POOLED_OUTER_1_5",
        "statistic": statistic,
    }
    return int.from_bytes(bytes.fromhex(stable_hash(key))[:4], "big", signed=False)


def _negative_control_seed_family(control_id: str) -> str:
    family, control = control_id.split("::", 1)
    if control == "CONSTANT":
        return "CONSTANT"
    if control == "SIGN_REVERSED":
        return f"SIGN_REVERSED_{family}"
    if control == "TIME_SHIFTED_FEATURES":
        return f"TIME_SHIFTED_{family}"
    match = re.fullmatch(r"SHUFFLED_TARGET_(\d{2})", control)
    if match is None:
        raise RuntimeError("entry pooled reconstruction control seed identity drift")
    return f"SHUFFLED_TARGET_{family}_{match.group(1)}"


def _action_gate_from_observations(
    rows: Iterable[Mapping[str, Any]],
    *,
    sessions: tuple[str, ...],
    model_family: str,
) -> dict[str, dict[str, Any]]:
    """Recompute ENTER/WAIT coverage and decile bootstrap from journal facts."""

    import numpy as np

    session_set = set(sessions)
    values = [dict(row) for row in rows]
    identities: set[tuple[str, str, str]] = set()
    for row in values:
        identity = (
            str(row.get("action")),
            str(row.get("session")),
            str(row.get("trajectory_or_episode_id")),
        )
        if row.get("session") not in session_set or identity in identities:
            raise RuntimeError("entry pooled reconstruction pooled action identity drift")
        identities.add(identity)
    result: dict[str, dict[str, Any]] = {}
    for action in ("ENTER", "WAIT"):
        action_rows = [row for row in values if row["action"] == action]
        action_rows.sort(
            key=lambda row: (
                row["predicted_mean_micros"],
                row["session"],
                row["trajectory_or_episode_id"],
            )
        )
        count = len(action_rows)
        base_size, remainder = divmod(count, 10)
        bins: list[list[dict[str, Any]]] = []
        offset = 0
        for index in range(10):
            size = base_size + int(index < remainder)
            bins.append(action_rows[offset : offset + size])
            offset += size
        if offset != count:
            raise AssertionError("entry pooled action decile partition drift")
        session_means: list[dict[str, float]] = []
        for bucket in bins:
            grouped: dict[str, list[int]] = {}
            for row in bucket:
                grouped.setdefault(row["session"], []).append(row["realized_micros"])
            session_means.append(
                {
                    session: float(sum(observations)) / float(len(observations))
                    for session, observations in grouped.items()
                }
            )
        rng = np.random.Generator(
            np.random.PCG64(
                _seed_derivation_integer(
                    purpose="action_decile_bootstrap",
                    model_family=model_family,
                    statistic=f"ACTION::{action}",
                )
            )
        )
        valid_counts: list[int] = []
        total_draws: list[int] = []
        upper_bounds: list[float] = []
        for lower_index in range(9):
            lower = session_means[lower_index]
            upper = session_means[lower_index + 1]
            draws = 0
            differences: list[float] = []
            while len(differences) < 5_000 and draws < 50_000:
                draws += 1
                sampled = rng.integers(0, len(sessions), size=len(sessions))
                sampled_sessions = [sessions[int(index)] for index in sampled]
                lower_values = [lower[item] for item in sampled_sessions if item in lower]
                upper_values = [upper[item] for item in sampled_sessions if item in upper]
                if not lower_values or not upper_values:
                    continue
                differences.append(
                    float(sum(lower_values)) / float(len(lower_values))
                    - float(sum(upper_values)) / float(len(upper_values))
                )
            valid_counts.append(len(differences))
            total_draws.append(draws)
            if differences:
                ordered = sorted(differences)
                rank = min(
                    len(ordered),
                    max(1, int(math.ceil((len(ordered) + 1) * 0.90))),
                )
                upper_bounds.append(float(ordered[rank - 1]))
            else:
                upper_bounds.append(0.0)
        result[action] = {
            "distinct_trajectory_count": count,
            "coverage_numerator": sum(
                row["realized_micros"] >= row["predicted_lower_micros"]
                for row in action_rows
            ),
            "coverage_denominator": count,
            "decile_trajectory_counts": [len(bucket) for bucket in bins],
            "decile_distinct_session_counts": [
                len({row["session"] for row in bucket}) for bucket in bins
            ],
            "adjacent_valid_bootstrap_replicates": valid_counts,
            "adjacent_total_draws": total_draws,
            "adjacent_upper_bounds_micros": upper_bounds,
        }
    return result


def _action_gate_passes(action: Mapping[str, Any]) -> bool:
    return (
        action["distinct_trajectory_count"] >= 30
        and action["coverage_denominator"] == action["distinct_trajectory_count"]
        and action["coverage_denominator"] > 0
        and 100 * action["coverage_numerator"]
        >= 85 * action["coverage_denominator"]
        and all(item >= 3 for item in action["decile_trajectory_counts"])
        and all(item >= 3 for item in action["decile_distinct_session_counts"])
        and all(item == 5_000 for item in action["adjacent_valid_bootstrap_replicates"])
        and all(5_000 <= item <= 50_000 for item in action["adjacent_total_draws"])
        and all(item <= 0.0 for item in action["adjacent_upper_bounds_micros"])
    )


def _cell_total_pnl(cell: Mapping[str, Any]) -> int:
    values = cell["evaluation"]["session_pnl_micros"]
    if any(type(item) is not int for item in values):
        raise RuntimeError("entry pooled reconstruction PnL type drift")
    return sum(values)


def _sum_survival_counts(
    cells: Iterable[Mapping[str, Any]], *, outer_fold: int
) -> dict[str, int]:
    keys = entry_pooled_gate_input_spec()["closed_shapes"][
        "survival_violation_counts"
    ]["keys_in_order"]
    total = {key: 0 for key in keys}
    seen: set[str] = set()
    for cell in cells:
        digest = cell["cell_sha256"]
        if digest in seen:
            continue
        seen.add(digest)
        counts = _validate_survival_records(cell, outer_fold=outer_fold)
        for key in keys:
            total[key] += counts[key]
    return total


def _validate_large_improvement_check_receipt(
    *,
    path_label: str,
    check_name: str,
    check_index: int,
    authorization: FrozenResultAggregationAuthorization,
    economic_inputs_sha256: str,
) -> tuple[str, str | None]:
    path = REPO_ROOT / path_label
    if not os.path.lexists(path):
        return "MISSING", None
    canonical = _canonical_repo_regular_file(path_label)
    receipt = read_json(canonical)
    expected_keys = {
        "schema_version", "status", "check_name", "check_index",
        "economic_inputs_sha256", "outer_result_receipts_sha256",
        "evidence_paths", "evidence_sha256s", "findings_path",
        "findings_sha256", "receipt_sha256",
    }
    evidence_paths = receipt.get("evidence_paths")
    evidence_hashes = receipt.get("evidence_sha256s")
    if (
        type(receipt) is not dict
        or set(receipt) != expected_keys
        or receipt.get("schema_version")
        != "pathd.entry_large_improvement_check_receipt.v1"
        or receipt.get("status") not in {"PASS", "FAIL"}
        or receipt.get("check_name") != check_name
        or receipt.get("check_index") != check_index
        or receipt.get("economic_inputs_sha256") != economic_inputs_sha256
        or receipt.get("outer_result_receipts_sha256")
        != stable_hash(list(authorization.outer_result_receipts_sha256))
        or type(evidence_paths) is not list
        or not evidence_paths
        or type(evidence_hashes) is not list
        or len(evidence_hashes) != len(evidence_paths)
        or not _is_hex64(receipt.get("findings_sha256"))
    ):
        raise RuntimeError("entry pooled reconstruction large-audit receipt drift")
    for source_label, expected_hash in zip(
        evidence_paths, evidence_hashes, strict=True
    ):
        source = _canonical_repo_regular_file(source_label)
        if not _is_hex64(expected_hash) or sha256_path(source) != expected_hash:
            raise RuntimeError("entry pooled reconstruction large-audit evidence drift")
    findings = _canonical_repo_regular_file(receipt.get("findings_path"))
    if sha256_path(findings) != receipt["findings_sha256"]:
        raise RuntimeError("entry pooled reconstruction large-audit findings drift")
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    return str(receipt["status"]), sha256_path(canonical)


def _large_improvement_audit_from_sources(
    *,
    folds: list[dict[str, Any]],
    authorization: FrozenResultAggregationAuthorization,
) -> dict[str, Any]:
    import numpy as np

    candidate_sessions: list[int] = []
    p5_sessions: list[int] = []
    random_session_sums: list[int] = []
    random_complete = True
    for source in folds:
        candidate = source["replay"]["candidate"][2]["evaluation"][
            "session_pnl_micros"
        ]
        p5 = source["replay"]["p5"][2]["evaluation"]["session_pnl_micros"]
        random_cells = source["replay"]["random"][0]
        candidate_sessions.extend(candidate)
        p5_sessions.extend(p5)
        for session_index in range(len(candidate)):
            total = 0
            for seed_rows in random_cells:
                cell = seed_rows[2]
                if cell is None:
                    random_complete = False
                    continue
                total += cell["evaluation"]["session_pnl_micros"][session_index]
            random_session_sums.append(total)
    if len(candidate_sessions) != len(authorization.sessions):
        raise RuntimeError("entry pooled reconstruction large-audit session drift")
    candidate_scaled_total = 8 * sum(candidate_sessions)
    p5_scaled_total = 8 * sum(p5_sessions)
    random_scaled_total = sum(random_session_sums)
    if random_complete and random_scaled_total > p5_scaled_total:
        comparator = "MATCHED_RANDOM_AGGREGATE"
        comparator_scaled = random_session_sums
        statistic = "ECON::CANDIDATE_VS_MATCHED_RANDOM"
    else:
        comparator = "P5"
        comparator_scaled = [8 * value for value in p5_sessions]
        statistic = "ECON::CANDIDATE_VS_P5"
    comparator_total = sum(comparator_scaled)
    paired = [
        8 * candidate - control
        for candidate, control in zip(
            candidate_sessions, comparator_scaled, strict=True
        )
    ]
    ratio_trigger = (
        candidate_scaled_total > 0
        and candidate_scaled_total > 2 * max(comparator_total, 0)
    )
    rng = np.random.Generator(
        np.random.PCG64(
            _seed_derivation_integer(
                purpose="pooled_economic_lcb",
                model_family="GLOBAL_HGB_ENTRY",
                statistic=statistic,
            )
        )
    )
    paired_array = np.asarray(paired, dtype=np.float64)
    bootstrap_totals = np.empty(10_000, dtype=np.float64)
    for index in range(10_000):
        selection = rng.integers(0, len(paired), size=len(paired))
        bootstrap_totals[index] = float(paired_array[selection].sum())
    bootstrap_std = float(bootstrap_totals.std(ddof=1))
    bootstrap_trigger = float(sum(paired)) > 2.0 * bootstrap_std
    triggered = ratio_trigger or bootstrap_trigger
    economic_inputs = {
        "candidate_session_pnl_micros": candidate_sessions,
        "p5_session_pnl_micros": p5_sessions,
        "matched_random_eighth_session_pnl_micros": random_session_sums,
        "matched_random_complete": random_complete,
        "selected_best_comparator": comparator,
        "paired_scaled_session_delta_micros": paired,
        "ratio_trigger": ratio_trigger,
        "bootstrap_standard_deviation_scaled_micros": bootstrap_std,
        "bootstrap_trigger": bootstrap_trigger,
    }
    economic_hash = stable_hash(economic_inputs)
    required_names = read_json(PREREG_PATH)["metrics_and_gates"][
        "large_improvement_audit"
    ]["required_checks"]
    if len(required_names) != len(entry_large_improvement_check_paths()):
        raise RuntimeError("entry pooled reconstruction large-audit registry drift")
    statuses: list[str] = []
    receipt_hashes: list[str | None] = []
    if triggered:
        for index, (name, path_label) in enumerate(
            zip(required_names, entry_large_improvement_check_paths(), strict=True),
            1,
        ):
            status, receipt_hash = _validate_large_improvement_check_receipt(
                path_label=path_label,
                check_name=name,
                check_index=index,
                authorization=authorization,
                economic_inputs_sha256=economic_hash,
            )
            statuses.append(status)
            receipt_hashes.append(receipt_hash)
    else:
        statuses = ["NOT_TRIGGERED"] * len(required_names)
        receipt_hashes = [None] * len(required_names)
    return {
        "ratio_trigger": ratio_trigger,
        "bootstrap_trigger": bootstrap_trigger,
        "triggered": triggered,
        "required_check_names": required_names,
        "required_check_statuses": statuses,
        "required_check_receipt_sha256s": receipt_hashes,
        "complete": (not triggered) or all(status == "PASS" for status in statuses),
    }


def _reconstruct_entry_pooled_gate_inputs_from_outer_artifacts(
    *, authorization: FrozenResultAggregationAuthorization
) -> dict[str, Any]:
    """Reopen the five typed outer artifacts and derive every pooled gate cell."""

    payload = read_json(PREREG_PATH)
    assignments = read_json(SESSION_PATH)
    folds: list[dict[str, Any]] = []
    all_cell_hashes: set[str] = set()
    from v4.research.pathd_evidence_gate import (
        read_frozen_entry_evidence_authorization,
    )

    for outer_fold in range(1, 6):
        evidence_authorization = read_frozen_entry_evidence_authorization(
            role="outer_test_primary", outer_fold=outer_fold, inner_fold=None
        )
        dataset_receipt = _validate_entry_evidence_dataset_receipt(
            _outer_fold_artifact_path(
                outer_fold, "outer_primary_dataset_receipt.json"
            ),
            authorization=evidence_authorization,
        )
        config = read_json(
            _outer_fold_artifact_path(outer_fold, "control_replay_config.json")
        )
        replay = _validate_entry_control_replay_result_immutable(
            read_json(
                _outer_fold_artifact_path(outer_fold, "control_replay_result.json")
            ),
            config=config,
            authorization=evidence_authorization,
            dataset_receipt=dataset_receipt,
        )
        manifest = read_json(
            _outer_fold_artifact_path(outer_fold, "negative_control_manifest.json")
        )
        negative = _validate_entry_negative_control_panel_immutable(
            read_json(
                _outer_fold_artifact_path(outer_fold, "negative_control_panel.json")
            ),
            manifest=manifest,
            authorization=evidence_authorization,
            dataset_receipt=dataset_receipt,
        )
        current_hashes = [
            cell["cell_sha256"]
            for cell in (*replay["candidate"], *replay["p5"])
        ]
        current_hashes.extend(
            cell["cell_sha256"]
            for policy in replay["random"]
            for seed in policy
            for cell in seed
            if cell is not None
        )
        current_hashes.extend(
            cell["cell_sha256"]
            for control in negative["cells"]
            for cell in control
            if cell is not None
        )
        if any(digest in all_cell_hashes for digest in current_hashes):
            raise RuntimeError("entry pooled reconstruction cross-fold replay reuse")
        all_cell_hashes.update(current_hashes)
        folds.append(
            {
                "outer_fold": outer_fold,
                "authorization": evidence_authorization,
                "dataset_receipt": dataset_receipt,
                "replay": replay,
                "negative": negative,
            }
        )

    per_fold: list[dict[str, int]] = []
    for source in folds:
        replay = source["replay"]
        candidate = replay["candidate"]
        p5 = replay["p5"]
        random_hgb = replay["random"][0]
        random_hold3 = sum(
            _cell_total_pnl(seed[0]) for seed in random_hgb if seed[0] is not None
        )
        random_shared3 = sum(
            _cell_total_pnl(seed[2]) for seed in random_hgb if seed[2] is not None
        )
        random_shared4 = sum(
            _cell_total_pnl(seed[3]) for seed in random_hgb if seed[3] is not None
        )
        per_fold.append(
            {
                "outer_fold": source["outer_fold"],
                "primary_session_count": len(source["authorization"].sessions),
                "shared_exit_candidate_completed_trades": len(
                    candidate[2]["evaluation"]["completed_trade_ids"]
                ),
                "time300_oof_trajectory_count": len(replay["time300"]),
                "holdflat_hgb_minus_p5_micros": _cell_total_pnl(candidate[0])
                - _cell_total_pnl(p5[0]),
                "holdflat_hgb_minus_random_eighth_micros": 8
                * _cell_total_pnl(candidate[0])
                - random_hold3,
                "shared_hgb_minus_p5_micros": _cell_total_pnl(candidate[2])
                - _cell_total_pnl(p5[2]),
                "shared_hgb_minus_random_eighth_micros": 8
                * _cell_total_pnl(candidate[2])
                - random_shared3,
                "fee4_shared_hgb_minus_p5_micros": _cell_total_pnl(candidate[3])
                - _cell_total_pnl(p5[3]),
                "fee4_shared_hgb_minus_random_eighth_micros": 8
                * _cell_total_pnl(candidate[3])
                - random_shared4,
            }
        )
    pooled_keys = entry_pooled_gate_input_spec()["closed_shapes"]["pooled"][
        "keys_in_order"
    ]
    pooled = {key: sum(row[key] for row in per_fold) for key in pooled_keys}
    action = _action_gate_from_observations(
        (
            row
            for source in folds
            for row in source["replay"]["action"]
        ),
        sessions=authorization.sessions,
        model_family="GLOBAL_HGB_ENTRY",
    )

    owner_ids = entry_matched_random_owner_policy_ids()
    random_completion: list[list[list[list[bool]]]] = []
    random_receipts: list[list[list[list[str | None]]]] = []
    for policy_index, _owner in enumerate(owner_ids):
        policy_completion: list[list[list[bool]]] = []
        policy_receipts: list[list[list[str | None]]] = []
        for source in folds:
            fold_completion: list[list[bool]] = []
            fold_receipts: list[list[str | None]] = []
            for seed_rows in source["replay"]["random"][policy_index]:
                fold_completion.append([cell is not None for cell in seed_rows])
                fold_receipts.append(
                    [None if cell is None else cell["cell_sha256"] for cell in seed_rows]
                )
            policy_completion.append(fold_completion)
            policy_receipts.append(fold_receipts)
        random_completion.append(policy_completion)
        random_receipts.append(policy_receipts)
    random_panel = {
        "policy_ids": list(owner_ids),
        "seeds": list(MATCHED_RANDOM_SEEDS),
        "channels": list(ENTRY_REPLAY_CHANNELS),
        "complete_by_policy_fold_seed_channel": random_completion,
        "evaluation_receipt_sha256s_by_policy_fold_seed_channel": random_receipts,
    }

    controls = entry_negative_control_policy_ids()
    negative_completion: list[list[list[bool]]] = []
    negative_receipts: list[list[list[str | None]]] = []
    negative_full_gate: list[bool] = []
    for control_index, control in enumerate(controls):
        completion_by_fold: list[list[bool]] = []
        receipts_by_fold: list[list[str | None]] = []
        standard_complete = True
        random_complete = True
        control_rows: list[dict[str, int]] = []
        control_actions: list[dict[str, Any]] = []
        control_survival = {
            key: 0
            for key in entry_pooled_gate_input_spec()["closed_shapes"][
                "survival_violation_counts"
            ]["keys_in_order"]
        }
        owner_index = owner_ids.index(control)
        for source in folds:
            cells = source["negative"]["cells"][control_index]
            completion_by_fold.append([cell is not None for cell in cells])
            receipts_by_fold.append(
                [None if cell is None else cell["cell_sha256"] for cell in cells]
            )
            if any(cell is None for cell in cells):
                standard_complete = False
                continue
            typed_cells = [cell for cell in cells if cell is not None]
            random_rows = source["replay"]["random"][owner_index]
            if any(cell is None for seed in random_rows for cell in seed):
                random_complete = False
            random_hold3 = sum(
                _cell_total_pnl(seed[0])
                for seed in random_rows
                if seed[0] is not None
            )
            random_shared3 = sum(
                _cell_total_pnl(seed[2])
                for seed in random_rows
                if seed[2] is not None
            )
            random_shared4 = sum(
                _cell_total_pnl(seed[3])
                for seed in random_rows
                if seed[3] is not None
            )
            p5 = source["replay"]["p5"]
            control_rows.append(
                {
                    "primary_session_count": len(source["authorization"].sessions),
                    "shared_exit_candidate_completed_trades": len(
                        typed_cells[2]["evaluation"]["completed_trade_ids"]
                    ),
                    "time300_oof_trajectory_count": len(
                        source["negative"]["time300"][control_index]
                    ),
                    "holdflat_hgb_minus_p5_micros": _cell_total_pnl(typed_cells[0])
                    - _cell_total_pnl(p5[0]),
                    "holdflat_hgb_minus_random_eighth_micros": 8
                    * _cell_total_pnl(typed_cells[0])
                    - random_hold3,
                    "shared_hgb_minus_p5_micros": _cell_total_pnl(typed_cells[2])
                    - _cell_total_pnl(p5[2]),
                    "shared_hgb_minus_random_eighth_micros": 8
                    * _cell_total_pnl(typed_cells[2])
                    - random_shared3,
                    "fee4_shared_hgb_minus_p5_micros": _cell_total_pnl(typed_cells[3])
                    - _cell_total_pnl(p5[3]),
                    "fee4_shared_hgb_minus_random_eighth_micros": 8
                    * _cell_total_pnl(typed_cells[3])
                    - random_shared4,
                }
            )
            control_actions.extend(source["negative"]["action"][control_index])
            counts = _sum_survival_counts(
                [
                    *typed_cells,
                    *(
                        cell
                        for seed in random_rows
                        for cell in seed
                        if cell is not None
                    ),
                ],
                outer_fold=source["outer_fold"],
            )
            for key, count in counts.items():
                control_survival[key] += count
        negative_completion.append(completion_by_fold)
        negative_receipts.append(receipts_by_fold)
        full_gate = False
        if standard_complete and random_complete and len(control_rows) == 5:
            control_pooled = {
                key: sum(row[key] for row in control_rows)
                for key in pooled_keys
            }
            primary_economic_and_power = (
                all(row["primary_session_count"] >= 25 for row in control_rows)
                and all(
                    row["shared_exit_candidate_completed_trades"] >= 40
                    for row in control_rows
                )
                and control_pooled["shared_exit_candidate_completed_trades"] >= 200
                and all(row["time300_oof_trajectory_count"] >= 50 for row in control_rows)
                and control_pooled["time300_oof_trajectory_count"] >= 300
                and control_pooled["holdflat_hgb_minus_p5_micros"] > 0
                and sum(row["holdflat_hgb_minus_p5_micros"] > 0 for row in control_rows) >= 4
                and control_pooled["holdflat_hgb_minus_random_eighth_micros"] > 0
                and sum(row["holdflat_hgb_minus_random_eighth_micros"] > 0 for row in control_rows) >= 4
                and control_pooled["shared_hgb_minus_p5_micros"] > 0
                and sum(row["shared_hgb_minus_p5_micros"] > 0 for row in control_rows) >= 4
                and control_pooled["shared_hgb_minus_random_eighth_micros"] > 0
                and sum(row["shared_hgb_minus_random_eighth_micros"] > 0 for row in control_rows) >= 4
                and control_pooled["fee4_shared_hgb_minus_p5_micros"] > 0
                and control_pooled["fee4_shared_hgb_minus_random_eighth_micros"] > 0
                and all(value == 0 for value in control_survival.values())
            )
            if primary_economic_and_power:
                control_action = _action_gate_from_observations(
                    control_actions,
                    sessions=authorization.sessions,
                    model_family=_negative_control_seed_family(control),
                )
                full_gate = all(
                    _action_gate_passes(control_action[name])
                    for name in ("ENTER", "WAIT")
                )
        negative_full_gate.append(full_gate)
    negative_panel = {
        "control_ids": list(controls),
        "channels": list(ENTRY_REPLAY_CHANNELS),
        "complete_by_control_fold_channel": negative_completion,
        "full_gate_passed_by_control": negative_full_gate,
        "evaluation_sha256s_by_control_fold_channel": negative_receipts,
    }

    survival_keys = entry_pooled_gate_input_spec()["closed_shapes"][
        "survival_violation_counts"
    ]["keys_in_order"]
    survival = {key: 0 for key in survival_keys}
    for source in folds:
        replay_cells = [
            *source["replay"]["candidate"],
            *source["replay"]["p5"],
            *(
                cell
                for policy in source["replay"]["random"]
                for seed in policy
                for cell in seed
                if cell is not None
            ),
            *(
                cell
                for control in source["negative"]["cells"]
                for cell in control
                if cell is not None
            ),
        ]
        counts = _sum_survival_counts(
            replay_cells, outer_fold=source["outer_fold"]
        )
        for key in survival_keys:
            survival[key] += counts[key]
    reconstructed = {
        "schema_version": "pathd.entry_pooled_gate_inputs.v1",
        "fold_order": [1, 2, 3, 4, 5],
        "per_fold": per_fold,
        "pooled": pooled,
        "action_calibration": action,
        "matched_random": random_panel,
        "negative_controls": negative_panel,
        "large_improvement_audit": _large_improvement_audit_from_sources(
            folds=folds, authorization=authorization
        ),
        "survival_violation_counts": survival,
    }
    _assert_strict_json_value(reconstructed)
    return reconstructed


def _all_nested_exact_bool(value: Any) -> bool:
    if type(value) is bool:
        return value
    if type(value) is list:
        return bool(value) and all(_all_nested_exact_bool(item) for item in value)
    raise RuntimeError("entry pooled acceptance Boolean tensor drift")


def _validate_nested_tensor(
    value: Any, dimensions: tuple[int, ...], leaf_ok: Any, owner: str
) -> None:
    if not dimensions:
        if not leaf_ok(value):
            raise RuntimeError(f"entry pooled acceptance {owner} tensor drift")
        return
    if type(value) is not list or len(value) != dimensions[0]:
        raise RuntimeError(f"entry pooled acceptance {owner} shape drift")
    for item in value:
        _validate_nested_tensor(item, dimensions[1:], leaf_ok, owner)


def _validate_entry_pooled_gate_inputs(value: Any) -> tuple[dict[str, Any], dict[str, bool], str, str | None, bool]:
    if type(value) is not dict:
        raise RuntimeError("entry pooled acceptance gate inputs are not an object")
    _assert_strict_json_value(value)
    spec = entry_pooled_gate_input_spec()
    if set(value) != set(spec["top_level_keys_in_order"]):
        raise RuntimeError("entry pooled acceptance gate-input key drift")
    if value.get("schema_version") != spec["schema_version"]:
        raise RuntimeError("entry pooled acceptance gate-input schema drift")
    if value.get("fold_order") != [1, 2, 3, 4, 5]:
        raise RuntimeError("entry pooled acceptance fold order drift")
    per_fold = value.get("per_fold")
    pooled = value.get("pooled")
    fold_keys = spec["closed_shapes"]["per_fold"]["row_keys_in_order"]
    pooled_keys = spec["closed_shapes"]["pooled"]["keys_in_order"]
    if (
        type(per_fold) is not list
        or len(per_fold) != 5
        or any(type(row) is not dict or set(row) != set(fold_keys) for row in per_fold)
        or type(pooled) is not dict
        or set(pooled) != set(pooled_keys)
    ):
        raise RuntimeError("entry pooled acceptance fold/pooled shape drift")
    for index, row in enumerate(per_fold, 1):
        if row["outer_fold"] != index or any(type(row[key]) is not int for key in fold_keys):
            raise RuntimeError("entry pooled acceptance fold value drift")
    if any(type(pooled[key]) is not int for key in pooled_keys):
        raise RuntimeError("entry pooled acceptance pooled value drift")
    for key in pooled_keys:
        if pooled[key] != sum(row[key] for row in per_fold):
            raise RuntimeError("entry pooled acceptance pooled arithmetic drift")

    action = value.get("action_calibration")
    action_keys = spec["closed_shapes"]["action_calibration"][
        "action_keys_in_order"
    ]
    if type(action) is not dict or set(action) != {"ENTER", "WAIT"}:
        raise RuntimeError("entry pooled acceptance action object drift")
    action_pass: dict[str, bool] = {}
    action_underpowered = False
    for name in ("ENTER", "WAIT"):
        row = action[name]
        if type(row) is not dict or set(row) != set(action_keys):
            raise RuntimeError("entry pooled acceptance action key drift")
        for key in (
            "distinct_trajectory_count", "coverage_numerator", "coverage_denominator"
        ):
            if type(row[key]) is not int or row[key] < 0:
                raise RuntimeError("entry pooled acceptance action count drift")
        for key, size in (
            ("decile_trajectory_counts", 10),
            ("decile_distinct_session_counts", 10),
            ("adjacent_valid_bootstrap_replicates", 9),
            ("adjacent_total_draws", 9),
        ):
            vector = row[key]
            if (
                type(vector) is not list
                or len(vector) != size
                or any(type(item) is not int or item < 0 for item in vector)
            ):
                raise RuntimeError("entry pooled acceptance action vector drift")
        upper = row["adjacent_upper_bounds_micros"]
        if (
            type(upper) is not list
            or len(upper) != 9
            or any(type(item) is not float or not math.isfinite(item) for item in upper)
        ):
            raise RuntimeError("entry pooled acceptance action bound drift")
        powered = (
            row["distinct_trajectory_count"] >= 30
            and row["coverage_denominator"] == row["distinct_trajectory_count"]
            and all(item >= 3 for item in row["decile_trajectory_counts"])
            and all(item >= 3 for item in row["decile_distinct_session_counts"])
            and all(item == 5_000 for item in row["adjacent_valid_bootstrap_replicates"])
            and all(5_000 <= item <= 50_000 for item in row["adjacent_total_draws"])
        )
        action_underpowered = action_underpowered or not powered
        action_pass[name] = (
            powered
            and row["coverage_denominator"] > 0
            and 100 * row["coverage_numerator"]
            >= 85 * row["coverage_denominator"]
            and all(item <= 0.0 for item in upper)
        )

    random_panel = value.get("matched_random")
    random_spec = spec["closed_shapes"]["matched_random"]
    if (
        type(random_panel) is not dict
        or set(random_panel) != set(random_spec["keys_in_order"])
        or random_panel.get("policy_ids") != random_spec["policy_ids"]
        or random_panel.get("seeds") != random_spec["seeds"]
        or random_panel.get("channels") != random_spec["channels"]
    ):
        raise RuntimeError("entry pooled matched-random identity drift")
    random_complete = _all_nested_exact_bool(
        random_panel["complete_by_policy_fold_seed_channel"]
    )
    random_dimensions = (len(random_spec["policy_ids"]), 5, 8, 4)
    _validate_nested_tensor(
        random_panel["complete_by_policy_fold_seed_channel"],
        random_dimensions,
        lambda item: type(item) is bool,
        "matched-random completion",
    )
    _validate_nested_tensor(
        random_panel["evaluation_receipt_sha256s_by_policy_fold_seed_channel"],
        random_dimensions,
        lambda item: item is None or _is_hex64(item),
        "matched-random receipt",
    )
    random_completion = random_panel["complete_by_policy_fold_seed_channel"]
    random_receipts = random_panel[
        "evaluation_receipt_sha256s_by_policy_fold_seed_channel"
    ]
    for policy_index in range(random_dimensions[0]):
        for fold_index in range(5):
            for seed_index in range(8):
                for channel_index in range(4):
                    complete = random_completion[policy_index][fold_index][seed_index][channel_index]
                    receipt_hash = random_receipts[policy_index][fold_index][seed_index][channel_index]
                    if complete is not (receipt_hash is not None):
                        raise RuntimeError(
                            "entry pooled matched-random completion/receipt drift"
                        )

    negative = value.get("negative_controls")
    negative_spec = spec["closed_shapes"]["negative_controls"]
    if (
        type(negative) is not dict
        or set(negative) != set(negative_spec["keys_in_order"])
        or negative.get("control_ids") != negative_spec["control_ids"]
        or negative.get("channels") != negative_spec["channels"]
        or type(negative.get("full_gate_passed_by_control")) is not list
        or len(negative["full_gate_passed_by_control"]) != 22
        or any(type(item) is not bool for item in negative["full_gate_passed_by_control"])
    ):
        raise RuntimeError("entry pooled negative-control identity drift")
    negative_complete = _all_nested_exact_bool(
        negative["complete_by_control_fold_channel"]
    )
    _validate_nested_tensor(
        negative["complete_by_control_fold_channel"],
        (22, 5, 4),
        lambda item: type(item) is bool,
        "negative-control completion",
    )
    _validate_nested_tensor(
        negative["evaluation_sha256s_by_control_fold_channel"],
        (22, 5, 4),
        lambda item: item is None or _is_hex64(item),
        "negative-control evaluation",
    )
    negative_completion = negative["complete_by_control_fold_channel"]
    negative_receipts = negative["evaluation_sha256s_by_control_fold_channel"]
    for control_index in range(22):
        for fold_index in range(5):
            for channel_index in range(4):
                complete = negative_completion[control_index][fold_index][channel_index]
                receipt_hash = negative_receipts[control_index][fold_index][channel_index]
                if complete is not (receipt_hash is not None):
                    raise RuntimeError(
                        "entry pooled negative-control completion/hash drift"
                    )
    negative_none_pass = not any(negative["full_gate_passed_by_control"])

    audit = value.get("large_improvement_audit")
    audit_keys = spec["closed_shapes"]["large_improvement_audit"]["keys_in_order"]
    expected_check_names = preregistration_payload()[0]["metrics_and_gates"][
        "large_improvement_audit"
    ]["required_checks"]
    if (
        type(audit) is not dict
        or set(audit) != set(audit_keys)
        or any(type(audit[key]) is not bool for key in ("ratio_trigger", "bootstrap_trigger", "triggered", "complete"))
        or audit["triggered"] is not (audit["ratio_trigger"] or audit["bootstrap_trigger"])
        or audit.get("required_check_names") != expected_check_names
        or type(audit.get("required_check_statuses")) is not list
        or len(audit["required_check_statuses"]) != len(expected_check_names)
        or any(item not in {"PASS", "FAIL", "MISSING", "NOT_TRIGGERED"} for item in audit["required_check_statuses"])
        or type(audit.get("required_check_receipt_sha256s")) is not list
        or len(audit["required_check_receipt_sha256s"]) != len(expected_check_names)
        or any(
            item is not None and not _is_hex64(item)
            for item in audit["required_check_receipt_sha256s"]
        )
        or (
            audit["triggered"]
            and any(
                status == "NOT_TRIGGERED"
                or (status == "MISSING") is not (receipt_hash is None)
                for status, receipt_hash in zip(
                    audit["required_check_statuses"],
                    audit["required_check_receipt_sha256s"],
                    strict=True,
                )
            )
        )
        or (
            not audit["triggered"]
            and (
                any(status != "NOT_TRIGGERED" for status in audit["required_check_statuses"])
                or any(item is not None for item in audit["required_check_receipt_sha256s"])
            )
        )
        or audit["complete"] is not (
            (not audit["triggered"])
            or all(item == "PASS" for item in audit["required_check_statuses"])
        )
    ):
        raise RuntimeError("entry pooled large-improvement audit drift")
    survival = value.get("survival_violation_counts")
    survival_keys = spec["closed_shapes"]["survival_violation_counts"][
        "keys_in_order"
    ]
    if (
        type(survival) is not dict
        or list(survival) != survival_keys
        or any(type(item) is not int or item < 0 for item in survival.values())
    ):
        raise RuntimeError("entry pooled survival ledger drift")

    criteria = {
        "outer_session_power_each": all(row["primary_session_count"] >= 25 for row in per_fold),
        "entry_trade_power_each": all(row["shared_exit_candidate_completed_trades"] >= 40 for row in per_fold),
        "entry_trade_power_pooled": pooled["shared_exit_candidate_completed_trades"] >= 200,
        "time300_trajectory_power_each": all(row["time300_oof_trajectory_count"] >= 50 for row in per_fold),
        "time300_trajectory_power_pooled": pooled["time300_oof_trajectory_count"] >= 300,
        "holdflat_vs_p5_pooled_positive": pooled["holdflat_hgb_minus_p5_micros"] > 0,
        "holdflat_vs_p5_positive_at_least_4_of_5": sum(row["holdflat_hgb_minus_p5_micros"] > 0 for row in per_fold) >= 4,
        "holdflat_vs_random_pooled_positive": pooled["holdflat_hgb_minus_random_eighth_micros"] > 0,
        "holdflat_vs_random_positive_at_least_4_of_5": sum(row["holdflat_hgb_minus_random_eighth_micros"] > 0 for row in per_fold) >= 4,
        "shared_vs_p5_pooled_positive": pooled["shared_hgb_minus_p5_micros"] > 0,
        "shared_vs_p5_positive_at_least_4_of_5": sum(row["shared_hgb_minus_p5_micros"] > 0 for row in per_fold) >= 4,
        "shared_vs_random_pooled_positive": pooled["shared_hgb_minus_random_eighth_micros"] > 0,
        "shared_vs_random_positive_at_least_4_of_5": sum(row["shared_hgb_minus_random_eighth_micros"] > 0 for row in per_fold) >= 4,
        "enter_action_gate": action_pass["ENTER"],
        "wait_action_gate": action_pass["WAIT"],
        "matched_random_all_complete": random_complete,
        "negative_controls_all_complete": negative_complete,
        "negative_controls_none_pass_full_gate": negative_none_pass,
        "fee4_vs_p5_pooled_positive": pooled["fee4_shared_hgb_minus_p5_micros"] > 0,
        "fee4_vs_random_pooled_positive": pooled["fee4_shared_hgb_minus_random_eighth_micros"] > 0,
        "large_improvement_audit_complete_if_triggered": audit["complete"],
        "survival_all_zero": all(item == 0 for item in survival.values()),
    }
    if list(criteria) != list(spec["pass_criteria_exact_keys_and_formulas"]):
        raise AssertionError("entry pooled criterion registry drift")
    power_names = list(criteria)[:5]
    if (
        any(not criteria[name] for name in power_names)
        or action_underpowered
        or not criteria["matched_random_all_complete"]
        or not criteria["negative_controls_all_complete"]
    ):
        verdict, stop_reason, full = "insufficient_evidence", "insufficient_entry_evidence", False
    elif any(
        not criteria[name]
        for name in (
            "fee4_vs_p5_pooled_positive", "fee4_vs_random_pooled_positive",
            "large_improvement_audit_complete_if_triggered", "survival_all_zero",
        )
    ):
        verdict, stop_reason, full = "owner_decision_required", None, False
    elif not all(criteria.values()):
        verdict, stop_reason, full = "no_genuine_signal", "no_genuine_entry_signal", False
    else:
        verdict, stop_reason, full = "PASS", None, True
    return dict(value), criteria, verdict, stop_reason, full


def validate_entry_pooled_acceptance_result(
    result: Any, /, *, authorization: FrozenResultAggregationAuthorization
) -> dict[str, Any]:
    """Independently validate the typed pooled result under immutable authority."""

    current = assert_entry_result_aggregation_ready(role="pooled_outer_primary")
    if type(authorization) is not FrozenResultAggregationAuthorization or authorization != current:
        raise RuntimeError("entry pooled acceptance authorization is stale or forged")
    if is_dataclass(result) and not isinstance(result, type):
        value = asdict(result)
    elif type(result) is dict:
        value = dict(result)
    else:
        raise RuntimeError("entry pooled acceptance result type drift")
    expected_fields = entry_future_api_contract()["dataclass_fields"][
        "EntryPooledAcceptanceResultV1"
    ]
    if set(value) != set(expected_fields):
        raise RuntimeError("entry pooled acceptance result field drift")
    payload = read_json(PREREG_PATH)
    assignments = read_json(SESSION_PATH)
    if (
        value.get("schema_version") != "pathd.entry_pooled_acceptance_result.v1"
        or value.get("holdout_caveat") != HOLDOUT_CAVEAT
        or value.get("aggregation_authorization_sha256")
        != stable_hash(authorization.to_dict())
        or value.get("outer_result_receipts_sha256")
        != stable_hash(list(authorization.outer_result_receipts_sha256))
        or value.get("sessions") != list(authorization.sessions)
        or value.get("sessions_sha256_newline")
        != authorization.sessions_sha256_newline
        or value.get("candidate_family") != "HGB"
        or value.get("gate_spec_sha256") != _entry_pooled_gate_spec_sha256(payload)
    ):
        raise RuntimeError("entry pooled acceptance upstream binding drift")
    expected_hgb: list[str] = []
    expected_control_exit: list[str] = []
    expected_replay: list[str] = []
    expected_negative: list[str] = []
    for fold in range(1, 6):
        _validate_outer_result_receipt_one(payload, assignments, fold)
        outer = read_json(_outer_fold_artifact_path(fold, "outer_primary_result.json"))
        outer_payload = outer.get("payload", {})
        expected_hgb.append(outer_payload.get("hgb", {}).get("result_sha256"))
        expected_control_exit.append(
            sha256_path(_outer_fold_artifact_path(fold, "control_exit.json"))
        )
        expected_replay.append(
            sha256_path(_outer_fold_artifact_path(fold, "control_replay_result.json"))
        )
        expected_negative.append(
            sha256_path(_outer_fold_artifact_path(fold, "negative_control_panel.json"))
        )
    if (
        value.get("candidate_outer_evaluation_sha256s") != expected_hgb
        or value.get("control_exit_sha256s") != expected_control_exit
        or value.get("control_replay_result_sha256s") != expected_replay
        or value.get("negative_control_panel_sha256s") != expected_negative
    ):
        raise RuntimeError("entry pooled acceptance fold artifact binding drift")
    reconstructed = _reconstruct_entry_pooled_gate_inputs_from_outer_artifacts(
        authorization=authorization
    )
    if value.get("reconstructed_gate_inputs") != reconstructed:
        raise RuntimeError(
            "entry pooled acceptance caller-authored gate inputs do not reconstruct"
        )
    inputs, criteria, verdict, stop_reason, full = _validate_entry_pooled_gate_inputs(
        reconstructed
    )
    if (
        value.get("gate_inputs_sha256") != stable_hash(inputs)
        or value.get("pass_criteria_recomputed") != criteria
        or value.get("verdict") != verdict
        or value.get("stop_reason") != stop_reason
        or value.get("full_fit_authorized") is not full
    ):
        raise RuntimeError("entry pooled acceptance recomputation drift")
    semantic = dict(value)
    digest = semantic.pop("result_sha256", None)
    if digest != stable_hash(semantic):
        raise RuntimeError("entry pooled acceptance self-hash drift")
    return value


def read_frozen_entry_pooled_acceptance(*, require_pass: bool = False) -> dict[str, Any]:
    if type(require_pass) is not bool:
        raise RuntimeError("entry pooled acceptance require_pass type drift")
    authorization = assert_entry_result_aggregation_ready(role="pooled_outer_primary")
    result_path = _canonical_repo_regular_file(
        repo_path_label(ENTRY_POOLED_ACCEPTANCE_RESULT_PATH)
    )
    receipt_path = _canonical_repo_regular_file(
        repo_path_label(ENTRY_POOLED_ACCEPTANCE_RECEIPT_PATH)
    )
    result = validate_entry_pooled_acceptance_result(
        read_json(result_path), authorization=authorization
    )
    receipt = read_json(receipt_path)
    expected_keys = {
        "schema_version", "status", "result_path", "result_sha256",
        "aggregation_authorization_sha256", "outer_result_receipts_sha256",
        "verdict", "full_fit_authorized", "receipt_sha256",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != "pathd.entry_pooled_acceptance_receipt.v1"
        or receipt.get("status") != "FROZEN_ENTRY_POOLED_ACCEPTANCE"
        or receipt.get("result_path") != repo_path_label(result_path)
        or receipt.get("result_sha256") != sha256_path(result_path)
        or receipt.get("aggregation_authorization_sha256")
        != stable_hash(authorization.to_dict())
        or receipt.get("outer_result_receipts_sha256")
        != stable_hash(list(authorization.outer_result_receipts_sha256))
        or receipt.get("verdict") != result["verdict"]
        or receipt.get("full_fit_authorized") is not result["full_fit_authorized"]
    ):
        raise RuntimeError("entry pooled acceptance receipt drift")
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    if require_pass and (
        result["verdict"] != "PASS" or result["full_fit_authorized"] is not True
    ):
        raise RuntimeError("entry full fit blocked: pooled entry acceptance did not PASS")
    return result


def _write_canonical_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_text_exclusive(path: Path, value: str) -> None:
    """Durably create one UTF-8 fixed artifact without overwrite semantics."""

    if type(value) is not str:
        raise TypeError("exclusive text artifact must be a string")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value.encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _validated_context_diagnostic_inventory_receipt(
    *,
    payload: dict[str, Any],
    preregistration_sha256: str,
    outer_result_receipt_sha256s: tuple[str, ...],
    pooled_acceptance_receipt_sha256: str,
) -> dict[str, Any]:
    """Validate the post-primary 604-file hash inventory without granting fit authority."""

    path = _canonical_repo_regular_file(
        repo_path_label(CONTEXT_DIAGNOSTIC_INVENTORY_RECEIPT_PATH)
    )
    receipt = read_json(path)
    spec = context_diagnostic_inventory_receipt_spec()
    partition = payload["corpus"]["integrity_contract"]["partitions"][
        CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION
    ]
    contracts = payload["source_hash_policy"]["receipt_contracts"]
    if (
        type(receipt) is not dict
        or set(receipt) != set(spec["fields_in_order"])
        or receipt.get("schema_version") != spec["schema_version"]
        or receipt.get("status") != spec["status"]
        or receipt.get("preregistration_sha256") != preregistration_sha256
        or receipt.get("source_hash_policy_sha256")
        != stable_hash(payload["source_hash_policy"])
        or receipt.get("command")
        != contracts["required_context_diagnostic_inventory_command"]
        or receipt.get("manifest_sha256")
        != payload["corpus"]["integrity_contract"]["manifest_sha256"]
        or receipt.get("partition_id")
        != CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION
        or receipt.get("partition_contract_sha256") != stable_hash(partition)
        or receipt.get("outer_result_receipt_sha256s")
        != list(outer_result_receipt_sha256s)
        or receipt.get("pooled_acceptance_receipt_sha256")
        != pooled_acceptance_receipt_sha256
        or receipt.get("verified_file_count") != 604
        or receipt.get("verified_total_bytes") != 8_843_589
        or receipt.get("verified_ordered_entries_semantic_sha256")
        != "11605b5f8cd88957b5b0227ef45ec8380d80cfbfae16ff0f0111c55857d8ae1c"
        or receipt.get("mismatches") != []
        or receipt.get("unreceipted_paths") != []
        or receipt.get("cache_paths_opened") != []
        or receipt.get("source_decode_performed") is not False
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T[^\s]+Z",
            str(receipt.get("verification_completed_at_utc", "")),
        )
        is None
    ):
        raise RuntimeError("entry context diagnostic inventory receipt drift")
    _validate_self_hashed_receipt(receipt, field="receipt_sha256")
    return receipt


def _context_policy_binding_row(
    *,
    outer_fold: int,
    policy_id: str,
    policy_order_index: int,
    source_kind: str,
    outer_result_receipt_sha256: str,
    source_artifact_sha256: str,
    replay_cell_sha256: str | None,
    evaluation: Mapping[str, Any],
    matched_random_seed: int | None,
) -> dict[str, Any]:
    semantic = {
        "outer_fold": outer_fold,
        "policy_id": policy_id,
        "policy_order_index": policy_order_index,
        "source_kind": source_kind,
        "outer_result_receipt_sha256": outer_result_receipt_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "replay_cell_sha256": replay_cell_sha256,
        "evaluation_sha256": evaluation.get("result_sha256"),
        "terminal_journal_sha256": evaluation.get("terminal_journal_sha256"),
        "matched_random_seed": matched_random_seed,
    }
    if (
        any(
            re.fullmatch(r"[0-9a-f]{64}", str(semantic[name])) is None
            for name in (
                "outer_result_receipt_sha256", "source_artifact_sha256",
                "evaluation_sha256", "terminal_journal_sha256",
            )
        )
        or (
            replay_cell_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", replay_cell_sha256) is None
        )
    ):
        raise RuntimeError("entry context policy journal binding hash drift")
    return {**semantic, "binding_sha256": stable_hash(semantic)}


def _reconstruct_context_policy_journal_bindings_one(
    payload: dict[str, Any], assignments: dict[str, Any], outer_fold: int
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Reopen only fixed embedded evaluations and validate every complete journal."""

    if type(outer_fold) is not int or outer_fold not in range(1, 6):
        raise RuntimeError("entry context policy journal fold drift")
    _validate_outer_result_receipt_one(payload, assignments, outer_fold)
    outer_result_path = _outer_fold_artifact_path(
        outer_fold, "outer_primary_result.json"
    )
    outer_receipt_path = _outer_fold_artifact_path(
        outer_fold, "outer_result_receipt.json"
    )
    replay_path = _outer_fold_artifact_path(outer_fold, "control_replay_result.json")
    replay_config_path = _outer_fold_artifact_path(
        outer_fold, "control_replay_config.json"
    )
    dataset_receipt_path = _outer_fold_artifact_path(
        outer_fold, "outer_primary_dataset_receipt.json"
    )
    outer = read_json(_canonical_repo_regular_file(repo_path_label(outer_result_path)))
    outer_payload = outer.get("payload")
    if type(outer_payload) is not dict:
        raise RuntimeError("entry context outer payload drift")
    from v4.research.pathd_evidence_gate import (
        read_frozen_entry_evidence_authorization,
    )
    evidence_authorization = read_frozen_entry_evidence_authorization(
        role="outer_test_primary", outer_fold=outer_fold, inner_fold=None
    )
    dataset_receipt = _validate_entry_evidence_dataset_receipt(
        _canonical_repo_regular_file(repo_path_label(dataset_receipt_path)),
        authorization=evidence_authorization,
    )
    replay = _validate_entry_control_replay_result_immutable(
        read_json(_canonical_repo_regular_file(repo_path_label(replay_path))),
        config=read_json(
            _canonical_repo_regular_file(repo_path_label(replay_config_path))
        ),
        authorization=evidence_authorization,
        dataset_receipt=dataset_receipt,
    )
    if (
        replay["raw"].get("candidate_outer_evaluation_sha256")
        != outer_payload.get("hgb", {}).get("result_sha256")
        or replay["raw"].get("neural_outer_evaluation_sha256")
        != outer_payload.get("neural", {}).get("result_sha256")
    ):
        raise RuntimeError("entry context replay/outer evaluation binding drift")
    channel_index = ENTRY_REPLAY_CHANNELS.index("SHARED_TRANSPARENT_FEE3")
    owner_index = entry_matched_random_owner_policy_ids().index("HGB")
    candidate_cell = replay["candidate"][channel_index]
    p5_cell = replay["p5"][channel_index]
    neural = outer_payload.get("neural")
    random_cells = [
        replay["random"][owner_index][seed_index][channel_index]
        for seed_index in range(len(MATCHED_RANDOM_SEEDS))
    ]
    if (
        type(neural) is not dict
        or any(cell is None for cell in random_cells)
    ):
        raise RuntimeError("entry context required policy journal is incomplete")
    outer_receipt_sha256 = sha256_path(outer_receipt_path)
    replay_artifact_sha256 = replay["raw"]["artifact_sha256"]
    rows: list[tuple[str, str, dict[str, Any], int | None, str | None, str]] = [
        (
            "HGB_SHARED_TRANSPARENT_FEE3", "REPLAY_CELL",
            candidate_cell["evaluation"], None, candidate_cell["cell_sha256"],
            replay_artifact_sha256,
        ),
        (
            "NEURAL_PRIMARY", "OUTER_PRIMARY_EMBEDDED", neural, None, None,
            sha256_path(outer_result_path),
        ),
        (
            "P5_SHARED_TRANSPARENT_FEE3", "REPLAY_CELL",
            p5_cell["evaluation"], None, p5_cell["cell_sha256"],
            replay_artifact_sha256,
        ),
    ]
    rows.extend(
        (
            f"MATCHED_RANDOM_HGB_SEED_{seed}_SHARED_TRANSPARENT_FEE3",
            "MATCHED_RANDOM_REPLAY_CELL", cell["evaluation"], seed,
            cell["cell_sha256"], replay_artifact_sha256,
        )
        for seed, cell in zip(MATCHED_RANDOM_SEEDS, random_cells, strict=True)
        if cell is not None
    )
    expected_ids = context_policy_journal_spec()["policy_ids_in_order"]
    if [row[0] for row in rows] != expected_ids:
        raise RuntimeError("entry context policy journal order drift")
    bindings = tuple(
        _context_policy_binding_row(
            outer_fold=outer_fold,
            policy_id=policy_id,
            policy_order_index=index,
            source_kind=source_kind,
            outer_result_receipt_sha256=outer_receipt_sha256,
            source_artifact_sha256=source_artifact_sha256,
            replay_cell_sha256=cell_sha256,
            evaluation=evaluation,
            matched_random_seed=seed,
        )
        for index, (
            policy_id, source_kind, evaluation, seed, cell_sha256,
            source_artifact_sha256,
        ) in enumerate(rows)
    )
    journals = tuple(
        {
            "policy_id": policy_id,
            "binding_sha256": binding["binding_sha256"],
            "evaluation": dict(evaluation),
        }
        for binding, (
            policy_id, _source_kind, evaluation, _seed, _cell_sha256,
            _source_artifact_sha256,
        ) in zip(bindings, rows, strict=True)
    )
    return bindings, journals


def _reconstruct_context_policy_journal_bindings_all(
    payload: dict[str, Any], assignments: dict[str, Any]
) -> tuple[tuple[tuple[dict[str, Any], ...], ...], tuple[str, ...], str]:
    by_fold = tuple(
        _reconstruct_context_policy_journal_bindings_one(
            payload, assignments, outer_fold
        )[0]
        for outer_fold in range(1, 6)
    )
    roots = tuple(stable_hash(list(rows)) for rows in by_fold)
    return by_fold, roots, stable_hash(list(roots))


def _expected_context_diagnostics_authorization() -> FrozenContextDiagnosticsAuthorizationV1:
    from v4.research.pathd_holdout_gate import _assert_protected_holdout_unopened

    _assert_protected_holdout_unopened()
    prereg = assert_preregistration_frozen()
    payload = read_json(PREREG_PATH)
    assignments = read_json(SESSION_PATH)
    _verify_immutable_sources(payload)
    lineage_receipt = assert_lineage_implementation_frozen()
    machinery_receipt = _assert_entry_machinery_frozen(
        prereg, payload, lineage_receipt
    )
    _validated_corpus_integrity_receipt(payload, prereg, machinery_receipt)
    read_frozen_entry_pooled_acceptance(require_pass=False)
    aggregation = assert_entry_result_aggregation_ready(role="pooled_outer_primary")
    outer_paths = tuple(
        repo_path_label(_outer_fold_artifact_path(fold, "outer_result_receipt.json"))
        for fold in range(1, 6)
    )
    outer_hashes = tuple(sha256_path(REPO_ROOT / path) for path in outer_paths)
    pooled_acceptance_receipt_sha256 = sha256_path(
        ENTRY_POOLED_ACCEPTANCE_RECEIPT_PATH
    )
    _validated_context_diagnostic_inventory_receipt(
        payload=payload,
        preregistration_sha256=prereg["preregistration_sha256"],
        outer_result_receipt_sha256s=outer_hashes,
        pooled_acceptance_receipt_sha256=pooled_acceptance_receipt_sha256,
    )
    sessions_by_fold = tuple(
        tuple(assignments["folds"][fold - 1]["outer_test_primary_1555_complete"])
        for fold in range(1, 6)
    )
    if tuple(session for fold in sessions_by_fold for session in fold) != aggregation.sessions:
        raise RuntimeError("entry context authorization session reconstruction drift")
    _, policy_binding_roots, policy_bindings_root = (
        _reconstruct_context_policy_journal_bindings_all(payload, assignments)
    )
    semantic = {
        "schema_version": FrozenContextDiagnosticsAuthorizationV1.SCHEMA_VERSION,
        "status": "FROZEN_POST_ENTRY_PRIMARY_READ_ONLY",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "plan_sha256": payload["binding_plan"]["sha256"],
        "session_assignments_sha256": sha256_path(SESSION_PATH),
        "source_hash_policy_sha256": stable_hash(payload["source_hash_policy"]),
        "corpus_integrity_receipt_sha256": sha256_path(
            CORPUS_INTEGRITY_RECEIPT_PATH
        ),
        "machinery_receipt_sha256": sha256_path(ENTRY_MACHINERY_RECEIPT_PATH),
        "holdout_open_count": 0,
        "diagnostic_inventory_receipt_path": repo_path_label(
            CONTEXT_DIAGNOSTIC_INVENTORY_RECEIPT_PATH
        ),
        "diagnostic_inventory_receipt_sha256": sha256_path(
            CONTEXT_DIAGNOSTIC_INVENTORY_RECEIPT_PATH
        ),
        "context_access_receipt_path": repo_path_label(
            ENTRY_CONTEXT_DIAGNOSTICS_ACCESS_RECEIPT_PATH
        ),
        "context_access_count": 0,
        "pooled_acceptance_receipt_path": repo_path_label(
            ENTRY_POOLED_ACCEPTANCE_RECEIPT_PATH
        ),
        "pooled_acceptance_receipt_sha256": pooled_acceptance_receipt_sha256,
        "outer_result_receipt_paths": outer_paths,
        "outer_result_receipt_sha256s": outer_hashes,
        "outer_sessions_by_fold": sessions_by_fold,
        "outer_sessions_root_sha256": stable_hash(
            [list(sessions) for sessions in sessions_by_fold]
        ),
        "policy_journal_binding_roots_by_fold": policy_binding_roots,
        "policy_journal_bindings_root_sha256": policy_bindings_root,
        "diagnostic_spec_sha256": stable_hash(context_diagnostics_spec()),
    }
    return FrozenContextDiagnosticsAuthorizationV1(
        **semantic, authorization_sha256=stable_hash(
            {
                **semantic,
                "outer_result_receipt_paths": list(outer_paths),
                "outer_result_receipt_sha256s": list(outer_hashes),
                "outer_sessions_by_fold": [list(row) for row in sessions_by_fold],
                "policy_journal_binding_roots_by_fold": list(
                    policy_binding_roots
                ),
            }
        )
    )


def assert_context_diagnostics_authorization_current(
    authorization: FrozenContextDiagnosticsAuthorizationV1, /
) -> FrozenContextDiagnosticsAuthorizationV1:
    if type(authorization) is not FrozenContextDiagnosticsAuthorizationV1:
        raise RuntimeError("entry context diagnostics authorization type drift")
    current = _expected_context_diagnostics_authorization()
    if authorization != current:
        raise RuntimeError("entry context diagnostics authorization is stale or forged")
    path = _canonical_repo_regular_file(
        repo_path_label(ENTRY_CONTEXT_DIAGNOSTICS_AUTHORIZATION_PATH)
    )
    if read_json(path) != current.to_dict():
        raise RuntimeError("entry context diagnostics authorization artifact drift")
    if os.path.lexists(ENTRY_CONTEXT_DIAGNOSTICS_ABORT_RECEIPT_PATH):
        raise RuntimeError("entry context diagnostics authorization was terminally aborted")
    return current


def read_validated_context_policy_journals(
    authorization: FrozenContextDiagnosticsAuthorizationV1,
    /,
    *,
    outer_fold: int,
) -> tuple[dict[str, Any], ...]:
    """Return only immutable-main-revalidated policy journals for one fixed fold."""

    current = assert_context_diagnostics_authorization_current(authorization)
    if type(outer_fold) is not int or outer_fold not in range(1, 6):
        raise RuntimeError("entry context policy journal fold drift")
    bindings, journals = _reconstruct_context_policy_journal_bindings_one(
        read_json(PREREG_PATH), read_json(SESSION_PATH), outer_fold
    )
    observed_root = stable_hash(list(bindings))
    if observed_root != current.policy_journal_binding_roots_by_fold[outer_fold - 1]:
        raise RuntimeError("entry context policy journal authorization root drift")
    return journals


def assert_entry_context_diagnostics_ready() -> FrozenContextDiagnosticsAuthorizationV1:
    """Mint the sole post-entry, non-fit context authority after frozen acceptance."""

    expected = _expected_context_diagnostics_authorization()
    if os.path.lexists(ENTRY_CONTEXT_DIAGNOSTICS_AUTHORIZATION_PATH):
        persisted = read_json(
            _canonical_repo_regular_file(
                repo_path_label(ENTRY_CONTEXT_DIAGNOSTICS_AUTHORIZATION_PATH)
            )
        )
        if persisted != expected.to_dict():
            raise RuntimeError("entry context diagnostics authorization artifact drift")
        return assert_context_diagnostics_authorization_current(expected)
    reserved = [
        ENTRY_CONTEXT_DIAGNOSTICS_ACCESS_RECEIPT_PATH,
        ENTRY_CONTEXT_DIAGNOSTICS_ABORT_RECEIPT_PATH,
        ENTRY_POOLED_CONTEXT_DIAGNOSTICS_PATH,
        ENTRY_POOLED_CONTEXT_DIAGNOSTICS_RECEIPT_PATH,
        *[
            _outer_fold_artifact_path(fold, name)
            for fold in range(1, 6)
            for name in (
                "entry_context_diagnostics.json",
                "entry_context_diagnostics_receipt.json",
            )
        ],
    ]
    if any(os.path.lexists(path) for path in reserved):
        raise RuntimeError("entry context diagnostics reserved path is occupied")
    _write_canonical_json_exclusive(
        ENTRY_CONTEXT_DIAGNOSTICS_AUTHORIZATION_PATH, expected.to_dict()
    )
    return assert_context_diagnostics_authorization_current(expected)


def assert_fit_authorization_current(
    authorization: FrozenFitAuthorization,
) -> FrozenFitAuthorization:
    """Revalidate immediately before a dataset read or model/calibrator fit."""

    if type(authorization) is not FrozenFitAuthorization:
        raise RuntimeError("entry fitting blocked: untrusted fit authorization type")
    current = assert_entry_fit_ready(
        role=authorization.role,
        outer_fold=authorization.outer_fold,
        inner_fold=authorization.inner_fold,
    )
    if current != authorization:
        raise RuntimeError("entry fitting blocked: fit authorization is stale or altered")
    return current


def validate_negative_lineage_fixture(name: str, payload: dict[str, Any]) -> None:
    """Fail closed for the five mandatory model-input lineage negatives."""

    if name == "ibkr_bid_renamed_pnl" and (
        payload.get("vendor") == "IBKR" or "ibkr_bid" in payload.get("leaves", [])
    ):
        raise ValueError(name)
    if name == "vendor_greek_renamed_internal_delta" and payload.get("raw_vendor_greek"):
        raise ValueError(name)
    if name == "future_ts_recv" and payload.get("available_after_decision"):
        raise ValueError(name)
    if name == "unregistered_feature" and payload.get("name") not in FEATURE_NAMES:
        raise ValueError(name)
    if name == "feature_without_live_twin":
        namespace = payload.get("namespace")
        feature_name = payload.get("name")
        if namespace in {"entry", "exit"} and type(feature_name) is str:
            record = live_twin_record(namespace, feature_name)
            if record.live_twin_class == NO_INTRADAY_LIVE_TWIN:
                raise ValueError(name)
        if not payload.get("future_live_twin"):
            raise ValueError(name)


def require_all_negative_fixtures_rejected() -> dict[str, bool]:
    fixtures = {
        "ibkr_bid_renamed_pnl": {"name": "pnl", "vendor": "IBKR", "leaves": ["ibkr_bid"]},
        "vendor_greek_renamed_internal_delta": {
            "name": "E.bs.delta",
            "raw_vendor_greek": True,
        },
        "future_ts_recv": {"name": FEATURE_NAMES[0], "available_after_decision": True},
        "unregistered_feature": {"name": "mystery_alpha"},
        "feature_without_live_twin": {
            "namespace": "exit",
            "name": "last_causal_open_interest",
            "usage": "intraday_90_second_carry",
            "future_live_twin": "fake.but.nonempty::intraday_open_interest",
        },
    }
    result: dict[str, bool] = {}
    for name, payload in fixtures.items():
        try:
            validate_negative_lineage_fixture(name, payload)
        except ValueError:
            result[name] = True
        else:
            result[name] = False
    if not all(result.values()):
        raise AssertionError(f"lineage negative fixture escaped: {result}")
    return result


def assert_no_holdout_sessions(sessions: Iterable[str]) -> None:
    assignments = session_assignments()
    protected = set(assignments["protected_holdout_30"])
    overlap = sorted(protected.intersection(str(value) for value in sessions))
    if overlap:
        raise RuntimeError(f"protected holdout access before one-shot gate: {overlap}")


def assert_no_evidence_firewall_sessions(sessions: Iterable[str]) -> None:
    """Block all indices 216-251 from fit/calibration/economic evidence pre-open."""

    assignments = session_assignments()
    firewall = set(assignments["firewall_raw_indices_216_251"])
    overlap = sorted(firewall.intersection(str(value) for value in sessions))
    if overlap:
        raise RuntimeError(f"firewall session entered pre-holdout evidence: {overlap}")
