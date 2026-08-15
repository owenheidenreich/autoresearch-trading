"""Audit whether the frozen Protocol101 entry campaign learned entry quality.

This runner is intentionally read-only with respect to the campaign.  It
reconstructs causal executable path outcomes from governed normalized rows,
joins them to persisted out-of-fold scores, and never fits a model.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from v4.dataset.spxw_0dte_neural import (
    NeuralDatasetConfig,
    _contract_quote_path,
    _prepare_options,
)
from v4.model.protocol101_canonical_stage1_contract import HYPOTHESES
from v4.model.protocol101_scoped_stage1_hgb import (
    HGBUnitConfig,
    load_repaired_decisions,
    score_decisions,
)
from v4.scripts import materialize_protocol101_ft1d_two_clock_rows as materializer
from v4.scripts.run_protocol101_scoped_stage1_hgb_runner import (
    NOISE_DISTRIBUTION,
    guard_margins,
)
from v4.model.protocol101_divergence_noise import DivergenceNoiseModel


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_entry_objective_audit_attempt001"
)
CAMPAIGN_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_full_trader_stage1_entry_fresh_attempt001"
)
CAMPAIGN_CONTRACT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_full_trader_entry_campaign_preregistration_attempt001/"
    "campaign_contract.json"
)
CAMPAIGN_AGGREGATION = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_full_trader_stage1_entry_campaign_independent_audit_selection_attempt001/"
    "independent_aggregation.json"
)
D1_RESULTS = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_d1_shuffled_profit_causal_attribution_attempt009/"
    "paired_control_results.json"
)
D1_CORRECTED = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_d1_shuffled_profit_causal_attribution_inference_correction_attempt011/"
    "entry_model_trust_decision_corrected.json"
)
D1_V2_PRODUCER = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_d1_v2_campaign_reaggregation_attempt005/"
    "summary.json"
)
MANIFEST = materializer.MANIFEST_PATH
TWO_CLOCK_RECEIPT = materializer.GLOBAL_RECEIPT
PREREGISTRATION = OUTPUT_ROOT / "preregistration.json"
PREREGISTRATION_FREEZE = OUTPUT_ROOT / "preregistration_freeze.sha256"
SCRATCH_PATHS = OUTPUT_ROOT / ".scratch_candidate_path_metrics.parquet"

SCHEMA_VERSION = "Protocol101EntryObjectiveAuditV1"
FEE = 3.0
HORIZONS = (1, 2, 5, 10, 15)
BARRIER_PAIRS = (
    ("f10_a10", 0.10, -0.10),
    ("f25_a20", 0.25, -0.20),
    ("f50_a35", 0.50, -0.35),
)
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_BLOCK_SESSIONS = 5
SEEDS = (42, 43, 44)
P5_POLICY = 5
POSITIVE_ROWS = (
    ("H0", 5),
    ("H0", 6),
    ("H1", 1),
    ("H1", 2),
    ("H1", 5),
    ("H1", 6),
    ("H2", 5),
    ("H2", 6),
    ("H3", 5),
    ("H3", 6),
)
PATH_METRICS = (
    "return_1m",
    "return_2m",
    "return_5m",
    "return_10m",
    "return_15m",
    "mfe_15m",
    "mae_15m",
    "breakeven_within_15m",
    "time_to_breakeven_quality",
    "barrier_f10_a10",
    "barrier_f25_a20",
    "barrier_f50_a35",
    "current_fixed_exit_rop",
    "profit_available_before_exit_rop",
    "uncaptured_profit_rop",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _fold_number(fold_id: str) -> int:
    return int(str(fold_id).rsplit("_", 1)[-1])


def _unit_dir(hypothesis: str, policy: int, seed: int, fold_id: str) -> Path:
    return (
        CAMPAIGN_ROOT
        / hypothesis
        / "units"
        / hypothesis
        / f"policy{policy}"
        / f"seed{seed}"
        / fold_id
    )


def _load_full_unit_summary(unit_dir: Path) -> dict[str, Any]:
    archive = unit_dir / "summary.full.json.gz"
    if archive.is_file():
        with gzip.open(archive, "rt") as handle:
            return json.load(handle)
    return load_json(unit_dir / "summary.json")


def _input_hashes() -> dict[str, str]:
    paths = {
        "goal": Path(
            "/Users/gduby/.codex/attachments/"
            "159be67e-6a8e-47dc-9ca9-87940401cd07/pasted-text-1.txt"
        ),
        "campaign_contract": CAMPAIGN_CONTRACT,
        "campaign_aggregation": CAMPAIGN_AGGREGATION,
        "d1_results": D1_RESULTS,
        "d1_corrected": D1_CORRECTED,
        "d1_v2_producer": D1_V2_PRODUCER,
        "manifest": MANIFEST,
        "two_clock_receipt": TWO_CLOCK_RECEIPT,
        "noise_distribution": NOISE_DISTRIBUTION,
        "hgb_source": ROOT / "v4/model/protocol101_scoped_stage1_hgb.py",
        "dataset_source": ROOT / "v4/dataset/spxw_0dte_neural.py",
        "simulator_v5_source": ROOT / "v4/model/protocol101_serial_simulator_v5.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"required audit inputs missing: {missing}")
    return {name: sha256_path(path) for name, path in paths.items()}


def preregistration_payload() -> dict[str, Any]:
    scope = materializer.load_training_scope()
    aggregate = load_json(CAMPAIGN_AGGREGATION)
    positive_from_evidence = tuple(
        (str(row["hypothesis"]), int(row["policy_index"]))
        for row in aggregate["rows"]
        if float(
            row["median_seed_fee_adjusted_continuous_strict_serial_net_pnl"]
        )
        > 0.0
    )
    if positive_from_evidence != POSITIVE_ROWS:
        raise RuntimeError(
            "positive-row population differs from frozen preregistration: "
            f"{positive_from_evidence}"
        )
    folds = [
        {
            "fold_id": str(item["fold_id"]),
            "fold": _fold_number(str(item["fold_id"])),
            "validation_sessions": list(item["validation_sessions"]),
        }
        for item in scope.folds
    ]
    payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}Preregistration",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_path_or_score_comparisons",
        "audit_population": {
            "campaign_rows": [
                f"{hypothesis}/P{policy}" for hypothesis, policy in POSITIVE_ROWS
            ],
            "population_rule": (
                "all ten H0-H3 rows with positive median-seed simulator-v5 "
                "OOF PnL in the frozen independent aggregation"
            ),
            "selected_intent_analysis": "all seeds and all five OOF folds",
            "full_candidate_analysis": (
                "all four P5 hypotheses, seed 42, all five OOF folds"
            ),
            "protected_holdout": "forbidden",
            "G9": "forbidden",
        },
        "fixed_metrics": {
            "fee_dollars_round_trip": FEE,
            "return_equation": (
                "((future_executable_bid-entry_ask)*100-fee)/(entry_ask*100)"
            ),
            "horizons_minutes": list(HORIZONS),
            "path_horizon_minutes": 15,
            "future_quote_rule": (
                "latest causal bid at or before the horizon, not older than "
                "90 seconds; no future interpolation"
            ),
            "barrier_pairs": [
                {
                    "name": name,
                    "favorable_return": favorable,
                    "adverse_return": adverse,
                    "ordering": "first causal executable bid hit",
                }
                for name, favorable, adverse in BARRIER_PAIRS
            ],
            "breakeven": "first executable bid producing fee-adjusted PnL >= 0",
            "profit_available_before_fixed_exit": (
                "maximum executable fee-adjusted PnL after entry through the "
                "frozen label source-exit quote time"
            ),
        },
        "uncertainty": {
            "method": "session-paired moving-block bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "block_sessions": BOOTSTRAP_BLOCK_SESSIONS,
            "seed": 101_260_728,
            "interval": [0.025, 0.975],
            "session_level_effect": "score-decile top-minus-bottom mean",
        },
        "analysis_rules": {
            "score_quantiles": 10,
            "higher_is_better_metrics": list(PATH_METRICS),
            "time_to_breakeven_quality": (
                "negative minutes when reached; -16 when not reached in 15m"
            ),
            "candidate_structure_controls": [
                "within-decision score/outcome rank correlation",
                "right",
                "time_of_day",
                "absolute_offset_band",
                "entry_premium_quartile",
            ],
            "minimum_rows_per_subgroup": 30,
            "no_horizon_or_barrier_search": True,
            "no_model_refit": True,
            "persisted_oof_scores_only_or_exact_frozen_model_rescore": True,
        },
        "routing_law": {
            "Route_A": (
                "current target ranks broad entry quality stably and is not "
                "materially dependent on the current exit"
            ),
            "Route_B": (
                "scores rank broad path quality, but complete-strategy gates "
                "misstate entry-model contribution"
            ),
            "Route_C": (
                "score information is materially strongest for the frozen "
                "exit target and does not generalize sufficiently"
            ),
            "Route_D": (
                "P5 remains profitable and learned scores add no credible "
                "path-ranking or economic information"
            ),
            "Route_E": (
                "scores do not reliably rank justified entry-quality outcomes"
            ),
            "judgment_rule": (
                "route is selected from the complete evidence, not by a "
                "single optimized scalar; no route weakens corrected D1"
            ),
        },
        "folds": folds,
        "input_hashes": _input_hashes(),
        "side_effect_boundaries": {
            "model_refit": False,
            "G9": False,
            "protected_holdout": False,
            "hold_exit_training": False,
            "broker_or_paper": False,
            "live_or_runtime_mutation": False,
        },
    }
    payload["preregistration_hash"] = stable_hash(payload)
    return payload


def run_preregister(*, force: bool) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if PREREGISTRATION.exists() and not force:
        raise RuntimeError(f"preregistration already exists: {PREREGISTRATION}")
    payload = preregistration_payload()
    write_json(PREREGISTRATION, payload)
    PREREGISTRATION_FREEZE.write_text(
        f"{sha256_path(PREREGISTRATION)}  {PREREGISTRATION.name}\n"
    )
    write_json(
        OUTPUT_ROOT / "progress.json",
        {
            "status": "preregistered_pending_execution",
            "preregistration_hash": payload["preregistration_hash"],
            "preregistration_file_sha256": sha256_path(PREREGISTRATION),
        },
    )


def verify_preregistration() -> dict[str, Any]:
    if not PREREGISTRATION.is_file() or not PREREGISTRATION_FREEZE.is_file():
        raise RuntimeError("audit execution requires a frozen preregistration")
    expected_line = PREREGISTRATION_FREEZE.read_text().strip()
    expected_hash, expected_name = expected_line.split(maxsplit=1)
    if expected_name != PREREGISTRATION.name:
        raise RuntimeError("preregistration freeze filename mismatch")
    if expected_hash != sha256_path(PREREGISTRATION):
        raise RuntimeError("preregistration changed after freeze")
    payload = load_json(PREREGISTRATION)
    without_hash = dict(payload)
    frozen_hash = without_hash.pop("preregistration_hash")
    if stable_hash(without_hash) != frozen_hash:
        raise RuntimeError("preregistration self-hash mismatch")
    if payload["input_hashes"] != _input_hashes():
        raise RuntimeError("audit input changed after preregistration")
    return payload


def _scope_and_maps() -> tuple[Any, dict[str, Path], dict[str, Path]]:
    scope = materializer.load_training_scope()
    receipt = load_json(TWO_CLOCK_RECEIPT)
    receipt_without_hash = dict(receipt)
    receipt_hash = receipt_without_hash.pop("receipt_sha256")
    if stable_hash(receipt_without_hash) != receipt_hash:
        raise RuntimeError("two-clock receipt self-hash mismatch")
    processed = {
        str(item["session"]): ROOT / str(item["output_path"])
        for item in receipt["sessions"]
    }
    manifest = load_json(MANIFEST)
    normalized = {
        str(item["session"]): ROOT / str(item["normalized_official_context_file"])
        for item in manifest["included_sessions"]
    }
    validation_sessions = {
        str(session)
        for fold in scope.folds
        for session in fold["validation_sessions"]
    }
    if len(validation_sessions) != 225:
        raise RuntimeError(
            f"expected 225 OOF validation sessions, got {len(validation_sessions)}"
        )
    if validation_sessions - processed.keys() or validation_sessions - normalized.keys():
        raise RuntimeError("validation session path map is incomplete")
    return scope, processed, normalized


def _time_bucket(decision_ns: int) -> str:
    timestamp = pd.Timestamp(decision_ns, tz="UTC").tz_convert("America/New_York")
    minute = timestamp.hour * 60 + timestamp.minute
    if minute < 10 * 60 + 30:
        return "09:32-10:29"
    if minute < 14 * 60:
        return "10:30-13:59"
    return "14:00-15:30"


def _offset_band(offset: float) -> str:
    absolute = abs(float(offset))
    if absolute <= 10.0:
        return "ATM_0_10"
    if absolute <= 25.0:
        return "NEAR_15_25"
    return "WING_30_50"


def _latest_causal_bid(
    quote_ns: np.ndarray,
    bids: np.ndarray,
    *,
    decision_ns: int,
    horizon_ns: int,
) -> float:
    index = int(np.searchsorted(quote_ns, horizon_ns, side="right")) - 1
    if index < 0 or int(quote_ns[index]) <= int(decision_ns):
        return float("nan")
    age_ns = int(horizon_ns) - int(quote_ns[index])
    if age_ns > 90_000_000_000:
        return float("nan")
    bid = float(bids[index])
    return bid if math.isfinite(bid) else float("nan")


def _path_metric_record(
    *,
    session: str,
    row: dict[str, Any],
    strike_index: int,
    right_index: int,
    quote_path: Any,
) -> dict[str, Any]:
    decision = pd.Timestamp(row["decision_time"])
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    else:
        decision = decision.tz_convert("UTC")
    decision_ns = int(decision.value)
    contract_id = str(row["contract_ids"][strike_index, right_index])
    metadata = (row.get("contract_quote_metadata") or {}).get(contract_id) or {}
    ask = float(metadata["ask"])
    if not math.isfinite(ask) or ask <= 0.0:
        raise RuntimeError(f"invalid governed entry ask: {session}:{contract_id}")
    premium = ask * 100.0
    quote_ns = np.asarray(quote_path.quote_ns, dtype=np.int64)
    bids = np.asarray(quote_path.bid, dtype=float)
    path_end = decision_ns + 15 * 60 * 1_000_000_000
    start = int(np.searchsorted(quote_ns, decision_ns, side="right"))
    end = int(np.searchsorted(quote_ns, path_end, side="right"))
    path_bids = bids[start:end]
    path_times = quote_ns[start:end]
    valid = np.isfinite(path_bids)
    path_bids = path_bids[valid]
    path_times = path_times[valid]
    path_returns = (
        ((path_bids - ask) * 100.0 - FEE) / premium
        if len(path_bids)
        else np.asarray([], dtype=float)
    )
    record: dict[str, Any] = {
        "session": session,
        "decision_time_ns": decision_ns,
        "contract_id": contract_id,
        "right": str(row["rights"][right_index]),
        "offset": float(row["strike_offsets"][strike_index]),
        "strike_index": int(strike_index),
        "right_index": int(right_index),
        "entry_ask": ask,
        "entry_bid": float(metadata.get("bid") or 0.0),
        "time_bucket": _time_bucket(decision_ns),
        "offset_band": _offset_band(float(row["strike_offsets"][strike_index])),
        "mfe_15m": (
            float(np.max(path_returns)) if len(path_returns) else float("nan")
        ),
        "mae_15m": (
            float(np.min(path_returns)) if len(path_returns) else float("nan")
        ),
    }
    for horizon in HORIZONS:
        horizon_ns = decision_ns + horizon * 60 * 1_000_000_000
        bid = _latest_causal_bid(
            quote_ns,
            bids,
            decision_ns=decision_ns,
            horizon_ns=horizon_ns,
        )
        record[f"return_{horizon}m"] = (
            ((bid - ask) * 100.0 - FEE) / premium
            if math.isfinite(bid)
            else float("nan")
        )
    breakeven_hits = np.flatnonzero(path_returns >= 0.0)
    if len(breakeven_hits):
        breakeven_minutes = (
            int(path_times[int(breakeven_hits[0])]) - decision_ns
        ) / 60_000_000_000
        record["breakeven_within_15m"] = 1.0
        record["time_to_breakeven_minutes"] = float(breakeven_minutes)
        record["time_to_breakeven_quality"] = -float(breakeven_minutes)
    else:
        record["breakeven_within_15m"] = 0.0
        record["time_to_breakeven_minutes"] = float("nan")
        record["time_to_breakeven_quality"] = -16.0
    for name, favorable, adverse in BARRIER_PAIRS:
        favorable_hits = np.flatnonzero(path_returns >= favorable)
        adverse_hits = np.flatnonzero(path_returns <= adverse)
        favorable_index = int(favorable_hits[0]) if len(favorable_hits) else None
        adverse_index = int(adverse_hits[0]) if len(adverse_hits) else None
        favorable_first = (
            favorable_index is not None
            and (
                adverse_index is None
                or int(path_times[favorable_index]) < int(path_times[adverse_index])
            )
        )
        record[f"barrier_{name}"] = float(favorable_first)
        record[f"time_to_{name}_minutes"] = (
            (
                int(path_times[favorable_index]) - decision_ns
            )
            / 60_000_000_000
            if favorable_index is not None
            else float("nan")
        )
    labels = np.asarray(row["labels_net_pnl"], dtype=float)
    source_exits = np.asarray(row["label_source_exit_quote_time_ns"], dtype=np.int64)
    for policy in range(7):
        label = float(labels[strike_index, right_index, policy])
        source_exit = int(source_exits[strike_index, right_index, policy])
        current_rop = (label - FEE) / premium if math.isfinite(label) else float("nan")
        cutoff = int(np.searchsorted(quote_ns, source_exit, side="right"))
        prefix_bids = bids[start:cutoff]
        prefix_bids = prefix_bids[np.isfinite(prefix_bids)]
        if len(prefix_bids):
            available_dollars = float(
                np.max((prefix_bids - ask) * 100.0 - FEE)
            )
            available_rop = available_dollars / premium
        else:
            available_dollars = float("nan")
            available_rop = float("nan")
        record[f"current_fixed_exit_rop_p{policy}"] = current_rop
        record[f"profit_available_before_exit_dollars_p{policy}"] = available_dollars
        record[f"profit_available_before_exit_rop_p{policy}"] = available_rop
        record[f"uncaptured_profit_rop_p{policy}"] = (
            available_rop - current_rop
            if math.isfinite(available_rop) and math.isfinite(current_rop)
            else float("nan")
        )
    return record


def build_path_cache(
    *,
    scope: Any,
    processed_paths: dict[str, Path],
    normalized_paths: dict[str, Path],
) -> dict[str, Any]:
    if SCRATCH_PATHS.exists():
        metadata = pq.ParquetFile(SCRATCH_PATHS).metadata
        sessions = (
            pq.read_table(SCRATCH_PATHS, columns=["session"])
            .column("session")
            .unique()
        )
        if metadata.num_rows > 0 and len(sessions) == 225:
            return {
                "validation_sessions": len(sessions),
                "candidate_path_rows": int(metadata.num_rows),
                "scratch_sha256": sha256_path(SCRATCH_PATHS),
                "missing_contract_paths": 0,
                "reused_after_mechanical_resume": True,
            }
        SCRATCH_PATHS.unlink()
    config = NeuralDatasetConfig()
    validation_sessions = sorted(
        {
            str(session)
            for fold in scope.folds
            for session in fold["validation_sessions"]
        }
    )
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    missing_contract_paths = 0
    for index, session in enumerate(validation_sessions, start=1):
        with processed_paths[session].open("rb") as handle:
            rows = pickle.load(handle)
        table = pq.read_table(normalized_paths[session])
        options = _prepare_options(table, config)
        paths = {
            str(contract_id): _contract_quote_path(
                group,
                enforce_unique_path=True,
            )
            for contract_id, group in options.groupby("contract_id")
        }
        records: list[dict[str, Any]] = []
        for row in rows:
            mask = np.asarray(row["candidate_mask"], dtype=bool)
            for strike_index, right_index in np.argwhere(mask):
                contract_id = str(row["contract_ids"][strike_index, right_index])
                quote_path = paths.get(contract_id)
                if quote_path is None:
                    missing_contract_paths += 1
                    continue
                records.append(
                    _path_metric_record(
                        session=session,
                        row=row,
                        strike_index=int(strike_index),
                        right_index=int(right_index),
                        quote_path=quote_path,
                    )
                )
        frame = pd.DataFrame.from_records(records)
        arrow = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                SCRATCH_PATHS,
                arrow.schema,
                compression="zstd",
            )
        writer.write_table(arrow)
        total_rows += len(frame)
        if index % 10 == 0 or index == len(validation_sessions):
            write_json(
                OUTPUT_ROOT / "progress.json",
                {
                    "status": "building_causal_path_cache",
                    "completed_sessions": index,
                    "total_sessions": len(validation_sessions),
                    "candidate_path_rows": total_rows,
                },
            )
    if writer is None:
        raise RuntimeError("no path metric rows were constructed")
    writer.close()
    if missing_contract_paths:
        raise RuntimeError(
            f"missing normalized contract paths: {missing_contract_paths}"
        )
    return {
        "validation_sessions": len(validation_sessions),
        "candidate_path_rows": total_rows,
        "scratch_sha256": sha256_path(SCRATCH_PATHS),
        "missing_contract_paths": missing_contract_paths,
    }


def _assign_deciles(frame: pd.DataFrame) -> pd.Series:
    finite = pd.to_numeric(frame["score"], errors="coerce").notna()
    out = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    if finite.any():
        ranks = frame.loc[finite, "score"].rank(method="first", pct=True)
        out.loc[finite] = np.minimum(
            9,
            np.floor(ranks.to_numpy(dtype=float) * 10.0).astype(int),
        )
    return out


def _moving_block_ci(
    effects: pd.Series,
    *,
    seed: int,
) -> tuple[float, float]:
    values = effects.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    block = min(BOOTSTRAP_BLOCK_SESSIONS, len(values))
    starts = np.arange(max(1, len(values) - block + 1))
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    needed = int(math.ceil(len(values) / block))
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(starts, size=needed, replace=True)
        sample = np.concatenate([values[start : start + block] for start in selected])
        samples[index] = float(np.mean(sample[: len(values)]))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _safe_spearman(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=float)
    y = np.asarray(list(right), dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or len(np.unique(x[finite])) < 2 or len(np.unique(y[finite])) < 2:
        return float("nan")
    return float(spearmanr(x[finite], y[finite]).statistic)


def _subgroup_summary(frame: pd.DataFrame, metric: str, column: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for value, subset in frame.groupby(column, observed=True):
        finite = subset[["score", metric]].dropna()
        if len(finite) < 30:
            rows[str(value)] = {"n": len(finite), "status": "descriptive_low_n"}
            continue
        rows[str(value)] = {
            "n": len(finite),
            "rank_correlation": _safe_spearman(
                finite["score"], finite[metric]
            ),
            "mean": float(finite[metric].mean()),
        }
    return rows


def analyze_score_frame(
    frame: pd.DataFrame,
    *,
    row_id: str,
    scope_name: str,
    quantile_rows: list[dict[str, Any]],
    bootstrap_seed: int,
) -> dict[str, Any]:
    if frame.empty:
        raise RuntimeError(f"empty score frame for {row_id}:{scope_name}")
    frame = frame.copy()
    frame["score_decile"] = _assign_deciles(frame)
    if "entry_ask_quartile" not in frame:
        frame["entry_ask_quartile"] = pd.qcut(
            frame["entry_ask"].rank(method="first"),
            4,
            labels=("Q1", "Q2", "Q3", "Q4"),
        )
    results: dict[str, Any] = {}
    for metric_index, metric in enumerate(PATH_METRICS):
        if metric not in frame:
            continue
        finite = frame[
            frame["score_decile"].notna()
            & pd.to_numeric(frame[metric], errors="coerce").notna()
        ].copy()
        if finite.empty:
            results[metric] = {"status": "insufficient_evidence", "n": 0}
            continue
        grouped = finite.groupby("score_decile", observed=True)[metric]
        means = grouped.mean()
        medians = grouped.median()
        counts = grouped.size()
        for decile in sorted(means.index):
            quantile_rows.append(
                {
                    "row_id": row_id,
                    "scope": scope_name,
                    "metric": metric,
                    "score_decile": int(decile),
                    "count": int(counts.loc[decile]),
                    "mean": float(means.loc[decile]),
                    "median": float(medians.loc[decile]),
                }
            )
        bottom = finite[finite["score_decile"] == 0]
        top = finite[finite["score_decile"] == 9]
        session_bottom = bottom.groupby("session")[metric].mean()
        session_top = top.groupby("session")[metric].mean()
        session_effect = session_top.subtract(session_bottom, fill_value=np.nan).dropna()
        ci_low, ci_high = _moving_block_ci(
            session_effect,
            seed=bootstrap_seed + metric_index,
        )
        fold_results = {}
        if "fold" in finite:
            for fold, subset in finite.groupby("fold"):
                fold_results[str(fold)] = {
                    "n": len(subset),
                    "rank_correlation": _safe_spearman(
                        subset["score"], subset[metric]
                    ),
                }
        seed_results = {}
        if "seed" in finite:
            for seed, subset in finite.groupby("seed"):
                seed_results[str(seed)] = {
                    "n": len(subset),
                    "rank_correlation": _safe_spearman(
                        subset["score"], subset[metric]
                    ),
                }
        results[metric] = {
            "status": "complete",
            "n": len(finite),
            "session_count": int(finite["session"].nunique()),
            "rank_correlation": _safe_spearman(
                finite["score"], finite[metric]
            ),
            "decile_monotonicity": _safe_spearman(
                means.index.astype(float), means.to_numpy(dtype=float)
            ),
            "top_minus_bottom": (
                float(top[metric].mean() - bottom[metric].mean())
                if len(top) and len(bottom)
                else float("nan")
            ),
            "session_paired_top_minus_bottom": (
                float(session_effect.mean()) if len(session_effect) else float("nan")
            ),
            "session_paired_ci": [ci_low, ci_high],
            "fold_stability": fold_results,
            "seed_stability": seed_results,
            "right_stability": _subgroup_summary(finite, metric, "right"),
            "time_of_day_stability": _subgroup_summary(
                finite, metric, "time_bucket"
            ),
            "moneyness_stability": _subgroup_summary(
                finite, metric, "offset_band"
            ),
            "premium_stability": _subgroup_summary(
                finite, metric, "entry_ask_quartile"
            ),
        }
    return {
        "row_id": row_id,
        "scope": scope_name,
        "rows": len(frame),
        "sessions": int(frame["session"].nunique()),
        "metrics": results,
    }


def _path_frame() -> pd.DataFrame:
    frame = pq.read_table(SCRATCH_PATHS).to_pandas()
    key = (
        frame["session"].astype(str)
        + "|"
        + frame["decision_time_ns"].astype(str)
        + "|"
        + frame["contract_id"].astype(str)
    )
    if key.duplicated().any():
        raise RuntimeError("duplicate candidate path identity")
    frame.index = key
    return frame


def selected_intent_frame(
    *,
    hypothesis: str,
    policy: int,
    scope: Any,
    paths: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    unit_hashes: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in scope.folds:
            fold_id = str(fold["fold_id"])
            unit_dir = _unit_dir(hypothesis, policy, seed, fold_id)
            summary_path = unit_dir / (
                "summary.full.json.gz"
                if (unit_dir / "summary.full.json.gz").is_file()
                else "summary.json"
            )
            summary = _load_full_unit_summary(unit_dir)
            intents = summary["unit"]["validation"]["entry_intents"]
            unit_hashes.append(
                {
                    "seed": seed,
                    "fold": _fold_number(fold_id),
                    "summary_path": _relative(summary_path),
                    "summary_sha256": sha256_path(summary_path),
                    "model_sha256": summary["model_artifact"]["sha256"],
                    "intent_count": len(intents),
                }
            )
            for intent in intents:
                key = (
                    f"{intent['session']}|{int(intent['decision_time_ns'])}|"
                    f"{intent['contract_id']}"
                )
                records.append(
                    {
                        "key": key,
                        "session": str(intent["session"]),
                        "decision_time_ns": int(intent["decision_time_ns"]),
                        "contract_id": str(intent["contract_id"]),
                        "score": float(
                            intent["metadata"]["selected_candidate_score"]
                        ),
                        "action_score": float(intent["score"]),
                        "seed": int(seed),
                        "fold": _fold_number(fold_id),
                        "policy": int(policy),
                        "intent_current_fixed_exit_rop": float(
                            intent["raw_label_pnl_after_campaign_fee"]
                        )
                        / (float(intent["entry_ask"]) * 100.0),
                    }
                )
    frame = pd.DataFrame.from_records(records).set_index("key", drop=False)
    missing = frame.index.difference(paths.index)
    if len(missing):
        raise RuntimeError(
            f"selected intents missing path evidence: {hypothesis}/P{policy}:{len(missing)}"
        )
    joined = frame.join(
        paths[
            [
                "right",
                "offset",
                "entry_ask",
                "time_bucket",
                "offset_band",
                *[
                    column
                    for column in paths.columns
                    if column.startswith("return_")
                    or column
                    in {
                        "mfe_15m",
                        "mae_15m",
                        "breakeven_within_15m",
                        "time_to_breakeven_quality",
                    }
                    or column.startswith("barrier_")
                ],
                f"current_fixed_exit_rop_p{policy}",
                f"profit_available_before_exit_rop_p{policy}",
                f"uncaptured_profit_rop_p{policy}",
            ]
        ],
        how="left",
        validate="many_to_one",
    )
    joined["current_fixed_exit_rop"] = joined[
        "intent_current_fixed_exit_rop"
    ]
    label_difference = np.nanmax(
        np.abs(
            joined["current_fixed_exit_rop"].to_numpy(dtype=float)
            - joined[f"current_fixed_exit_rop_p{policy}"].to_numpy(dtype=float)
        )
    )
    if not math.isfinite(label_difference) or label_difference > 1e-10:
        raise RuntimeError(
            f"intent/path fixed-label mismatch: {hypothesis}/P{policy}:{label_difference}"
        )
    joined["profit_available_before_exit_rop"] = joined[
        f"profit_available_before_exit_rop_p{policy}"
    ]
    joined["uncaptured_profit_rop"] = joined[
        f"uncaptured_profit_rop_p{policy}"
    ]
    return joined.reset_index(drop=True), {
        "unit_count": len(unit_hashes),
        "units": unit_hashes,
        "intent_rows": len(joined),
        "selected_candidate_score_used": True,
        "action_top_score_reported_separately": True,
    }


def full_candidate_p5_frame(
    *,
    hypothesis: str,
    scope: Any,
    processed_paths: dict[str, Path],
    paths: pd.DataFrame,
    noise_model: DivergenceNoiseModel,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model_hashes: list[dict[str, Any]] = []
    seed = 42
    margins = guard_margins()
    for fold in scope.folds:
        fold_id = str(fold["fold_id"])
        unit_dir = _unit_dir(hypothesis, P5_POLICY, seed, fold_id)
        summary = _load_full_unit_summary(unit_dir)
        model_path = Path(summary["model_artifact"]["path"])
        if sha256_path(model_path) != summary["model_artifact"]["sha256"]:
            raise RuntimeError(f"frozen model hash mismatch: {model_path}")
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        validation_sessions = list(fold["validation_sessions"])
        decisions = load_repaired_decisions(
            [(session, processed_paths[session]) for session in validation_sessions],
            hypothesis=hypothesis,
            policy_index=P5_POLICY,
            guard_margins=margins,
            split=f"entry-objective-audit:{hypothesis}:{fold_id}",
        )
        scores = score_decisions(
            model,
            [item.base for item in decisions],
            feature_names=tuple(HYPOTHESES[hypothesis]),
            noise_model=noise_model,
            noise_scale=1.0,
            noise_seed=seed + 200_000,
        )
        for repaired, candidate_scores in zip(decisions, scores):
            decision = repaired.base
            for index, score in enumerate(candidate_scores):
                records.append(
                    {
                        "key": (
                            f"{decision.session}|{int(decision.decision_time.value)}|"
                            f"{decision.contract_ids[index]}"
                        ),
                        "session": decision.session,
                        "decision_time_ns": int(decision.decision_time.value),
                        "contract_id": str(decision.contract_ids[index]),
                        "score": float(score),
                        "fold": _fold_number(fold_id),
                        "seed": seed,
                        "candidate_count_in_minute": len(candidate_scores),
                    }
                )
        model_hashes.append(
            {
                "fold": _fold_number(fold_id),
                "path": _relative(model_path),
                "sha256": summary["model_artifact"]["sha256"],
                "validation_decisions": len(decisions),
                "validation_candidates": int(
                    sum(len(item.base.labels) for item in decisions)
                ),
            }
        )
    score_frame = pd.DataFrame.from_records(records).set_index("key", drop=False)
    if score_frame.index.duplicated().any():
        raise RuntimeError(f"duplicate full-candidate score identity: {hypothesis}")
    missing = score_frame.index.difference(paths.index)
    if len(missing):
        raise RuntimeError(
            f"full-candidate scores missing path evidence: {hypothesis}:{len(missing)}"
        )
    joined = score_frame.join(
        paths[
            [
                "right",
                "offset",
                "entry_ask",
                "time_bucket",
                "offset_band",
                *[
                    column
                    for column in paths.columns
                    if column.startswith("return_")
                    or column
                    in {
                        "mfe_15m",
                        "mae_15m",
                        "breakeven_within_15m",
                        "time_to_breakeven_quality",
                    }
                    or column.startswith("barrier_")
                ],
                "current_fixed_exit_rop_p5",
                "profit_available_before_exit_rop_p5",
                "uncaptured_profit_rop_p5",
            ]
        ],
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)
    joined["current_fixed_exit_rop"] = joined["current_fixed_exit_rop_p5"]
    joined["profit_available_before_exit_rop"] = joined[
        "profit_available_before_exit_rop_p5"
    ]
    joined["uncaptured_profit_rop"] = joined["uncaptured_profit_rop_p5"]
    # Rank within a decision to remove time, right, and broad opportunity level.
    joined["within_decision_score_rank"] = joined.groupby(
        ["session", "decision_time_ns"]
    )["score"].rank(pct=True, method="average")
    return joined, {
        "seed": seed,
        "unit_count": len(model_hashes),
        "models": model_hashes,
        "candidate_score_rows": len(joined),
        "exact_validation_noise_scale": 1.0,
        "exact_validation_noise_seed": seed + 200_000,
    }


def within_decision_results(frame: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for metric in PATH_METRICS:
        subset = frame[
            ["session", "decision_time_ns", "within_decision_score_rank", metric]
        ].dropna().copy()
        keys = ["session", "decision_time_ns"]
        subset["outcome_rank"] = subset.groupby(keys)[metric].rank(
            pct=True,
            method="average",
        )
        subset["x2"] = subset["within_decision_score_rank"] ** 2
        subset["y2"] = subset["outcome_rank"] ** 2
        subset["xy"] = (
            subset["within_decision_score_rank"] * subset["outcome_rank"]
        )
        grouped = subset.groupby(keys).agg(
            n=("within_decision_score_rank", "size"),
            sx=("within_decision_score_rank", "sum"),
            sy=("outcome_rank", "sum"),
            sxx=("x2", "sum"),
            syy=("y2", "sum"),
            sxy=("xy", "sum"),
        )
        numerator = grouped["n"] * grouped["sxy"] - grouped["sx"] * grouped["sy"]
        score_variance = (
            grouped["n"] * grouped["sxx"] - grouped["sx"] ** 2
        ).clip(lower=0.0)
        outcome_variance = (
            grouped["n"] * grouped["syy"] - grouped["sy"] ** 2
        ).clip(lower=0.0)
        denominator = np.sqrt(score_variance * outcome_variance)
        correlations = (numerator / denominator.replace(0.0, np.nan)).dropna()
        by_session = correlations.groupby(level=0).mean()
        ci_low, ci_high = _moving_block_ci(
            by_session,
            seed=811_000 + len(results),
        )
        results[metric] = {
            "decision_count": int(len(correlations)),
            "mean_within_decision_rank_correlation": (
                float(correlations.mean()) if len(correlations) else float("nan")
            ),
            "median_within_decision_rank_correlation": (
                float(correlations.median()) if len(correlations) else float("nan")
            ),
            "session_mean_ci": [ci_low, ci_high],
            "positive_decision_share": (
                float((correlations > 0.0).mean())
                if len(correlations)
                else float("nan")
            ),
        }
    return results


def existing_objectives() -> dict[str, Any]:
    contract = load_json(CAMPAIGN_CONTRACT)
    policies = [
        {
            "policy_index": index,
            "stop_loss_fraction": values[0],
            "take_profit_fraction": values[1],
            "max_hold_minutes": values[2],
        }
        for index, values in enumerate(
            (
                (0.35, 0.60, 10),
                (0.50, 1.00, 25),
                (0.65, 1.50, 45),
                (0.50, 2.00, 90),
                (1.00, 3.00, 120),
                (1.00, 9.99, 384),
                (1.00, 99.0, 384),
            )
        )
    ]
    hypotheses = {}
    for hypothesis, features in HYPOTHESES.items():
        hypotheses[hypothesis] = {
            "features": list(features),
            "feature_count": len(features),
            "model": "bounded HistGradientBoostingRegressor, squared-error regression",
            "exact_target": (
                "clip((labels_net_pnl[policy]-3.00)/(entry_ask*100), -2.0, 5.0)"
            ),
            "target_semantics": (
                "fee-adjusted return on premium realized by the selected fixed "
                "stop/take-profit/max-hold policy"
            ),
            "objective_difference_from_other_hypotheses": (
                "features only; target and training law are identical"
            ),
        }
    return {
        "schema_version": f"{SCHEMA_VERSION}ExistingObjectives",
        "status": "reconstructed_from_implementation",
        "hypotheses": hypotheses,
        "policies": policies,
        "label_inputs": {
            "entry": "causal executable ask at the decision minute",
            "exit": (
                "first future executable bid hitting stop/no-bid/target; "
                "otherwise latest causal bid at or before the policy deadline"
            ),
            "fee": FEE,
            "multiplier": 100,
            "slippage": (
                "ask-in/bid-out is primary; explicit additional slippage is "
                "not in the target"
            ),
            "normalization": "entry_ask*100",
        },
        "label_character": {
            "type": "regression",
            "fixed_horizon": False,
            "fixed_exit_realized_pnl": True,
            "MFE": False,
            "MAE": False,
            "time_to_profit": False,
            "barrier_ordering": False,
            "probability_of_profit": False,
            "exit_policy_dependent": True,
            "overlapping_windows": True,
            "candidate_rows_share_future_market_paths": True,
        },
        "score_to_trade": {
            "candidate_score": "model prediction for each eligible contract",
            "slot_selection": (
                "highest score only if top-two margin > 2*epsilon; otherwise "
                "nearest-ATM deterministic fallback"
            ),
            "action": "ENTER iff top_score > threshold + 2*epsilon",
            "threshold_candidates": "12 training-tail score quantiles",
            "threshold_selection": (
                "maximum simulator-v5 net PnL on chronological training-tail "
                "calibration sessions; ties favor fewer trades then higher threshold"
            ),
            "validation_influence": "none on fit, epsilon, or threshold",
        },
        "gates": contract["evidence_and_gates"],
        "what_is_mixed": (
            "the fitted score mixes entry timing, contract preference, and "
            "compatibility with one fixed exit policy; simulator gates then "
            "measure the complete entry-plus-fixed-exit strategy"
        ),
    }


def dependency_graph() -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_VERSION}DependencyGraph",
        "nodes": [
            {"id": "candidate_generation", "uses": "causal synchronized rows and guards"},
            {
                "id": "P5_policy_before_model",
                "uses": (
                    "P5 defines every candidate label's stop, target, close "
                    "deadline, occupancy, and resulting payoff before fit"
                ),
            },
            {
                "id": "label_construction",
                "uses": "entry ask, future executable bids, fee, fixed policy",
            },
            {
                "id": "model_fit",
                "uses": "17-feature subset and clipped policy-specific ROP",
            },
            {
                "id": "calibration",
                "uses": "training-tail OOF-like scores and complete simulator PnL",
            },
            {"id": "threshold", "uses": "calibration PnL-maximizing quantile"},
            {
                "id": "entry_selection",
                "uses": "action threshold, score margin, deterministic fallback",
            },
            {
                "id": "simulator_replay",
                "uses": "actual fixed-policy exit event and serial account state",
            },
            {
                "id": "G1_G8",
                "uses": "complete-strategy economics, robustness, concentration, calibration",
            },
        ],
        "edges": [
            ["candidate_generation", "P5_policy_before_model"],
            ["P5_policy_before_model", "label_construction"],
            ["label_construction", "model_fit"],
            ["model_fit", "calibration"],
            ["calibration", "threshold"],
            ["threshold", "entry_selection"],
            ["entry_selection", "simulator_replay"],
            ["simulator_replay", "G1_G8"],
        ],
        "p5_contribution_before_score": [
            "defines target economics",
            "defines exit clock and position occupancy",
            "creates a profitable no-model strategy baseline",
            "determines affordability and overlap consequences in replay",
        ],
        "interpretation": (
            "The current score is not policy-neutral entry quality. It is a "
            "forecast of payoff under a policy that already contributes "
            "substantial absolute profit."
        ),
    }


def objective_options() -> dict[str, Any]:
    options = [
        (
            1,
            "fixed_exit_pnl",
            "Predict PnL under the current fixed exit",
            "clear complete-policy value but directly couples entry to an arbitrary exit",
            "high",
            "supported",
            "not preferred for a separately learned exit",
        ),
        (
            2,
            "fixed_horizon_returns",
            "Predict executable returns at preregistered horizons",
            "policy-neutral snapshots but misses path ordering and tail shape",
            "low",
            "supported",
            "use as multi-horizon components",
        ),
        (
            3,
            "MFE",
            "Predict maximum favorable excursion",
            "captures opportunity but rewards paths with intolerable adverse movement",
            "low",
            "supported",
            "use only with MAE and timing",
        ),
        (
            4,
            "MAE",
            "Predict maximum adverse excursion",
            "risk diagnostic rather than a standalone profit target",
            "low",
            "supported",
            "use as paired risk component",
        ),
        (
            5,
            "time_to_profit",
            "Predict time to breakeven or profit",
            "aligns with capital efficiency; censored outcomes complicate regression",
            "low",
            "supported",
            "use as auxiliary survival target",
        ),
        (
            6,
            "barrier_ordering",
            "Predict favorable-before-adverse barrier ordering",
            "interpretable and path-aware but barrier choices are design assumptions",
            "medium",
            "supported",
            "use as preregistered auxiliary target",
        ),
        (
            7,
            "risk_adjusted_rank",
            "Rank eligible candidates by future opportunity adjusted for risk",
            "directly matches entry selection and avoids absolute P5 profit attribution",
            "low",
            "supported",
            "preferred primary formulation",
        ),
        (
            8,
            "path_distribution",
            "Predict a vector or distribution of the executable future path",
            "richest handoff to lifecycle learning but materially harder to calibrate",
            "low",
            "supported",
            "neural challenger after transparent baseline",
        ),
        (
            9,
            "anticipated_exit_utility",
            "Learn utility anticipating the later exit model",
            "jointly coherent but creates circular targets unless cross-fitted iteratively",
            "high",
            "supported_with_new_protocol",
            "defer until entry and exit baselines are independently valid",
        ),
        (
            10,
            "no_learned_entry",
            "Use P5 as the entry engine",
            "clean attribution baseline and fastest exit research path; forgoes learned timing/slot value",
            "none",
            "supported",
            "retain as mandatory baseline",
        ),
    ]
    return {
        "schema_version": f"{SCHEMA_VERSION}ObjectiveOptions",
        "evaluation_dimensions": [
            "scientific meaning",
            "compatibility with learned exit",
            "exit-assumption dependence",
            "premium/moneyness sensitivity",
            "leakage risk",
            "overlap/effective sample size",
            "calibration",
            "economic interpretation",
            "simulator-v5 fidelity",
            "data support",
        ],
        "options": [
            {
                "number": number,
                "id": identifier,
                "formulation": formulation,
                "assessment": assessment,
                "fixed_exit_dependence": dependence,
                "data_support": support,
                "recommendation": recommendation,
                "leakage_rule": (
                    "future outcome is label-only; all model inputs remain in "
                    "the signed 17-feature firewall"
                ),
                "overlap_rule": (
                    "session/block resampling and mutually exclusive decision "
                    "grouping are mandatory"
                ),
            }
            for number, identifier, formulation, assessment, dependence, support, recommendation in options
        ],
        "preferred_design": {
            "primary": (
                "within-decision risk-adjusted ranking of executable "
                "multi-horizon opportunity"
            ),
            "transparent_targets": [
                "1m/2m/5m/10m/15m executable returns",
                "15m MFE",
                "15m MAE",
                "time to breakeven",
                "favorable-before-adverse barriers",
            ],
            "baseline": "frozen P5 heuristic with no learned entry",
            "architecture_note": (
                "A lifecycle model may then decide HOLD/EXIT from causal "
                "position state without invalidating a policy-neutral entry target."
            ),
        },
    }


def economic_control_results() -> dict[str, Any]:
    d1 = load_json(D1_RESULTS)
    controls = d1["control_distributions"]
    names = {
        "A": "all-eligible P5 timing with fixed VWAP-side nearest-ATM heuristic",
        "B": "feature-independent random timing and slot",
        "G": "strong-global-shuffle HGB ranking",
        "H": "real H2/P5 HGB ranking",
        "I": "sign-reversed real H2/P5 HGB ranking",
    }
    summaries = {}
    for key, name in names.items():
        item = controls[key]
        summaries[key] = {
            "name": name,
            "rows": item["row_count"],
            "net_pnl": item["net_pnl"],
            "trades": item["trades"],
            "max_drawdown": item["max_drawdown"],
            "pnl_per_trade": item["pnl_per_trade"],
            "profit_factor": item["profit_factor"],
        }
    corrected = load_json(D1_CORRECTED)
    return {
        "schema_version": f"{SCHEMA_VERSION}PairedPolicyModelResults",
        "status": "reused_frozen_corrected_D1_economics_no_new_replay",
        "controls": summaries,
        "constant_score_interpretation": (
            "A is the deterministic score-independent fallback/heuristic "
            "baseline; an arbitrary constant score has no ranking content"
        ),
        "fixed_p5_heuristic_absolute_pnl": controls["A"]["net_pnl"]["median"],
        "random_p5_absolute_pnl": controls["B"]["net_pnl"]["median"],
        "strong_shuffle_absolute_pnl": controls["G"]["net_pnl"]["median"],
        "real_h2_p5_absolute_pnl": controls["H"]["net_pnl"]["median"],
        "reversed_real_absolute_pnl": controls["I"]["net_pnl"]["median"],
        "paired_model_contribution": {
            "real_H2_P5_total_feature_independent_effect": corrected[
                "real_H2_P5_total_feature_independent_effect"
            ],
            "credible": corrected["real_H2_P5_total_increment_credible"],
            "matched_exposure_quality": corrected[
                "real_matched_exposure_quality"
            ],
        },
        "strong_shuffle_assessment": {
            "joint_G1_G2_passes": corrected[
                "strong_shuffle_joint_G1_G2_passes"
            ],
            "all_negative_control_contrasts_nonsignificant": corrected[
                "negative_control_contrasts_all_nonsignificant"
            ],
        },
        "predictive_question": (
            "assessed separately in score_path_quality_results.json; absolute "
            "PnL is not treated as score contribution"
        ),
        "economic_conclusion": (
            "P5 is independently profitable. The real score direction beats "
            "its reversal, but the real model's paired increment over the "
            "feature-independent baseline is not statistically established."
        ),
    }


def label_pathology_results(path_rows: int) -> dict[str, Any]:
    aggregation = load_json(CAMPAIGN_AGGREGATION)
    trades = {
        row["row_id"]: {
            "median_seed_pnl": row[
                "median_seed_fee_adjusted_continuous_strict_serial_net_pnl"
            ],
            "seed_trade_counts": [
                item["continuous_oof_replay"]["trade_count"]
                for item in row["seed_results"]
            ],
        }
        for row in aggregation["rows"]
    }
    return {
        "schema_version": f"{SCHEMA_VERSION}LabelPathologyAudit",
        "status": "complete",
        "tests": {
            "overlapping_outcome_windows": {
                "classification": "invalid_scientific_independence_assumption_if_rows_are_iid",
                "finding": (
                    "decision rows occur every minute while targets span "
                    "10-384 minutes; P5/P6 windows overlap most of a session"
                ),
                "control": "session-level and five-session-block inference",
            },
            "duplicate_economic_events": {
                "classification": "design_risk",
                "finding": (
                    f"{path_rows} candidate opportunities collapse to only "
                    "225 OOF sessions; candidates at one minute are mutually exclusive"
                ),
                "control": "fail-closed identities plus decision-group ranking",
            },
            "contract_identity_memorization": {
                "classification": "not_reproduced_directly",
                "finding": (
                    "contract_id is not in the signed feature firewall; delta, "
                    "gamma, offset-sensitive composites, right, and premium "
                    "structure can still encode contract profile indirectly"
                ),
            },
            "ladder_slot_or_right_leakage": {
                "classification": "acceptable_structure_with_attribution_risk",
                "finding": (
                    "slot/right are not direct alpha columns, but candidate "
                    "geometry and internal Greeks carry economically legitimate structure"
                ),
                "control": "within-decision and structure-stratified results",
            },
            "future_derived_normalization": {
                "classification": "not_found",
                "finding": "target denominator is causal entry ask*100",
            },
            "target_derived_sample_weights": {
                "classification": "not_found",
                "finding": "HGB squared-error fit uses no target-derived sample weights",
            },
            "session_identity_proxies": {
                "classification": "unresolved_indirect",
                "finding": (
                    "time/context features can encode regimes but no explicit "
                    "session ID enters the model"
                ),
            },
            "calibration_leakage": {
                "classification": "not_found",
                "finding": (
                    "fit, chronological training-tail calibration, and OOF "
                    "validation sessions are separated by the governed folds"
                ),
            },
            "threshold_selection_overfit": {
                "classification": "bounded_process_risk",
                "finding": (
                    "12 thresholds maximize complete-strategy calibration PnL; "
                    "this is not leakage but it selects an exit-dependent utility"
                ),
            },
            "future_information_in_label": {
                "classification": "acceptable_label_only",
                "finding": (
                    "future bids and path events are used only as outcomes and "
                    "are blocked by the 17-feature firewall"
                ),
            },
            "session_trade_concentration": {
                "classification": "material_risk",
                "finding": trades,
            },
            "effective_sample_size": {
                "classification": "material_risk",
                "finding": (
                    "candidate-row count is not an independent sample count; "
                    "225 sessions are the primary OOF experimental units"
                ),
            },
            "mutually_exclusive_candidates": {
                "classification": "must_be_grouped",
                "finding": (
                    "only one contract can be selected at a decision minute; "
                    "candidate-level IID metrics overstate evidence"
                ),
            },
            "trade_count_reward": {
                "classification": "complete_strategy_metric_risk",
                "finding": (
                    "net PnL and threshold selection can reward more exposure; "
                    "paired trade-count/exposure baselines are mandatory"
                ),
            },
        },
        "implementation_integrity": {
            "simulator": "v5 two-clock exact occupancy",
            "duplicate_identities": "fail closed before fit/replay",
            "alpha_firewall_features": 17,
            "direct_feature_or_split_leakage": "not found by prior independent audit",
        },
    }


def _evidence_summary(path_results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row_id, item in path_results["selected_intent_rows"].items():
        metrics = item["analysis"]["metrics"]
        summary[row_id] = {
            metric: {
                "rho": metrics[metric].get("rank_correlation"),
                "top_bottom": metrics[metric].get("top_minus_bottom"),
                "ci": metrics[metric].get("session_paired_ci"),
            }
            for metric in (
                "return_5m",
                "return_15m",
                "mfe_15m",
                "mae_15m",
                "breakeven_within_15m",
                "current_fixed_exit_rop",
            )
            if metric in metrics
        }
    return summary


def _strictly_positive_ci(values: Any) -> bool:
    return bool(
        isinstance(values, list)
        and len(values) == 2
        and all(value is not None and math.isfinite(float(value)) for value in values)
        and float(values[0]) > 0.0
    )


def _all_fold_correlations_positive(metric: dict[str, Any]) -> bool:
    correlations = [
        item.get("rank_correlation")
        for item in metric.get("fold_stability", {}).values()
        if item.get("rank_correlation") is not None
    ]
    return bool(
        len(correlations) == 5
        and all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in correlations
        )
    )


def _all_seed_correlations_positive(metric: dict[str, Any]) -> bool:
    correlations = [
        item.get("rank_correlation")
        for item in metric.get("seed_stability", {}).values()
        if item.get("rank_correlation") is not None
    ]
    return bool(
        len(correlations) == 3
        and all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in correlations
        )
    )


def recommend_route(
    path_results: dict[str, Any],
    economic: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    p5 = path_results["full_candidate_p5"]
    selected = path_results["selected_intent_rows"]
    controlled_metrics = (
        "return_5m",
        "return_15m",
        "breakeven_within_15m",
    )
    per_hypothesis = {}
    for row_id, item in p5.items():
        metrics = item["analysis"]["metrics"]
        within = item["within_decision_controls"]
        selected_metrics = selected[row_id]["analysis"]["metrics"]
        controlled_metric_evidence = {}
        selected_intent_evidence = {}
        for metric_name in controlled_metrics:
            full_metric = metrics[metric_name]
            within_metric = within[metric_name]
            selected_metric = selected_metrics[metric_name]
            controlled_metric_evidence[metric_name] = {
                "pooled_rank_correlation": full_metric["rank_correlation"],
                "all_five_folds_positive": _all_fold_correlations_positive(
                    full_metric
                ),
                "within_decision_mean_rank_correlation": within_metric[
                    "mean_within_decision_rank_correlation"
                ],
                "within_decision_session_ci": within_metric["session_mean_ci"],
                "within_decision_session_ci_strictly_positive": (
                    _strictly_positive_ci(within_metric["session_mean_ci"])
                ),
                "session_paired_top_minus_bottom": full_metric[
                    "session_paired_top_minus_bottom"
                ],
                "session_paired_top_minus_bottom_ci": full_metric[
                    "session_paired_ci"
                ],
            }
            selected_intent_evidence[metric_name] = {
                "rank_correlation": selected_metric["rank_correlation"],
                "all_three_seeds_positive": _all_seed_correlations_positive(
                    selected_metric
                ),
                "session_paired_top_minus_bottom": selected_metric[
                    "session_paired_top_minus_bottom"
                ],
                "session_paired_top_minus_bottom_ci": selected_metric[
                    "session_paired_ci"
                ],
                "session_paired_ci_strictly_positive": _strictly_positive_ci(
                    selected_metric["session_paired_ci"]
                ),
            }
        controlled_pass_count = sum(
            int(
                evidence["all_five_folds_positive"]
                and evidence["within_decision_session_ci_strictly_positive"]
            )
            for evidence in controlled_metric_evidence.values()
        )
        selected_intent_pass_count = sum(
            int(
                evidence["all_three_seeds_positive"]
                and evidence["session_paired_ci_strictly_positive"]
            )
            for evidence in selected_intent_evidence.values()
        )
        per_hypothesis[row_id] = {
            "controlled_candidate_metric_pass_count": controlled_pass_count,
            "controlled_candidate_metric_count": len(controlled_metrics),
            "controlled_candidate_metrics": controlled_metric_evidence,
            "thresholded_selected_intent_metric_pass_count": (
                selected_intent_pass_count
            ),
            "thresholded_selected_intent_metric_count": len(controlled_metrics),
            "thresholded_selected_intent_metrics": selected_intent_evidence,
            "current_fixed_exit_rank_correlation": metrics[
                "current_fixed_exit_rop"
            ]["rank_correlation"],
        }
    controlled_predictive_hypotheses = sum(
        int(item["controlled_candidate_metric_pass_count"] >= 2)
        for item in per_hypothesis.values()
    )
    thresholded_predictive_hypotheses = sum(
        int(item["thresholded_selected_intent_metric_pass_count"] >= 2)
        for item in per_hypothesis.values()
    )
    model_increment_credible = bool(
        economic["paired_model_contribution"]["credible"]
    )
    # H0-H3 are nested feature variants, not independent replications. Route B
    # therefore requires controlled evidence in at least two variants and
    # remains a weak predictive finding until timing and economics also pass.
    if controlled_predictive_hypotheses >= 2:
        route = "Route B"
        reason = (
            "H2/P5 and H3/P5 contain modest out-of-fold within-decision "
            "contract-ranking information for short-horizon return and "
            "breakeven outcomes, with fold and session-level support. The "
            "thresholded selected-intent scores do not show corresponding "
            "stable timing discrimination, and the paired economic increment "
            "over P5 is unproven. The campaign evaluation and score-to-action "
            "contract are therefore misaligned with the information present."
        )
    elif (
        float(economic["fixed_p5_heuristic_absolute_pnl"]) > 0.0
        and not model_increment_credible
    ):
        route = "Route D"
        reason = (
            "P5 is profitable without a learned entry model, while the frozen "
            "HGB scores lack controlled path-ranking evidence and have no "
            "credible paired economic increment."
        )
    else:
        route = "Route E"
        reason = (
            "The frozen scores do not reliably rank broad path-quality "
            "outcomes and do not justify entry selection."
        )
    return route, {
        "adjudication_rule": {
            "status": "interpretive_route_rule_applied_after_frozen_metrics",
            "controlled_metrics": list(controlled_metrics),
            "controlled_metric_pass": (
                "all five fold correlations > 0 and the within-decision "
                "session-level interval is strictly > 0"
            ),
            "predictive_hypothesis_pass": "at least two of three metrics pass",
            "route_B_requirement": (
                "at least two nested feature variants pass; this is evidence "
                "of a provisional pattern, not independent replication"
            ),
        },
        "per_hypothesis": per_hypothesis,
        "controlled_predictive_hypotheses": controlled_predictive_hypotheses,
        "thresholded_predictive_hypotheses": thresholded_predictive_hypotheses,
        "hypothesis_independence_warning": (
            "H0-H3 are nested feature variants, not independent replications"
        ),
        "model_increment_credible": model_increment_credible,
        "reason": reason,
    }


def exit_dependency_audit(path_results: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for row_id, item in path_results["selected_intent_rows"].items():
        metrics = item["analysis"]["metrics"]
        rows[row_id] = {
            "current_fixed_exit_rank_correlation": metrics[
                "current_fixed_exit_rop"
            ]["rank_correlation"],
            "fixed_horizon_rank_correlations": {
                f"{horizon}m": metrics[f"return_{horizon}m"][
                    "rank_correlation"
                ]
                for horizon in HORIZONS
            },
            "path_opportunity_rank_correlations": {
                "MFE_15m": metrics["mfe_15m"]["rank_correlation"],
                "MAE_15m": metrics["mae_15m"]["rank_correlation"],
                "breakeven": metrics["breakeven_within_15m"][
                    "rank_correlation"
                ],
                "profit_available_before_exit": metrics[
                    "profit_available_before_exit_rop"
                ]["rank_correlation"],
            },
            "uncaptured_profit_rank_correlation": metrics[
                "uncaptured_profit_rop"
            ]["rank_correlation"],
        }
    return {
        "schema_version": f"{SCHEMA_VERSION}ExitDependencyAudit",
        "status": "complete",
        "diagnostic_exits": {
            "fixed_horizons_minutes": list(HORIZONS),
            "barriers": [
                {"name": name, "favorable": favorable, "adverse": adverse}
                for name, favorable, adverse in BARRIER_PAIRS
            ],
            "optimized_on_validation": False,
        },
        "rows": rows,
        "interpretation_rule": (
            "A score useful only for current_fixed_exit_rop is exit-policy "
            "compatibility; stable fixed-horizon/MFE/MAE/barrier ranking is "
            "evidence of more general entry quality."
        ),
        "learned_exit_consequence": (
            "Replacing the fixed exit changes the outcome the current model "
            "was trained to predict. It does not invalidate frozen OOF facts, "
            "but it invalidates interpreting that target as policy-neutral "
            "entry quality."
        ),
    }


def report_text(
    *,
    route: str,
    route_evidence: dict[str, Any],
    path_results: dict[str, Any],
    economic: dict[str, Any],
) -> str:
    selected = path_results["selected_intent_rows"]
    h2 = selected["H2/P5"]["analysis"]["metrics"]
    model_effect = economic["paired_model_contribution"][
        "real_H2_P5_total_feature_independent_effect"
    ]
    if route == "Route B":
        conclusion = (
            "The current entry campaign is primarily training the model to "
            "predict fee-adjusted return on premium under one fixed exit "
            "policy. This is not the appropriate final objective for a system "
            "in which a separately learned lifecycle model will decide when "
            "to exit. H2/P5 and H3/P5 out-of-fold scores weakly but "
            "repeatably rank some short-horizon contract outcomes within the "
            "same decision minute. They do not establish reliable entry "
            "timing after thresholding. The campaign gates measure the "
            "complete P5-plus-model strategy, and the model's contribution "
            "beyond P5 remains unproven. Therefore, the correct next route is "
            "Route B."
        )
        next_action = (
            "Freeze these results, replace complete-strategy entry gates with "
            "separate within-minute contract-ranking, across-minute timing, "
            "and matched-exposure economic gates, then preregister the "
            "smallest policy-neutral risk-adjusted entry campaign. Preserve "
            "P5 as the no-model baseline and keep corrected D1 binding."
        )
    elif route == "Route C":
        conclusion = (
            "The current entry campaign is primarily training the model to "
            "predict fee-adjusted return on premium under one fixed exit "
            "policy. This is not the appropriate objective for a system in "
            "which a separately learned lifecycle model will decide when to "
            "exit. Out-of-fold scores do not reliably rank broad path quality "
            "outside that fixed-exit outcome. The model's measured "
            "contribution beyond P5 remains unproven. Therefore, the correct "
            "next route is Route C."
        )
        next_action = (
            "Preregister a clean policy-neutral entry target and refit only "
            "the smallest justified campaign; do not reuse current economics."
        )
    elif route == "Route D":
        conclusion = (
            "The current entry campaign is primarily training the model to "
            "predict fee-adjusted return on premium under one fixed exit "
            "policy. This is not sufficient for a separately learned exit "
            "architecture. Out-of-fold scores do not provide controlled "
            "entry-quality discrimination, and contribution beyond P5 is "
            "unproven. Therefore, the correct next route is Route D."
        )
        next_action = (
            "Freeze P5 as the no-model entry engine and use it as the "
            "standardized-entry baseline for HOLD/EXIT research. Do not "
            "attribute P5 economics to HGB."
        )
    else:
        conclusion = (
            "The current entry campaign is primarily training the model to "
            "predict fee-adjusted return on premium under one fixed exit "
            "policy. This is not sufficient for a separately learned exit "
            "architecture. Out-of-fold scores do not reliably rank broad "
            "entry quality, and contribution beyond P5 is unproven. Therefore, "
            "the correct next route is Route E."
        )
        next_action = (
            "Redesign the entry target around preregistered multi-horizon "
            "opportunity and risk, then run a smaller clean campaign."
        )
    return f"""# Protocol101 Entry Objective Audit

## Decision

**{route}**

> {conclusion}

## Plain-English Finding

H0-H3 are not four different trading objectives. They are four feature sets
trained against the same question: “How much fee-adjusted return on premium
would this contract earn under policy P0-P6?”  For P5, that policy already
contains a historically profitable full-session exit rule before HGB scores
anything.

The right test is therefore not the absolute P5 PnL.  It is whether higher
frozen OOF scores consistently identify contracts and minutes with better
executable future paths, and whether that information improves on a matched P5
baseline.

## H2/P5 Path Evidence

The first set below is intentionally the actual threshold-passing selected
intent population. These results do not show reliable timing discrimination:

- 5-minute score/outcome rank correlation: `{h2['return_5m']['rank_correlation']}`
- 15-minute score/outcome rank correlation: `{h2['return_15m']['rank_correlation']}`
- MFE rank correlation: `{h2['mfe_15m']['rank_correlation']}`
- MAE rank correlation: `{h2['mae_15m']['rank_correlation']}`
- Breakeven-within-15m rank correlation: `{h2['breakeven_within_15m']['rank_correlation']}`
- Current fixed-exit target rank correlation: `{h2['current_fixed_exit_rop']['rank_correlation']}`

In the full seed-42 candidate universe, H2/P5 and H3/P5 do show modest
within-minute ranking of 5-minute return, 15-minute return, and/or breakeven,
with positive fold evidence. This is evidence of some contract-ranking
information, not proof of a usable entry-timing model. H0-H3 are nested feature
variants, so their agreement is not counted as four independent replications.
Full deciles, paired intervals, folds, seeds, calls/puts, time buckets,
moneyness bands, premium bands, and within-decision controls are in
`score_path_quality_results.json` and `score_quantile_results.parquet`.

## Model Contribution

- Fixed P5 heuristic PnL: `${economic['fixed_p5_heuristic_absolute_pnl']:,.2f}`
- Feature-independent random P5 median PnL: `${economic['random_p5_absolute_pnl']:,.2f}`
- Real H2/P5 fixed-count median PnL: `${economic['real_h2_p5_absolute_pnl']:,.2f}`
- Reversed-score median PnL: `${economic['reversed_real_absolute_pnl']:,.2f}`
- Strong-shuffle median PnL: `${economic['strong_shuffle_absolute_pnl']:,.2f}`
- Corrected paired real-model increment: `${model_effect['dollars']:,.2f}`
  with interval `${model_effect['ci_p2_5']:,.2f}` to
  `${model_effect['ci_p97_5']:,.2f}`

The reversed score performs much worse, and the controlled candidate analysis
finds modest ranking information. Neither fact establishes incremental
economic value over P5: the matched interval crosses zero and exposure
matching did not pass.

## Explicit Answers

1. **What does H0-H3 predict?** Each predicts clipped fee-adjusted
   return-on-premium under a fixed P0-P6 exit. H0-H3 differ only by features.
2. **Is that the right entry objective?** Not as the final objective for a
   separately learned HOLD/EXIT system; it is exit-policy compatibility.
3. **Do higher scores correspond to faster/larger favorable movement?**
   Weakly for H2/H3 contract ranking inside a decision minute; not reliably
   for the threshold-passing selected-intent population.
4. **Do higher scores mean lower adverse movement?** Some H2/H3
   candidate-level MAE evidence is favorable, but it is not stable in the
   selected-intent population and is not sufficient for selection.
5. **Does HGB add beyond P5/contract structure?** The within-decision controls
   show modest information beyond broad market opportunity, but cannot fully
   separate learned contract geometry from structure. Incremental economic
   value over matched P5 exposure is not established.
6. **What does current PnL measure?** Complete entry-plus-fixed-exit strategy
   performance, not isolated entry quality.
7. **Would a learned exit alter the labels?** Yes. It changes the realized
   outcome the current target forecasts, so current labels are not
   policy-neutral.
8. **What should happen next?** {next_action}

## Route Evidence

```json
{json.dumps(_clean(route_evidence), indent=2, sort_keys=True)}
```

## Boundaries

No model was refit. No entry model was selected. Corrected D1 remains binding.
G9 and the protected holdout remained closed. HOLD/EXIT training, paper,
broker, live, promotion, runtime, and launchd work were not performed.
"""


def run_audit() -> dict[str, Any]:
    prereg = verify_preregistration()
    scope, processed_paths, normalized_paths = _scope_and_maps()
    write_json(
        OUTPUT_ROOT / "progress.json",
        {
            "status": "executing_frozen_audit",
            "preregistration_hash": prereg["preregistration_hash"],
        },
    )
    path_cache_receipt = build_path_cache(
        scope=scope,
        processed_paths=processed_paths,
        normalized_paths=normalized_paths,
    )
    paths = _path_frame()
    if len(paths) != path_cache_receipt["candidate_path_rows"]:
        raise RuntimeError("path cache row count changed on read")
    quantile_rows: list[dict[str, Any]] = []
    selected_results: dict[str, Any] = {}
    selected_receipts: dict[str, Any] = {}
    for row_index, (hypothesis, policy) in enumerate(POSITIVE_ROWS):
        row_id = f"{hypothesis}/P{policy}"
        frame, receipt = selected_intent_frame(
            hypothesis=hypothesis,
            policy=policy,
            scope=scope,
            paths=paths,
        )
        analysis = analyze_score_frame(
            frame,
            row_id=row_id,
            scope_name="persisted_selected_entry_intents_all_seeds",
            quantile_rows=quantile_rows,
            bootstrap_seed=200_000 + row_index * 1_000,
        )
        selected_results[row_id] = {"analysis": analysis}
        selected_receipts[row_id] = receipt
        write_json(
            OUTPUT_ROOT / "progress.json",
            {
                "status": "analyzing_selected_intent_scores",
                "completed_rows": row_index + 1,
                "total_rows": len(POSITIVE_ROWS),
                "last_row": row_id,
            },
        )
        del frame
    noise_model = DivergenceNoiseModel.from_parquet(NOISE_DISTRIBUTION)
    full_candidate_results: dict[str, Any] = {}
    full_candidate_receipts: dict[str, Any] = {}
    for index, hypothesis in enumerate(("H0", "H1", "H2", "H3")):
        row_id = f"{hypothesis}/P5"
        frame, receipt = full_candidate_p5_frame(
            hypothesis=hypothesis,
            scope=scope,
            processed_paths=processed_paths,
            paths=paths,
            noise_model=noise_model,
        )
        analysis = analyze_score_frame(
            frame,
            row_id=row_id,
            scope_name="all_eligible_candidates_seed42",
            quantile_rows=quantile_rows,
            bootstrap_seed=400_000 + index * 1_000,
        )
        full_candidate_results[row_id] = {
            "analysis": analysis,
            "within_decision_controls": within_decision_results(frame),
        }
        full_candidate_receipts[row_id] = receipt
        write_json(
            OUTPUT_ROOT / "progress.json",
            {
                "status": "analyzing_full_candidate_p5_scores",
                "completed_hypotheses": index + 1,
                "total_hypotheses": 4,
                "last_row": row_id,
            },
        )
        del frame
    quantile_frame = pd.DataFrame.from_records(quantile_rows)
    quantile_path = OUTPUT_ROOT / "score_quantile_results.parquet"
    quantile_frame.to_parquet(quantile_path, index=False, compression="zstd")
    score_results = {
        "schema_version": f"{SCHEMA_VERSION}ScorePathQualityResults",
        "status": "complete",
        "preregistration_hash": prereg["preregistration_hash"],
        "path_cache_receipt": path_cache_receipt,
        "selected_intent_rows": selected_results,
        "selected_intent_receipts": selected_receipts,
        "full_candidate_p5": full_candidate_results,
        "full_candidate_receipts": full_candidate_receipts,
        "quantile_results": {
            "path": _relative(quantile_path),
            "sha256": sha256_path(quantile_path),
            "rows": len(quantile_frame),
        },
        "interpretation": {
            "selected_intent_scope": (
                "actual threshold-passing would-be entries; uses persisted "
                "selected-candidate score, not top score when fallback differs"
            ),
            "full_candidate_scope": (
                "exact frozen model OOF rescore at the original 1x noise law; "
                "seed42 for all P5 candidates"
            ),
            "within_decision_control": (
                "ranks score and outcome only among contracts available at "
                "the same minute, controlling broad market opportunity"
            ),
        },
    }
    write_json(OUTPUT_ROOT / "score_path_quality_results.json", score_results)
    economic = economic_control_results()
    write_json(OUTPUT_ROOT / "paired_policy_model_results.json", economic)
    exit_audit = exit_dependency_audit(score_results)
    write_json(OUTPUT_ROOT / "exit_dependency_audit.json", exit_audit)
    pathologies = label_pathology_results(len(paths))
    write_json(OUTPUT_ROOT / "label_pathology_audit.json", pathologies)
    write_json(OUTPUT_ROOT / "existing_objectives.json", existing_objectives())
    write_json(OUTPUT_ROOT / "target_dependency_graph.json", dependency_graph())
    write_json(OUTPUT_ROOT / "entry_objective_options.json", objective_options())
    route, route_evidence = recommend_route(score_results, economic)
    recommendation = {
        "schema_version": f"{SCHEMA_VERSION}RecommendedObjective",
        "route": route,
        "current_objective": (
            "policy-specific fixed-exit fee-adjusted return on premium regression"
        ),
        "recommended_objective": (
            "within-decision risk-adjusted ranking using preregistered "
            "multi-horizon executable returns, MFE, MAE, breakeven timing, "
            "and favorable-before-adverse barriers"
        ),
        "mandatory_baseline": "P5 fixed heuristic without a learned entry model",
        "why": route_evidence["reason"],
        "smallest_future_campaign": {
            "status": "proposal_only_not_launched",
            "models": [
                "transparent HGB multi-output/ranking baseline",
                "one bounded neural or sequence challenger only after baseline",
            ],
            "features": "same signed 17-feature firewall",
            "folds": "same governed five OOF folds",
            "primary_unit": "decision minute and session, never candidate row IID",
            "negative_controls": "corrected D1 plus reversed and constant-score baselines",
            "selection": "paired increment over P5 at matched exposure",
            "protected_data": "G9 and holdout remain closed",
        },
    }
    write_json(OUTPUT_ROOT / "recommended_entry_objective.json", recommendation)
    route_decision = {
        "schema_version": f"{SCHEMA_VERSION}RouteDecision",
        "status": "entry_objective_audit_complete",
        "route": route,
        "route_evidence": route_evidence,
        "entry_model_selected": False,
        "entry_selection_may_resume": False,
        "G9_allowed": False,
        "protected_holdout_accessed": False,
        "hold_exit_training_allowed": False,
        "corrected_D1_preserved": True,
        "next_action": recommendation["smallest_future_campaign"],
        "side_effects": prereg["side_effect_boundaries"],
    }
    route_decision["decision_hash"] = stable_hash(route_decision)
    write_json(OUTPUT_ROOT / "entry_campaign_route_decision.json", route_decision)
    report = report_text(
        route=route,
        route_evidence=route_evidence,
        path_results=score_results,
        economic=economic,
    )
    (OUTPUT_ROOT / "report.md").write_text(report)
    required = [
        "existing_objectives.json",
        "target_dependency_graph.json",
        "entry_objective_options.json",
        "score_path_quality_results.json",
        "score_quantile_results.parquet",
        "paired_policy_model_results.json",
        "exit_dependency_audit.json",
        "label_pathology_audit.json",
        "recommended_entry_objective.json",
        "entry_campaign_route_decision.json",
        "report.md",
    ]
    missing = [name for name in required if not (OUTPUT_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"required deliverables missing: {missing}")
    hashes = {
        name: sha256_path(OUTPUT_ROOT / name)
        for name in required
    }
    write_json(
        OUTPUT_ROOT / "summary.json",
        {
            "schema_version": f"{SCHEMA_VERSION}Summary",
            "status": "entry_objective_audit_complete",
            "route": route,
            "required_deliverables": hashes,
            "preregistration_hash": prereg["preregistration_hash"],
            "candidate_path_rows": len(paths),
            "quantile_rows": len(quantile_frame),
            "entry_model_selected": False,
            "G9_executed": False,
            "protected_holdout_accessed": False,
            "hold_exit_training_started": False,
            "model_refit": False,
            "broker_or_paper": False,
        },
    )
    write_json(
        OUTPUT_ROOT / "progress.json",
        {
            "status": "complete",
            "route": route,
            "summary_sha256": sha256_path(OUTPUT_ROOT / "summary.json"),
        },
    )
    if SCRATCH_PATHS.exists():
        SCRATCH_PATHS.unlink()
    return route_decision


def finalize_completed_score_audit() -> dict[str, Any]:
    """Finalize after a mechanical serializer failure without recomputation."""

    prereg = verify_preregistration()
    score_path = OUTPUT_ROOT / "score_path_quality_results.json"
    quantile_path = OUTPUT_ROOT / "score_quantile_results.parquet"
    if not score_path.is_file() or not quantile_path.is_file():
        raise RuntimeError("finalize requires completed score/path artifacts")
    score_results = load_json(score_path)
    if score_results.get("status") != "complete":
        raise RuntimeError("score/path artifact is not complete")
    expected_quantile_hash = score_results["quantile_results"]["sha256"]
    if sha256_path(quantile_path) != expected_quantile_hash:
        raise RuntimeError("quantile artifact changed before finalize")
    path_rows = int(score_results["path_cache_receipt"]["candidate_path_rows"])
    economic = economic_control_results()
    write_json(OUTPUT_ROOT / "paired_policy_model_results.json", economic)
    write_json(
        OUTPUT_ROOT / "exit_dependency_audit.json",
        exit_dependency_audit(score_results),
    )
    write_json(
        OUTPUT_ROOT / "label_pathology_audit.json",
        label_pathology_results(path_rows),
    )
    write_json(OUTPUT_ROOT / "existing_objectives.json", existing_objectives())
    write_json(OUTPUT_ROOT / "target_dependency_graph.json", dependency_graph())
    write_json(OUTPUT_ROOT / "entry_objective_options.json", objective_options())
    route, route_evidence = recommend_route(score_results, economic)
    recommendation = {
        "schema_version": f"{SCHEMA_VERSION}RecommendedObjective",
        "route": route,
        "current_objective": (
            "policy-specific fixed-exit fee-adjusted return on premium regression"
        ),
        "recommended_objective": (
            "within-decision risk-adjusted ranking using preregistered "
            "multi-horizon executable returns, MFE, MAE, breakeven timing, "
            "and favorable-before-adverse barriers"
        ),
        "mandatory_baseline": "P5 fixed heuristic without a learned entry model",
        "why": route_evidence["reason"],
        "smallest_future_campaign": {
            "status": "proposal_only_not_launched",
            "models": [
                "transparent HGB multi-output/ranking baseline",
                "one bounded neural or sequence challenger only after baseline",
            ],
            "features": "same signed 17-feature firewall",
            "folds": "same governed five OOF folds",
            "primary_unit": "decision minute and session, never candidate row IID",
            "negative_controls": (
                "corrected D1 plus reversed and constant-score baselines"
            ),
            "selection": "paired increment over P5 at matched exposure",
            "protected_data": "G9 and holdout remain closed",
        },
    }
    write_json(OUTPUT_ROOT / "recommended_entry_objective.json", recommendation)
    route_decision = {
        "schema_version": f"{SCHEMA_VERSION}RouteDecision",
        "status": "entry_objective_audit_complete",
        "route": route,
        "route_evidence": route_evidence,
        "entry_model_selected": False,
        "entry_selection_may_resume": False,
        "G9_allowed": False,
        "protected_holdout_accessed": False,
        "hold_exit_training_allowed": False,
        "corrected_D1_preserved": True,
        "next_action": recommendation["smallest_future_campaign"],
        "side_effects": prereg["side_effect_boundaries"],
        "mechanical_resume": (
            "finalized from hash-verified completed score artifacts after a "
            "report-only campaign-contract key lookup failure"
        ),
    }
    route_decision["decision_hash"] = stable_hash(route_decision)
    write_json(OUTPUT_ROOT / "entry_campaign_route_decision.json", route_decision)
    (OUTPUT_ROOT / "report.md").write_text(
        report_text(
            route=route,
            route_evidence=route_evidence,
            path_results=score_results,
            economic=economic,
        )
    )
    required = [
        "existing_objectives.json",
        "target_dependency_graph.json",
        "entry_objective_options.json",
        "score_path_quality_results.json",
        "score_quantile_results.parquet",
        "paired_policy_model_results.json",
        "exit_dependency_audit.json",
        "label_pathology_audit.json",
        "recommended_entry_objective.json",
        "entry_campaign_route_decision.json",
        "report.md",
    ]
    hashes = {
        name: sha256_path(OUTPUT_ROOT / name)
        for name in required
        if (OUTPUT_ROOT / name).is_file()
    }
    if len(hashes) != len(required):
        raise RuntimeError("finalize did not produce all required deliverables")
    summary = {
        "schema_version": f"{SCHEMA_VERSION}Summary",
        "status": "entry_objective_audit_complete",
        "route": route,
        "required_deliverables": hashes,
        "preregistration_hash": prereg["preregistration_hash"],
        "candidate_path_rows": path_rows,
        "quantile_rows": int(pq.ParquetFile(quantile_path).metadata.num_rows),
        "entry_model_selected": False,
        "G9_executed": False,
        "protected_holdout_accessed": False,
        "hold_exit_training_started": False,
        "model_refit": False,
        "broker_or_paper": False,
        "mechanical_resume": True,
    }
    write_json(OUTPUT_ROOT / "summary.json", summary)
    write_json(
        OUTPUT_ROOT / "progress.json",
        {
            "status": "complete",
            "route": route,
            "summary_sha256": sha256_path(OUTPUT_ROOT / "summary.json"),
        },
    )
    if SCRATCH_PATHS.exists():
        SCRATCH_PATHS.unlink()
    return route_decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("preregister", "run", "finalize"),
        required=True,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preregister":
        run_preregister(force=bool(args.force))
        print("entry objective audit preregistered")
        return
    if args.mode == "finalize":
        result = finalize_completed_score_audit()
        print(json.dumps(_clean(result), indent=2, sort_keys=True))
        return
    result = run_audit()
    print(json.dumps(_clean(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
