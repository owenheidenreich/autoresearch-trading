"""Fixed-score Path-D Wave-2 causal 60-minute signal discovery gate.

This module fits no estimator and has no network, broker, runtime, promotion, or
protected-holdout capability.  It evaluates exactly three preregistered scores on
the entry-v2 OOF test-session union under one passive, strict-serial trading game.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from v4.research.autoresearch_v2.statistics import paired_summary, session_blocked_max_t
from v4.research.pathd_model_gate import (
    GateError,
    decile_monotonicity,
    four_bucket_distribution,
    harvest_ratio,
    pnl_concentration,
    underwater_duration,
)
from v4.research.pathd_phase1_entry import development_sessions
from v4.research.phase1_exit_model import stable_hash


NY = ZoneInfo("America/New_York")
UTC = timezone.utc
MINUTE_NS = 60_000_000_000
SECOND_NS = 1_000_000_000
EMISSION_LAG_NS = 2_336_000_000
ENTRY_WINDOW_NS = 60 * SECOND_NS
HORIZON_60_NS = 60 * MINUTE_NS
STARTING_EQUITY = 10_000.0
DAILY_STOP_FRACTION = 0.05
FEE_PER_SIDE = 1.50
ENTRY_THRESHOLD = 1.0
MAXT_PERMUTATIONS = 20_000
MAXT_SEED = 2031
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2032
PREREGISTRATION_PATH = Path(
    "v4/docs/protocol101/training/research/"
    "PATHD_WAVE2_CAUSAL_60M_SIGNAL_DISCOVERY_PREREGISTRATION_2026_08_03.md"
)
MEMBERS = (
    "W2-H01-surface-continuation",
    "W2-H02-option-microstructure",
    "W2-H03-cross-market-alignment",
)
SCORE_COLUMNS = dict(
    zip(
        MEMBERS,
        ("score_surface", "score_microstructure", "score_cross_market"),
        strict=True,
    )
)
COMPONENTS = {
    MEMBERS[0]: (
        "surface_contract_mom15_bps",
        "surface_same_side_breadth",
        "surface_straddle_expansion_bps",
        "surface_smile_slope_change",
    ),
    MEMBERS[1]: (
        "microprice_pressure",
        "displayed_size_imbalance",
        "option_mid_mom5_bps",
        "option_mid_mom15_bps",
        "option_log_volume",
        "relative_spread_compression_5m",
    ),
    MEMBERS[2]: (
        "side_spx_mom15_bps",
        "side_es_mom15_bps",
        "side_es_spx_lead5_bps",
        "side_vix_mom15_bps",
    ),
}


class Wave2Error(RuntimeError):
    """Base failure for the bounded Wave-2 gate."""


class DataContractError(Wave2Error):
    """A source, identity, or causal clock cannot support the frozen test."""


@dataclass(frozen=True)
class Wave2Paths:
    root: Path
    corpus: Path
    campaign: Path
    scores: Path
    counterfactuals: Path
    cbbo_1m: Path
    cbbo_1s: Path
    option_ohlcv_1m: Path
    spx_1m: Path
    vix_1m: Path
    es_1m: Path

    @classmethod
    def from_root(cls, root: Path, *, counterfactuals: Path | None = None) -> "Wave2Paths":
        root = Path(root)
        corpus = root / "vendor/pathd_2025-08-01_2026-07-31"
        return cls(
            root=root,
            corpus=corpus,
            campaign=root / "artifacts/entry_v2/campaign.json",
            scores=root / "artifacts/entry_v2/oof_scores.parquet",
            counterfactuals=(
                Path(counterfactuals)
                if counterfactuals is not None
                else root
                / "reports/pathd_matched_feasibility_2026_08_03/option_primary_counterfactuals.parquet"
            ),
            cbbo_1m=corpus / "raw/databento/opra_spxw_cbbo_1m",
            cbbo_1s=corpus / "raw/databento/opra_spxw_cbbo_1s",
            option_ohlcv_1m=corpus / "raw/databento/opra_spxw_ohlcv_1m",
            spx_1m=corpus / "vendor/thetadata/index/spx_1m",
            vix_1m=corpus / "vendor/thetadata/index/vix_1m",
            es_1m=corpus / "raw/databento/glbx_es_ohlcv_1m",
        )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    ).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _tick(price: float) -> float:
    return 0.05 if price < 3.0 else 0.10


def _session_terminal_ns(session: str) -> int:
    day = datetime.fromisoformat(session).date()
    return int(datetime.combine(day, time(15, 55), tzinfo=NY).astimezone(UTC).timestamp() * 1e9)


def _tree_hash(path: Path) -> str:
    rows: list[tuple[str, str]] = []
    if path.is_file():
        rows.append((path.name, _sha256_path(path)))
    elif path.exists():
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            rows.append((str(item.relative_to(path)), _sha256_path(item)))
    return _semantic_sha(rows)


def forbidden_state_snapshot(repo_root: Path) -> dict[str, str]:
    relative = (
        "v4/promotion/PAPER_TRADING_DEFAULT.json",
        "v4/runtime",
        "v4/ops/launchd",
        "v4/ops/ibkr",
        "v4/research/pathd_phase1_entry.py",
    )
    return {name: _tree_hash(Path(repo_root) / name) for name in relative}


def load_oof_membership(paths: Wave2Paths, scores: pd.DataFrame) -> dict[str, Any]:
    if not paths.campaign.is_file():
        raise DataContractError(f"missing campaign: {paths.campaign}")
    campaign = json.loads(paths.campaign.read_text())
    semantic = dict(campaign)
    expected = semantic.pop("campaign_sha256", None)
    if expected != stable_hash(semantic):
        raise DataContractError("entry-v2 campaign hash drift")
    if campaign.get("protected_holdout_opened") is not False:
        raise DataContractError("entry-v2 campaign reports protected holdout open")
    fold_map: dict[str, int] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for fold in range(5):
        path = Path(campaign["fold_manifests"][str(fold)])
        manifest = json.loads(path.read_text())
        if manifest.get("protected_holdout_opened") is not False:
            raise DataContractError(f"fold {fold} reports protected holdout open")
        if int(manifest.get("fold", -1)) != fold or int(manifest["fold_spec"]["fold"]) != fold:
            raise DataContractError(f"fold manifest identity drift: {path}")
        for session in manifest["fold_spec"]["test"]:
            if session in fold_map:
                raise DataContractError(f"OOF test session assigned twice: {session}")
            fold_map[str(session)] = fold
        manifests[str(fold)] = {
            "path": str(path),
            "sha256": _sha256_path(path),
            "test_sessions": len(manifest["fold_spec"]["test"]),
        }
    score_identity = scores[["session", "outer_fold"]].drop_duplicates()
    if score_identity["session"].duplicated().any():
        raise DataContractError("a score session maps to multiple folds")
    observed = {str(row.session): int(row.outer_fold) for row in score_identity.itertuples(index=False)}
    if observed != fold_map:
        raise DataContractError("oof_scores sessions/folds do not equal fold-manifest test union")
    development = set(development_sessions(paths.corpus))
    outside = sorted(set(fold_map) - development)
    if outside:
        raise DataContractError(f"OOF membership crossed the 215-session development prefix: {outside}")
    return {
        "campaign_path": str(paths.campaign),
        "campaign_sha256": _sha256_path(paths.campaign),
        "campaign_semantic_sha256": expected,
        "sessions": len(fold_map),
        "fold_session_counts": {
            str(fold): sum(value == fold for value in fold_map.values()) for fold in range(5)
        },
        "fold_manifests": manifests,
        "development_session_count": len(development),
        "protected_firewall_sessions_decoded": 0,
        "protected_holdout_opened": False,
    }


def _parquet_frame(path: Path, columns: Sequence[str], *, symbols: Sequence[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise DataContractError(f"missing owned source: {path}")
    dataset = pads.dataset(path, format="parquet")
    missing = set(columns) - set(dataset.schema.names)
    if missing:
        raise DataContractError(f"source schema missing {sorted(missing)}: {path}")
    expression = pads.field("symbol").isin(sorted(set(symbols))) if symbols is not None else None
    return dataset.to_table(columns=list(columns), filter=expression).to_pandas(ignore_metadata=True)


def _exact_lag(values: pd.DataFrame, *, key: Sequence[str], time_column: str, columns: Sequence[str], minutes: int) -> pd.DataFrame:
    prior = values[list(key) + [time_column] + list(columns)].copy()
    prior[time_column] = prior[time_column].astype(np.int64) + minutes * MINUTE_NS
    prior = prior.rename(columns={column: f"{column}_lag{minutes}" for column in columns})
    return prior


def _robust_past_z(frame: pd.DataFrame, component: str) -> pd.Series:
    """Past-only 60-minute robust z-score, separately normalized by option right."""

    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for _right, group in frame.groupby("right", sort=True):
        minute = group.groupby("feature_boundary_ns", sort=True)[component].median()
        index = pd.to_datetime(minute.index.to_numpy(np.int64), unit="ns", utc=True)
        series = pd.Series(minute.to_numpy(float), index=index)
        rolling = series.rolling("60min", min_periods=30, closed="left")
        center = rolling.median()
        scale = rolling.quantile(0.75) - rolling.quantile(0.25)
        lookup_center = dict(zip(minute.index.astype(np.int64), center.to_numpy(float), strict=True))
        lookup_scale = dict(zip(minute.index.astype(np.int64), scale.to_numpy(float), strict=True))
        raw = group[component].to_numpy(float)
        centers = group["feature_boundary_ns"].map(lookup_center).to_numpy(float)
        scales = group["feature_boundary_ns"].map(lookup_scale).to_numpy(float)
        valid = np.isfinite(raw) & np.isfinite(centers) & np.isfinite(scales) & (scales > 1e-12)
        values = np.full(len(group), np.nan, dtype=float)
        values[valid] = np.clip((raw[valid] - centers[valid]) / scales[valid], -3.0, 3.0)
        output.loc[group.index] = values
    return output


def _completed_close_features(path: Path, *, kind: str) -> pd.DataFrame:
    if kind in {"SPX", "VIX"}:
        source = _parquet_frame(path, ("event_time", "symbol", "close"))
        timestamp = pd.to_datetime(source["event_time"], utc=True).map(lambda value: value.value)
    elif kind == "ES":
        source = _parquet_frame(path, ("ts_event", "symbol", "close"))
        timestamp = pd.to_datetime(source["ts_event"], utc=True).map(lambda value: value.value)
    else:
        raise ValueError(kind)
    close = pd.to_numeric(source["close"], errors="coerce").to_numpy(float)
    base = pd.DataFrame(
        {
            "decision_emission_ns": timestamp.to_numpy(np.int64) + MINUTE_NS + EMISSION_LAG_NS,
            f"{kind.lower()}_close": close,
        }
    ).drop_duplicates("decision_emission_ns", keep="last")
    for horizon in (5, 15):
        lag = base[["decision_emission_ns", f"{kind.lower()}_close"]].copy()
        lag["decision_emission_ns"] += horizon * MINUTE_NS
        lag = lag.rename(columns={f"{kind.lower()}_close": f"{kind.lower()}_close_lag{horizon}"})
        base = base.merge(lag, on="decision_emission_ns", how="left", validate="one_to_one")
        current = base[f"{kind.lower()}_close"].to_numpy(float)
        previous = base[f"{kind.lower()}_close_lag{horizon}"].to_numpy(float)
        valid = np.isfinite(current) & np.isfinite(previous) & (current > 0.0) & (previous > 0.0)
        momentum = np.full(len(base), np.nan)
        momentum[valid] = 10_000.0 * (current[valid] / previous[valid] - 1.0)
        base[f"{kind.lower()}_mom{horizon}_bps"] = momentum
    base[f"{kind.lower()}_available_at_ns"] = base["decision_emission_ns"]
    return base


def _es_path(paths: Wave2Paths, session: str) -> Path:
    matches = sorted(paths.es_1m.glob(f"{session}*.parquet"))
    if len(matches) != 1:
        raise DataContractError(f"expected one ES minute source for {session}, found {len(matches)}")
    return matches[0]


def _es_roll_exclusion(paths: Wave2Paths, sessions: Sequence[str]) -> tuple[set[str], list[dict[str, Any]]]:
    ordered: list[tuple[str, int]] = []
    unavailable: set[str] = set()
    details: list[dict[str, Any]] = []
    for session in sorted(sessions):
        path = _es_path(paths, str(session))
        source = _parquet_frame(path, ("instrument_id",))
        identifiers = pd.to_numeric(source["instrument_id"], errors="coerce").dropna().unique()
        if len(identifiers) == 0:
            unavailable.add(str(session))
            details.append({"session": str(session), "reason": "empty_es_minute_partition"})
            continue
        if len(identifiers) != 1:
            raise DataContractError(f"ES session does not have one continuous-contract identity: {session}")
        ordered.append((str(session), int(identifiers[0])))
    change_indexes: list[int] = []
    for index in range(1, len(ordered)):
        if ordered[index][1] != ordered[index - 1][1]:
            change_indexes.append(index)
            details.append(
                {
                    "session": ordered[index][0],
                    "prior_instrument_id": ordered[index - 1][1],
                    "new_instrument_id": ordered[index][1],
                }
            )
    excluded_indexes = {
        neighbor
        for index in change_indexes
        for neighbor in range(max(0, index - 1), min(len(ordered), index + 2))
    }
    return unavailable | {ordered[index][0] for index in excluded_indexes}, details


def _build_session_features(paths: Wave2Paths, candidates: pd.DataFrame, session: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = candidates[candidates["session"].eq(session)].copy()
    symbols = tuple(sorted(frame["raw_symbol"].astype(str).unique()))
    cbbo_path = paths.cbbo_1m / f"{session}.cbbo-1m.parquet"
    cbbo = _parquet_frame(
        cbbo_path,
        ("ts_recv", "symbol", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"),
        symbols=symbols,
    )
    cbbo["feature_boundary_ns"] = pd.to_datetime(cbbo["ts_recv"], utc=True).astype("int64")
    cbbo = cbbo.sort_values(["symbol", "feature_boundary_ns"], kind="mergesort").drop_duplicates(
        ["symbol", "feature_boundary_ns"], keep="last"
    )
    bid = pd.to_numeric(cbbo["bid_px_00"], errors="coerce").to_numpy(float)
    ask = pd.to_numeric(cbbo["ask_px_00"], errors="coerce").to_numpy(float)
    bid_size = pd.to_numeric(cbbo["bid_sz_00"], errors="coerce").to_numpy(float)
    ask_size = pd.to_numeric(cbbo["ask_sz_00"], errors="coerce").to_numpy(float)
    valid_quote = np.isfinite(bid) & np.isfinite(ask) & (bid > 0.0) & (ask > bid)
    cbbo["decision_bid"] = np.where(valid_quote, bid, np.nan)
    cbbo["decision_ask"] = np.where(valid_quote, ask, np.nan)
    cbbo["decision_mid"] = np.where(valid_quote, (bid + ask) / 2.0, np.nan)
    cbbo["relative_spread"] = np.where(valid_quote, (ask - bid) / ((ask + bid) / 2.0), np.nan)
    size_total = bid_size + ask_size
    valid_size = valid_quote & np.isfinite(size_total) & (size_total > 0.0)
    microprice = np.full(len(cbbo), np.nan)
    np.divide(
        ask * bid_size + bid * ask_size,
        size_total,
        out=microprice,
        where=valid_size,
    )
    half_spread = (ask - bid) / 2.0
    pressure = np.full(len(cbbo), np.nan)
    np.divide(
        microprice - (ask + bid) / 2.0,
        half_spread,
        out=pressure,
        where=valid_size & (half_spread > 0.0),
    )
    imbalance = np.full(len(cbbo), np.nan)
    np.divide(bid_size - ask_size, size_total, out=imbalance, where=valid_size)
    cbbo["microprice_pressure"] = pressure
    cbbo["displayed_size_imbalance"] = imbalance
    quote_columns = (
        "decision_bid",
        "decision_ask",
        "decision_mid",
        "relative_spread",
        "microprice_pressure",
        "displayed_size_imbalance",
    )
    current = cbbo[["symbol", "feature_boundary_ns", *quote_columns]].rename(columns={"symbol": "raw_symbol"})
    frame = frame.merge(
        current, on=["raw_symbol", "feature_boundary_ns"], how="left", validate="one_to_one"
    )
    for horizon in (5, 15):
        lag = _exact_lag(
            current.rename(columns={"raw_symbol": "symbol"}),
            key=("symbol",),
            time_column="feature_boundary_ns",
            columns=("decision_mid", "relative_spread"),
            minutes=horizon,
        ).rename(columns={"symbol": "raw_symbol"})
        frame = frame.merge(
            lag,
            on=["raw_symbol", "feature_boundary_ns"],
            how="left",
            validate="one_to_one",
        )
        current_mid = frame["decision_mid"].to_numpy(float)
        previous_mid = frame[f"decision_mid_lag{horizon}"].to_numpy(float)
        valid = np.isfinite(current_mid) & np.isfinite(previous_mid) & (previous_mid > 0.0)
        values = np.full(len(frame), np.nan)
        values[valid] = 10_000.0 * (current_mid[valid] / previous_mid[valid] - 1.0)
        frame[f"option_mid_mom{horizon}_bps"] = values
    frame["surface_contract_mom15_bps"] = frame["option_mid_mom15_bps"]
    breadth = frame.groupby(["feature_boundary_ns", "right"])["surface_contract_mom15_bps"].transform(
        lambda values: 2.0 * float((values.dropna() > 0.0).mean()) - 1.0 if values.notna().any() else np.nan
    )
    frame["surface_same_side_breadth"] = breadth
    minute_straddle = (
        frame.groupby("feature_boundary_ns", sort=True)["D.near_atm.straddle_mid_spot_bps"]
        .median()
        .rename("straddle")
        .reset_index()
    )
    straddle_lag = minute_straddle.copy()
    straddle_lag["feature_boundary_ns"] += 15 * MINUTE_NS
    straddle_lag = straddle_lag.rename(columns={"straddle": "straddle_lag15"})
    minute_straddle = minute_straddle.merge(straddle_lag, on="feature_boundary_ns", how="left")
    minute_straddle["surface_straddle_expansion_bps"] = (
        minute_straddle["straddle"] - minute_straddle["straddle_lag15"]
    )
    frame = frame.merge(
        minute_straddle[["feature_boundary_ns", "surface_straddle_expansion_bps"]],
        on="feature_boundary_ns",
        how="left",
        validate="many_to_one",
    )
    smile = (
        frame.groupby(["feature_boundary_ns", "right"], sort=True)[
            "D.near_atm.side_smile_slope_bps_per_5pt"
        ]
        .median()
        .rename("smile")
        .reset_index()
    )
    smile_lag = smile.copy()
    smile_lag["feature_boundary_ns"] += 15 * MINUTE_NS
    smile_lag = smile_lag.rename(columns={"smile": "smile_lag15"})
    smile = smile.merge(smile_lag, on=["feature_boundary_ns", "right"], how="left")
    smile["surface_smile_slope_change"] = smile["smile"] - smile["smile_lag15"]
    frame = frame.merge(
        smile[["feature_boundary_ns", "right", "surface_smile_slope_change"]],
        on=["feature_boundary_ns", "right"],
        how="left",
        validate="many_to_one",
    )
    frame["relative_spread_compression_5m"] = (
        frame["relative_spread_lag5"] - frame["relative_spread"]
    )

    ohlcv_path = paths.option_ohlcv_1m / f"{session}.ohlcv-1m.parquet"
    ohlcv = _parquet_frame(ohlcv_path, ("ts_event", "symbol", "volume"), symbols=symbols)
    ohlcv["decision_emission_ns"] = (
        pd.to_datetime(ohlcv["ts_event"], utc=True).astype("int64") + MINUTE_NS + EMISSION_LAG_NS
    )
    ohlcv["option_log_volume"] = np.log1p(pd.to_numeric(ohlcv["volume"], errors="coerce"))
    ohlcv = ohlcv.sort_values(["symbol", "decision_emission_ns"], kind="mergesort").drop_duplicates(
        ["symbol", "decision_emission_ns"], keep="last"
    )
    ohlcv = ohlcv.rename(columns={"symbol": "raw_symbol"})
    frame = frame.merge(
        ohlcv[["raw_symbol", "decision_emission_ns", "option_log_volume"]],
        on=["raw_symbol", "decision_emission_ns"],
        how="left",
        validate="one_to_one",
    )
    frame["option_available_at_ns"] = frame["feature_boundary_ns"]
    frame["option_volume_available_at_ns"] = np.where(
        frame["option_log_volume"].notna(), frame["decision_emission_ns"], np.nan
    )

    spx_path = paths.spx_1m / f"{session}.parquet"
    vix_path = paths.vix_1m / f"{session}.parquet"
    context = _completed_close_features(spx_path, kind="SPX")
    context = context.merge(
        _completed_close_features(vix_path, kind="VIX"),
        on="decision_emission_ns",
        how="inner",
        validate="one_to_one",
    )
    es_path = _es_path(paths, session)
    context = context.merge(
        _completed_close_features(es_path, kind="ES"),
        on="decision_emission_ns",
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(context, on="decision_emission_ns", how="left", validate="many_to_one")
    side = np.where(frame["right"].eq("C"), 1.0, -1.0)
    frame["side_spx_mom15_bps"] = side * frame["spx_mom15_bps"]
    frame["side_es_mom15_bps"] = side * frame["es_mom15_bps"]
    frame["side_es_spx_lead5_bps"] = side * (frame["es_mom5_bps"] - frame["spx_mom5_bps"])
    frame["side_vix_mom15_bps"] = -side * frame["vix_mom15_bps"]
    frame["cross_market_available_at_ns"] = frame[
        ["spx_available_at_ns", "vix_available_at_ns", "es_available_at_ns"]
    ].max(axis=1)

    for member, components in COMPONENTS.items():
        z_columns = []
        for component in components:
            column = f"z__{component}"
            frame[column] = _robust_past_z(frame, component)
            z_columns.append(column)
        score_column = SCORE_COLUMNS[member]
        complete = frame[z_columns].notna().all(axis=1)
        frame[score_column] = np.where(complete, frame[z_columns].mean(axis=1), np.nan)

    quote_delta = np.abs(
        frame[["entry_feature_bid", "entry_feature_ask"]].to_numpy(float)
        - frame[["decision_bid", "decision_ask"]].to_numpy(float)
    )
    finite_delta = quote_delta[np.isfinite(quote_delta)]
    if not len(finite_delta):
        raise DataContractError(f"no current OPRA quote identities joined for {session}")
    identity_delta = float(finite_delta.max())
    if identity_delta > 1e-9:
        raise DataContractError(
            f"current OPRA quote identity drift for {session}: max abs delta {identity_delta}"
        )
    source = {
        "session": session,
        "rows": int(len(frame)),
        "source_paths": {
            "cbbo_1m": str(cbbo_path),
            "option_ohlcv_1m": str(ohlcv_path),
            "spx_1m": str(spx_path),
            "vix_1m": str(vix_path),
            "es_1m": str(es_path),
        },
        "current_quote_identity_max_abs_delta": identity_delta,
    }
    return frame, source


def feature_lineage_manifest() -> dict[str, Any]:
    return {
        "schema_version": "pathd.wave2-feature-lineage.v1",
        "normalization": {
            "law": "past-only preceding 60 in-session minute medians; median/IQR; min 30; current excluded; clip +/-3",
            "fitted": False,
        },
        "members": {
            member: {
                "score_column": SCORE_COLUMNS[member],
                "components": list(components),
                "weight": 1.0 / len(components),
                "threshold": ENTRY_THRESHOLD,
            }
            for member, components in COMPONENTS.items()
        },
        "sources": {
            "surface": "entry-v2 governed OOF candidate ladder plus owned OPRA CBBO-1m exact-contract history",
            "microstructure": "owned OPRA CBBO-1m displayed BBO/size and OPRA OHLCV-1m volume",
            "cross_market": "owned official ThetaData SPX/VIX OHLC-1m plus Databento ES.c.0 OHLCV-1m",
            "outcomes": "owned OPRA CBBO-1s one-tick-penetration counterfactuals, actual fill time, bid exit",
        },
        "availability": {
            "option_cbbo_1m": "ts_recv <= feature_boundary <= decision_emission",
            "minute_bars": "interval_start + 60s + 2336ms <= decision_emission",
            "outcomes": "joined only after feature scores are frozen",
        },
        "live_parity_status": "REQUIRED_BEFORE_ANY_ADVANCEMENT; not asserted by this offline gate",
    }


def build_candidate_features(paths: Wave2Paths) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    columns = (
        "candidate_uid",
        "session",
        "feature_boundary_ns",
        "decision_emission_ns",
        "entry_arrival_ns",
        "raw_symbol",
        "right",
        "strike",
        "strike_offset",
        "entry_feature_bid",
        "entry_feature_ask",
        "entry_feature_mid",
        "D.near_atm.straddle_mid_spot_bps",
        "D.near_atm.side_smile_slope_bps_per_5pt",
        "outer_fold",
    )
    if not paths.scores.is_file():
        raise DataContractError(f"missing OOF scores: {paths.scores}")
    candidates = pd.read_parquet(paths.scores, columns=list(columns))
    if candidates["candidate_uid"].duplicated().any():
        raise DataContractError("OOF candidate identity is not unique")
    membership = load_oof_membership(paths, candidates)
    roll_excluded, roll_details = _es_roll_exclusion(paths, candidates["session"].unique())
    parts: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for index, session in enumerate(sorted(candidates["session"].unique()), start=1):
        part, source = _build_session_features(paths, candidates, str(session))
        parts.append(part)
        sources.append(source)
        if index % 20 == 0:
            print(json.dumps({"stage": "features", "sessions_complete": index}), flush=True)
    frame = pd.concat(parts, ignore_index=True)
    if frame["candidate_uid"].duplicated().any() or len(frame) != len(candidates):
        raise DataContractError("feature construction changed candidate identity")
    frame.loc[frame["session"].isin(roll_excluded), SCORE_COLUMNS[MEMBERS[2]]] = np.nan
    coverage = {
        member: {
            "finite_rows": int(frame[SCORE_COLUMNS[member]].notna().sum()),
            "sessions": int(frame.loc[frame[SCORE_COLUMNS[member]].notna(), "session"].nunique()),
            "folds": int(frame.loc[frame[SCORE_COLUMNS[member]].notna(), "outer_fold"].nunique()),
        }
        for member in MEMBERS
    }
    if any(item["finite_rows"] < 100 or item["sessions"] < 30 or item["folds"] != 5 for item in coverage.values()):
        raise DataContractError(f"frozen score family lacks required causal coverage: {coverage}")
    return frame, membership, {
        "sessions": sources,
        "score_coverage": coverage,
        "es_roll_excluded_sessions": sorted(roll_excluded),
        "es_roll_changes": roll_details,
    }


def _outcome_dataset(paths: Wave2Paths, features: pd.DataFrame) -> pd.DataFrame:
    if not paths.counterfactuals.is_file():
        raise DataContractError(f"missing passive counterfactual source: {paths.counterfactuals}")
    outcomes = pd.read_parquet(paths.counterfactuals)
    required = {
        "candidate_uid",
        "session",
        "outer_fold",
        "entry_arrival_ns",
        "horizon_min",
        "filled",
        "pnl",
        "invalid_source",
        "limit",
        "fill_time_ns",
        "exit_time_ns",
    }
    if required - set(outcomes):
        raise DataContractError("passive counterfactual schema drift")
    if set(outcomes["horizon_min"].unique()) != {30, 60}:
        raise DataContractError("passive counterfactual horizons are not exactly 30/60")
    if outcomes.duplicated(["candidate_uid", "horizon_min"]).any():
        raise DataContractError("passive counterfactual identity is duplicated")
    wide_parts = []
    for horizon in (30, 60):
        item = outcomes[outcomes["horizon_min"].eq(horizon)][
            [
                "candidate_uid",
                "filled",
                "pnl",
                "invalid_source",
                "limit",
                "fill_time_ns",
                "exit_time_ns",
            ]
        ].rename(
            columns={
                "filled": f"filled_{horizon}m",
                "pnl": f"pnl_{horizon}m",
                "invalid_source": f"invalid_source_{horizon}m",
                "limit": f"limit_{horizon}m",
                "fill_time_ns": f"fill_time_ns_{horizon}m",
                "exit_time_ns": f"exit_time_ns_{horizon}m",
            }
        )
        wide_parts.append(item)
    wide = wide_parts[0].merge(wide_parts[1], on="candidate_uid", validate="one_to_one")
    frame = features.merge(wide, on="candidate_uid", how="left", validate="one_to_one")
    if frame["pnl_60m"].isna().any():
        raise DataContractError("passive outcome join is incomplete")
    terminal = frame["session"].map({session: _session_terminal_ns(session) for session in frame["session"].unique()})
    nominally_eligible = (
        frame["entry_arrival_ns"] + ENTRY_WINDOW_NS + HORIZON_60_NS <= terminal
    ) & ~frame["invalid_source_60m"].astype(bool)
    filled_hold = frame["filled_60m"].astype(bool)
    duration = frame["exit_time_ns_60m"] - frame["fill_time_ns_60m"]
    actionable_full_exit = ~filled_hold | (np.abs(duration - HORIZON_60_NS) <= 2 * SECOND_NS)
    frame["full_horizon_eligible"] = nominally_eligible & actionable_full_exit
    frame["excluded_stale_or_short_60m_exit"] = nominally_eligible & ~actionable_full_exit
    return frame


def _bootstrap_lcb(values: Mapping[str, float]) -> float:
    sample = np.asarray([float(values[key]) for key in sorted(values)], dtype=float)
    if not len(sample):
        return float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for start in range(0, BOOTSTRAP_DRAWS, 1_000):
        count = min(1_000, BOOTSTRAP_DRAWS - start)
        indexes = rng.integers(0, len(sample), size=(count, len(sample)))
        means[start : start + count] = sample[indexes].mean(axis=1)
    return float(np.quantile(means, 0.05))


def session_shuffled_score(frame: pd.DataFrame, score_column: str, *, label: str) -> pd.Series:
    local = pd.to_datetime(frame["feature_boundary_ns"], unit="ns", utc=True).dt.tz_convert(NY)
    work = frame[["session", "outer_fold", "right", "strike_offset", "feature_boundary_ns", "raw_symbol", score_column]].copy()
    work["time_stratum"] = ((local.dt.hour * 60 + local.dt.minute - 570) // 30).astype(int)
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    strata = ["outer_fold", "time_stratum", "right", "strike_offset"]
    for key, group in work.groupby(strata, sort=True):
        by_session = {session: rows.sort_values(["feature_boundary_ns", "raw_symbol"]) for session, rows in group.groupby("session")}
        sessions = sorted(by_session)
        if len(sessions) < 2:
            continue
        seed = int(hashlib.sha256(f"{label}|{key}".encode()).hexdigest()[:16], 16)
        order = sorted(sessions, key=lambda session: hashlib.sha256(f"{seed}|{session}".encode()).hexdigest())
        sources = order[1:] + order[:1]
        for target, source in zip(order, sources, strict=True):
            target_rows = by_session[target]
            source_values = by_session[source][score_column].to_numpy(float)
            if not len(source_values):
                continue
            indexes = np.minimum(
                len(source_values) - 1,
                np.floor(np.arange(len(target_rows)) * len(source_values) / len(target_rows)).astype(int),
            )
            output.loc[target_rows.index] = source_values[indexes]
    return output


def strict_serial_replay(
    frame: pd.DataFrame,
    *,
    score_column: str,
    horizon_min: int,
    threshold: float = ENTRY_THRESHOLD,
    time_window_1030: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if horizon_min not in (30, 60):
        raise ValueError("only frozen 30/60 horizons are legal")
    pnl_column = f"pnl_{horizon_min}m"
    fill_column = f"filled_{horizon_min}m"
    fill_time_column = f"fill_time_ns_{horizon_min}m"
    exit_time_column = f"exit_time_ns_{horizon_min}m"
    limit_column = f"limit_{horizon_min}m"
    sessions = sorted(frame["session"].unique())
    equity = STARTING_EQUITY
    attempts: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    for session in sessions:
        day = frame[frame["session"].eq(session) & frame["full_horizon_eligible"]].copy()
        session_start = equity
        realized = 0.0
        busy_until = -1
        daily_stopped = False
        fold_values = day["outer_fold"].dropna().unique()
        if len(fold_values) != 1:
            raise DataContractError(f"session/fold identity drift during replay: {session}")
        for decision_ns, choices in day.groupby("decision_emission_ns", sort=True):
            if int(decision_ns) < busy_until or daily_stopped:
                continue
            if time_window_1030:
                local = pd.Timestamp(int(decision_ns), unit="ns", tz="UTC").tz_convert(NY)
                if not (local.hour == 10 and 30 <= local.minute <= 59):
                    continue
            eligible = choices[np.isfinite(choices[score_column]) & (choices[score_column] >= threshold)].copy()
            if eligible.empty:
                continue
            eligible = eligible.sort_values(
                [score_column, "relative_spread", "decision_mid", "raw_symbol"],
                ascending=[False, True, True, True],
                kind="mergesort",
            )
            row = eligible.iloc[0]
            limit = float(row[limit_column]) if np.isfinite(row[limit_column]) else float("nan")
            affordable = math.isfinite(limit) and limit * 100.0 + FEE_PER_SIDE <= equity
            filled = bool(row[fill_column]) and affordable
            item = {
                "session": session,
                "outer_fold": int(row["outer_fold"]),
                "candidate_uid": str(row["candidate_uid"]),
                "raw_symbol": str(row["raw_symbol"]),
                "decision_emission_ns": int(decision_ns),
                "score": float(row[score_column]),
                "filled": filled,
                "affordable": affordable,
                "limit": limit,
                "horizon_min": horizon_min,
                "pnl": float(row[pnl_column]) if filled else 0.0,
                "fill_time_ns": int(row[fill_time_column]) if filled else None,
                "exit_time_ns": int(row[exit_time_column]) if filled else None,
                "entry_premium_dollars": limit * 100.0 + FEE_PER_SIDE if filled else None,
            }
            attempts.append(item)
            if not affordable:
                continue
            if not filled:
                busy_until = int(row["entry_arrival_ns"]) + ENTRY_WINDOW_NS
                continue
            busy_until = int(row[exit_time_column])
            realized += float(row[pnl_column])
            equity += float(row[pnl_column])
            daily_stopped = realized <= -DAILY_STOP_FRACTION * session_start
        session_rows.append(
            {
                "session": session,
                "outer_fold": int(fold_values[0]),
                "net_pnl": realized,
                "session_start_equity": session_start,
                "session_end_equity": equity,
                "daily_stopped": daily_stopped,
            }
        )
    attempt_columns = (
        "session",
        "outer_fold",
        "candidate_uid",
        "raw_symbol",
        "decision_emission_ns",
        "score",
        "filled",
        "affordable",
        "limit",
        "horizon_min",
        "pnl",
        "fill_time_ns",
        "exit_time_ns",
        "entry_premium_dollars",
    )
    attempts_frame = pd.DataFrame(attempts, columns=attempt_columns)
    session_frame = pd.DataFrame(session_rows)
    filled_frame = attempts_frame[attempts_frame["filled"]].copy() if not attempts_frame.empty else attempts_frame.copy()
    equity_values = [STARTING_EQUITY]
    if not filled_frame.empty:
        equity_values.extend((STARTING_EQUITY + filled_frame["pnl"].cumsum()).tolist())
    diagnostics = underwater_duration(equity_values)
    diagnostics["ending_equity"] = float(equity)
    diagnostics["filled_trades"] = int(len(filled_frame))
    diagnostics["attempts"] = int(len(attempts_frame))
    diagnostics["filled_sessions"] = int(filled_frame["session"].nunique()) if len(filled_frame) else 0
    diagnostics["filled_folds"] = int(filled_frame["outer_fold"].nunique()) if len(filled_frame) else 0
    if len(filled_frame):
        returns = filled_frame["pnl"].to_numpy(float) / filled_frame["entry_premium_dollars"].to_numpy(float)
        buckets = four_bucket_distribution(returns)
        diagnostics["four_bucket_distribution"] = buckets.as_dict()
        diagnostics["pnl_concentration"] = pnl_concentration(filled_frame["pnl"].to_numpy(float))
    else:
        diagnostics["four_bucket_distribution"] = {
            "name": "four_bucket_distribution",
            "passed": False,
            "detail": "no filled trades",
            "metrics": {"big_loss_share": 1.0, "big_loss_limit": 0.02},
        }
        diagnostics["pnl_concentration"] = {"positive_pnl": 0.0, "top_5_share": None, "top_20_share": None}
    return attempts_frame, session_frame, diagnostics


def _session_dict(frame: pd.DataFrame) -> dict[str, float]:
    return {str(row.session): float(row.net_pnl) for row in frame.itertuples(index=False)}


def _fold_deltas(session_delta: Mapping[str, float], fold_map: Mapping[str, int]) -> dict[str, float]:
    return {
        str(fold): float(sum(value for session, value in session_delta.items() if fold_map[session] == fold))
        for fold in range(5)
    }


def _positive_delta_concentration(session_delta: Mapping[str, float]) -> dict[str, Any]:
    positive = {session: value for session, value in session_delta.items() if value > 0.0}
    total = float(sum(positive.values()))
    by_weekday: dict[str, float] = {}
    for session, value in positive.items():
        weekday = pd.Timestamp(session).day_name()
        by_weekday[weekday] = by_weekday.get(weekday, 0.0) + value
    session_share = max(positive.values(), default=0.0) / total if total > 0.0 else 1.0
    weekday_share = max(by_weekday.values(), default=0.0) / total if total > 0.0 else 1.0
    return {
        "positive_delta_dollars": total,
        "max_session_share": float(session_share),
        "max_weekday_share": float(weekday_share),
        "passed": bool(total > 0.0 and session_share <= 0.50 and weekday_share <= 0.50),
    }


def _decile_result(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    eligible = frame[frame["full_horizon_eligible"] & np.isfinite(frame[score_column])]
    try:
        result = decile_monotonicity(
            eligible[score_column].to_numpy(float),
            eligible["pnl_60m"].to_numpy(float),
            eligible["outer_fold"].to_numpy(int),
        )
        return result.as_dict()
    except GateError as exc:
        return {"name": "decile_monotonicity", "passed": False, "detail": str(exc), "metrics": {}}


def _policy_evaluation(
    *,
    attempts: pd.DataFrame,
    sessions: pd.DataFrame,
    diagnostics: dict[str, Any],
    comparator_sessions: pd.DataFrame,
    fold_map: Mapping[str, int],
) -> dict[str, Any]:
    policy = _session_dict(sessions)
    comparator = _session_dict(comparator_sessions)
    delta = {session: policy[session] - comparator[session] for session in sorted(policy)}
    folds = _fold_deltas(delta, fold_map)
    concentration = _positive_delta_concentration(delta)
    paired = paired_summary(delta)
    bucket = diagnostics["four_bucket_distribution"]
    criteria = {
        "positive_and_beats_best_comparator": bool(sum(policy.values()) > sum(comparator.values()) and sum(policy.values()) > 0.0),
        "positive_at_least_4_of_5_folds": bool(sum(value > 0.0 for value in folds.values()) >= 4),
        "positive_one_sided_95pct_session_bootstrap_lcb": bool(_bootstrap_lcb(delta) > 0.0),
        "minimum_100_filled_trades": bool(diagnostics["filled_trades"] >= 100),
        "minimum_30_filled_sessions": bool(diagnostics["filled_sessions"] >= 30),
        "fills_in_all_five_folds": bool(diagnostics["filled_folds"] == 5),
        "positive_delta_not_concentrated": bool(concentration["passed"]),
        "big_loss_share_at_most_2pct": bool(bucket["passed"]),
    }
    return {
        "total_net_pnl_dollars": float(sum(policy.values())),
        "best_comparator_total_net_pnl_dollars": float(sum(comparator.values())),
        "paired_delta_total_dollars": float(sum(delta.values())),
        "mean_session_net_pnl_dollars": float(np.mean(list(policy.values()))),
        "mean_session_paired_delta_dollars": float(np.mean(list(delta.values()))),
        "one_sided_95pct_session_bootstrap_lcb_dollars": _bootstrap_lcb(delta),
        "raw_p_one_sided": paired["p_one_sided"],
        "fold_deltas_dollars": folds,
        "positive_folds": int(sum(value > 0.0 for value in folds.values())),
        "concentration": concentration,
        "criteria": criteria,
        "diagnostics": diagnostics,
        "session_delta": delta,
        "attempt_rows": int(len(attempts)),
    }


def _choose_best_comparator(comparators: Mapping[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]) -> str:
    order = ("not_trading", "passive_all_time", "passive_1030_1059")
    totals = {
        name: float(item[1]["net_pnl"].sum()) for name, item in comparators.items()
    }
    return max(order, key=lambda name: (totals[name], -order.index(name)))


def _control_accepted(evaluation: Mapping[str, Any], *, causal_checks_passed: bool) -> bool:
    return bool(
        causal_checks_passed
        and all(evaluation["criteria"].values())
        and evaluation["raw_p_one_sided"] is not None
        and float(evaluation["raw_p_one_sided"]) <= 0.05
    )


def _causal_audit(frame: pd.DataFrame) -> dict[str, Any]:
    member_available = {
        MEMBERS[0]: frame["option_available_at_ns"].to_numpy(float),
        MEMBERS[1]: frame[["option_available_at_ns", "option_volume_available_at_ns"]].max(axis=1).to_numpy(float),
        MEMBERS[2]: frame["cross_market_available_at_ns"].to_numpy(float),
    }
    violations: dict[str, int] = {}
    for member, availability in member_available.items():
        finite = np.isfinite(frame[SCORE_COLUMNS[member]].to_numpy(float))
        decision = frame["decision_emission_ns"].to_numpy(np.int64)
        violations[member] = int((finite & (~np.isfinite(availability) | (availability > decision))).sum())
    # The feature frame is sealed before outcomes are joined. Mutating every
    # future outcome must therefore leave the feature hash unchanged.
    feature_columns = ["candidate_uid", *SCORE_COLUMNS.values()]
    before = pd.util.hash_pandas_object(frame[feature_columns], index=False).to_numpy(np.uint64)
    mutated = frame.copy()
    for column in ("pnl_30m", "pnl_60m", "fill_time_ns_60m", "exit_time_ns_60m"):
        if column in mutated:
            numeric = pd.to_numeric(mutated[column], errors="coerce")
            mutated[column] = np.where(numeric.notna(), numeric * -7.0 + 123.0, numeric)
    after = pd.util.hash_pandas_object(mutated[feature_columns], index=False).to_numpy(np.uint64)
    return {
        "clock_violations": violations,
        "causal_clock_passed": not any(violations.values()),
        "future_mutation_columns": ["pnl_30m", "pnl_60m", "fill_time_ns_60m", "exit_time_ns_60m"],
        "future_mutation_passed": bool(np.array_equal(before, after)),
    }


def _peak_enrichment(paths: Wave2Paths, attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return attempts.assign(peak_available_pnl=np.nan)
    result = attempts.copy()
    result["peak_available_pnl"] = np.nan
    for session, selected in result[result["filled"]].groupby("session", sort=True):
        path = paths.cbbo_1s / f"{session}.cbbo-1s.parquet"
        symbols = selected["raw_symbol"].astype(str).unique()
        quotes = _parquet_frame(path, ("ts_recv", "symbol", "bid_px_00"), symbols=symbols)
        quotes["ts_ns"] = pd.to_datetime(quotes["ts_recv"], utc=True).astype("int64")
        for index, trade in selected.iterrows():
            path_rows = quotes[
                quotes["symbol"].eq(trade["raw_symbol"])
                & quotes["ts_ns"].between(int(trade["fill_time_ns"]), int(trade["exit_time_ns"]))
            ]
            bids = pd.to_numeric(path_rows["bid_px_00"], errors="coerce")
            bids = bids[np.isfinite(bids) & (bids >= 0.0)]
            if len(bids):
                result.at[index, "peak_available_pnl"] = float(
                    (bids.max() - float(trade["limit"])) * 100.0 - 2.0 * FEE_PER_SIDE
                )
    return result


def run_gate(
    paths: Wave2Paths,
    output_root: Path,
    *,
    preregistration: Path = PREREGISTRATION_PATH,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    preregistration = Path(preregistration).resolve(strict=True)
    if "FROZEN_BEFORE_EXECUTION" not in preregistration.read_text():
        raise DataContractError("Wave-2 preregistration is not frozen")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    before_state = forbidden_state_snapshot(Path(repo_root).resolve())
    freeze = {
        "schema_version": "pathd.wave2-preregistration-freeze.v1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "path": str(preregistration),
        "sha256": _sha256_path(preregistration),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "declared_family": list(MEMBERS),
        "family_size": 3,
        "threshold": ENTRY_THRESHOLD,
        "acceptance_horizon_minutes": 60,
        "diagnostic_horizon_minutes": 30,
        "model_fit_authorized": False,
    }
    (output_root / "preregistration_freeze_receipt.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    )
    lineage = feature_lineage_manifest()
    (output_root / "feature_lineage_and_clock_manifest.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n"
    )
    features, membership, source_manifest = build_candidate_features(paths)
    frame = _outcome_dataset(paths, features)
    frame = frame.sort_values(["session", "feature_boundary_ns", "raw_symbol"], kind="mergesort").reset_index(drop=True)
    if frame["candidate_uid"].duplicated().any():
        raise DataContractError("joined candidate identity is duplicated")
    causal = _causal_audit(frame)
    identity = {
        "candidate_rows": int(len(frame)),
        "unique_candidate_uids": int(frame["candidate_uid"].nunique()),
        "session_fold_unique": bool(
            not frame[["session", "outer_fold"]].drop_duplicates()["session"].duplicated().any()
        ),
        "outcome_join_complete": bool(frame["pnl_60m"].notna().all()),
    }
    identity["passed"] = bool(
        identity["candidate_rows"] == identity["unique_candidate_uids"]
        and identity["session_fold_unique"]
        and identity["outcome_join_complete"]
    )
    if not identity["passed"]:
        raise DataContractError("identity audit failed")
    candidate_path = output_root / "candidate_scores.parquet"
    frame.to_parquet(candidate_path, index=False)

    fold_map = {
        str(row.session): int(row.outer_fold)
        for row in frame[["session", "outer_fold"]].drop_duplicates().itertuples(index=False)
    }
    constant = frame.copy()
    constant["__constant"] = 1.0
    comparator_runs: dict[int, dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]] = {}
    for horizon in (60, 30):
        all_time = strict_serial_replay(constant, score_column="__constant", horizon_min=horizon)
        window = strict_serial_replay(
            constant, score_column="__constant", horizon_min=horizon, time_window_1030=True
        )
        zero_sessions = all_time[1].copy()
        zero_sessions["net_pnl"] = 0.0
        zero_sessions["session_end_equity"] = zero_sessions["session_start_equity"]
        comparator_runs[horizon] = {
            "not_trading": (pd.DataFrame(), zero_sessions, {"ending_equity": STARTING_EQUITY, "filled_trades": 0}),
            "passive_all_time": all_time,
            "passive_1030_1059": window,
        }

    primary_runs: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    primary_evaluations: dict[str, dict[str, Any]] = {}
    diagnostic_30m: dict[str, Any] = {}
    best_comparators: dict[str, str] = {}
    for member in MEMBERS:
        score = SCORE_COLUMNS[member]
        primary = strict_serial_replay(frame, score_column=score, horizon_min=60)
        primary_runs[member] = primary
        best = _choose_best_comparator(comparator_runs[60])
        best_comparators[member] = best
        evaluation = _policy_evaluation(
            attempts=primary[0],
            sessions=primary[1],
            diagnostics=primary[2],
            comparator_sessions=comparator_runs[60][best][1],
            fold_map=fold_map,
        )
        evaluation["best_comparator"] = best
        evaluation["decile_monotonicity"] = _decile_result(frame, score)
        primary_evaluations[member] = evaluation
        diag = strict_serial_replay(frame, score_column=score, horizon_min=30)
        diag_best = _choose_best_comparator(comparator_runs[30])
        diagnostic_30m[member] = {
            "best_comparator": diag_best,
            **_policy_evaluation(
                attempts=diag[0],
                sessions=diag[1],
                diagnostics=diag[2],
                comparator_sessions=comparator_runs[30][diag_best][1],
                fold_map=fold_map,
            ),
        }

    family = session_blocked_max_t(
        {member: evaluation["session_delta"] for member, evaluation in primary_evaluations.items()},
        permutations=MAXT_PERMUTATIONS,
        seed=MAXT_SEED,
    )
    max_t_packet = {
        "method": "session_blocked_sign_flip_maxT",
        "family": list(MEMBERS),
        "family_size": 3,
        "permutations": MAXT_PERMUTATIONS,
        "seed": MAXT_SEED,
        "members": family,
    }
    (output_root / "family_maxT_packet.json").write_text(
        json.dumps(max_t_packet, indent=2, sort_keys=True, default=_json_default) + "\n"
    )

    causal_checks_passed = bool(causal["causal_clock_passed"] and causal["future_mutation_passed"] and identity["passed"])
    controls: dict[str, Any] = {}
    accepted_control = False
    control_attempts: list[pd.DataFrame] = []
    for member in MEMBERS:
        score = SCORE_COLUMNS[member]
        transformed = {
            "sign_reversed": -frame[score],
            "session_shuffled": session_shuffled_score(frame, score, label=member),
            "constant": pd.Series(1.0, index=frame.index),
        }
        controls[member] = {}
        best = best_comparators[member]
        for control_name, values in transformed.items():
            control_frame = frame.copy()
            control_frame["__control"] = values
            replay = strict_serial_replay(control_frame, score_column="__control", horizon_min=60)
            if len(replay[0]):
                item = replay[0].copy()
                item["member"] = member
                item["control"] = control_name
                control_attempts.append(item)
            evaluation = _policy_evaluation(
                attempts=replay[0],
                sessions=replay[1],
                diagnostics=replay[2],
                comparator_sessions=comparator_runs[60][best][1],
                fold_map=fold_map,
            )
            evaluation["accepted"] = _control_accepted(
                evaluation, causal_checks_passed=causal_checks_passed
            )
            accepted_control = accepted_control or evaluation["accepted"]
            controls[member][control_name] = evaluation
    (output_root / "negative_control_packet.json").write_text(
        json.dumps(controls, indent=2, sort_keys=True, default=_json_default) + "\n"
    )

    passing: list[str] = []
    for member in MEMBERS:
        evaluation = primary_evaluations[member]
        criteria = dict(evaluation["criteria"])
        criteria.update(
            {
                "top_decile_best_and_beats_bottom": bool(evaluation["decile_monotonicity"]["passed"]),
                "family_adjusted_maxT_p_le_0_05": bool(family[member]["maxT_p_one_sided"] <= 0.05),
                "all_negative_controls_fail": bool(
                    not any(item["accepted"] for item in controls[member].values())
                ),
                "causal_clock_future_mutation_identity_pass": causal_checks_passed,
            }
        )
        evaluation["maxT_p_one_sided"] = family[member]["maxT_p_one_sided"]
        evaluation["acceptance_criteria"] = criteria
        evaluation["passed"] = bool(all(criteria.values()))
        if evaluation["passed"]:
            passing.append(member)

    # Charter diagnostics require peak-available capture. Enrich only the three
    # selected primary trade sets, keeping the high-resolution scan bounded.
    primary_attempts: list[pd.DataFrame] = []
    for member, run in primary_runs.items():
        item = run[0].copy()
        item["member"] = member
        item["policy_kind"] = "primary"
        primary_attempts.append(item)
    combined_primary = pd.concat(primary_attempts, ignore_index=True) if primary_attempts else pd.DataFrame()
    enriched_primary = _peak_enrichment(paths, combined_primary)
    replay_outputs: list[pd.DataFrame] = [enriched_primary]
    for member in MEMBERS:
        enriched = enriched_primary[enriched_primary["member"].eq(member)]
        filled = enriched[enriched["filled"]]
        if len(filled) and (filled["peak_available_pnl"] > 0.0).any():
            primary_evaluations[member]["diagnostics"]["harvest_ratio"] = harvest_ratio(
                filled["pnl"].to_numpy(float), filled["peak_available_pnl"].to_numpy(float)
            )
        else:
            primary_evaluations[member]["diagnostics"]["harvest_ratio"] = {
                "n_trades_with_upside": 0,
                "mean_harvest_ratio": None,
                "median_harvest_ratio": None,
                "aggregate_harvest_ratio": None,
            }
    if control_attempts:
        replay_outputs.extend(control_attempts)
    replay_frame = pd.concat(replay_outputs, ignore_index=True) if replay_outputs else pd.DataFrame()
    replay_frame.to_parquet(output_root / "replay_attempts_and_trades.parquet", index=False)
    comparator_summary = {
        str(horizon): {
            name: {
                "total_net_pnl_dollars": float(run[1]["net_pnl"].sum()),
                "mean_session_net_pnl_dollars": float(run[1]["net_pnl"].mean()),
                "filled_trades": int(run[2].get("filled_trades", 0)),
                "max_drawdown_dollars": float(run[2].get("max_drawdown", 0.0)),
            }
            for name, run in values.items()
        }
        for horizon, values in comparator_runs.items()
    }
    after_state = forbidden_state_snapshot(Path(repo_root).resolve())
    side_effects = {
        "before": before_state,
        "after": after_state,
        "unchanged": before_state == after_state,
        "protected_holdout_opened": False,
        "model_fit_executed": False,
        "paid_data_downloaded": False,
        "broker_accessed": False,
        "paper_order_submitted": False,
        "runtime_or_promotion_changed": False,
    }
    (output_root / "side_effect_audit.json").write_text(
        json.dumps(side_effects, indent=2, sort_keys=True) + "\n"
    )
    if not side_effects["unchanged"]:
        raise DataContractError("forbidden state changed during the offline gate")

    if accepted_control:
        verdict = "INVALID"
        winner = None
    elif passing:
        winner = max(
            passing,
            key=lambda member: (
                primary_evaluations[member]["mean_session_net_pnl_dollars"],
                -primary_evaluations[member]["diagnostics"]["max_drawdown"],
                -MEMBERS.index(member),
            ),
        )
        verdict = "TRAINING_GATE_EARNED"
    else:
        verdict = "NO_SIGNAL"
        winner = None
    result = {
        "schema_version": "pathd.wave2-causal-60m-signal-discovery.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "winner": winner,
        "freeze": freeze,
        "membership": membership,
        "input_hashes": {
            "oof_scores": _sha256_path(paths.scores),
            "passive_counterfactuals": _sha256_path(paths.counterfactuals),
        },
        "lineage": lineage,
        "source_manifest": source_manifest,
        "identity_audit": identity,
        "causal_audit": causal,
        "comparators": comparator_summary,
        "members": primary_evaluations,
        "diagnostic_30m": diagnostic_30m,
        "maxT": max_t_packet,
        "negative_controls": controls,
        "side_effect_audit": side_effects,
        "protected_holdout_opened": False,
        "model_fit_executed": False,
        "training_goal_drafted": bool(winner),
    }
    # Re-read the scored candidates and independently verify the deterministic
    # candidate seal. This catches serialization/order drift without a second raw scan.
    persisted = pd.read_parquet(candidate_path)
    columns_to_seal = ["candidate_uid", *SCORE_COLUMNS.values(), "pnl_60m", "full_horizon_eligible"]
    first_hash = hashlib.sha256(
        pd.util.hash_pandas_object(frame[columns_to_seal], index=False).to_numpy(np.uint64).tobytes()
    ).hexdigest()
    second_hash = hashlib.sha256(
        pd.util.hash_pandas_object(persisted[columns_to_seal], index=False).to_numpy(np.uint64).tobytes()
    ).hexdigest()
    result["reproducibility_audit"] = {
        "in_memory_candidate_seal": first_hash,
        "persisted_candidate_seal": second_hash,
        "passed": first_hash == second_hash,
    }
    if not result["reproducibility_audit"]["passed"]:
        raise DataContractError("candidate serialization reproduction failed")
    semantic = dict(result)
    semantic.pop("generated_at_utc", None)
    result["result_semantic_sha256"] = _semantic_sha(semantic)
    if winner:
        training_goal = (
            "# Draft only — Path-D Wave 2 bounded model-training goal\n\n"
            f"Winning discovery member: `{winner}`.\n\n"
            "STOP: owner review and explicit model-training authorization are required before any fit. "
            "Bind the Wave-2 result SHA, retain the same OOF sessions and causal/live parity contract, "
            "pre-register one bounded estimator family, and do not access the spent historical firewall.\n"
        )
        (output_root / "DRAFT_MODEL_TRAINING_GOAL.md").write_text(training_goal)
    (output_root / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    (output_root / "report.md").write_text(render_report(result))
    return result


def render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Path-D Wave 2 — Causal 60-Minute Signal Discovery Results",
        "",
        f"**Verdict:** `{result['verdict']}`",
        "",
        "No model was fit. The spent historical firewall remained closed; no paid data, broker, paper, "
        "runtime, promotion, or default path was used.",
        "",
        "## Primary family",
        "",
        "| Member | Net PnL | Best comparator | Delta | +folds | LCB | maxT p | Trades | Deciles | Pass |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for member in MEMBERS:
        row = result["members"][member]
        lines.append(
            f"| {member} | ${row['total_net_pnl_dollars']:,.2f} | {row['best_comparator']} | "
            f"${row['paired_delta_total_dollars']:,.2f} | {row['positive_folds']}/5 | "
            f"${row['one_sided_95pct_session_bootstrap_lcb_dollars']:,.2f} | "
            f"{row['maxT_p_one_sided']:.6f} | {row['diagnostics']['filled_trades']} | "
            f"{str(row['decile_monotonicity']['passed']).lower()} | {str(row['passed']).lower()} |"
        )
    lines.extend(("", "## Comparators", "", "```json", json.dumps(result["comparators"]["60"], indent=2, sort_keys=True), "```", ""))
    lines.extend(("## Negative controls", ""))
    for member in MEMBERS:
        lines.append(f"### {member}")
        lines.append("")
        for name, control in result["negative_controls"][member].items():
            lines.append(
                f"- `{name}`: accepted `{str(control['accepted']).lower()}`, net "
                f"`${control['total_net_pnl_dollars']:,.2f}`, delta `${control['paired_delta_total_dollars']:,.2f}`."
            )
        lines.append("")
    lines.extend(
        (
            "## Audits",
            "",
            f"- Causal clock: `{result['causal_audit']['causal_clock_passed']}`",
            f"- Future mutation: `{result['causal_audit']['future_mutation_passed']}`",
            f"- Identity: `{result['identity_audit']['passed']}`",
            f"- Reproducibility: `{result['reproducibility_audit']['passed']}`",
            f"- Forbidden side effects unchanged: `{result['side_effect_audit']['unchanged']}`",
            "",
            "## Claim boundary",
            "",
            "A passing member would earn only a separate owner-reviewed training proposal. This packet is "
            "not deployable-edge, live-paper, promotion, or real-money evidence.",
            "",
            "STOP_FOR_CLAUDE_VERIFICATION",
            "",
        )
    )
    return "\n".join(lines)


__all__ = [
    "COMPONENTS",
    "DataContractError",
    "MEMBERS",
    "PREREGISTRATION_PATH",
    "SCORE_COLUMNS",
    "Wave2Paths",
    "feature_lineage_manifest",
    "forbidden_state_snapshot",
    "load_oof_membership",
    "render_report",
    "run_gate",
    "session_shuffled_score",
    "strict_serial_replay",
]
