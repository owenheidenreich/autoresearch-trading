"""Data integrity validation layer.

Validates the full data chain before training: manifest (data.pt),
sidecars (per-day files), and normalized features. Catches garbage
inputs before they corrupt model training.

Usage:
    python -m v2.core.data_integrity [--data v2/data.pt] [--sample N]
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from v2.core.chain_data import (
    QUALITY_VALID, QUALITY_PARTIAL, QUALITY_CORRUPT,
    contract_snapshot, load_sidecar_cached, padded_snapshot,
)
from v2.core.features import FEATURE_NAMES, NUM_FEATURES, _NO_NORMALIZE


DEFAULT_RAW_INPUTS = {
    "spx": (
        os.path.expanduser("~/.cache/autoresearch-trading/data/spx_1min.pkl"),
        ("spx_open", "spx_high", "spx_low", "spx_close"),
    ),
    "spy": (
        os.path.expanduser("~/.cache/autoresearch-trading/data/spy_1min.pkl"),
        ("open", "high", "low", "close"),
    ),
    "vix": (
        os.path.expanduser("~/.cache/autoresearch-trading/data/vix_1min.pkl"),
        ("vix_open", "vix_high", "vix_low", "vix_close"),
    ),
}
RAW_BREAK_THRESHOLD = 0.10
RAW_STAGNANT_STREAK = 45
SIDECAR_THIN_BAR_ALERT = 0.50
SIDECAR_LABEL_NAN_ALERT = 0.80
SIDECAR_ORDERING_ALERT = 50
SIDECAR_PRICE_ALERT = 100


@dataclass
class DateAnomaly:
    date: str
    source: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class SideBiasAuditSummary:
    mask_name: str
    rows: int = 0
    rows_with_valid_snapshot: int = 0
    bars_with_calls: int = 0
    bars_with_puts: int = 0
    bars_with_both_sides: int = 0
    best_call_bars: int = 0
    best_put_bars: int = 0
    tied_best_side_bars: int = 0
    target_call_mass_sum: float = 0.0
    target_put_mass_sum: float = 0.0
    valid_call_sum: int = 0
    valid_put_sum: int = 0
    model_scored_rows: int = 0
    model_traded_rows: int = 0
    model_call_count: int = 0
    model_put_count: int = 0
    model_oracle_eval_rows: int = 0
    model_oracle_call_rows: int = 0
    model_oracle_put_rows: int = 0
    model_direction_matches: int = 0
    model_direction_matches_oracle_call: int = 0
    model_direction_matches_oracle_put: int = 0
    model_selected_pnl_sum: float = 0.0
    model_selected_pnl_count: int = 0
    model_selected_pnl_sum_oracle_call: float = 0.0
    model_selected_pnl_count_oracle_call: int = 0
    model_selected_pnl_sum_oracle_put: float = 0.0
    model_selected_pnl_count_oracle_put: int = 0


@dataclass
class DataQualityReport:
    """Result of a validation check."""

    source: str = ""              # what was validated
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.errors:
            self.passed = False

    def print_report(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        print(f"  [{status}] {self.source}")
        for e in self.errors:
            print(f"    ERROR: {e}")
        for w in self.warnings:
            print(f"    WARN:  {w}")
        if self.stats:
            for k, v in sorted(self.stats.items()):
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = ["X", "dates", "bar_of_day", "spot_prices", "metadata"]


def validate_manifest(data: dict) -> DataQualityReport:
    """Validate the data.pt manifest structure and basic sanity."""
    r = DataQualityReport(source="manifest (data.pt)")

    # Required keys
    for key in _REQUIRED_KEYS:
        if key not in data:
            r.errors.append(f"missing required key: {key}")
    if r.errors:
        return r

    X = data["X"]
    n_bars = X.shape[0]
    r.stats["n_bars"] = n_bars

    # Shape checks
    if X.ndim != 2:
        r.errors.append(f"X should be 2D, got {X.ndim}D")
    elif X.shape[1] != NUM_FEATURES:
        r.errors.append(f"X has {X.shape[1]} features, expected {NUM_FEATURES}")

    if "X_sim" in data:
        if data["X_sim"].shape[0] != n_bars:
            r.errors.append(f"X_sim has {data['X_sim'].shape[0]} bars, X has {n_bars}")

    # Dates
    dates = data["dates"]
    if len(dates) != n_bars:
        r.errors.append(f"dates has {len(dates)} entries, X has {n_bars} bars")

    unique_days = sorted(set(dates))
    r.stats["n_unique_days"] = len(unique_days)

    # bar_of_day bounds
    bod = data["bar_of_day"]
    if hasattr(bod, "numpy"):
        bod = bod.numpy()
    bod_min, bod_max = int(bod.min()), int(bod.max())
    if bod_min < 0 or bod_max > 389:
        r.errors.append(f"bar_of_day out of range: [{bod_min}, {bod_max}], expected [0, 389]")

    # spot_prices
    sp = data["spot_prices"]
    if hasattr(sp, "numpy"):
        sp = sp.numpy()
    n_nan = int(np.isnan(sp).sum())
    n_neg = int((sp <= 0).sum())
    if n_nan > 0:
        r.errors.append(f"spot_prices has {n_nan} NaN values")
    if n_neg > 0:
        r.errors.append(f"spot_prices has {n_neg} non-positive values")
    r.stats["spot_min"] = float(np.nanmin(sp))
    r.stats["spot_max"] = float(np.nanmax(sp))

    # Metadata
    meta = data.get("metadata", {})
    if "chain_sidecar_dir" not in meta:
        r.warnings.append("metadata missing chain_sidecar_dir")
    if "max_contracts_per_bar" not in meta:
        r.warnings.append("metadata missing max_contracts_per_bar")
    if "fingerprint" not in meta:
        r.warnings.append("metadata missing fingerprint")

    # Mask keys
    for mask_name in ["train_mask", "val_mask", "promote_mask"]:
        if mask_name in data:
            mask = data[mask_name]
            if hasattr(mask, "numpy"):
                mask = mask.numpy()
            n_true = int(mask.sum())
            r.stats[f"{mask_name}_bars"] = n_true
            if n_true == 0:
                r.warnings.append(f"{mask_name} has zero True values")
        else:
            r.warnings.append(f"missing mask: {mask_name}")

    return r


# ---------------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------------

def validate_features(X: np.ndarray) -> DataQualityReport:
    """Validate normalized feature matrix for quality issues."""
    r = DataQualityReport(source="feature matrix")

    if X.ndim != 2 or X.shape[1] != NUM_FEATURES:
        r.errors.append(f"unexpected shape {X.shape}, expected (N, {NUM_FEATURES})")
        return r

    n_bars = X.shape[0]

    for j in range(NUM_FEATURES):
        name = FEATURE_NAMES[j]
        col = X[:, j].astype(np.float64)

        # NaN rate
        nan_rate = float(np.isnan(col).sum()) / n_bars
        if nan_rate > 0.05:
            r.warnings.append(f"feature '{name}' has {nan_rate:.1%} NaN rate (> 5%)")

        # For z-scored features, check clip saturation
        if name not in _NO_NORMALIZE:
            at_pos_clip = float((col >= 4.99).sum()) / n_bars
            at_neg_clip = float((col <= -4.99).sum()) / n_bars
            total_clip = at_pos_clip + at_neg_clip
            if total_clip > 0.10:
                r.warnings.append(
                    f"feature '{name}' has {total_clip:.1%} values at clip boundary (> 10%)"
                )

            # Dead feature check
            finite = col[np.isfinite(col)]
            if len(finite) > 100:
                std = float(np.std(finite))
                if std < 0.01:
                    r.warnings.append(f"feature '{name}' appears dead (std={std:.4f} < 0.01)")

                # Distribution sanity for z-scored
                mean = float(np.mean(finite))
                r.stats[f"feat_{name}_mean"] = mean
                r.stats[f"feat_{name}_std"] = std

    return r


# ---------------------------------------------------------------------------
# Sidecar validation
# ---------------------------------------------------------------------------

def validate_sidecar(sidecar: dict, date: str) -> DataQualityReport:
    """Validate a single day's sidecar file."""
    r = DataQualityReport(source=f"sidecar ({date})")

    # Shape integrity
    if "bar_ptrs" not in sidecar:
        r.errors.append("missing bar_ptrs")
        return r

    ptrs = sidecar["bar_ptrs"]
    if len(ptrs) != 391:
        r.errors.append(f"bar_ptrs has {len(ptrs)} entries, expected 391")

    if "contract_strike" not in sidecar or "contract_mid" not in sidecar:
        r.errors.append("missing contract_strike or contract_mid")
        return r

    n_contracts = len(sidecar["contract_strike"])
    mid_shape = sidecar["contract_mid"].shape
    if mid_shape[0] != n_contracts:
        r.errors.append(f"contract_mid has {mid_shape[0]} contracts, strike has {n_contracts}")

    r.stats["n_contracts"] = n_contracts

    # Price sanity (supervised window bars 30-270)
    bid = sidecar.get("contract_bid")
    mid = sidecar["contract_mid"]
    ask = sidecar.get("contract_ask")
    quality = sidecar.get("contract_quality")

    price_violations = 0
    ordering_violations = 0

    for c_idx in range(min(n_contracts, 50)):  # sample up to 50 contracts
        for bar in range(30, min(270, mid_shape[1] if mid.ndim > 1 else 390)):
            m = float(mid[c_idx, bar]) if mid.ndim > 1 else 0.0
            q = int(quality[c_idx, bar]) if quality is not None and quality.ndim > 1 else 0

            if q < QUALITY_PARTIAL:
                continue

            if not np.isfinite(m) or m <= 0:
                price_violations += 1
                continue

            if bid is not None and ask is not None and bid.ndim > 1:
                b = float(bid[c_idx, bar])
                a = float(ask[c_idx, bar])
                if np.isfinite(b) and np.isfinite(a):
                    if b > m + 0.01 or m > a + 0.01:
                        ordering_violations += 1

    if price_violations > 100:
        r.warnings.append(f"{price_violations} price violations (non-positive mid for valid quality)")
    if ordering_violations > 50:
        r.warnings.append(f"{ordering_violations} bid/mid/ask ordering violations")

    r.stats["price_violations"] = price_violations
    r.stats["ordering_violations"] = ordering_violations

    # Chain completeness: count executable contracts per bar
    thin_bars = 0
    total_checked = 0
    for bar in range(30, 270):
        feats, labels, _ = contract_snapshot(sidecar, bar)
        n_exec = int((feats[:, 0] > 0.5).sum()) if len(feats) > 0 else 0
        total_checked += 1
        if n_exec < 5:
            thin_bars += 1

    thin_pct = thin_bars / max(total_checked, 1)
    r.stats["thin_bar_pct"] = thin_pct
    if thin_pct > 0.50:
        r.warnings.append(f"{thin_pct:.0%} of supervised bars have < 5 executable contracts")

    # Label consistency
    if "bar_best_contract_idx" in sidecar and "bar_labelable" in sidecar:
        labelable = sidecar["bar_labelable"]
        best_idx = sidecar["bar_best_contract_idx"]
        label_issues = 0
        n_labelable = 0
        for bar in range(30, 270):
            if not labelable[bar]:
                continue
            n_labelable += 1
            idx = int(best_idx[bar])
            if idx < 0 or idx >= n_contracts:
                label_issues += 1

        r.stats["n_labelable_bars"] = n_labelable
        if label_issues > 0:
            r.warnings.append(f"{label_issues} bars with invalid best_contract_idx")

    # NaN in row_labels for labelable bars
    if "row_labels" in sidecar:
        rl = sidecar["row_labels"]
        nan_rate = float(np.isnan(rl).sum()) / max(len(rl), 1)
        r.stats["row_labels_nan_rate"] = nan_rate
        if nan_rate > 0.30:
            r.warnings.append(f"row_labels NaN rate is {nan_rate:.0%} (> 30%)")

    return r


# ---------------------------------------------------------------------------
# Label quality scoring
# ---------------------------------------------------------------------------

def compute_label_quality_scores(sidecar: dict) -> np.ndarray:
    """Compute per-bar label quality (margin between top and second-best).

    Returns (390,) array. High margin = clear oracle winner. Low margin = noise.
    """
    quality = np.zeros(390, dtype=np.float32)

    for bar in range(390):
        feats, labels, _ = contract_snapshot(sidecar, bar)
        if len(labels) == 0:
            continue

        valid = np.isfinite(labels) & (feats[:, 0] > 0.5)
        if valid.sum() < 1:
            continue

        valid_labels = labels[valid]
        if len(valid_labels) < 2:
            quality[bar] = float(valid_labels[0]) if len(valid_labels) == 1 else 0.0
            continue

        sorted_desc = np.sort(valid_labels)[::-1]
        quality[bar] = float(sorted_desc[0] - sorted_desc[1])

    return quality


def _max_equal_close_run(values: np.ndarray) -> int:
    max_run = 0
    current_run = 0
    prev = None
    for value in values:
        if not np.isfinite(value):
            current_run = 0
            prev = None
            continue
        if prev is not None and value == prev:
            current_run += 1
        else:
            current_run = 1
        prev = value
        if current_run > max_run:
            max_run = current_run
    return max_run


def audit_raw_minute_file(source: str, path: str, price_cols: tuple[str, str, str, str]) -> tuple[DataQualityReport, list[DateAnomaly]]:
    report = DataQualityReport(source=f"raw minute audit ({source})")
    anomalies: list[DateAnomaly] = []

    if not os.path.exists(path):
        report.errors.append(f"raw input not found: {path}")
        return report, anomalies

    try:
        frame = pd.read_pickle(path)
    except NotImplementedError as exc:
        report.errors.append(
            f"could not load {path}; use the project .venv Python (>=3.10). Loader error: {exc}"
        )
        return report, anomalies
    except Exception as exc:  # pragma: no cover - defensive for local environment drift
        report.errors.append(f"failed to load {path}: {exc}")
        return report, anomalies

    required_cols = ["date", *price_cols]
    missing_cols = [col for col in required_cols if col not in frame.columns]
    if missing_cols:
        report.errors.append(f"missing required columns: {missing_cols}")
        return report, anomalies

    open_col, _, _, close_col = price_cols
    price_frame = frame.loc[:, required_cols].copy()
    price_frame["date"] = price_frame["date"].astype(str)
    prev_close = price_frame[close_col].shift(1)
    same_day_prev = price_frame["date"].eq(price_frame["date"].shift(1))
    valid_gap = same_day_prev & price_frame[open_col].notna() & prev_close.notna() & (prev_close.abs() > 1e-10)
    structural_break = pd.Series(False, index=price_frame.index)
    structural_break.loc[valid_gap] = (
        (price_frame.loc[valid_gap, open_col] / prev_close.loc[valid_gap] - 1.0).abs() > RAW_BREAK_THRESHOLD
    )

    grouped = price_frame.groupby("date", sort=True)
    for day, day_frame in grouped:
        missing_price_rows = int(day_frame[list(price_cols)].isna().any(axis=1).sum())
        structural_breaks = int(structural_break.loc[day_frame.index].sum())
        max_stagnant = _max_equal_close_run(day_frame[close_col].to_numpy(dtype=np.float64, copy=False))
        reasons = []
        if missing_price_rows > 0:
            reasons.append("missing_prices")
        if structural_breaks > 0:
            reasons.append("structural_break")
        if max_stagnant >= RAW_STAGNANT_STREAK:
            reasons.append("stagnant_close_run")
        if reasons:
            anomalies.append(DateAnomaly(
                date=day,
                source=source,
                reasons=reasons,
                metrics={
                    "bars": float(len(day_frame)),
                    "missing_price_rows": float(missing_price_rows),
                    "structural_breaks": float(structural_breaks),
                    "max_stagnant_close_run": float(max_stagnant),
                },
            ))

    report.stats["rows"] = float(len(price_frame))
    report.stats["unique_days"] = float(price_frame["date"].nunique())
    report.stats["flagged_days"] = float(len({a.date for a in anomalies}))
    report.stats["findings"] = float(len(anomalies))
    return report, anomalies


def audit_raw_inputs(raw_inputs: dict[str, tuple[str, tuple[str, str, str, str]]] | None = None) -> tuple[list[DataQualityReport], list[DateAnomaly]]:
    reports: list[DataQualityReport] = []
    anomalies: list[DateAnomaly] = []
    for source, (path, price_cols) in (raw_inputs or DEFAULT_RAW_INPUTS).items():
        report, file_anomalies = audit_raw_minute_file(source, path, price_cols)
        reports.append(report)
        anomalies.extend(file_anomalies)
    return reports, anomalies


def audit_sidecar_dates(sidecar_dir: str) -> tuple[DataQualityReport, list[DateAnomaly]]:
    report = DataQualityReport(source="sidecar anomaly audit")
    anomalies: list[DateAnomaly] = []

    if not os.path.isdir(sidecar_dir):
        report.errors.append(f"sidecar directory not found: {sidecar_dir}")
        return report, anomalies

    sidecar_files = sorted(name for name in os.listdir(sidecar_dir) if name.endswith(".pt"))
    for filename in sidecar_files:
        date_str = filename[:-3]
        sidecar = torch.load(os.path.join(sidecar_dir, filename), map_location="cpu", weights_only=False)
        try:
            sc_report = validate_sidecar(sidecar, date_str)
            stats = sc_report.stats
            reasons = []
            if stats.get("thin_bar_pct", 0.0) > SIDECAR_THIN_BAR_ALERT:
                reasons.append("extreme_thin_bars")
            if stats.get("row_labels_nan_rate", 0.0) > SIDECAR_LABEL_NAN_ALERT:
                reasons.append("high_label_nan_rate")
            if stats.get("ordering_violations", 0.0) > SIDECAR_ORDERING_ALERT:
                reasons.append("ordering_violations")
            if stats.get("price_violations", 0.0) > SIDECAR_PRICE_ALERT:
                reasons.append("non_positive_valid_mids")
            if reasons:
                anomalies.append(DateAnomaly(
                    date=date_str,
                    source="sidecar",
                    reasons=reasons,
                    metrics={
                        "thin_bar_pct": float(stats.get("thin_bar_pct", 0.0)),
                        "row_labels_nan_rate": float(stats.get("row_labels_nan_rate", 0.0)),
                        "ordering_violations": float(stats.get("ordering_violations", 0.0)),
                        "price_violations": float(stats.get("price_violations", 0.0)),
                    },
                ))
        except Exception as exc:  # pragma: no cover - anomaly path for broken sidecars
            anomalies.append(DateAnomaly(
                date=date_str,
                source="sidecar",
                reasons=["sidecar_schema_break"],
                metrics={},
            ))
            report.warnings.append(f"{date_str}: validate_sidecar crashed ({exc})")

    report.stats["sidecars_checked"] = float(len(sidecar_files))
    report.stats["flagged_days"] = float(len({a.date for a in anomalies}))
    report.stats["findings"] = float(len(anomalies))
    return report, anomalies


def summarize_trace_overlap(anomalies: list[DateAnomaly], trace_path: str, worst_n: int = 10) -> dict | None:
    if not trace_path or not os.path.exists(trace_path):
        return None
    try:
        trace_df = pd.read_csv(trace_path)
    except Exception:
        return None
    if "date" not in trace_df.columns or "delta_pnl" not in trace_df.columns:
        return None

    delta = pd.to_numeric(trace_df["delta_pnl"], errors="coerce")
    trace_df = trace_df.assign(delta_pnl=delta).dropna(subset=["delta_pnl"])
    if trace_df.empty:
        return None

    worst_days = (
        trace_df.groupby("date", sort=True)["delta_pnl"]
        .mean()
        .sort_values()
        .head(worst_n)
        .index
        .tolist()
    )
    flagged_dates = {a.date for a in anomalies}
    overlap = sorted(flagged_dates.intersection(worst_days))
    return {
        "worst_trace_days": worst_days,
        "overlap_dates": overlap,
        "overlap_count": len(overlap),
    }


def print_date_anomalies(title: str, report: DataQualityReport, anomalies: list[DateAnomaly], limit: int = 20) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    report.print_report()
    if not anomalies:
        print("  No flagged dates.")
        return

    print(f"  Flagged dates: {len({a.date for a in anomalies})}  |  Findings: {len(anomalies)}")
    for anomaly in anomalies[:limit]:
        metric_text = ", ".join(f"{k}={v:.4f}" for k, v in anomaly.metrics.items())
        print(f"  {anomaly.date} [{anomaly.source}] {'/'.join(anomaly.reasons)}")
        if metric_text:
            print(f"    {metric_text}")
    remaining = len(anomalies) - limit
    if remaining > 0:
        print(f"  ... {remaining} more findings not shown")


# ---------------------------------------------------------------------------
# Side-bias audit
# ---------------------------------------------------------------------------

def _safe_ratio(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def _soft_target_side_mass(labels: torch.Tensor, valid: torch.Tensor, is_put: torch.Tensor, soft_temp: float) -> tuple[float, float]:
    logits = labels.clone()
    logits[~valid] = -1e9
    probs = torch.softmax(logits / soft_temp, dim=0)
    call_mass = float(probs[valid & ~is_put].sum().item())
    put_mass = float(probs[valid & is_put].sum().item())
    return call_mass, put_mass


def _oracle_side_from_labels(labels: torch.Tensor, valid_calls: torch.Tensor, valid_puts: torch.Tensor) -> int | None:
    has_calls = bool(valid_calls.any())
    has_puts = bool(valid_puts.any())
    if not has_calls and not has_puts:
        return None
    if has_calls and not has_puts:
        return 0
    if has_puts and not has_calls:
        return 1

    best_call = float(labels[valid_calls].max().item())
    best_put = float(labels[valid_puts].max().item())
    if np.isclose(best_call, best_put, atol=1e-8):
        return None
    return 1 if best_put > best_call else 0


def _flush_side_bias_batch(
    summary: SideBiasAuditSummary,
    batch_items: list[dict],
    model,
    device: str,
) -> None:
    if not batch_items or model is None:
        return

    from v2.core.policy import DEFAULT_POLICY

    batch_x = torch.stack([item["window"] for item in batch_items]).to(device)
    batch_c = torch.stack([item["contracts"] for item in batch_items]).to(device)
    label_valid = torch.stack([item["label_valid"] for item in batch_items])
    is_put = torch.stack([item["is_put"] for item in batch_items])
    labels = torch.stack([item["labels"] for item in batch_items])

    with torch.no_grad():
        outputs = model(batch_x, batch_c)

    scores = outputs["contract_scores"].detach().cpu()
    replay_valid = outputs["valid_mask"].detach().cpu() & label_valid & (batch_c[:, :, 14].detach().cpu() >= QUALITY_PARTIAL)

    if "gate_logit" in outputs and "direction_logit" in outputs:
        gate_trade = outputs["gate_logit"].detach().cpu() > DEFAULT_POLICY.gate_threshold
        pred_put = outputs["direction_logit"].detach().cpu() > 0
        replay_valid = replay_valid & (is_put == pred_put.unsqueeze(1))
        masked_scores = scores.clone()
        masked_scores[~replay_valid] = -1e9
        best_scores, pred_idx = masked_scores.max(dim=-1)
        pred_trade = gate_trade & replay_valid.any(dim=-1) & torch.isfinite(best_scores)
    else:
        no_trade = outputs["no_trade_score"].detach().cpu()
        masked_scores = scores.clone()
        masked_scores[~replay_valid] = -1e9
        best_scores, pred_idx = masked_scores.max(dim=-1)
        pred_trade = best_scores > no_trade

    summary.model_scored_rows += len(batch_items)

    for row_idx, item in enumerate(batch_items):
        if not bool(pred_trade[row_idx].item()):
            continue

        summary.model_traded_rows += 1
        pred_row = int(pred_idx[row_idx].item())
        pred_is_put = bool(is_put[row_idx, pred_row].item())
        if pred_is_put:
            summary.model_put_count += 1
        else:
            summary.model_call_count += 1

        selected_label = labels[row_idx, pred_row]
        if torch.isfinite(selected_label):
            pnl_value = float(selected_label.item())
            summary.model_selected_pnl_sum += pnl_value
            summary.model_selected_pnl_count += 1
        else:
            pnl_value = None

        oracle_side = item["oracle_side"]
        if oracle_side is None:
            continue

        summary.model_oracle_eval_rows += 1
        if oracle_side == 0:
            summary.model_oracle_call_rows += 1
            if not pred_is_put:
                summary.model_direction_matches += 1
                summary.model_direction_matches_oracle_call += 1
            if pnl_value is not None:
                summary.model_selected_pnl_sum_oracle_call += pnl_value
                summary.model_selected_pnl_count_oracle_call += 1
        else:
            summary.model_oracle_put_rows += 1
            if pred_is_put:
                summary.model_direction_matches += 1
                summary.model_direction_matches_oracle_put += 1
            if pnl_value is not None:
                summary.model_selected_pnl_sum_oracle_put += pnl_value
                summary.model_selected_pnl_count_oracle_put += 1


def run_side_bias_audit(
    data_path: str = "v2/data.pt",
    model_path: str | None = None,
    batch_size: int = 512,
) -> list[SideBiasAuditSummary]:
    from v2.train import LOOKBACK, SOFT_TEMP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if model_path:
        from v2.replay import load_model_from_path

        model = load_model_from_path(model_path, device=device).to(device)
        model.eval()

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    X = torch.as_tensor(data["X"], dtype=torch.float32)
    dates = data["dates"]
    bar_of_day = torch.as_tensor(data["bar_of_day"], dtype=torch.long)
    meta = data.get("metadata", {})
    sidecar_dir = meta["chain_sidecar_dir"]
    max_contracts = int(meta["max_contracts_per_bar"])

    sidecar_cache: dict[str, dict] = {}
    summaries: list[SideBiasAuditSummary] = []

    for mask_key in ("train_mask", "val_mask", "promote_mask"):
        mask = data.get(mask_key)
        if mask is None:
            continue
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
        indices = np.arange(LOOKBACK, len(X))[mask_np[LOOKBACK:]]
        summary = SideBiasAuditSummary(mask_name=mask_key.replace("_mask", ""))
        batch_items: list[dict] = []

        for idx in indices:
            summary.rows += 1
            day = dates[int(idx)]
            local_bar = int(bar_of_day[int(idx)].item())
            if day not in sidecar_cache:
                sidecar_cache[day] = torch.load(
                    os.path.join(sidecar_dir, f"{day}.pt"),
                    map_location="cpu",
                    weights_only=False,
                )
            sidecar = sidecar_cache[day]
            contracts_np, labels_np, _ = padded_snapshot(sidecar, local_bar, max_contracts)
            contracts = torch.from_numpy(contracts_np)
            labels = torch.from_numpy(labels_np)

            label_valid = (contracts[:, 0] > 0.5) & torch.isfinite(labels)
            if not bool(label_valid.any()):
                continue

            summary.rows_with_valid_snapshot += 1
            is_put = contracts[:, 2] > 0.5
            valid_calls = label_valid & ~is_put
            valid_puts = label_valid & is_put

            n_valid_calls = int(valid_calls.sum().item())
            n_valid_puts = int(valid_puts.sum().item())
            summary.valid_call_sum += n_valid_calls
            summary.valid_put_sum += n_valid_puts

            if n_valid_calls > 0:
                summary.bars_with_calls += 1
            if n_valid_puts > 0:
                summary.bars_with_puts += 1
            if n_valid_calls > 0 and n_valid_puts > 0:
                summary.bars_with_both_sides += 1
                best_call = float(labels[valid_calls].max().item())
                best_put = float(labels[valid_puts].max().item())
                if np.isclose(best_call, best_put, atol=1e-8):
                    summary.tied_best_side_bars += 1
                elif best_put > best_call:
                    summary.best_put_bars += 1
                else:
                    summary.best_call_bars += 1

            call_mass, put_mass = _soft_target_side_mass(labels, label_valid, is_put, SOFT_TEMP)
            summary.target_call_mass_sum += call_mass
            summary.target_put_mass_sum += put_mass

            if model is not None:
                batch_items.append(
                    {
                        "window": X[int(idx) - LOOKBACK:int(idx)],
                        "contracts": contracts.to(torch.float32),
                        "labels": labels.to(torch.float32),
                        "label_valid": label_valid,
                        "is_put": is_put,
                        "oracle_side": _oracle_side_from_labels(labels, valid_calls, valid_puts),
                    }
                )
                if len(batch_items) >= batch_size:
                    _flush_side_bias_batch(summary, batch_items, model, device)
                    batch_items = []

        if batch_items:
            _flush_side_bias_batch(summary, batch_items, model, device)

        summaries.append(summary)

    return summaries


def print_side_bias_audit(summary: SideBiasAuditSummary) -> None:
    valid_rows = max(summary.rows_with_valid_snapshot, 1)
    oracle_rows = max(summary.bars_with_both_sides, 1)
    traded_rows = max(summary.model_traded_rows, 1)

    print(f"\n{'=' * 60}")
    print(f"  SIDE-BIAS AUDIT — {summary.mask_name.upper()}")
    print(f"{'=' * 60}")
    print(f"  Rows audited: {summary.rows}  |  Valid snapshots: {summary.rows_with_valid_snapshot}")
    print(
        f"  Best side (both sides present): "
        f"call={_safe_ratio(summary.best_call_bars, oracle_rows):.1%}  "
        f"put={_safe_ratio(summary.best_put_bars, oracle_rows):.1%}  "
        f"ties={_safe_ratio(summary.tied_best_side_bars, oracle_rows):.1%}"
    )
    print(
        f"  Soft target mass: "
        f"call={summary.target_call_mass_sum / valid_rows:.3f}  "
        f"put={summary.target_put_mass_sum / valid_rows:.3f}"
    )
    print(
        f"  Valid contracts/bar: "
        f"call={summary.valid_call_sum / valid_rows:.2f}  "
        f"put={summary.valid_put_sum / valid_rows:.2f}"
    )
    if summary.model_scored_rows == 0:
        print("  Model metrics: skipped (no model supplied)")
        return

    print(
        f"  Model trade rate: {_safe_ratio(summary.model_traded_rows, summary.model_scored_rows):.1%}  "
        f"Direction={summary.model_call_count}C / {summary.model_put_count}P"
    )
    print(
        f"  Model traded side share: "
        f"call={_safe_ratio(summary.model_call_count, traded_rows):.1%}  "
        f"put={_safe_ratio(summary.model_put_count, traded_rows):.1%}"
    )
    print(
        f"  Direction accuracy: "
        f"overall={_safe_ratio(summary.model_direction_matches, summary.model_oracle_eval_rows):.1%}  "
        f"oracle_call={_safe_ratio(summary.model_direction_matches_oracle_call, summary.model_oracle_call_rows):.1%}  "
        f"oracle_put={_safe_ratio(summary.model_direction_matches_oracle_put, summary.model_oracle_put_rows):.1%}"
    )
    print(
        f"  Selected label pnl: "
        f"overall={_safe_ratio(summary.model_selected_pnl_sum, summary.model_selected_pnl_count):.4f}  "
        f"oracle_call={_safe_ratio(summary.model_selected_pnl_sum_oracle_call, summary.model_selected_pnl_count_oracle_call):.4f}  "
        f"oracle_put={_safe_ratio(summary.model_selected_pnl_sum_oracle_put, summary.model_selected_pnl_count_oracle_put):.4f}"
    )


# ---------------------------------------------------------------------------
# Full pipeline validation
# ---------------------------------------------------------------------------

def run_full_validation(
    data_path: str = "v2/data.pt",
    sidecar_sample: int = 10,
    seed: int = 42,
) -> list[DataQualityReport]:
    """Run all validation checks. Returns list of reports."""
    reports = []

    # 1. Manifest
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    manifest_report = validate_manifest(data)
    reports.append(manifest_report)

    # 2. Features
    X = data["X"]
    if hasattr(X, "numpy"):
        X = X.numpy()
    feat_report = validate_features(X)
    reports.append(feat_report)

    # 3. Sample sidecars
    sidecar_dir = data.get("metadata", {}).get("chain_sidecar_dir", "v2/data_sidecars")
    dates = sorted(set(data["dates"]))
    rng = random.Random(seed)
    sample_dates = rng.sample(dates, min(sidecar_sample, len(dates)))

    for day in sample_dates:
        path = os.path.join(sidecar_dir, f"{day}.pt")
        if not os.path.exists(path):
            r = DataQualityReport(source=f"sidecar ({day})")
            r.errors.append(f"sidecar file not found: {path}")
            reports.append(r)
            continue
        sidecar = torch.load(path, map_location="cpu", weights_only=False)
        sc_report = validate_sidecar(sidecar, day)
        reports.append(sc_report)

    return reports


def print_validation_summary(reports: list[DataQualityReport]) -> None:
    """Print a summary of all validation reports."""
    print(f"\n{'=' * 60}")
    print(f"  DATA INTEGRITY REPORT")
    print(f"{'=' * 60}")

    n_pass = sum(1 for r in reports if r.passed)
    n_fail = sum(1 for r in reports if not r.passed)
    n_warn = sum(len(r.warnings) for r in reports)

    print(f"  Checks: {len(reports)} total, {n_pass} passed, {n_fail} failed, {n_warn} warnings")
    print()

    for r in reports:
        r.print_report()

    overall = "PASS" if n_fail == 0 else "FAIL"
    print(f"\n  Overall: {overall}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data integrity validation")
    parser.add_argument("--data", type=str, default="v2/data.pt")
    parser.add_argument("--sample", type=int, default=10, help="Number of sidecars to sample")
    parser.add_argument("--raw-audit", action="store_true", help="Run the separate raw minute-bar audit track")
    parser.add_argument("--sidecar-audit", action="store_true", help="Run the full sidecar anomaly audit across all dates")
    parser.add_argument("--side-bias-audit", action="store_true", help="Audit label structure, side availability, and model side collapse")
    parser.add_argument("--model", type=str, default="", help="Optional model checkpoint path for side-bias audit")
    parser.add_argument("--trace-path", type=str, default="", help="Optional replay trace CSV for overlap reporting")
    args = parser.parse_args()

    reports = run_full_validation(data_path=args.data, sidecar_sample=args.sample)
    print_validation_summary(reports)

    trace_path = args.trace_path
    if trace_path and not os.path.isabs(trace_path):
        trace_path = os.path.join(os.getcwd(), trace_path)

    audit_errors = []
    if args.raw_audit:
        raw_reports, raw_anomalies = audit_raw_inputs()
        for report in raw_reports:
            if not report.passed:
                audit_errors.extend(report.errors)
        for report, source in zip(raw_reports, DEFAULT_RAW_INPUTS.keys()):
            source_anomalies = [a for a in raw_anomalies if a.source == source]
            print_date_anomalies(f"RAW AUDIT — {source.upper()}", report, source_anomalies)
        overlap = summarize_trace_overlap(raw_anomalies, trace_path) if trace_path else None
        if overlap is not None:
            print(f"  Trace overlap: {overlap['overlap_count']} of worst days overlap flagged raw dates")
            print(f"  Overlap dates: {overlap['overlap_dates']}")

    if args.sidecar_audit:
        data = torch.load(args.data, map_location="cpu", weights_only=False)
        sidecar_dir = data.get("metadata", {}).get("chain_sidecar_dir", "v2/data_sidecars")
        sidecar_report, sidecar_anomalies = audit_sidecar_dates(sidecar_dir)
        if not sidecar_report.passed:
            audit_errors.extend(sidecar_report.errors)
        print_date_anomalies("SIDECAR DATE AUDIT", sidecar_report, sidecar_anomalies)
        overlap = summarize_trace_overlap(sidecar_anomalies, trace_path) if trace_path else None
        if overlap is not None:
            print(f"  Trace overlap: {overlap['overlap_count']} of worst days overlap flagged sidecar dates")
            print(f"  Overlap dates: {overlap['overlap_dates']}")

    if args.side_bias_audit:
        model_path = args.model
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), model_path)
        summaries = run_side_bias_audit(data_path=args.data, model_path=model_path or None)
        for summary in summaries:
            print_side_bias_audit(summary)

    if any(not r.passed for r in reports) or audit_errors:
        exit(1)
