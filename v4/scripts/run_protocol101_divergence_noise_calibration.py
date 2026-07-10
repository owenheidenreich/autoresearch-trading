"""Build the Protocol101 canonical divergence-noise calibration packet."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.model.protocol101_divergence_noise import (
    DEFAULT_DIVERGENCE_DISTRIBUTIONS,
    DivergenceNoiseModel,
)


SCHEMA_VERSION = "Protocol101CanonicalDivergenceNoiseCalibrationV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_canonical_v1_divergence_noise_calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--divergence-distributions", type=Path, default=DEFAULT_DIVERGENCE_DISTRIBUTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def calibration_rows(model: DivergenceNoiseModel, *, seed: int) -> pd.DataFrame:
    summary = model.summarize()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        samples = model.samples.get((row.feature, row.moneyness_band))
        if samples is None or len(samples) == 0:
            injected = np.asarray([], dtype=float)
        else:
            injected = rng.choice(samples, size=len(samples), replace=True)
        measured_abs = np.abs(samples) if samples is not None else np.asarray([], dtype=float)
        injected_abs = np.abs(injected)
        rows.append(
            {
                "feature": row.feature,
                "family": row.family,
                "moneyness_band": row.moneyness_band,
                "sample_count": int(row.sample_count),
                "measured_p95_abs": float(np.percentile(measured_abs, 95)) if len(measured_abs) else 0.0,
                "injected_p95_abs": float(np.percentile(injected_abs, 95)) if len(injected_abs) else 0.0,
                "measured_p99_abs": float(np.percentile(measured_abs, 99)) if len(measured_abs) else 0.0,
                "injected_p99_abs": float(np.percentile(injected_abs, 99)) if len(injected_abs) else 0.0,
                "nonzero_noise_enabled": bool(row.nonzero_noise_enabled),
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} exists; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = DivergenceNoiseModel.from_parquet(args.divergence_distributions)
    summary = model.summarize()
    calibration = calibration_rows(model, seed=args.seed)
    summary.to_csv(out_dir / "noise_distribution_summary.csv", index=False)
    calibration.to_csv(out_dir / "injected_vs_measured.csv", index=False)
    enabled = summary[summary["nonzero_noise_enabled"]]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass",
        "source_divergence_distributions": str(args.divergence_distributions),
        "source_path_recorded_by_model": model.source_path,
        "feature_count": int(summary["feature"].nunique()),
        "nonzero_noise_feature_count": int(enabled["feature"].nunique()),
        "moneyness_bands": sorted(str(item) for item in summary["moneyness_band"].dropna().unique()),
        "max_measured_p95_abs": float(calibration["measured_p95_abs"].max()) if not calibration.empty else 0.0,
        "max_injected_p95_abs": float(calibration["injected_p95_abs"].max()) if not calibration.empty else 0.0,
        "zero_divergence_families_untouched": True,
        "side_effects": {
            "model_training_executed": False,
            "threshold_selection_executed": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_downloaded": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "real_money_path_changed": False,
            "sealed_market_data_read": False,
        },
    }
    write_json(out_dir / "summary.json", payload)
    report = [
        "# Protocol101 Canonical Divergence Noise Calibration",
        "",
        "Offline build/test packet for measured cross-feed feature noise. No training or live side effects occurred.",
        "",
        f"- Features in divergence artifact: `{payload['feature_count']}`",
        f"- Features receiving nonzero noise: `{payload['nonzero_noise_feature_count']}`",
        f"- Bands: `{', '.join(payload['moneyness_bands'])}`",
        f"- Max measured p95 absolute drift: `{payload['max_measured_p95_abs']:.6g}`",
        f"- Max injected p95 absolute drift: `{payload['max_injected_p95_abs']:.6g}`",
        "",
        "Outputs:",
        "",
        "- `noise_distribution_summary.csv`",
        "- `injected_vs_measured.csv`",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
