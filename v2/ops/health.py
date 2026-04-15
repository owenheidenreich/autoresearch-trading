"""Health check: single command to verify pipeline integrity.

Usage:
    python -m v2.ops.health              # full check
    python -m v2.ops.health quick        # config + data + model only (< 5s)
    python -m v2.ops.health config       # RuntimeConfig consistency
    python -m v2.ops.health data         # dataset integrity
    python -m v2.ops.health model        # model checkpoint compatibility
    python -m v2.ops.health smoke        # 1-day replay smoke test

Created 2026-04-15 as part of the stage-decoupled architecture fix.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def check_config() -> list[str]:
    """Verify RuntimeConfig matches dataset metadata."""
    errors = []
    try:
        from v2.core.config import RUNTIME_CONFIG
        data_path = "v2/data.pt"
        if not os.path.exists(data_path):
            errors.append(f"data.pt not found at {data_path}")
            return errors

        data = torch.load(data_path, map_location="cpu", weights_only=False)
        meta = data.get("metadata", {})
        config_errors = RUNTIME_CONFIG.validate_dataset_metadata(meta)
        errors.extend(config_errors)

        # Check feature count matches tensor shape
        X = data.get("X")
        if X is not None and X.shape[1] != RUNTIME_CONFIG.num_features:
            errors.append(
                f"X tensor has {X.shape[1]} features, config expects {RUNTIME_CONFIG.num_features}"
            )
    except Exception as e:
        errors.append(f"config check failed: {e}")
    return errors


def check_data() -> list[str]:
    """Verify dataset integrity: shapes, sidecar sampling, schema version."""
    errors = []
    try:
        data_path = "v2/data.pt"
        if not os.path.exists(data_path):
            errors.append(f"data.pt not found at {data_path}")
            return errors

        data = torch.load(data_path, map_location="cpu", weights_only=False)
        meta = data.get("metadata", {})

        # Check required keys
        for key in ["X", "X_sim", "dates", "bar_of_day", "spot_prices", "metadata"]:
            if key not in data:
                errors.append(f"missing key '{key}' in data.pt")

        X = data.get("X")
        if X is not None:
            n_bars, n_feat = X.shape
            if n_feat != 52:
                errors.append(f"X has {n_feat} features, expected 52")
            if n_bars == 0:
                errors.append("X has 0 bars")

        # Check sidecar directory exists
        sidecar_dir = meta.get("chain_sidecar_dir", "v2/data_sidecars")
        if not os.path.isdir(sidecar_dir):
            errors.append(f"sidecar directory not found: {sidecar_dir}")
        else:
            # Spot-check first sidecar
            sidecars = sorted(f for f in os.listdir(sidecar_dir) if f.endswith(".pt"))
            if len(sidecars) == 0:
                errors.append("no .pt files in sidecar directory")
            else:
                sample = torch.load(
                    os.path.join(sidecar_dir, sidecars[0]),
                    map_location="cpu", weights_only=False,
                )
                sv = sample.get("schema_version", "")
                expected = meta.get("chain_schema_version", "v4_exact_chain_v2_paths")
                if sv != expected:
                    errors.append(f"sidecar schema '{sv}' != expected '{expected}'")

        # Check schema version
        if meta.get("chain_schema_version") != "v4_exact_chain_v2_paths":
            errors.append(f"unexpected chain_schema_version: {meta.get('chain_schema_version')}")

    except Exception as e:
        errors.append(f"data check failed: {e}")
    return errors


def check_model(model_path: str = "v2/models/model.pt") -> list[str]:
    """Verify model checkpoint is compatible with current config and data."""
    errors = []
    try:
        if not os.path.exists(model_path):
            errors.append(f"model not found at {model_path}")
            return errors

        from v2.core.config import RUNTIME_CONFIG
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        # Load dataset fingerprint for comparison
        data_path = "v2/data.pt"
        dataset_fp = None
        if os.path.exists(data_path):
            data = torch.load(data_path, map_location="cpu", weights_only=False)
            dataset_fp = data.get("metadata", {}).get("fingerprint")

        ckpt_errors = RUNTIME_CONFIG.validate_checkpoint(ckpt, dataset_fp)
        errors.extend(ckpt_errors)

        # Check score config fingerprint staleness
        from v2.core.metrics import score_config_fingerprint
        ckpt_score_fp = ckpt.get("score_config_fingerprint", "")
        current_score_fp = score_config_fingerprint()
        if ckpt_score_fp and ckpt_score_fp != current_score_fp:
            errors.append(
                f"model scored with evaluator {ckpt_score_fp[:8]}, "
                f"current evaluator is {current_score_fp[:8]} — scores not comparable"
            )

    except Exception as e:
        errors.append(f"model check failed: {e}")
    return errors


def check_smoke() -> list[str]:
    """Run a 1-day replay and verify basic sanity."""
    errors = []
    try:
        from v2.train import TradingModel, LOOKBACK
        from v2.core.policy import DEFAULT_POLICY
        from v2.replay import replay_validation

        data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
        model_path = "v2/models/model.pt"
        if not os.path.exists(model_path):
            errors.append("smoke test skipped: no model.pt")
            return errors

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model = TradingModel()
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        metrics, trades, _ = replay_validation(
            model, data, mask_key="promote_mask", max_days=1, policy=DEFAULT_POLICY,
        )

        # Sanity checks
        if metrics.total_trades == 0:
            errors.append("smoke test: 0 trades on 1-day replay (may be ok if model abstains)")
        if metrics.profit_factor < 0:
            errors.append(f"smoke test: negative profit_factor ({metrics.profit_factor})")
        if metrics.total_trades > 0 and metrics.max_account_drawdown == 0.0:
            errors.append("smoke test: DD is 0 with trades present (qty bug?)")
        for t in trades:
            if t.intent.qty < 1:
                errors.append(f"smoke test: trade has qty={t.intent.qty} (should be >= 1)")
                break

    except Exception as e:
        errors.append(f"smoke test failed: {e}")
    return errors


def run_health(mode: str = "full", model_path: str = "v2/models/model.pt") -> dict:
    """Run health checks and return structured results."""
    results = {}

    checks = {
        "config": check_config,
        "data": check_data,
        "model": lambda: check_model(model_path),
    }
    if mode == "full":
        checks["smoke"] = check_smoke

    for name, check_fn in checks.items():
        if mode == "quick" and name == "smoke":
            continue
        t0 = time.time()
        errors = check_fn()
        elapsed = time.time() - t0
        status = "PASS" if not errors else "FAIL"
        results[name] = {"status": status, "errors": errors, "seconds": round(elapsed, 2)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Pipeline health check")
    parser.add_argument("mode", nargs="?", default="full",
                        choices=["full", "quick", "config", "data", "model", "smoke"])
    parser.add_argument("--model", default="v2/models/model.pt")
    args = parser.parse_args()

    mode = args.mode
    if mode in ("config", "data", "model", "smoke"):
        # Run single check
        if mode == "config":
            errors = check_config()
        elif mode == "data":
            errors = check_data()
        elif mode == "model":
            errors = check_model(args.model)
        elif mode == "smoke":
            errors = check_smoke()
        else:
            errors = []

        if errors:
            print(f"{mode}:  FAIL")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        else:
            print(f"{mode}:  PASS")
            sys.exit(0)

    # Full or quick mode
    results = run_health(mode=mode, model_path=args.model)
    any_fail = False
    for name, result in results.items():
        status = result["status"]
        elapsed = result["seconds"]
        print(f"{name:8s} {status:4s}  ({elapsed:.1f}s)")
        if result["errors"]:
            any_fail = True
            for e in result["errors"]:
                print(f"           - {e}")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
