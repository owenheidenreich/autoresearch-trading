"""Local pre-run gate for the v3 pure-RL surface."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from v3.core.market_state import DEFAULT_MARKET_STATE_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "v3"
REQUIRED_PATHS = (
    V3_ROOT / "program.md",
    V3_ROOT / "docs" / "README.md",
    V3_ROOT / "train.py",
    V3_ROOT / "replay.py",
    V3_ROOT / "build_market_state.py",
    V3_ROOT / "core" / "env.py",
    V3_ROOT / "ops" / "run_experiment.py",
    V3_ROOT / "ops" / "requirements-gpu.txt",
)


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def check_required_paths(errors: list[str]) -> None:
    for path in REQUIRED_PATHS:
        if not path.exists():
            errors.append(f"missing required v3 path: {path.relative_to(PROJECT_ROOT)}")


def check_data_contract(data_path: str, errors: list[str]) -> None:
    if not os.path.exists(data_path):
        errors.append(f"data file not found: {data_path}")
        return
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    meta = data.get("metadata", {})
    for key in ("X", "spot_prices", "bar_of_day", "dates", "train_mask", "val_mask", "promote_mask"):
        if key not in data:
            errors.append(f"dataset missing required key: {key}")
    if meta.get("chain_schema_version") != "v4_exact_chain_v1":
        errors.append(f"unexpected chain schema version: {meta.get('chain_schema_version')}")
    sidecar_dir = meta.get("chain_sidecar_dir")
    if not sidecar_dir:
        errors.append("dataset metadata missing chain_sidecar_dir")
    elif not (PROJECT_ROOT / sidecar_dir).exists():
        errors.append(f"chain sidecar dir missing: {sidecar_dir}")


def check_py_compile(errors: list[str]) -> None:
    code, output = run(["python3", "-m", "py_compile", *[str(p.relative_to(PROJECT_ROOT)) for p in V3_ROOT.rglob("*.py")]])
    if code != 0:
        errors.append(f"py_compile failed:\n{output}")


def check_market_state(data_path: str, cache_path: str, errors: list[str]) -> None:
    if not os.path.exists(cache_path):
        errors.append(
            f"market-state cache missing: {cache_path}\n"
            f"build it with: python3 -m v3.build_market_state --data v2/data.pt --output {cache_path}"
        )
        return
    code, output = run(
        [
            "python3",
            "-c",
            (
                "from v3.core.market_state import load_market_state_cache;"
                f"load_market_state_cache(data_path='{data_path}', cache_path='{cache_path}')"
            ),
        ]
    )
    if code != 0:
        errors.append(f"market-state cache failed to load:\n{output}")


def check_replay_smoke(data_path: str, errors: list[str]) -> None:
    code, output = run(
        [
            "python3",
            "-m",
            "v3.train",
            "--data",
            data_path,
            "--market-state",
            DEFAULT_MARKET_STATE_PATH,
            "--updates",
            "1",
            "--rollout-days",
            "1",
            "--eval-interval",
            "1",
            "--max-eval-days",
            "1",
            "--device",
            "cpu",
            "--checkpoint",
            "/tmp/v3_gate_smoke.pt",
            "--experiment-id",
            "v3_gate_smoke",
        ]
    )
    run(["rm", "-f", "/tmp/v3_gate_smoke.pt"])
    if code != 0:
        errors.append(f"v3 train smoke failed:\n{output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 local pre-run gate")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--market-state", default=DEFAULT_MARKET_STATE_PATH)
    parser.add_argument("--skip-train-smoke", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    check_required_paths(errors)
    check_data_contract(args.data, errors)
    check_market_state(args.data, args.market_state, errors)
    check_py_compile(errors)
    code, output = run(["python3", "-m", "pytest", "-q", "tests/test_v3_env.py", "tests/test_v3_policy.py"])
    if code != 0:
        errors.append(f"v3 tests failed:\n{output}")
    elif not args.skip_train_smoke:
        check_replay_smoke(args.data, errors)

    if errors:
        print("v3 pre-run gate FAILED\n")
        for item in errors:
            print(f"- {item}")
        sys.exit(1)

    print("v3 pre-run gate PASSED")


if __name__ == "__main__":
    main()
