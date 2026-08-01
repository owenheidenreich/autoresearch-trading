"""Build and optionally install an immutable Protocol101 recorder runtime."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path.home() / ".autoresearch-trading/runtime-bundles/protocol101-parity-v1"
LAUNCH_AGENT_ROOT = Path.home() / "Library/LaunchAgents"
LOG_ROOT = Path.home() / "Library/Logs/autoresearch-trading"
LABEL_PREFIX = "com.autoresearch.protocol101.parityrecorder"
DEFAULT_PACKET_DATES = ("2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10")
OLD_LABEL_PREFIX = "com.autoresearch.protocol101.week20260623"
SEED_MODULES = (
    "v4.ops.ibkr.run_protocol101_ibkr_recorder",
    "v4.ops.ibkr.protocol101_recorder_control",
    "v4.ops.ibkr.protocol101_gateway_ready",
    "v4.scripts.run_protocol101_capture_replay",
    "v4.scripts.run_protocol101_captured_trace_readiness",
    "v4.scripts.run_protocol101_extract_captured_decision_traces",
    "v4.scripts.run_protocol101_paired_live_historical_diff",
    "v4.scripts.run_protocol101_fiveday_evidence_packet",
)
SHELL_FILES = (
    "v4/ops/ibkr/retry_command.sh",
    "v4/ops/ibkr/start_protocol101_recorder_gateway.sh",
    "v4/ops/ibkr/run_protocol101_recorder_packet.sh",
)
SURFACE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/model_artifacts/train_through_q4_2025_test_q1_2026/seed_11")
ENTRY_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/model_artifacts/fold3_train_q1_q2_q3_validate_q4_test_q1_2026/seed_1")
ENTRY_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/summary.json")
LIFECYCLE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts/model_artifacts/train_q1_2025_q2_2025_q3_2025_q4_2025_test_q1_2026/seed_1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-launchd", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument(
        "--packet-dates",
        nargs="+",
        default=list(DEFAULT_PACKET_DATES),
        help="Exact YYYY-MM-DD sessions to schedule. Later dates are gated on the development session.",
    )
    parser.add_argument(
        "--development-session",
        default=DEFAULT_PACKET_DATES[0],
        help="First ungated development session whose collection gate unlocks later packet dates.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=("development_session", "none"),
        default="development_session",
        help="Use 'none' for independent daily collection attempts without requiring a prior collection gate.",
    )
    return parser.parse_args()


def module_path(module: str) -> Path | None:
    path = REPO_ROOT / Path(*module.split("."))
    file_path = path.with_suffix(".py")
    if file_path.exists():
        return file_path
    init_path = path / "__init__.py"
    return init_path if init_path.exists() else None


def local_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return set()
    found: set[str] = set()
    relative = path.relative_to(REPO_ROOT)
    module_parts = list(relative.with_suffix("").parts)
    if module_parts[-1] == "__init__":
        package_parts = module_parts[:-1]
    else:
        package_parts = module_parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - node.level + 1
                base_parts = package_parts[:max(keep, 0)]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
                names = [base, *(f"{base}.{alias.name}" for alias in node.names if alias.name != "*" and base)]
            elif node.module:
                names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*")]
            else:
                names = []
        else:
            continue
        for name in names:
            if name == "v4" or name.startswith("v4.") or name == "v2" or name.startswith("v2."):
                found.add(name)
    return found


def required_python_files() -> list[Path]:
    queue = list(SEED_MODULES)
    seen_modules: set[str] = set()
    files: set[Path] = set()
    while queue:
        module = queue.pop()
        if module in seen_modules:
            continue
        seen_modules.add(module)
        path = module_path(module)
        if path is None:
            continue
        files.add(path)
        parts = module.split(".")
        for index in range(1, len(parts)):
            package_init = REPO_ROOT / Path(*parts[:index]) / "__init__.py"
            if package_init.exists():
                files.add(package_init)
                package_module = ".".join(parts[:index])
                if package_module not in seen_modules:
                    queue.append(package_module)
        queue.extend(sorted(local_imports(path) - seen_modules))
    return sorted(files)


def input_hash(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def rewrite_manifest(path: Path, replacements: dict[str, Path]) -> None:
    payload = json.loads(path.read_text())
    files = payload.get("files") if isinstance(payload.get("files"), dict) else {}
    for key, replacement in replacements.items():
        if key in files:
            files[key] = str(replacement)
    payload["files"] = files
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def deployment_files(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "deployment_manifest.json"):
        rows.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return rows


def build_bundle(runtime_root: Path) -> Path:
    python_files = required_python_files()
    input_files = python_files + [REPO_ROOT / path for path in SHELL_FILES]
    for directory in (SURFACE_DIR, ENTRY_DIR, LIFECYCLE_DIR):
        input_files.extend(sorted(item for item in (REPO_ROOT / directory).iterdir() if item.is_file()))
    input_files.append(REPO_ROOT / ENTRY_SUMMARY)
    bundle_hash = input_hash(input_files)
    target = runtime_root.expanduser() / bundle_hash
    runtime_root.expanduser().parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    staging = Path(tempfile.mkdtemp(prefix=f"protocol101-parity-{bundle_hash}-", dir=str(runtime_root.expanduser().parent)))
    try:
        for source in python_files:
            destination = staging / source.relative_to(REPO_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for relative in SHELL_FILES:
            source = REPO_ROOT / relative
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            destination.chmod(0o700)
        artifacts = staging / "artifacts"
        shutil.copytree(REPO_ROOT / SURFACE_DIR, artifacts / "surface")
        shutil.copytree(REPO_ROOT / ENTRY_DIR, artifacts / "protocol101")
        shutil.copy2(REPO_ROOT / ENTRY_SUMMARY, artifacts / "protocol101_summary.json")
        shutil.copytree(REPO_ROOT / LIFECYCLE_DIR, artifacts / "lifecycle")
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, target, dirs_exist_ok=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    surface_manifest = target / "artifacts/surface/manifest.json"
    lifecycle_manifest = target / "artifacts/lifecycle/manifest.json"
    rewrite_manifest(surface_manifest, {
        "entry_model": target / "artifacts/surface/entry_model.pt",
        "entry_standardizer": target / "artifacts/surface/entry_standardizer.json",
        "protocol054_risk_model": target / "artifacts/surface/protocol054_risk_model.pt",
        "protocol054_risk_scaler": target / "artifacts/surface/protocol054_risk_scaler.json",
        "manifest": surface_manifest,
    })
    rewrite_manifest(lifecycle_manifest, {
        "model": target / "artifacts/lifecycle/model.pt",
        "scaler": target / "artifacts/lifecycle/scaler.json",
        "threshold_sweep": target / "artifacts/lifecycle/threshold_sweep.json",
        "manifest": lifecycle_manifest,
    })
    runtime_dir = target / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    registry = {
        "feature_name": "Protocol101 recorder-first parity runtime",
        "current_model_id": "protocol101",
        "selection_policy": "Frozen Protocol101 baseline; recorder-only during market hours.",
        "models": {
            "protocol101": {
                "paper_trading_status": "frozen_baseline_recorder_only",
                "arguments": {
                    "surface_manifest": str(surface_manifest),
                    "protocol101_manifest": str(target / "artifacts/protocol101/manifest.json"),
                    "protocol101_summary": str(target / "artifacts/protocol101_summary.json"),
                    "lifecycle_manifest": str(lifecycle_manifest),
                },
            }
        },
    }
    (runtime_dir / "PAPER_TRADING_DEFAULT.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    (runtime_dir / "paper_only.json").write_text(json.dumps({
        "mode": "recorder-only",
        "paper_account_only": True,
        "paper_orders_enabled": False,
        "real_money_trading": False,
        "broker_order_endpoint_called": False,
        "ibkr_client_id": 159,
    }, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "Protocol101ParityDeploymentManifestV1",
        "bundle_hash": bundle_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(target),
        "market_hours_repo_reads_allowed": False,
        "paper_orders_enabled": False,
        "real_money_trading": False,
        "files": deployment_files(target),
    }
    (target / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    assert_no_repo_references(target)
    current = runtime_root.expanduser() / "current"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.unlink(missing_ok=True)
    current.symlink_to(target, target_is_directory=True)
    return target


def assert_no_repo_references(bundle: Path) -> None:
    forbidden = str(REPO_ROOT)
    failures = []
    for path in bundle.rglob("*"):
        if not path.is_file() or path.suffix in {".pt", ".parquet"}:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if forbidden in text:
            failures.append(str(path))
    if failures:
        raise RuntimeError(f"deployed files reference research repository: {failures}")


def calendar(date: str, hour: int, minute: int) -> dict[str, int]:
    parsed = datetime.strptime(date, "%Y-%m-%d")
    return {"Month": parsed.month, "Day": parsed.day, "Hour": hour, "Minute": minute}


def plist_payload(
    bundle: Path,
    suffix: str,
    action: str,
    schedules: list[dict[str, int]],
    *,
    packet_dates: tuple[str, ...],
    development_session: str,
    gate_mode: str,
    extra_environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    label = f"{LABEL_PREFIX}.{suffix}"
    environment = {
        "BUNDLE_ROOT": str(bundle),
        "CAPTURE_ROOT": str(Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture"),
        "MODEL_PYTHON": str(Path.home() / ".autoresearch-trading/runtime-venv/bin/python"),
        "RECORDER_PYTHON": "/usr/bin/python3",
        "PROTOCOL101_PACKET_ALLOWED_SESSIONS": ",".join(packet_dates),
        "PROTOCOL101_PACKET_DEVELOPMENT_SESSION": development_session,
        "PROTOCOL101_PACKET_GATE_MODE": gate_mode,
        "PROTOCOL101_RECORDER_LABEL_PREFIX": LABEL_PREFIX,
        "PYTHONPATH": str(bundle),
        "PYTHONUNBUFFERED": "1",
    }
    environment.update(extra_environment or {})
    return {
        "Label": label,
        "ProgramArguments": ["/bin/bash", str(bundle / "v4/ops/ibkr/run_protocol101_recorder_packet.sh"), action],
        "WorkingDirectory": str(bundle),
        "EnvironmentVariables": environment,
        "StartCalendarInterval": schedules,
        "RunAtLoad": False,
        "ProcessType": "Interactive",
        "ThrottleInterval": 10,
        "StandardOutPath": str(LOG_ROOT / f"protocol101-parityrecorder-{suffix}.out.log"),
        "StandardErrorPath": str(LOG_ROOT / f"protocol101-parityrecorder-{suffix}.err.log"),
    }


def launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], text=True, capture_output=True, check=False)


def install_launchd(bundle: Path, *, packet_dates: tuple[str, ...], development_session: str, gate_mode: str) -> None:
    LAUNCH_AGENT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    uid = os.getuid()
    for suffix in ("ibgateway.paper", "paper-preflight", "paper-session", "daily-monitor", "recovery-watchdog", "paper-shutdown", "evidence-postsession"):
        label = f"{OLD_LABEL_PREFIX}.{suffix}"
        launchctl("bootout", f"gui/{uid}/{label}")
        (LAUNCH_AGENT_ROOT / f"{label}.plist").unlink(missing_ok=True)
    for label in (
        "com.autoresearch.ibgateway.paper",
        "com.autoresearch.protocol101.paper-preflight",
        "com.autoresearch.protocol101.paper-session",
        "com.autoresearch.protocol101.daily-monitor",
        "com.autoresearch.ibgateway.paper-shutdown",
        "com.autoresearch.protocol101.fiveday.ibgateway.paper",
        "com.autoresearch.protocol101.fiveday.paper-preflight",
        "com.autoresearch.protocol101.fiveday.paper-session",
        "com.autoresearch.protocol101.fiveday.daily-monitor",
        "com.autoresearch.protocol101.fiveday.paper-shutdown",
    ):
        launchctl("bootout", f"gui/{uid}/{label}")

    definitions = {
        "gateway": ("gateway", 4, 45, {"IB_GATEWAY_WAIT_SECONDS": "2700"}),
        "preflight": ("preflight", 5, 25, {
            "RETRY_ATTEMPTS": "360",
            "RETRY_INITIAL_SLEEP_SECONDS": "2",
            "RETRY_MAX_SLEEP_SECONDS": "10",
        }),
        "recorder": ("recorder", 5, 40, {
            "RETRY_ATTEMPTS": "240",
            "RETRY_INITIAL_SLEEP_SECONDS": "2",
            "RETRY_MAX_SLEEP_SECONDS": "20",
        }),
        "shutdown": ("shutdown", 13, 5, {}),
        "finalize": ("finalize", 13, 10, {}),
        "audit": ("audit", 13, 15, {}),
        "watchdog": ("watchdog", 6, 5, {
            "WATCHDOG_RESTART_COOLDOWN_SECONDS": "120",
            "WATCHDOG_MAX_RESTART_COOLDOWN_SECONDS": "300",
        }),
    }
    payloads: dict[str, dict[str, Any]] = {}
    for suffix, (action, hour, minute, extra_environment) in definitions.items():
        payloads[suffix] = plist_payload(
            bundle,
            suffix,
            action,
            [calendar(date, hour, minute) for date in packet_dates],
            packet_dates=packet_dates,
            development_session=development_session,
            gate_mode=gate_mode,
            extra_environment=extra_environment,
        )
    health_schedules = [
        calendar(date, hour, minute)
        for date in packet_dates
        for hour, minute in ((5, 50), (6, 0), (6, 10), (6, 20), (6, 28))
    ]
    payloads["health"] = plist_payload(
        bundle,
        "health",
        "health",
        health_schedules,
        packet_dates=packet_dates,
        development_session=development_session,
        gate_mode=gate_mode,
    )
    for suffix, payload in payloads.items():
        label = payload["Label"]
        path = LAUNCH_AGENT_ROOT / f"{label}.plist"
        launchctl("bootout", f"gui/{uid}/{label}")
        with path.open("wb") as handle:
            plistlib.dump(payload, handle, sort_keys=False)
        result = launchctl("bootstrap", f"gui/{uid}", str(path))
        if result.returncode != 0:
            raise RuntimeError(f"failed to bootstrap {label}: {result.stderr.strip()}")
        launchctl("enable", f"gui/{uid}/{label}")


def main() -> int:
    args = parse_args()
    packet_dates = tuple(str(date) for date in args.packet_dates)
    if not packet_dates:
        raise SystemExit("at least one --packet-dates value is required")
    if str(args.development_session) not in packet_dates:
        raise SystemExit("--development-session must be included in --packet-dates")
    bundle = build_bundle(args.runtime_root)
    if args.install_launchd:
        install_launchd(
            bundle,
            packet_dates=packet_dates,
            development_session=str(args.development_session),
            gate_mode=str(args.gate_mode),
        )
    print(json.dumps({
        "status": "pass",
        "bundle": str(bundle),
        "launchd_installed": bool(args.install_launchd),
        "labels_prefix": LABEL_PREFIX,
        "packet_dates": list(packet_dates),
        "development_session": str(args.development_session),
        "gate_mode": str(args.gate_mode),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
