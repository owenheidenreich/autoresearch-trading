#!/usr/bin/env python3
"""Run the exact sterile Job-52 tests and seal paid precredential readiness."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
WORK = REPO / "v5/work/cmbp-tier0-paid-resume"
REPORT = WORK / "TEST_RESULTS_V1.xml"
READINESS = WORK / "LOCAL_READINESS_RECEIPT_V1.json"
TESTS = (
    "v5/tests/test_cmbp_stream.py",
    "v5/tests/test_cmbp_tier0.py",
    "v5/tests/test_cmbp_tier0_paid.py",
)


def main() -> int:
    if len(sys.argv) != 1:
        print("STOP_UNEXPECTED_ARGUMENTS: this paid sealer accepts no arguments", file=sys.stderr)
        return 2

    from v5.research.cmbp_tier0 import fsync_directory
    from v5.research.cmbp_tier0_paid import Tier0Error, seal_paid_readiness

    if REPORT.exists() or REPORT.is_symlink() or READINESS.exists() or READINESS.is_symlink():
        print("STOP_PAID_READINESS: V1 paid report/readiness path already exists", file=sys.stderr)
        return 2
    temporary = WORK / f".TEST_RESULTS_V1.{os.getpid()}.xml.tmp"
    if temporary.exists() or temporary.is_symlink():
        print("STOP_PAID_READINESS: temporary paid JUnit path already exists", file=sys.stderr)
        return 2
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    command = [
        str(REPO / ".venv/bin/python"),
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        f"--rootdir={REPO}",
        "-p",
        "no:cacheprovider",
        "-q",
        *TESTS,
        f"--junitxml={temporary}",
    ]
    completed = subprocess.run(command, cwd=REPO, env=environment, check=False)
    if completed.returncode != 0:
        print(
            f"STOP_PAID_LOCAL_TESTS: focused offline tests failed; diagnostic report retained at {temporary}",
            file=sys.stderr,
        )
        return completed.returncode or 1
    metadata = temporary.lstat() if temporary.exists() or temporary.is_symlink() else None
    if (
        metadata is None
        or temporary.is_symlink()
        or not temporary.is_file()
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
    ):
        print("STOP_PAID_LOCAL_TESTS: pytest did not emit a private regular JUnit report", file=sys.stderr)
        return 2
    descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, REPORT, follow_symlinks=False)
    except FileExistsError:
        print("STOP_PAID_READINESS: canonical paid JUnit path raced or exists", file=sys.stderr)
        return 2
    os.unlink(temporary)
    fsync_directory(WORK)
    try:
        path = seal_paid_readiness(REPO)
    except Tier0Error as exc:
        print(f"{exc.status}: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
