#!/usr/bin/env python3
"""Run the exact offline Job-51 tests and seal precredential readiness."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
WORK = REPO / "v5/work/cmbp-tier0-acquisition"
REPORT = WORK / "TEST_RESULTS_V1.xml"
READINESS = WORK / "LOCAL_READINESS_RECEIPT_V1.json"
TESTS = (
    "v5/tests/test_cmbp_stream.py",
    "v5/tests/test_cmbp_tier0.py",
)


def main() -> int:
    if len(sys.argv) != 1:
        print("STOP_UNEXPECTED_ARGUMENTS: this sealer accepts no arguments", file=sys.stderr)
        return 2

    from v5.research.cmbp_tier0 import Tier0Error, fsync_directory, seal_readiness

    if REPORT.exists() or REPORT.is_symlink() or READINESS.exists() or READINESS.is_symlink():
        print("STOP_READINESS_SEAL: V1 report/readiness path already exists", file=sys.stderr)
        return 2
    temporary = WORK / f".TEST_RESULTS_V1.{os.getpid()}.xml.tmp"
    if temporary.exists() or temporary.is_symlink():
        print("STOP_READINESS_SEAL: temporary JUnit path already exists", file=sys.stderr)
        return 2
    env = {
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
    completed = subprocess.run(command, cwd=REPO, env=env, check=False)
    if completed.returncode != 0:
        print(
            f"STOP_LOCAL_TESTS: focused offline tests failed; diagnostic report retained at {temporary}",
            file=sys.stderr,
        )
        return completed.returncode or 1
    temporary_metadata = temporary.lstat() if temporary.exists() or temporary.is_symlink() else None
    if (
        temporary_metadata is None
        or temporary.is_symlink()
        or not temporary.is_file()
        or temporary_metadata.st_nlink != 1
        or temporary_metadata.st_uid != os.getuid()
    ):
        print("STOP_LOCAL_TESTS: pytest did not emit a regular JUnit report", file=sys.stderr)
        return 2
    report_descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(report_descriptor)
    finally:
        os.close(report_descriptor)
    try:
        os.link(temporary, REPORT, follow_symlinks=False)
    except FileExistsError:
        print("STOP_READINESS_SEAL: canonical JUnit path raced or already exists", file=sys.stderr)
        return 2
    os.unlink(temporary)
    fsync_directory(WORK)
    try:
        path = seal_readiness(REPO)
    except Tier0Error as exc:
        print(f"{exc.status}: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
