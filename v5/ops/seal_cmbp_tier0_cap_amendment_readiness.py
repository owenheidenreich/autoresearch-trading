#!/usr/bin/env python3
"""Sterile offline readiness seal for Job 55; no credential or vendor access."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    if len(sys.argv) != 1:
        print("STOP_UNEXPECTED_ARGUMENTS: this Job-55 sealer accepts no arguments", file=sys.stderr, flush=True)
        return 2
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from v5.research import cmbp_tier0 as base
    from v5.research import cmbp_tier0_cap_amendment as job55

    paths = job55._paths(REPO)
    report = paths["test_report"]
    readiness = paths["readiness"]
    if readiness.exists() or readiness.is_symlink():
        try:
            receipt = job55.validate_job55_readiness_receipt(REPO, readiness)
        except Exception as exc:  # noqa: BLE001
            status = exc.status if isinstance(exc, job55.Tier0Error) else "STOP_JOB55_READINESS"
            print(f"{status}: {type(exc).__name__}", file=sys.stderr, flush=True)
            return 2
        print(f"SEALED {receipt['receipt_sha256']} {readiness}", flush=True)
        return 0
    if report.is_symlink():
        print("STOP_JOB55_LOCAL_TESTS: canonical JUnit path is a symlink", file=sys.stderr, flush=True)
        return 2
    recovering_report = report.exists()
    if recovering_report:
        try:
            job55._private_regular(report, status="STOP_JOB55_LOCAL_TESTS")
            job55._job55_junit_counts(report)
        except Exception as exc:  # noqa: BLE001
            status = exc.status if isinstance(exc, job55.Tier0Error) else "STOP_JOB55_LOCAL_TESTS"
            print(f"{status}: {type(exc).__name__}", file=sys.stderr, flush=True)
            return 2
    temporary = report.with_name(f".{report.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        print("STOP_JOB55_LOCAL_TESTS: temporary JUnit path already exists", file=sys.stderr, flush=True)
        return 2
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
        "v5/tests/test_cmbp_stream.py",
        "v5/tests/test_cmbp_tier0.py",
        "v5/tests/test_cmbp_tier0_paid.py",
        "v5/tests/test_cmbp_tier0_cap_amendment.py",
        f"--junitxml={temporary}",
    ]
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    try:
        completed = subprocess.run(command, cwd=REPO, env=environment, check=False)
        if completed.returncode != 0:
            raise job55.Tier0Error("sterile Job-55 tests failed", status="STOP_JOB55_LOCAL_TESTS")
        metadata = temporary.lstat()
        if temporary.is_symlink() or not temporary.is_file() or metadata.st_nlink != 1 or metadata.st_size <= 0:
            raise job55.Tier0Error("Job-55 JUnit output is unsafe", status="STOP_JOB55_LOCAL_TESTS")
        descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if recovering_report:
            if job55._job55_junit_counts(temporary) != job55._job55_junit_counts(report):
                raise job55.Tier0Error(
                    "recovered Job-55 JUnit differs from a fresh sterile run",
                    status="STOP_JOB55_LOCAL_TESTS",
                )
            temporary.unlink()
            descriptor = os.open(report, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            os.rename(temporary, report)
        base.fsync_directory(report.parent)
        sealed = job55.seal_job55_readiness(REPO)
        receipt = job55.validate_job55_readiness_receipt(REPO, sealed)
        print(f"SEALED {receipt['receipt_sha256']} {sealed}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        if temporary.exists() and not temporary.is_symlink():
            try:
                temporary.unlink()
            except OSError:
                pass
        status = exc.status if isinstance(exc, job55.Tier0Error) else "STOP_JOB55_UNEXPECTED_LOCAL_ERROR"
        print(f"{status}: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
