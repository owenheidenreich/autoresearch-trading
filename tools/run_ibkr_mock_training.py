#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path


def _previous_business_day(today: dt.date) -> dt.date:
    d = today - dt.timedelta(days=1)
    while d.weekday() >= 5:  # Sat/Sun
        d -= dt.timedelta(days=1)
    return d


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        logf.write(f"$ {' '.join(cmd)}\n\n")
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
        return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated mock training from IBKR historical data (ending on yesterday by default)."
    )
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (default: previous business day)")
    parser.add_argument("--calendar-days", type=int, default=20, help="Calendar lookback window before end date")
    parser.add_argument("--time-budget", type=int, default=90, help="TIME_BUDGET seconds for mock train")
    parser.add_argument("--ib-port", type=int, default=4002)
    parser.add_argument("--skip-train", action="store_true", help="Only build isolated dataset, skip training step")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    training_dir = repo_root / "training"
    if args.end_date:
        end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end_date = _previous_business_day(dt.date.today())
    start_date = end_date - dt.timedelta(days=max(args.calendar_days, 5))

    run_id = dt.datetime.utcnow().strftime("run-%Y-%m-%d-%H%M%S")
    run_root = repo_root / "results" / "mock-training" / run_id
    sandbox = run_root / "sandbox"
    logs = run_root / "logs"
    home_dir = run_root / "home"
    sandbox.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)

    # Isolate train/prepare so baseline best_model.pt and cache are untouched.
    shutil.copy2(training_dir / "prepare.py", sandbox / "prepare.py")
    shutil.copy2(training_dir / "train.py", sandbox / "train.py")

    env = dict(os.environ)
    env["HOME"] = str(home_dir)
    env["IB_PORT"] = str(args.ib_port)
    env["TIME_BUDGET"] = str(args.time_budget)
    env["PYTHONUNBUFFERED"] = "1"
    # Preserve user-site packages (numpy/torch/etc.) even with isolated HOME.
    user_site = site.getusersitepackages()
    if user_site:
        prior = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{user_site}:{prior}" if prior else user_site

    start_s = start_date.isoformat()
    end_s = end_date.isoformat()

    # Pre-warm SPXW option caches from IBKR in isolated HOME cache so prepare.py can consume them.
    prewarm_code = (
        "import os,pickle,prepare as p\n"
        f"start='{start_s}'\n"
        f"end='{end_s}'\n"
        "df=p.download_spy_bars_ibkr(start,end)\n"
        "df=df[df['date'].apply(p.is_0dte_day)].copy().reset_index(drop=True)\n"
        "os.makedirs(p.DATA_DIR, exist_ok=True)\n"
        "with open(os.path.join(p.DATA_DIR,'spy_1min.pkl'),'wb') as f: pickle.dump(df,f)\n"
        "dates=sorted(df['date'].unique())\n"
        "print(f'Prewarm 0DTE dates: {len(dates)}')\n"
        "if dates:\n"
        "    p.download_spxw_ibkr(df, dates=dates)\n"
    )

    rc_prewarm = _run(
        [sys.executable, "-c", prewarm_code],
        cwd=sandbox,
        env=env,
        log_path=logs / "01_prewarm.log",
    )
    if rc_prewarm != 0:
        print(f"Prewarm failed (rc={rc_prewarm}). See {logs / '01_prewarm.log'}")
        sys.exit(rc_prewarm)

    rc_prepare = _run(
        [
            sys.executable,
            "prepare.py",
            "--start",
            start_s,
            "--end",
            end_s,
            "--spy-source",
            "ibkr",
            "--use-spx",
            "--ib-port",
            str(args.ib_port),
        ],
        cwd=sandbox,
        env=env,
        log_path=logs / "02_prepare.log",
    )
    if rc_prepare != 0:
        print(f"prepare failed (rc={rc_prepare}). See {logs / '02_prepare.log'}")
        sys.exit(rc_prepare)

    rc_train = 0
    if not args.skip_train:
        rc_train = _run(
            [sys.executable, "train.py"],
            cwd=sandbox,
            env=env,
            log_path=logs / "03_train.log",
        )

    metadata = {
        "run_id": run_id,
        "start_date": start_s,
        "end_date": end_s,
        "calendar_days": int(args.calendar_days),
        "time_budget": int(args.time_budget),
        "ib_port": int(args.ib_port),
        "skip_train": bool(args.skip_train),
        "return_codes": {
            "prewarm": rc_prewarm,
            "prepare": rc_prepare,
            "train": rc_train,
        },
        "paths": {
            "run_root": str(run_root),
            "sandbox": str(sandbox),
            "logs": str(logs),
            "mock_home": str(home_dir),
            "mock_model": str(sandbox / "best_model.pt"),
        },
    }
    with (run_root / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(json.dumps(metadata, indent=2))
    if rc_train != 0:
        print(f"mock training failed (rc={rc_train}). See {logs / '03_train.log'}")
        sys.exit(rc_train)


if __name__ == "__main__":
    main()
