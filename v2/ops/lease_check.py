"""Pre-flight lease check for ART² experiment loop.

Run between experiments to ensure the Akash deployment won't expire mid-run.
Auto-funds if < 1 hour remaining and wallet has ACT. Exits with error if
wallet is empty and lease is low.

Usage:
    python v2/ops/lease_check.py          # check and auto-fund if needed
    python v2/ops/lease_check.py --status  # just print status, no action
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

AKASH_NODE = "https://akash-rpc.polkachu.com:443"
AKASH_OWNER = "akash155hphg6qyy3vtr584p38wlngtqxzdr0l6jutmp"
BLOCK_TIME_SEC = 6.2
MIN_HOURS_REMAINING = 1.0
AUTO_FUND_ACT = 3  # ACT to add when low
STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), ".deploy-state")


def _run(cmd: list[str], timeout: int = 30) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def get_dseq() -> str | None:
    if not os.path.exists(STATE_FILE):
        return None
    for line in open(STATE_FILE):
        if line.startswith("DSEQ="):
            return line.strip().split("=")[1]
    return None


def get_escrow_funds(dseq: str) -> float:
    out = _run([
        "provider-services", "query", "deployment", "get",
        "--dseq", dseq, "--owner", AKASH_OWNER,
        "--node", AKASH_NODE,
    ])
    # Parse funds from YAML output
    for i, line in enumerate(out.splitlines()):
        if "funds:" in line:
            for j in range(i + 1, min(i + 5, len(out.splitlines()))):
                funds_line = out.splitlines()[j]
                if "amount:" in funds_line and "uact" not in funds_line:
                    amount_str = funds_line.split('"')[1].split('.')[0]
                    return int(amount_str)
    return 0


def get_lease_rate(dseq: str) -> float:
    out = _run([
        "provider-services", "query", "market", "lease", "list",
        "--dseq", dseq, "--owner", AKASH_OWNER,
        "--node", AKASH_NODE,
    ])
    for i, line in enumerate(out.splitlines()):
        if "rate:" in line:
            for j in range(i + 1, min(i + 3, len(out.splitlines()))):
                rate_line = out.splitlines()[j]
                if "amount:" in rate_line:
                    amount_str = rate_line.split('"')[1].split('.')[0]
                    return float(amount_str)
    return 0


def get_wallet_act() -> float:
    out = _run([
        "provider-services", "query", "bank", "balances",
        AKASH_OWNER, "--node", AKASH_NODE,
    ])
    for line in out.splitlines():
        if "uact" in line:
            prev_line_idx = out.splitlines().index(line) - 1
            if prev_line_idx >= 0:
                amount_line = out.splitlines()[prev_line_idx]
                if "amount:" in amount_line:
                    return int(amount_line.split('"')[1]) / 1e6
    return 0


def fund_deployment(dseq: str, act_amount: int):
    deploy_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy.sh")
    result = subprocess.run(
        [deploy_sh, "fund", str(act_amount)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"WARNING: fund failed: {result.stderr}")
        return False
    print(f"  Funded {act_amount} ACT")
    return True


def check_lease(status_only: bool = False) -> dict:
    dseq = get_dseq()
    if not dseq:
        print("No active deployment found (.deploy-state missing)")
        return {"ok": False, "reason": "no_deployment"}

    funds_uact = get_escrow_funds(dseq)
    rate_per_block = get_lease_rate(dseq)

    if rate_per_block <= 0:
        print("Could not determine lease rate")
        return {"ok": False, "reason": "no_rate"}

    rate_per_hour = rate_per_block * 3600 / BLOCK_TIME_SEC
    remaining_hours = funds_uact / rate_per_hour
    wallet_act = get_wallet_act()

    print(f"  Escrow: {funds_uact/1e6:.2f} ACT")
    print(f"  Rate: {rate_per_hour/1e6:.2f} ACT/hr")
    print(f"  Remaining: {remaining_hours:.1f} hours (~{int(remaining_hours*60/30)} experiments)")
    print(f"  Wallet: {wallet_act:.2f} ACT")

    if remaining_hours >= MIN_HOURS_REMAINING:
        print(f"  Status: OK")
        return {"ok": True, "remaining_hours": remaining_hours, "wallet_act": wallet_act}

    # Low on time
    print(f"  WARNING: < {MIN_HOURS_REMAINING}h remaining!")

    if status_only:
        return {"ok": False, "reason": "low_time", "remaining_hours": remaining_hours}

    if wallet_act >= AUTO_FUND_ACT:
        print(f"  Auto-funding {AUTO_FUND_ACT} ACT...")
        if fund_deployment(dseq, AUTO_FUND_ACT):
            new_remaining = remaining_hours + (AUTO_FUND_ACT * 1e6 / rate_per_hour)
            print(f"  New remaining: ~{new_remaining:.1f} hours")
            return {"ok": True, "remaining_hours": new_remaining, "funded": AUTO_FUND_ACT}
        return {"ok": False, "reason": "fund_failed"}
    else:
        print(f"  STOP: wallet has only {wallet_act:.2f} ACT, can't auto-fund {AUTO_FUND_ACT}")
        return {"ok": False, "reason": "wallet_empty", "wallet_act": wallet_act}


def main():
    parser = argparse.ArgumentParser(description="Check Akash lease time remaining")
    parser.add_argument("--status", action="store_true", help="Status only, no auto-fund")
    args = parser.parse_args()

    result = check_lease(status_only=args.status)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
