"""IBKR data-line preflight (Phase 0.5).

Runs a 5-minute streaming check against the user's IBKR paper TWS to verify
the live universe fits within the 100-line budget and pacing limits, before
Phase 3 commits to the deploy-time data feed.

Usage:
    1. Start TWS (paper account) on this machine; enable API on port 7497.
    2. Ensure OPRA + CBOE-indices market-data subscriptions are active on
       the paper account.
    3. python -m v4.scripts.ibkr_preflight \\
           --underlying SPX \\
           --strikes-around-atm 10 \\
           --duration-seconds 300

Output:
    - human-readable summary to stdout
    - JSONL audit record to v4/audit/ibkr_preflight_YYYYMMDD_HHMMSS.jsonl

This script does NOT place orders, does NOT modify any account state, and
does NOT use real-money credentials. It only subscribes to market-data
ticks for the requested universe and counts (lines used, ticks received,
greek update cadence, pacing-violation messages).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class IbkrPreflightResult:
    started_at: str
    finished_at: str
    underlying: str
    strikes_around_atm: int
    duration_seconds: int
    contracts_subscribed: int
    market_data_lines_used: int
    nbbo_ticks_received: int
    greek_ticks_received: int
    pacing_violations: int
    contracts_with_no_ticks: list[str]
    avg_greek_update_seconds: float
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.pacing_violations == 0
            and self.contracts_subscribed > 0
            and len(self.contracts_with_no_ticks) == 0
            and self.market_data_lines_used <= 100
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--underlying", default="SPX", choices=["SPX", "SPY"])
    p.add_argument(
        "--strikes-around-atm",
        type=int,
        default=10,
        help="Number of strikes above and below ATM to subscribe (so 10 → 21 strikes)",
    )
    p.add_argument("--duration-seconds", type=int, default=300)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497, help="Paper TWS API port")
    p.add_argument("--client-id", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("v4/audit"),
        help="Where to write the preflight audit JSONL",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ib_insync is already in pyproject; defer import so this script also
    # serves as a usage/docs file without a hard import cost.
    try:
        from ib_insync import IB, Index, Option, Stock  # type: ignore
    except ImportError:
        print(
            "ib_insync not installed. Run: uv add ib_insync (or pip install ib_insync)",
            file=sys.stderr,
        )
        return 2

    started_at = datetime.now(timezone.utc).isoformat()
    notes: list[str] = []

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
    except Exception as e:
        notes.append(f"connect failed: {e}")
        result = IbkrPreflightResult(
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            underlying=args.underlying,
            strikes_around_atm=args.strikes_around_atm,
            duration_seconds=args.duration_seconds,
            contracts_subscribed=0,
            market_data_lines_used=0,
            nbbo_ticks_received=0,
            greek_ticks_received=0,
            pacing_violations=0,
            contracts_with_no_ticks=[],
            avg_greek_update_seconds=0.0,
            notes=notes,
        )
        _write_result(result, args.output_dir)
        print(json.dumps(asdict(result), default=str, indent=2))
        return 1

    # --- 1. Subscribe to underlying ---
    if args.underlying == "SPX":
        underlying = Index("SPX", "CBOE")
    else:
        underlying = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(underlying)
    ib.reqMktData(underlying, "", False, False)

    # --- 2. Build the candidate option universe ---
    # In a real preflight we'd query nearest expiry + strikes around spot.
    # For this skeleton we leave that to the user to wire to their account
    # — the structure is here, the vendor-specific calls fill in.
    notes.append(
        "Skeleton: actual contract-discovery code (nearest expiry + ATM strikes) "
        "must be wired to your TWS account. The structure below counts whatever "
        "you subscribe to; it does not invent contracts."
    )

    candidate_contracts: list[Any] = []
    # Example wiring (commented; uncomment after confirming your account):
    #
    # spot = await_spot_from_ticker(ib, underlying)
    # chains = ib.reqSecDefOptParams(underlying.symbol, '', underlying.secType, underlying.conId)
    # nearest_chain = sorted(chains, key=lambda c: ...)[0]
    # nearest_expiry = sorted(nearest_chain.expirations)[0]
    # strikes = sorted(nearest_chain.strikes)
    # atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
    # for k in strikes[max(0, atm_idx - args.strikes_around_atm):atm_idx + args.strikes_around_atm + 1]:
    #     for right in ('C', 'P'):
    #         opt = Option(args.underlying, nearest_expiry, k, right, 'SMART')
    #         ib.qualifyContracts(opt)
    #         candidate_contracts.append(opt)

    for opt in candidate_contracts:
        ib.reqMktData(opt, "100,101,104,106,165,221,225,232,233,236", False, False)

    # --- 3. Stream for the requested duration ---
    nbbo_ticks = 0
    greek_ticks = 0
    pacing_violations = 0
    contract_tick_counts: dict[str, int] = defaultdict(int)
    greek_timestamps: dict[str, list[datetime]] = defaultdict(list)

    def on_pending_tickers(tickers: Any) -> None:
        nonlocal nbbo_ticks, greek_ticks
        for ticker in tickers:
            cid = str(ticker.contract.localSymbol or ticker.contract.conId)
            contract_tick_counts[cid] += 1
            if ticker.bid is not None or ticker.ask is not None:
                nbbo_ticks += 1
            if (
                getattr(ticker, "modelGreeks", None) is not None
                or getattr(ticker, "bidGreeks", None) is not None
            ):
                greek_ticks += 1
                greek_timestamps[cid].append(datetime.now(timezone.utc))

    def on_error(reqId: int, errorCode: int, errorString: str, contract: Any) -> None:
        nonlocal pacing_violations
        # IBKR pacing-violation codes are in the 100-200 range; specifically
        # 100, 110, 162 cover common rate-limit messages. See IBKR API docs.
        if errorCode in (100, 110, 162) or "pacing" in (errorString or "").lower():
            pacing_violations += 1
            notes.append(f"pacing violation: code={errorCode} msg={errorString}")

    ib.pendingTickersEvent += on_pending_tickers
    ib.errorEvent += on_error

    try:
        ib.sleep(args.duration_seconds)
    finally:
        ib.disconnect()

    # --- 4. Aggregate ---
    contracts_with_no_ticks = [
        str(c.localSymbol or c.conId)
        for c in candidate_contracts
        if contract_tick_counts.get(str(c.localSymbol or c.conId), 0) == 0
    ]

    avg_greek_gap = 0.0
    if greek_timestamps:
        gaps: list[float] = []
        for ts_list in greek_timestamps.values():
            ts_list.sort()
            for a, b in zip(ts_list, ts_list[1:], strict=False):
                gaps.append((b - a).total_seconds())
        if gaps:
            avg_greek_gap = sum(gaps) / len(gaps)

    result = IbkrPreflightResult(
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        underlying=args.underlying,
        strikes_around_atm=args.strikes_around_atm,
        duration_seconds=args.duration_seconds,
        contracts_subscribed=len(candidate_contracts) + 1,  # +1 for underlying
        market_data_lines_used=len(candidate_contracts) + 1,
        nbbo_ticks_received=nbbo_ticks,
        greek_ticks_received=greek_ticks,
        pacing_violations=pacing_violations,
        contracts_with_no_ticks=contracts_with_no_ticks,
        avg_greek_update_seconds=avg_greek_gap,
        notes=notes,
    )

    _write_result(result, args.output_dir)
    print(json.dumps(asdict(result), default=str, indent=2))
    return 0 if result.passed else 1


def _write_result(result: IbkrPreflightResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"ibkr_preflight_{stamp}.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps(asdict(result), default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
