"""Inventory the causal inputs for the time-aware 0DTE day-trader study.

This is deliberately model-free.  It answers whether the owned quote year can
support the episode clock declared in ``work/entry-exit-attribution/PLAN.md``
before a label, policy, or threshold is built.

The quote files are completed-minute snapshots.  A decision stamped 09:35 may
use the quote snapshot stamped 09:35 and the ES candle stamped 09:34 (the bar
covering [09:34, 09:35)); it may not use the ES bar stamped 09:35.  This audit
checks the availability of both sides of that boundary without computing any
future outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from v5.ops.measure_fill_quality import QUOTE_CORPUS


ET = "America/New_York"
CONTRACT_MULTIPLIER = 100.0
SESSION_START_EQUITY_USD = 10_000.0
TICKET_CEILING_SHARE = 0.13
MAX_ENTRY_ASK_USD = SESSION_START_EQUITY_USD * TICKET_CEILING_SHARE
QUOTE_AGE_CAP_MS = 90_000.0
NEAR_ATM_POINTS = 25.0

FIRST_QUOTE_MINUTE = "09:31"
FIRST_DECISION_MINUTE = "09:35"
LAST_ENTRY_MINUTE = "15:00"
LAST_QUOTE_MINUTE = "16:00"
MORNING_END = "12:45"
AFTERNOON_START = "12:46"
KEY_MINUTES = ("09:50", "10:00", "10:10", "13:20", "13:30", "13:40")

SESSION_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")

QUOTE_COLUMNS = (
    "event_time",
    "expiry",
    "strike",
    "right",
    "bid",
    "ask",
    "mid",
    "bid_size",
    "ask_size",
    "quote_age_ms",
    "underlying_price",
    "volume",
    "open_interest",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
)


class CoverageError(RuntimeError):
    """The owned corpus violates a structural assumption of the audit."""


@dataclass(frozen=True)
class SourceFile:
    role: str
    session: str
    path: str
    size_bytes: int
    sha256: str


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def session_from_path(path: Path) -> str:
    found = SESSION_RE.search(path.name)
    if found is None:
        raise CoverageError(f"cannot extract session from {path}")
    return found.group(1)


def minute_range(first: str, last: str) -> tuple[str, ...]:
    values = pd.date_range(f"2000-01-01 {first}", f"2000-01-01 {last}", freq="min")
    return tuple(values.strftime("%H:%M"))


QUOTE_MINUTES = minute_range(FIRST_QUOTE_MINUTE, LAST_QUOTE_MINUTE)
ENTRY_MINUTES = minute_range(FIRST_DECISION_MINUTE, LAST_ENTRY_MINUTE)
MORNING_MINUTES = minute_range(FIRST_DECISION_MINUTE, MORNING_END)
AFTERNOON_MINUTES = minute_range(AFTERNOON_START, LAST_ENTRY_MINUTE)
ES_MINUTES = minute_range("09:30", "15:59")


def signed_moneyness(
    spot: np.ndarray, strike: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """ITM points: positive in the money, negative out of the money."""

    is_call = np.asarray(right).astype(str) == "C"
    return np.where(is_call, spot - strike, strike - spot)


def live_two_sided(frame: pd.DataFrame) -> pd.Series:
    bid = pd.to_numeric(frame["bid"], errors="coerce")
    ask = pd.to_numeric(frame["ask"], errors="coerce")
    age = pd.to_numeric(frame["quote_age_ms"], errors="coerce")
    return bid.gt(0.0) & ask.gt(bid) & age.between(0.0, QUOTE_AGE_CAP_MS)


def eligible_entry(frame: pd.DataFrame) -> pd.Series:
    """OTM side of near-ATM, live now, within the signed ticket ceiling."""

    spot = pd.to_numeric(frame["underlying_price"], errors="coerce").to_numpy(float)
    strike = pd.to_numeric(frame["strike"], errors="coerce").to_numpy(float)
    money = signed_moneyness(spot, strike, frame["right"].to_numpy())
    ask = pd.to_numeric(frame["ask"], errors="coerce").to_numpy(float)
    ask_size = pd.to_numeric(frame["ask_size"], errors="coerce").to_numpy(float)
    allowed = (
        live_two_sided(frame).to_numpy(bool)
        & np.isfinite(money)
        & (money >= -NEAR_ATM_POINTS)
        & (money < 0.0)
        & np.isfinite(ask)
        & (ask * CONTRACT_MULTIPLIER <= MAX_ENTRY_ASK_USD)
        & np.isfinite(ask_size)
        & (ask_size >= 1.0)
    )
    return pd.Series(allowed, index=frame.index)


def causal_es_bar_minute(decision_minute: str) -> str:
    """Bar timestamp whose right edge is the completed decision boundary."""

    here = pd.Timestamp(f"2000-01-01 {decision_minute}") - pd.Timedelta(minutes=1)
    return here.strftime("%H:%M")


def _share(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").notna().mean()) if len(frame) else 0.0


def _counts_at_minutes(frame: pd.DataFrame, mask: pd.Series) -> dict[str, int]:
    return {
        minute: int(mask[frame["minute"].eq(minute)].sum())
        for minute in KEY_MINUTES
    }


def _distribution(values: Iterable[float]) -> dict[str, float | None]:
    got = np.asarray(list(values), dtype=float)
    got = got[np.isfinite(got)]
    if not got.size:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    return {
        "min": float(np.min(got)),
        "p10": float(np.quantile(got, 0.10)),
        "median": float(np.median(got)),
        "p90": float(np.quantile(got, 0.90)),
        "max": float(np.max(got)),
    }


def inspect_quote_file(path: Path) -> tuple[dict[str, Any], list[int], list[int]]:
    session = session_from_path(path)
    frame = pd.read_parquet(path, columns=list(QUOTE_COLUMNS))
    if frame.empty:
        raise CoverageError(f"empty quote file: {path}")

    stamped = pd.to_datetime(frame["event_time"], utc=True).dt.tz_convert(ET)
    frame = frame.assign(minute=stamped.dt.strftime("%H:%M"))
    rth = frame[frame["minute"].isin(QUOTE_MINUTES)].copy()
    if rth.empty:
        raise CoverageError(f"no regular-session quote rows: {path}")

    live = live_two_sided(rth)
    eligible = eligible_entry(rth)
    live_per_minute = live.groupby(rth["minute"]).sum().reindex(QUOTE_MINUTES, fill_value=0)
    entry_per_minute = eligible.groupby(rth["minute"]).sum().reindex(ENTRY_MINUTES, fill_value=0)
    minute_set = set(rth["minute"].unique())

    expiry = pd.to_datetime(rth["expiry"], errors="coerce").dt.strftime("%Y-%m-%d")
    same_day = expiry.eq(session)
    quote_age = pd.to_numeric(rth["quote_age_ms"], errors="coerce")
    terminal = rth[rth["minute"].eq(LAST_QUOTE_MINUTE)]

    row = {
        "session": session,
        "quote_rows_all": int(len(frame)),
        "quote_rows_rth": int(len(rth)),
        "rth_quote_minutes": int(len(minute_set.intersection(QUOTE_MINUTES))),
        "missing_rth_quote_minutes": int(len(set(QUOTE_MINUTES) - minute_set)),
        "first_rth_quote_minute": min(minute_set) if minute_set else None,
        "last_rth_quote_minute": max(minute_set) if minute_set else None,
        "live_rows_rth": int(live.sum()),
        "live_minutes_rth": int((live_per_minute > 0).sum()),
        "live_contracts_min": int(live_per_minute.min()),
        "live_contracts_median": float(live_per_minute.median()),
        "live_contracts_max": int(live_per_minute.max()),
        "entry_minutes_with_eligible_contract": int((entry_per_minute > 0).sum()),
        "morning_minutes_with_eligible_contract": int(
            (entry_per_minute.reindex(MORNING_MINUTES, fill_value=0) > 0).sum()
        ),
        "afternoon_minutes_with_eligible_contract": int(
            (entry_per_minute.reindex(AFTERNOON_MINUTES, fill_value=0) > 0).sum()
        ),
        "first_decision_eligible_contracts": int(entry_per_minute.get(FIRST_DECISION_MINUTE, 0)),
        "same_day_expiry_share": float(same_day.mean()),
        "underlying_price_coverage": _share(rth, "underlying_price"),
        "bid_size_coverage": _share(rth, "bid_size"),
        "ask_size_coverage": _share(rth, "ask_size"),
        "volume_coverage": _share(rth, "volume"),
        "open_interest_coverage": _share(rth, "open_interest"),
        "vendor_iv_coverage": _share(rth, "iv"),
        "vendor_delta_coverage": _share(rth, "delta"),
        "vendor_gamma_coverage": _share(rth, "gamma"),
        "vendor_theta_coverage": _share(rth, "theta"),
        "vendor_vega_coverage": _share(rth, "vega"),
        "quote_age_zero_share": float(quote_age.eq(0.0).mean()),
        "quote_age_non_null_share": float(quote_age.notna().mean()),
        "terminal_underlying_price_rows": int(
            pd.to_numeric(terminal["underlying_price"], errors="coerce").notna().sum()
        ),
        "terminal_underlying_price_unique": int(
            pd.to_numeric(terminal["underlying_price"], errors="coerce").dropna().nunique()
        ),
    }
    for minute, count in _counts_at_minutes(rth, live).items():
        row[f"live_contracts_{minute.replace(':', '')}"] = count
    for minute, count in _counts_at_minutes(rth, eligible).items():
        row[f"eligible_entry_contracts_{minute.replace(':', '')}"] = count
    return row, live_per_minute.astype(int).tolist(), entry_per_minute.astype(int).tolist()


def inspect_es_file(path: Path) -> dict[str, Any]:
    session = session_from_path(path)
    frame = pd.read_parquet(path)
    index = pd.to_datetime(frame.index, utc=True).tz_convert(ET)
    minutes = tuple(index.strftime("%H:%M"))
    regular = tuple(minute for minute in minutes if "09:30" <= minute <= "15:59")
    expected = set(ES_MINUTES)
    present = set(regular)
    return {
        "session": session,
        "es_rows": int(len(frame)),
        "es_rth_rows": int(len(regular)),
        "es_first_minute": min(regular) if regular else None,
        "es_last_minute": max(regular) if regular else None,
        "es_missing_rth_minutes": int(len(expected - present)),
        "es_duplicate_rth_minutes": int(len(regular) - len(present)),
        "es_ohlcv_complete_share": float(
            frame[["open", "high", "low", "close", "volume"]].notna().all(axis=1).mean()
        ),
        "es_first_decision_causal_bar_present": causal_es_bar_minute(FIRST_DECISION_MINUTE) in present,
    }


def _manifest_digest(files: list[SourceFile]) -> str:
    return hashlib.sha256(canonical_json([asdict(item) for item in files])).hexdigest()


def run(
    quote_root: Path,
    es_root: Path,
    out_dir: Path,
    *,
    supersedes: str | None = None,
    correction: str | None = None,
) -> dict[str, Any]:
    quote_files = [
        path
        for path in sorted(quote_root.glob("*.parquet"))
        if "official_context" not in path.name
    ]
    context_files = sorted(quote_root.glob("*official_context*.parquet"))
    if not quote_files:
        raise CoverageError(f"no base quote files under {quote_root}")

    quote_rows: list[dict[str, Any]] = []
    live_counts: list[int] = []
    entry_counts: list[int] = []
    sources: list[SourceFile] = []
    for i, path in enumerate(quote_files, 1):
        session = session_from_path(path)
        row, live, eligible = inspect_quote_file(path)
        quote_rows.append(row)
        live_counts.extend(live)
        entry_counts.extend(eligible)
        sources.append(
            SourceFile("spxw_0dte_quote", session, str(path), path.stat().st_size, file_sha256(path))
        )
        if i % 25 == 0 or i == len(quote_files):
            print(f"quotes {i}/{len(quote_files)}", flush=True)

    coverage = pd.DataFrame(quote_rows).sort_values("session").reset_index(drop=True)
    es_rows: list[dict[str, Any]] = []
    for session in coverage["session"]:
        path = es_root / f"{session}.es_c_0.ohlcv-1m.parquet"
        if not path.exists():
            es_rows.append({"session": session, "es_missing_file": True})
            continue
        row = inspect_es_file(path)
        row["es_missing_file"] = False
        es_rows.append(row)
        sources.append(SourceFile("es_ohlcv_1m", session, str(path), path.stat().st_size, file_sha256(path)))

    coverage = coverage.merge(pd.DataFrame(es_rows), on="session", how="left", validate="one_to_one")
    # A day with no affordable contract at 09:35 is still an essential no-trade
    # episode.  Eligibility is a per-minute action fact, never a session
    # inclusion filter.  A partial/gapped clock or absent ES path is different:
    # that session cannot implement the declared full-day primary episode.
    coverage["included_for_episode_build"] = (
        coverage["same_day_expiry_share"].eq(1.0)
        & coverage["missing_rth_quote_minutes"].eq(0)
        & coverage["es_missing_file"].eq(False)  # noqa: E712 - pandas comparison
        & coverage["es_missing_rth_minutes"].fillna(1).eq(0)
        & coverage["es_duplicate_rth_minutes"].fillna(1).eq(0)
        & coverage["es_first_decision_causal_bar_present"].fillna(False)
    )

    sessions = coverage["session"]
    included = coverage[coverage["included_for_episode_build"]]
    context_sessions = {session_from_path(path) for path in context_files}
    source_manifest = [asdict(item) for item in sources]

    receipt: dict[str, Any] = {
        "schema_version": "v5.causal-day-coverage.v1",
        "created_on": "2026-08-14",
        "purpose": "model-free inventory for the causal time-aware 0DTE day trader",
        "research_cutoff": "2026-08-05",
        "economic_data_latest_session": str(sessions.max()),
        "roots": {"quotes": str(quote_root), "es": str(es_root)},
        "declarations": {
            "quote_clock": f"{FIRST_QUOTE_MINUTE}-{LAST_QUOTE_MINUTE} ET",
            "entry_clock": f"{FIRST_DECISION_MINUTE}-{LAST_ENTRY_MINUTE} ET",
            "morning": f"{FIRST_DECISION_MINUTE}-{MORNING_END} ET",
            "afternoon": f"{AFTERNOON_START}-{LAST_ENTRY_MINUTE} ET",
            "key_minutes": list(KEY_MINUTES),
            "entry_moneyness_itm_points": [-NEAR_ATM_POINTS, 0.0],
            "entry_moneyness_upper_bound_exclusive": True,
            "max_entry_ask_usd": MAX_ENTRY_ASK_USD,
            "minimum_entry_ask_size_contracts": 1.0,
            "quote_age_cap_ms": QUOTE_AGE_CAP_MS,
            "causal_es_bar_at_first_decision": causal_es_bar_minute(FIRST_DECISION_MINUTE),
        },
        "files": {
            "base_quote_files": len(quote_files),
            "official_context_companions": len(context_files),
            "base_sessions_with_context_companion": int(sum(s in context_sessions for s in sessions)),
            "source_manifest_rows": len(source_manifest),
            "source_manifest_sha256": _manifest_digest(sources),
        },
        "sessions": {
            "base_quote_sessions": int(coverage["session"].nunique()),
            "first": str(sessions.min()),
            "last": str(sessions.max()),
            "included_for_episode_build": int(coverage["included_for_episode_build"].sum()),
            "excluded": int((~coverage["included_for_episode_build"]).sum()),
            "excluded_sessions": coverage.loc[
                ~coverage["included_for_episode_build"], "session"
            ].tolist(),
            "all_have_0935_eligible_otm_contract": bool(
                coverage["first_decision_eligible_contracts"].gt(0).all()
            ),
            "sessions_without_0935_eligible_otm_contract": coverage.loc[
                coverage["first_decision_eligible_contracts"].eq(0), "session"
            ].tolist(),
            "included_sessions_without_0935_eligible_otm_contract": included.loc[
                included["first_decision_eligible_contracts"].eq(0), "session"
            ].tolist(),
            "all_have_complete_es_rth": bool(
                coverage["es_missing_file"].eq(False).all()  # noqa: E712
                and coverage["es_missing_rth_minutes"].fillna(1).eq(0).all()
            ),
        },
        "rows_and_minutes": {
            "quote_rows_all": int(coverage["quote_rows_all"].sum()),
            "quote_rows_rth": int(coverage["quote_rows_rth"].sum()),
            "live_rows_rth": int(coverage["live_rows_rth"].sum()),
            "expected_quote_minutes_per_session": len(QUOTE_MINUTES),
            "expected_entry_minutes_per_session": len(ENTRY_MINUTES),
            "live_contracts_per_session_minute": _distribution(live_counts),
            "eligible_entry_contracts_per_decision_minute": _distribution(entry_counts),
        },
        "field_coverage_mean_across_sessions": {
            column: float(coverage[column].mean())
            for column in (
                "underlying_price_coverage",
                "bid_size_coverage",
                "ask_size_coverage",
                "volume_coverage",
                "open_interest_coverage",
                "vendor_iv_coverage",
                "vendor_delta_coverage",
                "vendor_gamma_coverage",
                "vendor_theta_coverage",
                "vendor_vega_coverage",
                "quote_age_zero_share",
                "quote_age_non_null_share",
            )
        },
        "important_limits": [
            "quote_age_ms is zero in historical aligned snapshots and is not an observed arrival-latency measurement",
            "vendor greeks are coverage diagnostics only; the episode builder must recompute them causally",
            "official_context companion files are counted but excluded from the 0DTE action ladder",
            "this inventory computes no future outcome and fits no model",
        ],
        "terminal_settlement_source": {
            "status": "NOT_VALIDATED",
            "candidate": "underlying_price in the 16:00 aligned option snapshot",
            "candidate_present_all_included_sessions": bool(
                included["terminal_underlying_price_rows"].gt(0).all()
                and included["terminal_underlying_price_unique"].eq(1).all()
            ),
            "reason": (
                "the aligned underlying snapshot is available but is not certified here as the "
                "official SPXW cash-settlement value; a held contract with no executable terminal "
                "bid must remain in the blocked population until that identity is established"
            ),
        },
        "artifacts": {
            "session_coverage_csv": "session_coverage.csv",
            "source_manifest_json": "source_manifest.json",
        },
    }
    if supersedes is not None:
        receipt["supersedes"] = supersedes
    if correction is not None:
        receipt["correction"] = correction
    unsigned_hash = hashlib.sha256(canonical_json(receipt)).hexdigest()
    receipt["receipt_sha256"] = unsigned_hash

    out_dir.mkdir(parents=True, exist_ok=False)
    coverage.to_csv(out_dir / "session_coverage.csv", index=False)
    (out_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", type=Path, default=QUOTE_CORPUS)
    parser.add_argument(
        "--es",
        type=Path,
        default=Path(
            "/Users/och/.autoresearch-trading/es_1m_2016-08-01_2026-07-31/"
            "raw/databento/glbx_es_ohlcv_1m"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--supersedes", default=None)
    parser.add_argument("--correction", default=None)
    args = parser.parse_args()
    receipt = run(
        args.quotes,
        args.es,
        args.out_dir,
        supersedes=args.supersedes,
        correction=args.correction,
    )
    print(json.dumps(receipt["sessions"], indent=2, sort_keys=True))
    print(args.out_dir / "receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
