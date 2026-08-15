"""Protocol 118: historical no-order shadow rehearsal for frozen Protocol 101.

This script does not train, download paid data, call IBKR, or place orders. It
turns the already-audited Protocol 101 high-resolution replay rows into the
same kind of JSONL stream expected from a live no-order shadow router, then
checks parity, one-contract accounting, affordability, and end-flat behavior.
"""
from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from v4.live.shadow_parity import ShadowParityConfig, summarize_shadow_parity
from v4.scripts.run_protocol097_sequential_event_policy import build_events
from v4.scripts.run_protocol101_event_history_policy import FEATURE_COLUMNS, add_causal_history_features
from v4.sim.shadow_paper import ShadowPaperConfig, replay_shadow_paper


DEFAULT_HIGHRES_REPLAY = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_117_protocol101_targeted_highres_path_audit/report.json"
)
DEFAULT_PROTOCOL092_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy/serial_opportunity_dataset.parquet"
)
DEFAULT_Q4_EXTERNAL_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_107_protocol101_q4_2024_external_stress/"
    "q4_2024_external_serial_opportunity_dataset.parquet"
)
DEFAULT_SPX_DIRS = (
    Path("data/raw/index/spx_1m"),
    Path("data/vendor/thetadata/index/spx_1m"),
)
DEFAULT_VIX_DIRS = (
    Path("data/raw/index/vix_1m"),
    Path("data/vendor/thetadata/index/vix_1m"),
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_118_protocol101_shadow_rehearsal"
)
DEFAULT_LEDGER_PATH = Path("v4/ledger/RESEARCH_LEDGER.md")
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--highres-replay", type=Path, default=DEFAULT_HIGHRES_REPLAY)
    parser.add_argument("--protocol092-dataset", type=Path, default=DEFAULT_PROTOCOL092_DATASET)
    parser.add_argument("--q4-external-dataset", type=Path, default=DEFAULT_Q4_EXTERNAL_DATASET)
    parser.add_argument("--spx-dir", action="append", type=Path, default=None)
    parser.add_argument("--vix-dir", action="append", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--paper-seed", type=int, default=1)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--max-historical-quote-age-ms", type=int, default=5_000)
    parser.add_argument("--max-historical-context-age-ms", type=int, default=125_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spx_dirs = tuple(args.spx_dir or DEFAULT_SPX_DIRS)
    vix_dirs = tuple(args.vix_dir or DEFAULT_VIX_DIRS)

    highres_rows = _load_highres_rows(args.highres_replay, paper_seed=args.paper_seed)
    feature_map = _build_feature_map([args.protocol092_dataset, args.q4_external_dataset])
    observations, build_summary = build_shadow_observations(
        highres_rows=highres_rows,
        feature_map=feature_map,
        spx_dirs=spx_dirs,
        vix_dirs=vix_dirs,
        paper_seed=args.paper_seed,
    )
    if build_summary["missing_feature_rows"]:
        raise SystemExit(f"missing feature rows: {build_summary['missing_feature_rows'][:5]}")
    if build_summary["missing_context_rows"]:
        raise SystemExit(f"missing official context rows: {build_summary['missing_context_rows'][:5]}")

    observations_path = args.out_dir / "protocol101_shadow_observations.jsonl"
    observations_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in observations) + "\n"
    )

    parity = summarize_shadow_parity(
        observations,
        config=ShadowParityConfig(
            protocol_id="protocol101",
            max_quote_age_ms=args.max_historical_quote_age_ms,
            max_context_age_ms=args.max_historical_context_age_ms,
            intended_size=1,
            required_feature_columns=tuple(FEATURE_COLUMNS),
            allowed_actions=("hold", "exit", "stop", "forced_flat"),
        ),
    )
    shadow_paper = replay_shadow_paper(
        observations,
        config=ShadowPaperConfig(
            protocol_id="protocol101",
            require_all_closed=True,
            require_terminal_final=True,
            enforce_global_one_position=True,
        ),
    )
    account = _paper_account_check(
        highres_rows,
        starting_cash=float(args.starting_cash),
    )
    decision = _decision(parity, shadow_paper, account, build_summary)
    summary = {
        "protocol": "118_protocol101_shadow_rehearsal",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "paper_seed": int(args.paper_seed),
        "starting_cash": float(args.starting_cash),
        "source_highres_replay": str(args.highres_replay),
        "observations_path": str(observations_path),
        "build_summary": build_summary,
        "shadow_parity": _strip_row_results(parity),
        "shadow_paper": _strip_trade_ledgers(shadow_paper),
        "paper_account": account["summary"],
        "caveats": [
            "This is a historical no-order rehearsal, not live shadow parity.",
            "Historical SPX/VIX context is official one-minute context, so the rehearsal allows a wider context-age gate than live.",
            "The live gate remains stricter: fresh broker quotes/context and no order placement before paper approval.",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "trade_ledgers.json").write_text(
        json.dumps(shadow_paper.get("trade_ledgers", []), indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "paper_account_trades.json").write_text(
        json.dumps(account["trades"], indent=2, sort_keys=True) + "\n"
    )
    _write_report(args.out_dir / "report.md", summary)
    _append_ledger(args.ledger_path, summary)
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2))
    return 0


def build_shadow_observations(
    *,
    highres_rows: list[dict[str, Any]],
    feature_map: dict[tuple[str, int, str], dict[str, Any]],
    spx_dirs: tuple[Path, ...],
    vix_dirs: tuple[Path, ...],
    paper_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    missing_feature_rows: list[dict[str, Any]] = []
    missing_context_rows: list[dict[str, Any]] = []
    official_spx_count = 0
    official_vix_count = 0
    context_count = 0
    quote_age_values: list[int] = []
    context_age_values: list[int] = []

    sorted_rows = sorted(highres_rows, key=lambda row: (_ts(row["decision_time"]).value, row["candidate_uid"]))
    for row in sorted_rows:
        key = (str(row["split"]), int(row["seed"]), str(row["candidate_uid"]))
        features_source = feature_map.get(key)
        if features_source is None:
            missing_feature_rows.append(
                {
                    "split": row.get("split"),
                    "seed": row.get("seed"),
                    "candidate_uid": row.get("candidate_uid"),
                    "decision_time": row.get("decision_time"),
                }
            )
            continue
        entry_features = _entry_features(features_source, row)

        entry_ts = _ts(row["decision_time"])
        entry_quote_ts = _ts(row["entry_time_1s"])
        exit_ts = _ts(row["mandatory_exit_time_1s"])
        split = str(row["split"])
        session = str(row["session"])
        contract_id = str(row["contract_id"])
        trade_uid = f"{split}:seed{paper_seed}:{row['candidate_uid']}"

        entry_context = _lookup_context(session, entry_ts, spx_dirs=spx_dirs, vix_dirs=vix_dirs)
        exit_context = _lookup_context(session, exit_ts, spx_dirs=spx_dirs, vix_dirs=vix_dirs)
        if entry_context is None or exit_context is None:
            missing_context_rows.append(
                {
                    "split": split,
                    "session": session,
                    "candidate_uid": row.get("candidate_uid"),
                    "entry_context_missing": entry_context is None,
                    "exit_context_missing": exit_context is None,
                }
            )
            continue

        for context, observation_ts in ((entry_context, entry_ts), (exit_context, exit_ts)):
            context_count += 1
            official_spx_count += int(bool(context["spx_is_official"]))
            official_vix_count += int(bool(context["vix_is_official"]))
            context_age_values.append(int(observation_ts.value // 1_000_000 - context["timestamp_ms"]))
        quote_age_values.append(int(entry_ts.value // 1_000_000 - entry_quote_ts.value // 1_000_000))
        quote_age_values.append(0)

        observations.append(
            _observation(
                row=row,
                action="hold",
                reason="entry_selected_by_frozen_protocol101",
                decision_time=entry_ts,
                quote_time=entry_quote_ts,
                bid=float(row["entry_bid_1s"]),
                ask=float(row["entry_ask_1s"]),
                context=entry_context,
                features=entry_features,
                contract_id=contract_id,
                trade_uid=trade_uid,
                sequence_step_index=0,
            )
        )
        observations.append(
            _observation(
                row=row,
                action=_terminal_action(str(row.get("exit_reason_1s", ""))),
                reason=f"highres_replay_{row.get('exit_reason_1s', 'exit')}",
                decision_time=exit_ts,
                quote_time=exit_ts,
                bid=float(row["exit_bid_1s"]),
                ask=float(row["exit_ask_1s"]),
                context=exit_context,
                features=entry_features,
                contract_id=contract_id,
                trade_uid=trade_uid,
                sequence_step_index=1,
            )
        )

    build_summary = {
        "paper_seed": int(paper_seed),
        "source_trades": len(highres_rows),
        "shadow_observations": len(observations),
        "shadow_trades": len(observations) // 2,
        "missing_feature_rows": missing_feature_rows,
        "missing_context_rows": missing_context_rows,
        "official_spx_fraction": _fraction(official_spx_count, context_count),
        "official_vix_fraction": _fraction(official_vix_count, context_count),
        "quote_age_ms_summary": _numeric_summary(quote_age_values),
        "context_age_ms_summary": _numeric_summary(context_age_values),
        "feature_columns": list(FEATURE_COLUMNS),
    }
    return observations, build_summary


def _load_highres_rows(path: Path, *, paper_seed: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a rows list")
    raw_clean = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("audit_status") == "audited"
        and int(row.get("seed", -1)) == int(paper_seed)
    ]
    clean: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in raw_clean:
        key = (
            row.get("source_file"),
            row.get("split"),
            row.get("seed"),
            row.get("candidate_uid"),
            row.get("contract_id"),
            row.get("decision_time"),
            row.get("mandatory_exit_time_1s"),
        )
        if key in seen:
            continue
        seen.add(key)
        clean.append(row)
    required = {"entry_bid_1s", "entry_ask_1s", "exit_bid_1s", "exit_ask_1s", "mandatory_exit_time_1s"}
    missing = [row for row in clean if any(row.get(key) is None for key in required)]
    if missing:
        raise ValueError(f"{len(missing)} high-resolution rows are missing bid/ask replay fields")
    return clean


def _build_feature_map(paths: list[Path]) -> dict[tuple[str, int, str], dict[str, Any]]:
    frames = [pd.read_parquet(path) for path in paths]
    dataset = pd.concat(frames, ignore_index=True)
    dataset["decision_dt"] = pd.to_datetime(dataset["decision_time"], utc=True)
    dataset["candidate_exit_dt"] = pd.to_datetime(dataset["candidate_exit_time"], utc=True)
    events = build_events(dataset)
    add_causal_history_features(events)
    feature_map: dict[tuple[str, int, str], dict[str, Any]] = {}
    for event in events:
        candidates = event["candidates"]
        for _, row in candidates.iterrows():
            feature_map[(str(row["split"]), int(row["seed"]), str(row["candidate_uid"]))] = row.to_dict()
    return feature_map


def _entry_features(source: dict[str, Any], replay_row: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        value = _finite(source.get(column), 0.0)
        features[column] = float(value)

    bid = float(replay_row["entry_bid_1s"])
    ask = float(replay_row["entry_ask_1s"])
    mid = (bid + ask) / 2.0
    spread = ask - bid
    features["entry_bid"] = bid
    features["entry_ask"] = ask
    features["entry_mid"] = mid
    features["entry_spread"] = spread
    features["entry_spread_frac"] = spread / mid if mid > 0 else 0.0
    features["entry_spread_over_mid"] = spread / mid if mid > 0 else 0.0
    return features


def _observation(
    *,
    row: dict[str, Any],
    action: str,
    reason: str,
    decision_time: pd.Timestamp,
    quote_time: pd.Timestamp,
    bid: float,
    ask: float,
    context: dict[str, Any],
    features: dict[str, float],
    contract_id: str,
    trade_uid: str,
    sequence_step_index: int,
) -> dict[str, Any]:
    return {
        "protocol_id": "protocol101",
        "timestamp_ms": _timestamp_ms(decision_time),
        "decision_time": _iso(decision_time),
        "entry_decision_time": str(row["decision_time"]).replace(" ", "T"),
        "session": str(row["session"]),
        "reported_split": str(row.get("split", "")),
        "seed": int(row["seed"]),
        "trade_uid": trade_uid,
        "candidate_uid": str(row["candidate_uid"]),
        "sequence_step_index": int(sequence_step_index),
        "position_state": "holding",
        "intended_size": 1,
        "contract_id": contract_id,
        "raw_symbol": contract_id,
        "nbbo": {
            "bid": float(bid),
            "ask": float(ask),
            "timestamp_ms": _timestamp_ms(quote_time),
            "source": "databento_highres_replay",
        },
        "context": {
            "spx": float(context["spx"]),
            "vix": float(context["vix"]),
            "timestamp_ms": int(context["timestamp_ms"]),
            "source": "historical_official_1m_context",
            "spx_source": context["spx_source"],
            "vix_source": context["vix_source"],
        },
        "features": dict(features),
        "model": {
            "artifact": "v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/model_artifacts",
            "protocol": "101_event_history_policy",
            "score": float(row.get("score", 0.0)),
            "predicted_continuation_value": float(row.get("score", 0.0)),
            "override_threshold": float(row.get("threshold", 0.0)),
            "fold": row.get("fold"),
            "frozen_selected_trade_replay": True,
        },
        "decision": {"action": action, "reason": reason},
        "order_intent": None,
    }


def _lookup_context(
    session: str,
    ts: pd.Timestamp,
    *,
    spx_dirs: tuple[Path, ...],
    vix_dirs: tuple[Path, ...],
) -> dict[str, Any] | None:
    spx = _lookup_index_bar(session, ts, spx_dirs, official_suffix="official_spx")
    vix = _lookup_index_bar(session, ts, vix_dirs, official_suffix="official_vix")
    if spx is None or vix is None:
        return None
    context_ts = min(int(spx["timestamp_ms"]), int(vix["timestamp_ms"]))
    return {
        "spx": float(spx["close"]),
        "vix": float(vix["close"]),
        "timestamp_ms": context_ts,
        "spx_source": spx["source"],
        "vix_source": vix["source"],
        "spx_is_official": bool(spx["is_official"]),
        "vix_is_official": bool(vix["is_official"]),
    }


def _lookup_index_bar(
    session: str,
    ts: pd.Timestamp,
    dirs: tuple[Path, ...],
    *,
    official_suffix: str,
) -> dict[str, Any] | None:
    for root in dirs:
        paths = [
            root / f"{session}.{official_suffix}.parquet",
            root / f"{session}.parquet",
        ]
        for path in paths:
            if not path.exists():
                continue
            df = _read_index_context_file(str(path))
            if df.empty or "event_time" not in df.columns:
                continue
            eligible = df[df["event_time"] <= ts]
            if eligible.empty:
                continue
            row = eligible.iloc[-1]
            return {
                "close": float(row["close"]),
                "timestamp_ms": _timestamp_ms(pd.Timestamp(row["event_time"])),
                "source": str(row.get("context_source", path)),
                "is_official": bool(row.get("is_official_index_data", "official" in path.name)),
            }
    return None


@lru_cache(maxsize=2048)
def _read_index_context_file(path_text: str) -> pd.DataFrame:
    df = pd.read_parquet(path_text)
    if df.empty or "event_time" not in df.columns:
        return df
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df.sort_values("event_time").reset_index(drop=True)


def _paper_account_check(rows: list[dict[str, Any]], *, starting_cash: float) -> dict[str, Any]:
    cash = float(starting_cash)
    peak = cash
    max_drawdown = 0.0
    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (_ts(item["decision_time"]).value, str(item["candidate_uid"]))):
        entry_ask = float(row["entry_ask_1s"])
        exit_bid = float(row["exit_bid_1s"])
        cost = entry_ask * CONTRACT_MULTIPLIER
        proceeds = exit_bid * CONTRACT_MULTIPLIER
        if cost > cash + 1e-9:
            skipped.append(
                {
                    "candidate_uid": row["candidate_uid"],
                    "session": row["session"],
                    "decision_time": row["decision_time"],
                    "entry_cost": cost,
                    "cash_available": cash,
                    "reason": "insufficient_cash",
                }
            )
            continue
        before = cash
        cash -= cost
        cash += proceeds
        peak = max(peak, cash)
        max_drawdown = max(max_drawdown, peak - cash)
        trades.append(
            {
                "candidate_uid": row["candidate_uid"],
                "session": row["session"],
                "decision_time": row["decision_time"],
                "exit_time": row["mandatory_exit_time_1s"],
                "contract_id": row["contract_id"],
                "entry_ask": entry_ask,
                "exit_bid": exit_bid,
                "entry_cost": cost,
                "cash_before": before,
                "cash_after": cash,
                "realized_pnl": proceeds - cost,
                "exit_reason": row.get("exit_reason_1s"),
            }
        )
    summary = {
        "starting_cash": float(starting_cash),
        "ending_cash": float(cash),
        "trades": len(trades),
        "skipped_unaffordable_trades": len(skipped),
        "max_drawdown": float(max_drawdown),
        "min_cash_after_trade": min((trade["cash_after"] for trade in trades), default=float(starting_cash)),
        "max_entry_cost": max((trade["entry_cost"] for trade in trades), default=0.0),
        "total_realized_pnl": float(cash - starting_cash),
    }
    return {"summary": summary, "trades": trades, "skipped": skipped}


def _decision(
    parity: dict[str, Any],
    shadow_paper: dict[str, Any],
    account: dict[str, Any],
    build_summary: dict[str, Any],
) -> str:
    if build_summary["missing_feature_rows"] or build_summary["missing_context_rows"]:
        return "blocked_missing_rehearsal_inputs"
    if parity.get("status") not in {"pass", "warn"}:
        return "blocked_shadow_parity_failed"
    if shadow_paper.get("status") != "pass":
        return "blocked_shadow_paper_failed"
    if account["summary"]["skipped_unaffordable_trades"]:
        return "blocked_unaffordable_trades"
    if build_summary["official_spx_fraction"] < 1.0 or build_summary["official_vix_fraction"] < 1.0:
        return "blocked_incomplete_official_context"
    return "pass_historical_no_order_shadow_rehearsal_live_capture_next"


def _strip_row_results(parity: dict[str, Any]) -> dict[str, Any]:
    clean = dict(parity)
    clean.pop("row_results", None)
    return clean


def _strip_trade_ledgers(shadow_paper: dict[str, Any]) -> dict[str, Any]:
    clean = dict(shadow_paper)
    clean.pop("trade_ledgers", None)
    return clean


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    parity = summary["shadow_parity"]
    paper = summary["shadow_paper"]
    account = summary["paper_account"]
    build = summary["build_summary"]
    lines = [
        "# Protocol 118: Protocol101 No-Order Shadow Rehearsal",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "This is a historical rehearsal of the live shadow stream shape. It used the frozen Protocol 101 selected trades, the Protocol 117 high-resolution bid/ask replay, and official historical SPX/VIX one-minute context. It did not train, download paid data, call IBKR, or place orders.",
        "",
        "## Scope",
        "",
        f"- Paper seed: `{summary['paper_seed']}`",
        f"- Starting cash: `${summary['starting_cash']:,.2f}`",
        f"- Trades rehearsed: `{build['shadow_trades']}`",
        f"- JSONL observations: `{build['shadow_observations']}`",
        f"- Official SPX context fraction: `{build['official_spx_fraction']:.3f}`",
        f"- Official VIX context fraction: `{build['official_vix_fraction']:.3f}`",
        "",
        "## Checks",
        "",
        f"- Shadow parity status: `{parity['status']}`",
        f"- Shadow paper status: `{paper['status']}`",
        f"- Max concurrent positions: `{paper['max_concurrent_positions']}`",
        f"- Closed trades: `{paper['closed_trades']}` / `{paper['trades']}`",
        f"- Unaffordable trades from $10,000 cash: `{account['skipped_unaffordable_trades']}`",
        f"- Ending cash using high-resolution ask-entry/bid-exit: `${account['ending_cash']:,.2f}`",
        f"- Max entry premium: `${account['max_entry_cost']:,.2f}`",
        f"- Max drawdown on closed-trade cash: `${account['max_drawdown']:,.2f}`",
        "",
        "## Caveat",
        "",
        "Protocol 118 is not live paper approval. It proves the frozen Protocol 101 behavior can be represented and audited as a no-order shadow stream on historical high-resolution data. The next live gate still needs fresh IBKR/router observations with strict quote/context freshness before any order placement.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _append_ledger(path: Path, summary: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 118 Protocol101 No-Order Shadow Rehearsal"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Built a historical no-order shadow-router rehearsal for frozen Protocol 101 using the Protocol 117 targeted high-resolution replay.
Reason: Before broker-connected paper trading, the project needed to prove the frozen selected-trade behavior can be emitted as a strict no-order JSONL stream with one contract, no overlap, ask-entry/bid-exit accounting, official historical context, and end-flat behavior.
Data Used: Existing Protocol 117 high-resolution replay, existing Protocol 092/107 candidate datasets, and existing local official SPX/VIX context files only. No paid data was downloaded, no live broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {summary['decision']}. Rehearsed {summary['build_summary']['shadow_trades']} seed-{summary['paper_seed']} trades with max concurrent positions {summary['shadow_paper']['max_concurrent_positions']} and {summary['paper_account']['skipped_unaffordable_trades']} unaffordable trades from ${summary['starting_cash']:,.2f} starting cash.
Next Gate: Run a real no-order live shadow capture for Protocol 101 when IBKR market data/cash status is ready; keep order placement disabled until that live parity gate passes.
Owner: Codex
```
"""
    existing = path.read_text() if path.exists() else ""
    if marker not in existing:
        path.write_text(existing.rstrip() + entry + "\n")
        return

    start = existing.index(marker)
    next_start = existing.find("\n## ", start + len(marker))
    replacement = entry.strip() + "\n"
    if next_start == -1:
        path.write_text(existing[:start].rstrip() + "\n\n" + replacement)
    else:
        path.write_text(existing[:start].rstrip() + "\n\n" + replacement + existing[next_start:])


def _terminal_action(exit_reason: str) -> str:
    return "stop" if exit_reason == "stop" else "exit"


def _ts(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def _timestamp_ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).isoformat().replace(" ", "T")


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _numeric_summary(values: list[int]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return {"count": 0, "min": None, "median": None, "max": None}
    mid = len(clean) // 2
    median = clean[mid] if len(clean) % 2 else (clean[mid - 1] + clean[mid]) / 2.0
    return {"count": len(clean), "min": clean[0], "median": median, "max": clean[-1]}


if __name__ == "__main__":
    raise SystemExit(main())
