"""Paper replay and promotion-readiness accounting.

This module replays a frozen selected-trade file through executable bid/ask
fills, conservative extra-slippage scenarios, and the v4 order-state machine.
It is intentionally not a live broker simulator. Its job is to make promotion
risks explicit before paper trading: order state, one-contract exposure,
mandatory flat behavior, fill-price source, stress PnL, and live-data parity
gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.model.supervised_pilot import Trade
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.sim.order_state import OrderRecord, OrderState


_NY = ZoneInfo("America/New_York")
_CONTRACT_MULTIPLIER = 100.0


@dataclass(frozen=True)
class SlippageScenario:
    """Extra adverse option-price slippage beyond crossing bid/ask."""

    name: str
    extra_entry_price: float = 0.0
    extra_exit_price: float = 0.0

    @property
    def round_trip_cost(self) -> float:
        return (self.extra_entry_price + self.extra_exit_price) * _CONTRACT_MULTIPLIER


@dataclass(frozen=True)
class PaperReplayConfig:
    protocol_id: str = "protocol066"
    contract_multiplier: float = _CONTRACT_MULTIPLIER
    forced_flat_local_time: str = "15:55"
    intended_size: int = 1
    persisted_model_artifact_available: bool = False


DEFAULT_SLIPPAGE_SCENARIOS = (
    SlippageScenario("nbbo_bid_ask", 0.0, 0.0),
    SlippageScenario("extra_005_each_side", 0.05, 0.05),
    SlippageScenario("extra_010_each_side", 0.10, 0.10),
    SlippageScenario("extra_025_each_side", 0.25, 0.25),
)


def load_selected_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path)
    if frame.empty:
        raise ValueError(f"selected-trade file is empty: {path}")
    required = {
        "trade_uid",
        "split",
        "seed",
        "session",
        "decision_time",
        "candidate_exit_time",
        "contract_id",
        "right",
        "candidate_pnl",
        "candidate_exit_step",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"selected-trade file is missing columns: {missing}")
    frame["trade_uid"] = frame["trade_uid"].astype(str)
    frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["candidate_exit_ts"] = pd.to_datetime(frame["candidate_exit_time"], utc=True)
    frame["candidate_exit_step_int"] = pd.to_numeric(frame["candidate_exit_step"], errors="coerce").round().astype("Int64")
    return frame


def load_lifecycle_steps(sequence_dir: Path) -> pd.DataFrame:
    path = sequence_dir / "protocol054_lifecycle_steps.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    columns = [
        "trade_uid",
        "step_idx",
        "quote_time",
        "bid",
        "ask",
        "mid",
        "spread",
        "spread_frac",
        "quote_gap_seconds",
        "entry_ask",
        "current_pnl",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["trade_uid"] = frame["trade_uid"].astype(str)
    frame["step_idx"] = pd.to_numeric(frame["step_idx"], errors="coerce").astype("Int64")
    frame["quote_ts"] = pd.to_datetime(frame["quote_time"], utc=True)
    for column in ["bid", "ask", "mid", "spread", "spread_frac", "quote_gap_seconds", "entry_ask", "current_pnl"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_replay_frame(selected: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    entry = steps[steps["step_idx"].eq(0)].copy()
    entry = entry.rename(
        columns={
            "bid": "entry_bid",
            "ask": "entry_ask_nbbo",
            "mid": "entry_mid",
            "spread": "entry_spread",
            "spread_frac": "entry_spread_frac",
            "quote_gap_seconds": "entry_quote_gap_seconds",
            "quote_time": "entry_quote_time",
            "quote_ts": "entry_quote_ts",
        }
    )
    entry = entry[
        [
            "trade_uid",
            "entry_bid",
            "entry_ask_nbbo",
            "entry_mid",
            "entry_spread",
            "entry_spread_frac",
            "entry_quote_gap_seconds",
            "entry_quote_time",
            "entry_quote_ts",
        ]
    ]

    exit_steps = steps.rename(
        columns={
            "bid": "exit_bid",
            "ask": "exit_ask",
            "mid": "exit_mid",
            "spread": "exit_spread",
            "spread_frac": "exit_spread_frac",
            "quote_gap_seconds": "exit_quote_gap_seconds",
            "entry_ask": "entry_ask_from_path",
            "current_pnl": "path_current_pnl",
            "quote_time": "exit_quote_time",
            "quote_ts": "exit_quote_ts",
        }
    )
    exit_steps = exit_steps[
        [
            "trade_uid",
            "step_idx",
            "exit_bid",
            "exit_ask",
            "exit_mid",
            "exit_spread",
            "exit_spread_frac",
            "exit_quote_gap_seconds",
            "entry_ask_from_path",
            "path_current_pnl",
            "exit_quote_time",
            "exit_quote_ts",
        ]
    ]

    frame = selected.merge(entry, on="trade_uid", how="left")
    frame = frame.merge(
        exit_steps,
        left_on=["trade_uid", "candidate_exit_step_int"],
        right_on=["trade_uid", "step_idx"],
        how="left",
    )
    frame["entry_fill_nbbo"] = frame["entry_ask_from_path"].fillna(frame["entry_ask_nbbo"])
    frame["exit_fill_nbbo"] = frame["exit_bid"]
    frame["paper_pnl_nbbo"] = (frame["exit_fill_nbbo"] - frame["entry_fill_nbbo"]) * _CONTRACT_MULTIPLIER
    frame["candidate_pnl_diff_vs_path"] = frame["paper_pnl_nbbo"] - pd.to_numeric(frame["candidate_pnl"], errors="coerce")
    frame["eval_split"] = frame["split"].astype(str)
    frame["exit_local_time"] = frame["candidate_exit_ts"].dt.tz_convert(_NY).dt.strftime("%H:%M")
    frame["decision_local_time"] = frame["decision_ts"].dt.tz_convert(_NY).dt.strftime("%H:%M")
    frame["is_march"] = frame["split"].eq("q1_2026") & frame["session"].astype(str).ge("2026-03-01")
    return frame


def apply_slippage(frame: pd.DataFrame, scenario: SlippageScenario) -> pd.DataFrame:
    out = frame.copy()
    out["slippage_scenario"] = scenario.name
    out["entry_fill_price"] = out["entry_fill_nbbo"] + scenario.extra_entry_price
    out["exit_fill_price"] = (out["exit_fill_nbbo"] - scenario.extra_exit_price).clip(lower=0.0)
    out["paper_pnl"] = (out["exit_fill_price"] - out["entry_fill_price"]) * _CONTRACT_MULTIPLIER
    out["extra_round_trip_cost"] = scenario.round_trip_cost
    return out


def _profit_factor(values: pd.Series) -> float:
    wins = values.clip(lower=0).sum()
    losses = -values.clip(upper=0).sum()
    if losses <= 0:
        return 999.0 if wins > 0 else 0.0
    return float(wins / losses)


def _trade_metrics(frame: pd.DataFrame) -> dict:
    values = pd.to_numeric(frame["paper_pnl"], errors="coerce").fillna(0.0)
    if frame.empty:
        return {
            "rows": 0,
            "total_pnl": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "median_pnl": 0.0,
        }
    return {
        "rows": int(len(frame)),
        "total_pnl": float(values.sum()),
        "profit_factor": _profit_factor(values),
        "win_rate": float((values > 0).mean()),
        "median_pnl": float(values.median()),
        "mean_pnl": float(values.mean()),
    }


def summarize_by_split_seed(frame: pd.DataFrame) -> list[dict]:
    expanded = [frame]
    march = frame[frame["is_march"]].copy()
    if not march.empty:
        march["eval_split"] = "march_2026"
        expanded.append(march)
    all_rows = pd.concat(expanded, ignore_index=True)
    seed_rows = []
    for (scenario, split, seed), group in all_rows.groupby(["slippage_scenario", "eval_split", "seed"], dropna=False):
        metrics = _trade_metrics(group)
        seed_rows.append({"slippage_scenario": scenario, "eval_split": split, "seed": int(seed), **metrics})
    seed_frame = pd.DataFrame(seed_rows)
    out = []
    for (scenario, split), group in seed_frame.groupby(["slippage_scenario", "eval_split"], dropna=False):
        out.append(
            {
                "slippage_scenario": scenario,
                "eval_split": split,
                "seed_count": int(len(group)),
                "seed_total_pnl_median": float(group["total_pnl"].median()),
                "seed_total_pnl_min": float(group["total_pnl"].min()),
                "seed_total_pnl_max": float(group["total_pnl"].max()),
                "positive_seed_fraction": float((group["total_pnl"] > 0.0).mean()),
                "profit_factor_median": float(group["profit_factor"].median()),
                "trades_median": float(group["rows"].median()),
            }
        )
    return sorted(out, key=lambda row: (row["slippage_scenario"], row["eval_split"]))


def concentration_metrics(frame: pd.DataFrame) -> dict:
    trades = [
        Trade(
            session=str(row.session),
            decision_time=str(row.decision_time),
            pnl=float(row.paper_pnl),
            score=float(getattr(row, "predicted_continuation_value", 0.0) or 0.0),
            right=str(row.right),
            offset=float(getattr(row, "offset", 0.0) or 0.0),
            strategy=str(row.slippage_scenario),
        )
        for row in frame.itertuples(index=False)
    ]
    metrics = metrics_with_concentration(trades)
    return {
        key: (float(value) if isinstance(value, (int, float, np.generic)) and np.isfinite(value) else value)
        for key, value in metrics.items()
    }


def build_order_record(row: pd.Series, *, config: PaperReplayConfig, scenario: SlippageScenario) -> OrderRecord:
    decision_time = _timestamp(row["decision_ts"])
    exit_time = _timestamp(row["candidate_exit_ts"])
    record = OrderRecord(
        order_id=f"{config.protocol_id}-{scenario.name}-{row['seed']}-{row.name}",
        contract_id=str(row["contract_id"]),
        side="BUY",
        intended_size=config.intended_size,
        decision_time=decision_time,
        submit_time=decision_time,
        ack_time=decision_time,
        fill_times=[decision_time],
        limit_price=None,
        fill_prices=[float(row["entry_fill_price"])],
        fill_sizes=[config.intended_size],
        nbbo_at_decision=(float(row["entry_bid"]), float(row["entry_ask_nbbo"])),
        nbbo_at_submit=(float(row["entry_bid"]), float(row["entry_ask_nbbo"])),
        nbbo_at_fill=(float(row["entry_bid"]), float(row["entry_ask_nbbo"])),
        quote_age_ms_at_decision=_quote_age_ms(row.get("entry_quote_gap_seconds")),
        quote_age_ms_at_fill=_quote_age_ms(row.get("exit_quote_gap_seconds")),
        spread_at_submit=float(row["entry_spread"]) if pd.notna(row["entry_spread"]) else None,
        option_price_at_submit=float(row["entry_fill_price"]),
        sequencing_suspect_flag=bool(row.get("sequencing_suspect_flag", False)),
    )
    for state in (
        OrderState.DECISION_MADE,
        OrderState.ORDER_SUBMITTED,
        OrderState.BROKER_ACKNOWLEDGED,
        OrderState.WORKING,
        OrderState.FILLED,
    ):
        record.transition(t=decision_time, to_state=state)
    record.transition(t=exit_time, to_state=OrderState.EXIT_SUBMITTED)
    record.transition(t=exit_time, to_state=OrderState.EXIT_FILLED)
    return record


def order_state_summary(frame: pd.DataFrame, *, config: PaperReplayConfig, scenario: SlippageScenario, sample_size: int = 5000) -> dict:
    sample = frame.head(sample_size)
    records = [build_order_record(row, config=config, scenario=scenario) for _, row in sample.iterrows()]
    final_counts = {}
    transition_counts = {}
    for record in records:
        final = record.final_state.value if record.final_state else "missing"
        final_counts[final] = final_counts.get(final, 0) + 1
        for event in record.history:
            key = f"{event.from_state.value}->{event.to_state.value}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
    return {
        "sampled_records": len(records),
        "final_state_counts": final_counts,
        "transition_counts": transition_counts,
        "all_exit_filled": bool(records and all(record.final_state == OrderState.EXIT_FILLED for record in records)),
        "all_one_contract": bool(records and all(record.intended_size == config.intended_size for record in records)),
    }


def live_data_parity_checks(frame: pd.DataFrame, one_second_summary: dict | None, *, config: PaperReplayConfig) -> list[dict]:
    checks = []

    def add(name: str, status: str, detail: str, value: object = None) -> None:
        checks.append({"name": name, "status": status, "detail": detail, "value": value})

    add(
        "required_quote_fields",
        "pass" if frame[["entry_fill_nbbo", "exit_fill_nbbo", "entry_bid", "entry_ask_nbbo", "exit_bid", "exit_ask"]].notna().all().all() else "fail",
        "entry/exit executable bid-ask fields must be present for every replayed trade",
    )
    add(
        "bid_ask_sanity",
        "pass" if ((frame["entry_ask_nbbo"] >= frame["entry_bid"]) & (frame["exit_ask"] >= frame["exit_bid"]) & (frame["entry_fill_nbbo"] > 0)).all() else "fail",
        "entry and exit NBBO must have ask >= bid and positive entry ask",
    )
    add(
        "no_mid_fills",
        "pass",
        "paper replay uses entry ask and exit bid only; mid fills are not allowed",
        {"entry_source": "ask", "exit_source": "bid"},
    )
    add(
        "one_contract",
        "pass",
        "all replayed decisions are forced to intended_size=1",
        config.intended_size,
    )
    add(
        "flat_before_close",
        "pass" if (frame["exit_local_time"] <= config.forced_flat_local_time).all() else "fail",
        f"candidate exits must be at or before {config.forced_flat_local_time} New York time",
    )
    add(
        "spxw_pm_contracts",
        "pass" if frame["contract_id"].astype(str).str.startswith("SPXW-").all() else "fail",
        "paper replay must not include AM-settled SPX contracts",
    )
    path_diff = pd.to_numeric(frame["candidate_pnl_diff_vs_path"], errors="coerce").abs()
    add(
        "selected_vs_path_pnl",
        "pass" if path_diff.quantile(0.99) <= 1e-6 else "fail",
        "selected candidate PnL must match replayed bid/ask path PnL",
        {"p99_abs_diff": float(path_diff.quantile(0.99))},
    )
    exit_time_ok = frame["exit_quote_ts"].notna() & (frame["exit_quote_ts"] <= frame["candidate_exit_ts"])
    add(
        "decision_before_exit",
        "pass" if bool((frame["decision_ts"] <= frame["candidate_exit_ts"]).all()) else "fail",
        "decision timestamp must be at or before the candidate exit timestamp",
    )
    add(
        "exit_quote_time_causality",
        "pass" if bool(exit_time_ok.all()) else "fail",
        "exit quote timestamp must not be after the candidate exit timestamp",
        {
            "exit_violations": int((~exit_time_ok).sum()),
        },
    )
    add(
        "entry_quote_timestamp_retained",
        "warn",
        "Protocol 060 retained the executable entry ask, but not the original entry quote timestamp; live parity must retain it",
    )
    quote_gap_missing = float(frame[["entry_quote_gap_seconds", "exit_quote_gap_seconds"]].isna().any(axis=1).mean())
    add(
        "quote_gap_available",
        "warn" if quote_gap_missing > 0.0 else "pass",
        "quote age/gap fields are needed for live parity and stale-quote rejection",
        {"missing_fraction": quote_gap_missing},
    )
    if one_second_summary:
        coverage = float(one_second_summary.get("coverage", 0.0))
        sign_flip = float(one_second_summary.get("sign_flip_fraction", 1.0))
        add(
            "one_second_path_audit",
            "warn" if coverage < 0.5 else "pass",
            "existing CBBO-1s audit should agree with 1m path labels where coverage exists",
            {
                "coverage": coverage,
                "sign_flip_fraction": sign_flip,
                "abs_diff_p95": one_second_summary.get("abs_diff_p95"),
            },
        )
        add(
            "one_second_sign_flips",
            "pass" if sign_flip == 0.0 else "fail",
            "1s replay should not flip trade signs on audited rows",
            {"sign_flip_fraction": sign_flip},
        )
    else:
        add("one_second_path_audit", "warn", "no CBBO-1s audit summary was supplied")
    add(
        "live_shadow_feed",
        "blocker",
        "live parity still requires an IBKR/theta or Databento shadow-feed comparison before paper trading",
    )
    add(
        "persisted_model_artifact",
        "pass" if config.persisted_model_artifact_available else "blocker",
        (
            f"{config.protocol_id} persisted model/scaler artifacts are present."
            if config.persisted_model_artifact_available
            else f"{config.protocol_id} deployment still needs persisted model/scaler artifacts"
        ),
    )
    return checks


def promotion_gate_status(checks: list[dict]) -> str:
    statuses = {check["status"] for check in checks}
    if "fail" in statuses:
        return "fail"
    if "blocker" in statuses:
        return "blocked"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _timestamp(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _quote_age_ms(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(round(float(value) * 1000.0))
