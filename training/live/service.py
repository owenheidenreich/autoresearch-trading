from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from ib_insync import IB, Index, Stock

from training.prepare import ACTION_DO_NOTHING, FEATURE_NAMES
from training.live.context import LIVE_CONTEXT_DIR, load_latest_context_bundle, refresh_context_bundle
from training.live.decision import ModelDecisionEngine
from training.live.entitlements import EntitlementReport, probe_entitlements
from training.live.execution import OCOExecutionEngine
from training.live.features import FiveSecondMinuteAggregator, LiveFeatureEngine
from training.live.resolver import ACTION_TO_SPEC, SPXWContractResolver

ET_TZ = ZoneInfo("America/New_York")


@dataclass
class PaperLiveConfig:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 70
    model_path: str = "training/best_model.pt"
    train_py_path: str | None = None
    context_days: int = 30
    context_dir: str = LIVE_CONTEXT_DIR
    refresh_context: bool = True
    paper_auto: bool = True
    dry_run: bool = False
    kill_switch_path: str | None = None
    audit_path: str = os.path.join("results", "live", "audit.jsonl")
    poll_sleep_seconds: float = 0.5
    max_position_size: int = 1
    min_trade_prob: float = 0.55
    start_time_et: str = "09:30"
    end_time_et: str = "16:00"
    max_minutes: int | None = None


class IBKRMarketStream:
    """IBKR real-time stream using 5-second bars + option top-of-book/Greeks."""

    def __init__(self, ib: IB, resolver: SPXWContractResolver) -> None:
        self.ib = ib
        self.resolver = resolver
        self.agg = FiveSecondMinuteAggregator()
        self.rt_lists: dict[str, Any] = {}
        self.rt_index: dict[str, int] = {}
        self.option_contracts: dict[str, Any] = {}
        self.option_tickers: dict[str, Any] = {}

    def subscribe(self, spx_seed_price: float) -> None:
        spx = Index("SPX", "CBOE", "USD")
        spy = Stock("SPY", "ARCA", "USD")
        vix = Index("VIX", "CBOE", "USD")
        self.ib.qualifyContracts(spx, spy, vix)
        self.rt_lists["SPX"] = self.ib.reqRealTimeBars(spx, 5, "TRADES", True)
        self.rt_lists["SPY"] = self.ib.reqRealTimeBars(spy, 5, "TRADES", True)
        self.rt_lists["VIX"] = self.ib.reqRealTimeBars(vix, 5, "TRADES", True)
        self.rt_index = {k: 0 for k in self.rt_lists}

        contracts = self.resolver.resolve_all(spx_seed_price)
        for action, contract in contracts.items():
            label = ACTION_TO_SPEC[action].label
            self.option_contracts[label] = contract
            self.option_tickers[label] = self.ib.reqMktData(contract, "100,101,104,106", False, False)

    def poll(self) -> dict[str, Any] | None:
        for sym, bar_list in self.rt_lists.items():
            start = self.rt_index[sym]
            if len(bar_list) <= start:
                continue
            new_bars = bar_list[start:]
            self.rt_index[sym] = len(bar_list)
            for b in new_bars:
                self.agg.update(
                    sym,
                    b.time,
                    float(b.open_),
                    float(b.high),
                    float(b.low),
                    float(b.close),
                    float(b.volume),
                )

        completed = self.agg.pop_completed()
        if not completed:
            return None
        minute_ts, bars = completed
        if "SPX" not in bars or "SPY" not in bars:
            return None
        spx = bars["SPX"]
        spy = bars["SPY"]
        vix = bars.get("VIX")
        option_snapshot: dict[str, dict[str, Any]] = {}
        staleness: dict[str, float] = {}
        now = dt.datetime.now(dt.timezone.utc)
        for label, ticker in self.option_tickers.items():
            t = getattr(ticker, "time", None)
            if t is not None:
                if t.tzinfo is None:
                    t = t.replace(tzinfo=dt.timezone.utc)
                staleness[label] = float((now - t).total_seconds())
            else:
                staleness[label] = float("inf")

            option_snapshot[label] = {
                "bid": _f(ticker.bid),
                "ask": _f(ticker.ask),
                "last": _f(ticker.last),
                "size": _f(ticker.lastSize, 0.0),
                "strike": float(getattr(self.option_contracts[label], "strike", math.nan)),
                "delta": _greek(ticker, "delta"),
                "gamma": _greek(ticker, "gamma"),
                "theta": _greek(ticker, "theta"),
                "vega": _greek(ticker, "vega"),
                "impliedVol": _greek(ticker, "impliedVol"),
            }
        return {
            "minute_ts": minute_ts,
            "spx_bar": spx,
            "spy_bar": spy,
            "vix_bar": vix,
            "option_snapshot": option_snapshot,
            "staleness": staleness,
        }

    def close(self) -> None:
        for _, bl in self.rt_lists.items():
            self.ib.cancelRealTimeBars(bl)
        for _, contract in self.option_contracts.items():
            self.ib.cancelMktData(contract)


class PaperTradingService:
    def __init__(self, cfg: PaperLiveConfig) -> None:
        self.cfg = cfg
        os.makedirs(os.path.dirname(self.cfg.audit_path), exist_ok=True)

    def _audit(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "ts": dt.datetime.utcnow().isoformat(),
            "event": event,
            "payload": payload,
        }
        with open(self.cfg.audit_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def run_context_refresh(self, as_of_date: str | None = None) -> str:
        bundle, path = refresh_context_bundle(
            as_of_date=as_of_date,
            context_days=self.cfg.context_days,
            ib_port=self.cfg.port,
            base_dir=self.cfg.context_dir,
            include_polygon_options=True,
        )
        self._audit(
            "context_refresh",
            {
                "bundle_path": path,
                "bars": bundle.num_bars(),
                "as_of_date": bundle.as_of_date,
            },
        )
        return path

    def _load_or_refresh_context(self) -> Any:
        if self.cfg.refresh_context:
            self.run_context_refresh()
        bundle = load_latest_context_bundle(self.cfg.context_dir)
        if bundle is None:
            raise RuntimeError("No context bundle available; run context refresh first")
        return bundle

    def _probe_entitlements(self) -> EntitlementReport:
        report = probe_entitlements(
            host=self.cfg.host,
            port=self.cfg.port,
            client_id=self.cfg.client_id + 1,
            require_paper_account=True,
        )
        self._audit(
            "entitlement_probe",
            {
                "passed": report.passed,
                "warnings": report.warnings,
                "symbols": {k: vars(v) for k, v in report.symbols.items()},
            },
        )
        return report

    def run_session(self) -> None:
        session_id = f"session-{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        bundle = self._load_or_refresh_context()
        report = self._probe_entitlements()
        if not report.passed:
            raise RuntimeError("Entitlement probe failed; live session aborted")
        if not report.account or not str(report.account).startswith("DU"):
            raise RuntimeError(
                f"Paper-session guard failed: expected DU* paper account, got {report.account!r}"
            )

        decision = ModelDecisionEngine.from_checkpoint(
            model_path=self.cfg.model_path,
            train_py_path=self.cfg.train_py_path,
            min_trade_prob=self.cfg.min_trade_prob,
            max_qty=self.cfg.max_position_size,
        )
        if bundle.feature_contract_version != decision.feature_contract_version:
            raise RuntimeError(
                "Feature contract mismatch: "
                f"context={bundle.feature_contract_version}, model={decision.feature_contract_version}"
            )
        context_features = int(bundle.raw_features.shape[1]) if bundle.raw_features.ndim == 2 else 0
        if context_features < decision.num_features:
            raise RuntimeError(
                "Context feature width is too small for checkpoint: "
                f"context={context_features}, model={decision.num_features}. "
                "Refresh context with the matching feature contract."
            )
        if context_features > decision.num_features:
            self._audit(
                "feature_projection",
                {
                    "context_num_features": context_features,
                    "model_num_features": decision.num_features,
                    "mode": "truncate_context_to_model",
                },
            )

        ib = IB()
        ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id, timeout=20)

        resolver = SPXWContractResolver(ib=ib, auto_qualify=True)
        exec_engine = OCOExecutionEngine(
            ib=ib,
            dry_run=self.cfg.dry_run or (not self.cfg.paper_auto),
            max_position_size=self.cfg.max_position_size,
            kill_switch_path=self.cfg.kill_switch_path,
            audit_path=self.cfg.audit_path,
            session_id=session_id,
        )
        feat_engine = LiveFeatureEngine(bundle, target_num_features=decision.num_features)

        seed_spx = float(bundle.market_rows[-1]["close"])
        stream = IBKRMarketStream(ib, resolver)
        stream.subscribe(seed_spx)

        start_h, start_m = [int(x) for x in self.cfg.start_time_et.split(":")]
        end_h, end_m = [int(x) for x in self.cfg.end_time_et.split(":")]
        processed = 0
        current_position_id: str | None = None
        decision_seq = 0
        intent_seq = 0
        position_entry_bar = 0
        position_entry_price = 0.0
        counters = {
            "signals_generated": 0,
            "entry_intents": 0,
            "entries_applied": 0,
            "risk_update_intents": 0,
            "risk_updates_applied": 0,
            "exit_intents": 0,
            "exits_applied": 0,
            "bars_skipped_incomplete": 0,
        }

        self._audit(
            "session_start",
            {
                "session_id": session_id,
                "seed_spx": seed_spx,
                "dry_run": exec_engine.dry_run,
                "context_num_features": context_features,
                "model_num_features": decision.num_features,
                "account": report.account,
            },
        )
        try:
            while True:
                now_et = dt.datetime.now(dt.timezone.utc).astimezone(ET_TZ)
                if (now_et.hour, now_et.minute) < (start_h, start_m):
                    ib.sleep(self.cfg.poll_sleep_seconds)
                    continue
                if (now_et.hour, now_et.minute) >= (end_h, end_m):
                    break

                pkt = stream.poll()
                if pkt is None:
                    ib.sleep(self.cfg.poll_sleep_seconds)
                    continue

                ts_ms = feat_engine.append_live_minute(
                    pkt["minute_ts"],
                    pkt["spx_bar"],
                    pkt["spy_bar"],
                    pkt["vix_bar"],
                    pkt["option_snapshot"],
                )
                feat_engine.staleness = pkt["staleness"]
                snap = feat_engine.compute_snapshot(lookback=decision.lookback)
                if snap is None:
                    continue

                processed += 1
                self._audit(
                    "bar_snapshot",
                    {
                        "timestamp_ms": ts_ms,
                        "completeness": snap.completeness,
                        "present_feature_count": snap.present_feature_count,
                        "missing_feature_count": len(snap.missing_feature_indices),
                        "missing_feature_indices": snap.missing_feature_indices,
                        "missing_feature_names": [
                            FEATURE_NAMES[i]
                            for i in snap.missing_feature_indices
                            if 0 <= i < len(FEATURE_NAMES)
                        ],
                        "non_nan_mask": snap.non_nan_mask,
                        "staleness_seconds": snap.staleness_seconds,
                    },
                )
                if snap.completeness < 0.65:
                    counters["bars_skipped_incomplete"] += 1
                    self._audit("bar_skipped_incomplete", {"timestamp_ms": ts_ms, "completeness": snap.completeness})
                    if self.cfg.max_minutes and processed >= self.cfg.max_minutes:
                        break
                    continue

                # Update position state for gate head context
                if current_position_id is not None:
                    bars_held = processed - position_entry_bar
                    option_mid = None
                    state = exec_engine.positions.get(current_position_id)
                    if state and state.entry_price_reference and state.entry_price_reference > 0:
                        unrealized = 0.0  # default if no current price
                        # Try to get current option mid from execution state
                        if hasattr(state, 'last_price') and state.last_price:
                            unrealized = (state.last_price / state.entry_price_reference) - 1.0
                        decision.update_position_state(True, bars_held, unrealized)
                    else:
                        decision.update_position_state(True, bars_held, 0.0)
                else:
                    decision.update_position_state(False)

                inference = decision.infer(snap.normalized_window)
                counters["signals_generated"] += 1
                decision_seq += 1
                decision_id = f"{session_id}-d{decision_seq:05d}"
                latest_spx = float(pkt["spx_bar"]["close"])
                self._audit(
                    "model_inference",
                    {
                        "session_id": session_id,
                        "decision_id": decision_id,
                        "timestamp_ms": ts_ms,
                        "action": int(inference.action),
                        "confidence": float(inference.confidence),
                        "gate_trade_prob": float(inference.gate_trade_prob),
                        "direction_probs": [float(x) for x in inference.direction_probs],
                        "reason_codes": list(inference.reason_codes),
                    },
                )

                if current_position_id is None:
                    intent = decision.build_entry_intent(inference, resolver, latest_spx, snap.latest_raw_row,
                                                         bar_of_day=processed)
                    if intent is not None:
                        intent_seq += 1
                        intent.decision_id = decision_id
                        intent.intent_id = f"{session_id}-i{intent_seq:05d}"
                        counters["entry_intents"] += 1
                        self._audit(
                            "entry_intent",
                            {
                                "session_id": session_id,
                                "decision_id": intent.decision_id,
                                "intent_id": intent.intent_id,
                                "action": int(intent.action),
                                "qty": int(intent.qty),
                                "confidence": float(intent.confidence),
                                "stop_price": float(intent.stop_price),
                                "take_profit_price": float(intent.take_profit_price),
                                "reference_price": intent.reference_price,
                                "reason_codes": list(intent.reason_codes),
                                "contract": _contract_payload(intent.contract),
                            },
                        )
                        state = exec_engine.place_entry(intent)
                        counters["entries_applied"] += 1
                        current_position_id = state.position_id
                        position_entry_bar = processed
                        position_entry_price = intent.reference_price or 0.0
                        self._audit(
                            "entry_intent_applied",
                            {
                                "session_id": session_id,
                                "decision_id": intent.decision_id,
                                "intent_id": intent.intent_id,
                                "position_id": state.position_id,
                                "action": intent.action,
                                "confidence": intent.confidence,
                            },
                        )
                else:
                    if inference.action == ACTION_DO_NOTHING:
                        counters["exit_intents"] += 1
                        self._audit(
                            "exit_intent",
                            {
                                "session_id": session_id,
                                "decision_id": decision_id,
                                "position_id": current_position_id,
                                "reason_codes": list(inference.reason_codes),
                            },
                        )
                        if exec_engine.flatten_position(current_position_id, reason="model_exit"):
                            counters["exits_applied"] += 1
                            self._audit(
                                "model_exit",
                                {
                                    "session_id": session_id,
                                    "decision_id": decision_id,
                                    "position_id": current_position_id,
                                },
                            )
                            current_position_id = None
                    else:
                        state = exec_engine.positions.get(current_position_id)
                        if state and state.status == "OPEN":
                            mid = resolver.quote_mid(state.contract, timeout_s=0.2)
                            update = decision.build_risk_update_intent(state, mid, snap.latest_raw_row)
                            if update is not None:
                                intent_seq += 1
                                update.decision_id = decision_id
                                update.intent_id = f"{session_id}-i{intent_seq:05d}"
                                counters["risk_update_intents"] += 1
                                self._audit(
                                    "risk_update_intent",
                                    {
                                        "session_id": session_id,
                                        "decision_id": update.decision_id,
                                        "intent_id": update.intent_id,
                                        "position_id": update.position_id,
                                        "new_stop": update.new_stop_price,
                                        "new_take_profit": update.new_take_profit_price,
                                        "reason_codes": list(update.reason_codes),
                                    },
                                )
                                applied = exec_engine.apply_risk_update(update)
                                if applied:
                                    counters["risk_updates_applied"] += 1
                                self._audit(
                                    "risk_update",
                                    {
                                        "session_id": session_id,
                                        "decision_id": update.decision_id,
                                        "intent_id": update.intent_id,
                                        "position_id": current_position_id,
                                        "applied": applied,
                                        "new_stop": update.new_stop_price,
                                        "new_take_profit": update.new_take_profit_price,
                                    },
                                )
                            if state.status != "OPEN":
                                current_position_id = None

                if self.cfg.max_minutes and processed >= self.cfg.max_minutes:
                    break
                ib.sleep(self.cfg.poll_sleep_seconds)
        finally:
            if current_position_id is not None:
                exec_engine.flatten_position(current_position_id, reason="eod_flatten")
                self._audit("eod_flatten", {"position_id": current_position_id})
            stream.close()
            if ib.isConnected():
                ib.disconnect()
            self._audit(
                "session_end",
                {
                    "session_id": session_id,
                    "processed_minutes": processed,
                    "signals_generated": counters["signals_generated"],
                    "entry_intents": counters["entry_intents"],
                    "entries_applied": counters["entries_applied"],
                    "risk_update_intents": counters["risk_update_intents"],
                    "risk_updates_applied": counters["risk_updates_applied"],
                    "exit_intents": counters["exit_intents"],
                    "exits_applied": counters["exits_applied"],
                    "bars_skipped_incomplete": counters["bars_skipped_incomplete"],
                    "open_positions_final": len(
                        [p for p in exec_engine.positions.values() if p.status == "OPEN"]
                    ),
                },
            )


def _f(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default


def _greek(ticker: Any, key: str) -> float:
    for attr in ("modelGreeks", "lastGreeks", "bidGreeks", "askGreeks"):
        g = getattr(ticker, attr, None)
        if g is None:
            continue
        try:
            x = float(getattr(g, key))
            if not math.isnan(x):
                return x
        except Exception:
            continue
    return float("nan")


def _contract_payload(contract: Any) -> dict[str, Any]:
    return {
        "symbol": getattr(contract, "symbol", None),
        "secType": getattr(contract, "secType", None),
        "exchange": getattr(contract, "exchange", None),
        "currency": getattr(contract, "currency", None),
        "tradingClass": getattr(contract, "tradingClass", None),
        "lastTradeDateOrContractMonth": getattr(contract, "lastTradeDateOrContractMonth", None),
        "strike": getattr(contract, "strike", None),
        "right": getattr(contract, "right", None),
        "conId": getattr(contract, "conId", None),
    }
