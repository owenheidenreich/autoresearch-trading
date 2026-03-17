from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from typing import Any

from ib_insync import IB, Index, Stock

from training.prepare import ACTION_DO_NOTHING
from training.live.context import LIVE_CONTEXT_DIR, load_latest_context_bundle, refresh_context_bundle
from training.live.decision import ModelDecisionEngine
from training.live.entitlements import EntitlementReport, probe_entitlements
from training.live.execution import OCOExecutionEngine
from training.live.features import FiveSecondMinuteAggregator, LiveFeatureEngine
from training.live.resolver import ACTION_TO_SPEC, SPXWContractResolver


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
                    float(b.open),
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
        ib = IB()
        ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id, timeout=20)

        resolver = SPXWContractResolver(ib=ib, auto_qualify=True)
        exec_engine = OCOExecutionEngine(
            ib=ib,
            dry_run=self.cfg.dry_run or (not self.cfg.paper_auto),
            max_position_size=self.cfg.max_position_size,
            kill_switch_path=self.cfg.kill_switch_path,
            audit_path=self.cfg.audit_path,
        )
        feat_engine = LiveFeatureEngine(bundle)

        seed_spx = float(bundle.market_rows[-1]["close"])
        stream = IBKRMarketStream(ib, resolver)
        stream.subscribe(seed_spx)

        start_h, start_m = [int(x) for x in self.cfg.start_time_et.split(":")]
        end_h, end_m = [int(x) for x in self.cfg.end_time_et.split(":")]
        processed = 0
        current_position_id: str | None = None

        self._audit("session_start", {"seed_spx": seed_spx, "dry_run": exec_engine.dry_run})
        try:
            while True:
                now_et = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=-5)))
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
                        "staleness_seconds": snap.staleness_seconds,
                    },
                )
                if snap.completeness < 0.65:
                    self._audit("bar_skipped_incomplete", {"timestamp_ms": ts_ms, "completeness": snap.completeness})
                    if self.cfg.max_minutes and processed >= self.cfg.max_minutes:
                        break
                    continue

                inference = decision.infer(snap.normalized_window)
                latest_spx = float(pkt["spx_bar"]["close"])

                if current_position_id is None:
                    intent = decision.build_entry_intent(inference, resolver, latest_spx, snap.latest_raw_row)
                    if intent is not None:
                        state = exec_engine.place_entry(intent)
                        current_position_id = state.position_id
                        self._audit(
                            "entry_intent_applied",
                            {"position_id": state.position_id, "action": intent.action, "confidence": intent.confidence},
                        )
                else:
                    if inference.action == ACTION_DO_NOTHING:
                        if exec_engine.flatten_position(current_position_id, reason="model_exit"):
                            self._audit("model_exit", {"position_id": current_position_id})
                            current_position_id = None
                    else:
                        state = exec_engine.positions.get(current_position_id)
                        if state and state.status == "OPEN":
                            mid = resolver.quote_mid(state.contract, timeout_s=0.2)
                            update = decision.build_risk_update_intent(state, mid, snap.latest_raw_row)
                            if update is not None:
                                applied = exec_engine.apply_risk_update(update)
                                self._audit(
                                    "risk_update",
                                    {
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
            self._audit("session_end", {"processed_minutes": processed})


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
