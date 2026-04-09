# Hold Duration Edge Extended - 3 months
# Paste into QC Algorithm Lab
# Question: Does the vol-regime strategy (hi->calls, lo->puts) work
# better at short holds or long holds? Is there an optimal hold?
from AlgorithmImports import *
import math


class HoldExtended(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2025, 9, 15)
        self.SetEndDate(2025, 12, 19)
        self.SetCash(100000)
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-3, 3).Expiration(0, 5))
        self.day_date = None
        self.bar_count = 0
        self.day_highs = []
        self.day_lows = []
        self.day_closes = []
        self.pending = {}
        self.hold_durations = [10, 20, 30, 60, 90, 120]
        self.max_hold = 120
        # key = (hold, regime, right) -> list of pnl
        self.results = {}
        self.entry_bars = set(range(30, 271, 60))
        self.reported = False

    def OnData(self, data):
        current_date = self.Time.date()
        if current_date != self.day_date:
            self.day_date = current_date
            self.bar_count = 0
            self.day_highs = []
            self.day_lows = []
            self.day_closes = []
        self.bar_count += 1
        if data.Bars is not None and self.spx.Symbol in data.Bars:
            sb = data.Bars[self.spx.Symbol]
            self.day_highs.append(float(sb.High))
            self.day_lows.append(float(sb.Low))
            self.day_closes.append(float(sb.Close))
        # Check pending exits
        to_remove = []
        for sym, info in self.pending.items():
            info["bars_held"] += 1
            current_price = 0
            if self.Securities.ContainsKey(sym):
                current_price = float(self.Securities[sym].Price)
            if current_price > 0 and info["entry_price"] > 0:
                bh = info["bars_held"]
                if bh in self.hold_durations and bh not in info["recorded"]:
                    info["recorded"].add(bh)
                    pnl = (current_price - info["entry_price"]) / info["entry_price"] * 100
                    key = (bh, info["regime"], info["right"])
                    if key not in self.results:
                        self.results[key] = []
                    self.results[key].append(pnl)
            if info["bars_held"] >= self.max_hold:
                to_remove.append(sym)
        for sym in to_remove:
            del self.pending[sym]
        if not self.reported and str(current_date) == "2025-12-19" and self.bar_count > 350:
            self.reported = True
            self.report()
            return
        if self.bar_count not in self.entry_bars:
            return
        if data.OptionChains is None or len(data.OptionChains) == 0:
            return
        spx_price = self.spx.Price
        if spx_price <= 0:
            return
        # Compute regime from session range
        if self.day_highs:
            sr = (max(self.day_highs) - min(self.day_lows)) / spx_price * 100
        else:
            sr = 0
        regime = "hi" if sr > 0.3 else "lo"
        for chain_sym, chain in data.OptionChains.items():
            contracts = list(chain)
            if not contracts:
                continue
            best_call = None
            best_put = None
            min_cd = 99999
            min_pd = 99999
            for c in contracts:
                dist = abs(c.Strike - spx_price)
                if c.Right == OptionRight.Call and dist < min_cd:
                    min_cd = dist
                    best_call = c
                elif c.Right == OptionRight.Put and dist < min_pd:
                    min_pd = dist
                    best_put = c
            for contract, right in [(best_call, "call"), (best_put, "put")]:
                if contract is None:
                    continue
                price = contract.LastPrice
                if price <= 0 and contract.BidPrice > 0 and contract.AskPrice > 0:
                    price = (contract.BidPrice + contract.AskPrice) / 2.0
                if price <= 0:
                    continue
                self.pending[contract.Symbol] = {
                    "entry_price": float(price),
                    "right": right,
                    "bars_held": 0,
                    "regime": regime,
                    "recorded": set(),
                }
            break

    def report(self):
        if not self.results:
            self.Error("NO HOLD DATA")
            return

        def pf(pnls):
            w = sum(p for p in pnls if p > 0)
            l = abs(sum(p for p in pnls if p < 0))
            if l == 0:
                return 999.0
            return round(w / l, 3)

        def wr(pnls):
            if not pnls:
                return 0
            return round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 0)

        parts = []
        for hold in self.hold_durations:
            hi_calls = self.results.get((hold, "hi", "call"), [])
            lo_puts = self.results.get((hold, "lo", "put"), [])
            hi_puts = self.results.get((hold, "hi", "put"), [])
            lo_calls = self.results.get((hold, "lo", "call"), [])
            regime_combo = hi_calls + lo_puts
            anti_combo = hi_puts + lo_calls
            all_puts = hi_puts + lo_puts
            parts.append(
                "h" + str(hold) +
                " reg=" + str(pf(regime_combo)) +
                "/wr" + str(int(wr(regime_combo))) +
                "/n" + str(len(regime_combo)) +
                " anti=" + str(pf(anti_combo)) +
                "/n" + str(len(anti_combo)) +
                " put=" + str(pf(all_puts)) +
                "/n" + str(len(all_puts))
            )
        self.Error(" | ".join(parts))
