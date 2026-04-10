# Whipsaw Day Detection - paste into QC Algorithm Lab
from AlgorithmImports import *
import math


class WhipsawDetection(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2025, 12, 8)
        self.SetEndDate(2025, 12, 19)
        self.SetCash(100000)
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        self.vix = self.AddIndex("VIX", Resolution.Minute)
        option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-3, 3).Expiration(0, 5))
        self.day_date = None
        self.bar_count = 0
        self.day_highs = []
        self.day_lows = []
        self.day_closes = []
        self.day_opens = []
        self.prev_close = 0
        self.day_trades = []
        self.day_results = []
        self.pending = {}
        self.entry_bars = set(range(30, 301, 60))
        self.reported = False

    def OnData(self, data):
        current_date = self.Time.date()
        if current_date != self.day_date:
            if self.day_date is not None:
                self.finish_day()
            self.day_date = current_date
            self.bar_count = 0
            self.day_highs = []
            self.day_lows = []
            self.day_closes = []
            self.day_opens = []
            self.day_trades = []
            self.pending = {}
        self.bar_count += 1
        if data.Bars is not None and self.spx.Symbol in data.Bars:
            sb = data.Bars[self.spx.Symbol]
            self.day_highs.append(float(sb.High))
            self.day_lows.append(float(sb.Low))
            self.day_closes.append(float(sb.Close))
            self.day_opens.append(float(sb.Open))
        # Check pending exits
        to_remove = []
        for sym, info in self.pending.items():
            info["bars_held"] += 1
            if info["bars_held"] >= 30:
                exit_price = 0
                if data.Bars is not None and sym in data.Bars:
                    exit_price = float(data.Bars[sym].Close)
                elif self.Securities.ContainsKey(sym):
                    exit_price = float(self.Securities[sym].Price)
                if exit_price > 0 and info["entry_price"] > 0:
                    pnl = (exit_price - info["entry_price"]) / info["entry_price"] * 100
                    self.day_trades.append({"right": info["right"], "pnl": pnl})
                to_remove.append(sym)
        for sym in to_remove:
            del self.pending[sym]
        # Report on last day
        if not self.reported and str(current_date) == "2025-12-19" and self.bar_count > 350:
            self.finish_day()
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
                }
            break

    def finish_day(self):
        if not self.day_closes or not self.day_trades:
            return
        spx_open = self.day_closes[0]
        gap = 0
        if self.prev_close > 0:
            gap = (spx_open - self.prev_close) / self.prev_close * 100
        self.prev_close = self.day_closes[-1]
        # First hour range (bars 0-60)
        fh_end = min(60, len(self.day_highs))
        if fh_end > 0:
            fh_range = (max(self.day_highs[:fh_end]) - min(self.day_lows[:fh_end])) / spx_open * 100
        else:
            fh_range = 0
        # Full session range
        full_range = (max(self.day_highs) - min(self.day_lows)) / spx_open * 100
        # Range growth: range at bar 200 vs bar 60
        r60 = 0
        r200 = 0
        if len(self.day_highs) >= 60:
            r60 = (max(self.day_highs[:60]) - min(self.day_lows[:60])) / spx_open * 100
        if len(self.day_highs) >= 200:
            r200 = (max(self.day_highs[:200]) - min(self.day_lows[:200])) / spx_open * 100
        range_growth = r200 - r60 if r60 > 0 else 0
        # Mean bar range
        bar_ranges = []
        for i in range(len(self.day_highs)):
            if self.day_closes[i] > 0:
                bar_ranges.append((self.day_highs[i] - self.day_lows[i]) / self.day_closes[i] * 100)
        mean_br = sum(bar_ranges) / len(bar_ranges) if bar_ranges else 0
        # Direction reversals (crosses of opening price)
        reversals = 0
        above = self.day_closes[0] >= spx_open if self.day_closes else True
        for c in self.day_closes[1:]:
            now_above = c >= spx_open
            if now_above != above:
                reversals += 1
                above = now_above
        # VIX
        vix_price = self.vix.Price if self.vix.Price > 0 else 20.0
        # Trade results
        n_trades = len(self.day_trades)
        n_win = sum(1 for t in self.day_trades if t["pnl"] > 0)
        n_call = sum(1 for t in self.day_trades if t["right"] == "call")
        n_put = n_trades - n_call
        call_pnl = sum(t["pnl"] for t in self.day_trades if t["right"] == "call")
        put_pnl = sum(t["pnl"] for t in self.day_trades if t["right"] == "put")
        total_pnl = sum(t["pnl"] for t in self.day_trades)
        wr = round(n_win / n_trades * 100, 0) if n_trades > 0 else 0
        day_type = "WIN" if total_pnl > 0 else "WHIP"
        self.day_results.append(
            str(self.day_date) +
            " " + day_type +
            " gap=" + str(round(gap, 3)) +
            " fhr=" + str(round(fh_range, 3)) +
            " sr=" + str(round(full_range, 3)) +
            " rg=" + str(round(range_growth, 3)) +
            " mbr=" + str(round(mean_br, 4)) +
            " rev=" + str(reversals) +
            " vix=" + str(round(vix_price, 1)) +
            " t=" + str(n_trades) +
            " wr=" + str(int(wr)) +
            " cpnl=" + str(round(call_pnl, 1)) +
            " ppnl=" + str(round(put_pnl, 1))
        )

    def report(self):
        if not self.day_results:
            self.Error("NO DAY DATA")
            return
        self.Error(" | ".join(self.day_results))
