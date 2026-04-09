# SPX Signal Validation - paste into QC Algorithm Lab
from AlgorithmImports import *
import math


class SPXSignalValidation(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2025, 12, 8)
        self.SetEndDate(2025, 12, 19)
        self.SetCash(100000)
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        self.vix = self.AddIndex("VIX", Resolution.Minute)
        option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-3, 3).Expiration(0, 5))
        self.trades = []
        self.pending = {}
        self.day_date = None
        self.bar_count = 0
        self.day_highs = []
        self.day_lows = []
        self.day_closes = []
        self.entry_bars = set(range(30, 301, 60))
        self.days_with_data = set()
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
            spx_bar = data.Bars[self.spx.Symbol]
            self.day_highs.append(float(spx_bar.High))
            self.day_lows.append(float(spx_bar.Low))
            self.day_closes.append(float(spx_bar.Close))
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
                    pnl_pct = (exit_price - info["entry_price"]) / info["entry_price"] * 100
                    self.trades.append({
                        "right": info["right"],
                        "pnl_pct": pnl_pct,
                        "session_range_pct": info["session_range_pct"],
                        "realized_vol": info["realized_vol"],
                        "bar_range": info["bar_range"],
                        "vix": info["vix"],
                    })
                to_remove.append(sym)
        for sym in to_remove:
            del self.pending[sym]
        # Report on last day near close
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
        vix_price = self.vix.Price if self.vix.Price > 0 else 20.0
        if self.day_highs:
            session_range_pct = (max(self.day_highs) - min(self.day_lows)) / spx_price * 100
        else:
            session_range_pct = 0
        if len(self.day_closes) >= 21:
            recent = self.day_closes[-21:]
            returns = [(recent[i] - recent[i-1]) / recent[i-1] for i in range(1, len(recent))]
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            realized_vol = math.sqrt(var_r) * math.sqrt(390) * 100
        else:
            realized_vol = 0
        if data.Bars is not None and self.spx.Symbol in data.Bars:
            sb = data.Bars[self.spx.Symbol]
            bar_range = (float(sb.High) - float(sb.Low)) / spx_price * 100
        else:
            bar_range = 0
        for chain_sym, chain in data.OptionChains.items():
            contracts = list(chain)
            if not contracts:
                continue
            self.days_with_data.add(current_date)
            best_call = None
            best_put = None
            min_call_dist = 99999
            min_put_dist = 99999
            for c in contracts:
                dist = abs(c.Strike - spx_price)
                if c.Right == OptionRight.Call and dist < min_call_dist:
                    min_call_dist = dist
                    best_call = c
                elif c.Right == OptionRight.Put and dist < min_put_dist:
                    min_put_dist = dist
                    best_put = c
            for contract, right in [(best_call, "call"), (best_put, "put")]:
                if contract is None:
                    continue
                price = contract.LastPrice
                if price <= 0:
                    if contract.BidPrice > 0 and contract.AskPrice > 0:
                        price = (contract.BidPrice + contract.AskPrice) / 2.0
                if price <= 0:
                    continue
                self.pending[contract.Symbol] = {
                    "entry_price": float(price),
                    "right": right,
                    "bars_held": 0,
                    "session_range_pct": session_range_pct,
                    "realized_vol": realized_vol,
                    "bar_range": bar_range,
                    "vix": vix_price,
                }
            break

    def report(self):
        if not self.trades:
            self.Error("NO_TRADES days=" + str(len(self.days_with_data)))
            return

        def pf(tl):
            w = sum(t["pnl_pct"] for t in tl if t["pnl_pct"] > 0)
            l = abs(sum(t["pnl_pct"] for t in tl if t["pnl_pct"] < 0))
            if l == 0:
                return 999.0
            return round(w / l, 3)

        def wr(tl):
            if not tl:
                return 0
            return round(sum(1 for t in tl if t["pnl_pct"] > 0) / len(tl) * 100, 1)

        calls = [t for t in self.trades if t["right"] == "call"]
        puts = [t for t in self.trades if t["right"] == "put"]
        parts = []
        parts.append("t=" + str(len(self.trades)) + " d=" + str(len(self.days_with_data)))
        parts.append("call=" + str(pf(calls)) + "/" + str(len(calls)))
        parts.append("put=" + str(pf(puts)) + "/" + str(len(puts)))

        def med(vals):
            s = sorted(vals)
            return s[len(s) // 2] if s else 0

        range_med = med([t["session_range_pct"] for t in self.trades])
        vol_med = med([t["realized_vol"] for t in self.trades])
        br_med = med([t["bar_range"] for t in self.trades])
        vix_med = med([t["vix"] for t in self.trades])

        def regime(fk, mv, name, ppf):
            hc = [t for t in self.trades if t[fk] > mv and t["right"] == "call"]
            lp = [t for t in self.trades if t[fk] <= mv and t["right"] == "put"]
            combo = hc + lp
            parts.append(name + "=" + str(pf(combo)) + "(p" + ppf + ")n=" + str(len(combo)))
            return pf(combo)

        regime("session_range_pct", range_med, "RANGE", "1.435")
        regime("realized_vol", vol_med, "RVOL", "1.167")
        regime("bar_range", br_med, "BRANGE", "1.855")
        regime("vix", vix_med, "VIX", "N/A")
        self.Error(" | ".join(parts))
