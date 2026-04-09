# IV Crush Extended - 3 months, calls+puts, by regime
# Paste into QC Algorithm Lab
# Question: How do theta/gamma/IV evolve through the day?
# Split by vol regime to see when gamma dominates theta.
from AlgorithmImports import *
import math


class IVCrushExtended(QCAlgorithm):

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
        # Sample at these bars (every 30 min = 30 bars)
        self.sample_bars = set(range(30, 361, 30))
        # key = (bar_bucket, regime, right) -> list of dicts
        self.data = {}
        self.reported = False

    def OnData(self, data):
        current_date = self.Time.date()
        if current_date != self.day_date:
            self.day_date = current_date
            self.bar_count = 0
            self.day_highs = []
            self.day_lows = []
        self.bar_count += 1
        if data.Bars is not None and self.spx.Symbol in data.Bars:
            sb = data.Bars[self.spx.Symbol]
            self.day_highs.append(float(sb.High))
            self.day_lows.append(float(sb.Low))
        if not self.reported and str(current_date) == "2025-12-19" and self.bar_count > 350:
            self.reported = True
            self.report()
            return
        if self.bar_count not in self.sample_bars:
            return
        if data.OptionChains is None or len(data.OptionChains) == 0:
            return
        spx_price = self.spx.Price
        if spx_price <= 0:
            return
        # Compute regime from session range so far
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
            for contract, right in [(best_call, "C"), (best_put, "P")]:
                if contract is None:
                    continue
                iv = getattr(contract, "ImpliedVolatility", 0)
                greeks = getattr(contract, "Greeks", None)
                if not iv or iv <= 0:
                    continue
                if greeks is None:
                    continue
                theta = getattr(greeks, "Theta", 0)
                gamma = getattr(greeks, "Gamma", 0)
                delta = getattr(greeks, "Delta", 0)
                mid = 0
                if contract.BidPrice > 0 and contract.AskPrice > 0:
                    mid = (contract.BidPrice + contract.AskPrice) / 2.0
                key = (self.bar_count, regime, right)
                if key not in self.data:
                    self.data[key] = []
                self.data[key].append({
                    "iv": float(iv),
                    "theta": float(theta),
                    "gamma": float(gamma),
                    "delta": float(delta),
                    "mid": float(mid),
                    "sr": float(sr),
                })
            break

    def report(self):
        if not self.data:
            self.Error("NO DATA")
            return
        parts = []
        bars = sorted(set(k[0] for k in self.data.keys()))
        for b in bars:
            mtc = 390 - b
            for regime in ["hi", "lo"]:
                for right in ["C", "P"]:
                    key = (b, regime, right)
                    if key not in self.data:
                        continue
                    readings = self.data[key]
                    n = len(readings)
                    avg_iv = sum(r["iv"] for r in readings) / n
                    avg_t = sum(r["theta"] for r in readings) / n
                    avg_g = sum(r["gamma"] for r in readings) / n
                    avg_d = sum(r["delta"] for r in readings) / n
                    avg_mid = sum(r["mid"] for r in readings) / n
                    # theta per minute of close (how fast decaying)
                    t_per_min = avg_t / mtc if mtc > 0 else 0
                    parts.append(
                        "b" + str(b) + regime + right +
                        " n=" + str(n) +
                        " iv=" + str(round(avg_iv, 4)) +
                        " t=" + str(round(avg_t, 2)) +
                        " g=" + str(round(avg_g, 5)) +
                        " d=" + str(round(avg_d, 3)) +
                        " mid=" + str(round(avg_mid, 1)) +
                        " tpm=" + str(round(t_per_min, 4))
                    )
        self.Error(" | ".join(parts))
