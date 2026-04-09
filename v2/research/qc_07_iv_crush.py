# IV Crush Rate - paste into QC Algorithm Lab
from AlgorithmImports import *
import math


class IVCrush(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2025, 12, 8)
        self.SetEndDate(2025, 12, 19)
        self.SetCash(100000)
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-3, 3).Expiration(0, 5))
        self.day_date = None
        self.bar_count = 0
        # Track IV at each time bucket across days
        # key = bar_bucket -> list of IV readings
        self.iv_by_bar = {}
        self.theta_by_bar = {}
        self.gamma_by_bar = {}
        self.delta_by_bar = {}
        self.sample_bars = set(range(10, 381, 10))
        self.reported = False

    def OnData(self, data):
        current_date = self.Time.date()
        if current_date != self.day_date:
            self.day_date = current_date
            self.bar_count = 0
        self.bar_count += 1
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
        for chain_sym, chain in data.OptionChains.items():
            contracts = list(chain)
            if not contracts:
                continue
            # Find ATM call
            best_call = None
            min_dist = 99999
            for c in contracts:
                if c.Right != OptionRight.Call:
                    continue
                dist = abs(c.Strike - spx_price)
                if dist < min_dist:
                    min_dist = dist
                    best_call = c
            if best_call is None:
                continue
            iv = getattr(best_call, "ImpliedVolatility", 0)
            greeks = getattr(best_call, "Greeks", None)
            if iv and iv > 0:
                b = self.bar_count
                if b not in self.iv_by_bar:
                    self.iv_by_bar[b] = []
                self.iv_by_bar[b].append(float(iv))
            if greeks:
                b = self.bar_count
                theta = getattr(greeks, "Theta", 0)
                gamma = getattr(greeks, "Gamma", 0)
                delta = getattr(greeks, "Delta", 0)
                if b not in self.theta_by_bar:
                    self.theta_by_bar[b] = []
                    self.gamma_by_bar[b] = []
                    self.delta_by_bar[b] = []
                self.theta_by_bar[b].append(float(theta))
                self.gamma_by_bar[b].append(float(gamma))
                self.delta_by_bar[b].append(float(delta))
            break

    def report(self):
        if not self.iv_by_bar:
            self.Error("NO IV DATA")
            return
        parts = []
        sorted_bars = sorted(self.iv_by_bar.keys())
        for b in sorted_bars:
            ivs = self.iv_by_bar[b]
            avg_iv = sum(ivs) / len(ivs)
            mtc = 390 - b
            # Our model: theta_accel = 1/sqrt(mtc)
            model_accel = 1.0 / math.sqrt(max(mtc, 1))
            # Avg theta and gamma
            avg_theta = 0
            avg_gamma = 0
            avg_delta = 0
            if b in self.theta_by_bar:
                tvals = self.theta_by_bar[b]
                avg_theta = sum(tvals) / len(tvals)
            if b in self.gamma_by_bar:
                gvals = self.gamma_by_bar[b]
                avg_gamma = sum(gvals) / len(gvals)
            if b in self.delta_by_bar:
                dvals = self.delta_by_bar[b]
                avg_delta = sum(dvals) / len(dvals)
            parts.append(
                "b" + str(b) +
                " iv=" + str(round(avg_iv, 4)) +
                " t=" + str(round(avg_theta, 1)) +
                " g=" + str(round(avg_gamma, 5)) +
                " d=" + str(round(avg_delta, 3)) +
                " mtc=" + str(mtc) +
                " ta=" + str(round(model_accel, 4))
            )
        self.Error(" | ".join(parts))
