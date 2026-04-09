# Greeks Validation - paste into QC Algorithm Lab
from AlgorithmImports import *


class GreeksValidation(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2025, 12, 8)
        self.SetEndDate(2025, 12, 19)
        self.SetCash(100000)
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-3, 3).Expiration(0, 5))
        self.day_date = None
        self.bar_count = 0
        self.readings = []
        self.sample_bars = set([30, 120, 240, 360])
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
            for contract, label in [(best_call, "C"), (best_put, "P")]:
                if contract is None:
                    continue
                greeks = getattr(contract, "Greeks", None)
                iv = getattr(contract, "ImpliedVolatility", 0)
                if greeks is None:
                    continue
                delta = getattr(greeks, "Delta", 0)
                gamma = getattr(greeks, "Gamma", 0)
                theta = getattr(greeks, "Theta", 0)
                vega = getattr(greeks, "Vega", 0)
                mid = 0
                if contract.BidPrice > 0 and contract.AskPrice > 0:
                    mid = (contract.BidPrice + contract.AskPrice) / 2.0
                mtc = 390 - self.bar_count
                self.readings.append(
                    str(current_date) + " b" + str(self.bar_count) +
                    " " + label +
                    " d=" + str(round(delta, 4)) +
                    " g=" + str(round(gamma, 5)) +
                    " t=" + str(round(theta, 2)) +
                    " v=" + str(round(vega, 3)) +
                    " iv=" + str(round(iv, 4)) +
                    " mid=" + str(round(mid, 2)) +
                    " mtc=" + str(mtc)
                )
            break

    def report(self):
        if not self.readings:
            self.Error("NO GREEKS DATA")
            return
        # Pack all readings into one message
        # May be long but it's one Error call
        self.Error(" | ".join(self.readings))
