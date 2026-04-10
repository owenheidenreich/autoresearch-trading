# Greeks capture: single-day SPXW 0DTE
# Paste into QC Algorithm Lab
#
# CHANGE THIS DATE for each run (one run per day):
#   Run 1: 2025-12-15 (Mon)
#   Run 2: 2025-12-16 (Tue)
#   Run 3: 2025-12-17 (Wed)
#   Run 4: 2025-12-18 (Thu)
#   Run 5: 2025-12-19 (Fri)
#
# Output: pipe-delimited records with QC Greeks + BS inputs
# Save each run's output, then run compare_greeks.py on all of them.

from AlgorithmImports import *

# ---- CHANGE THIS DATE ----
TARGET_YEAR = 2025
TARGET_MONTH = 12
TARGET_DAY = 15
# ---------------------------


class Greeks1Day(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(TARGET_YEAR, TARGET_MONTH, TARGET_DAY)
        self.SetEndDate(TARGET_YEAR, TARGET_MONTH, TARGET_DAY)
        self.SetCash(100000)

        self.spx = self.AddIndex("SPX", Resolution.Minute)

        # SPXW must be added as a SEPARATE index -- AddIndexOption on SPX
        # only returns standard SPX options (Friday expiry), not SPXW dailies.
        # Confirmed by qc_10: Approach C (AddIndex("SPXW")) is what made
        # 0DTE data appear in data.OptionChains.
        spxw = self.AddIndex("SPXW", Resolution.Minute)
        option = self.AddIndexOption(spxw.Symbol, Resolution.Minute)
        option.SetFilter(
            lambda u: u.IncludeWeeklys().Strikes(-5, 5).Expiration(0, 0)
        )

        self.bar_count = 0
        self.readings = []
        self.reported = False
        self.sample_bars = set([10, 30, 60, 120, 180, 240, 300, 360, 380])

    def OnData(self, data):
        self.bar_count += 1

        if not self.reported and self.bar_count > 385:
            self.reported = True
            self._report()
            return

        if self.bar_count not in self.sample_bars:
            return

        if data.OptionChains is None or len(data.OptionChains) == 0:
            return

        spx_price = self.spx.Price
        if spx_price <= 0:
            return

        today = self.Time.date()

        for chain_sym, chain in data.OptionChains.items():
            contracts = [
                c for c in chain
                if c.Expiry.date() == today
            ]
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

                bid = float(contract.BidPrice) if contract.BidPrice else 0
                ask = float(contract.AskPrice) if contract.AskPrice else 0
                mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0
                mtc = 390 - self.bar_count

                self.readings.append(
                    str(today)
                    + "|" + str(self.bar_count)
                    + "|" + label
                    + "|" + str(round(float(spx_price), 2))
                    + "|" + str(round(float(contract.Strike), 1))
                    + "|" + str(round(mid, 2))
                    + "|" + str(round(bid, 2))
                    + "|" + str(round(ask, 2))
                    + "|" + str(round(float(iv), 6))
                    + "|" + str(round(float(delta), 6))
                    + "|" + str(round(float(gamma), 6))
                    + "|" + str(round(float(theta), 4))
                    + "|" + str(round(float(vega), 6))
                    + "|" + str(mtc)
                    + "|1"
                )
            break

    def _report(self):
        if not self.readings:
            self.Error("NO_DATA n_bars=" + str(self.bar_count))
            return
        header = "date|bar|right|SPX|strike|mid|bid|ask|IV|delta|gamma|theta|vega|mtc|0dte"
        self.Error(header + " || " + " || ".join(self.readings))
