# Greeks Comparison: QC computed Greeks vs Black-Scholes inputs
# Paste into QC Algorithm Lab (free tier)
#
# Purpose: Capture QC's computed Greeks AND the raw inputs needed to
# reproduce our Black-Scholes estimates locally. This lets us measure
# how far off our BS Greeks are from QC's properly computed values.
#
# Key fix vs qc_05: filters for TRUE 0DTE only (expiry == today),
# and outputs SPX price + strike so we can run _bs_greeks() locally.
#
# Output format (pipe-delimited records):
#   date|bar|right|SPX|strike|mid|bid|ask|IV|delta|gamma|theta|vega|mtc
#
# After running: copy self.Error() output to qc_08_results_raw.txt
# Then run compare_greeks.py locally to compute BS Greeks and compare.

from AlgorithmImports import *


class GreeksComparison(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2025, 12, 8)
        self.SetEndDate(2025, 12, 19)
        self.SetCash(100000)

        self.spx = self.AddIndex("SPX", Resolution.Minute)
        option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        # Wide expiration window -- we filter for 0DTE manually below
        option.SetFilter(
            lambda u: u.IncludeWeeklys().Strikes(-5, 5).Expiration(0, 5)
        )

        self.day_date = None
        self.bar_count = 0
        self.readings = []
        self.reported = False

        # Sample at these bars (covers open through near-close)
        # bar 10 = ~9:40, bar 30 = ~10:00, bar 60 = ~10:30,
        # bar 120 = ~11:30, bar 180 = ~12:30, bar 240 = ~1:30,
        # bar 300 = ~2:30, bar 360 = ~3:30, bar 380 = ~3:50
        self.sample_bars = set([10, 30, 60, 120, 180, 240, 300, 360, 380])

    def OnData(self, data):
        current_date = self.Time.date()
        if current_date != self.day_date:
            self.day_date = current_date
            self.bar_count = 0
        self.bar_count += 1

        # Report on last day near close
        if (not self.reported
                and str(current_date) == "2025-12-19"
                and self.bar_count > 385):
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

        for chain_sym, chain in data.OptionChains.items():
            contracts = list(chain)
            if not contracts:
                continue

            # Filter for TRUE 0DTE: expiry must be today
            today_contracts = [
                c for c in contracts
                if c.Expiry.date() == current_date
            ]
            if not today_contracts:
                # Fallback: no 0DTE found, try nearest expiry
                # but tag it so we know
                today_contracts = contracts

            # Find nearest-ATM call and put
            best_call = None
            best_put = None
            min_cd = 99999
            min_pd = 99999
            for c in today_contracts:
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

                # Check if this is truly 0DTE
                is_0dte = 1 if contract.Expiry.date() == current_date else 0

                # Compact format: pipe-delimited, minimal whitespace
                self.readings.append(
                    str(current_date)
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
                    + "|" + str(is_0dte)
                )
            break  # only first chain

    def _report(self):
        if not self.readings:
            self.Error("NO DATA COLLECTED")
            return

        # Header + all readings in one Error call
        header = "date|bar|right|SPX|strike|mid|bid|ask|IV|delta|gamma|theta|vega|mtc|0dte"
        self.Error(header + " || " + " || ".join(self.readings))
