# Expiry Diagnostic: figure out what QC returns for 0DTE SPXW
# Paste into QC Algorithm Lab
#
# Problem: qc_08 showed all options with 0dte=0, meaning
# contract.Expiry.date() != self.Time.date(). We need to know:
# 1. What expiry dates does QC actually return for SPXW?
# 2. How do we filter for true 0DTE?
# 3. Does the OPRA symbol contain the expiry date?
#
# This script samples one bar per day and dumps all contract details.

from AlgorithmImports import *


class ExpiryDiag(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2025, 12, 15)
        self.SetEndDate(2025, 12, 19)
        self.SetCash(100000)

        self.spx = self.AddIndex("SPX", Resolution.Minute)
        option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        # Try multiple filter configs
        option.SetFilter(
            lambda u: u.IncludeWeeklys().Strikes(-3, 3).Expiration(0, 5)
        )

        self.day_date = None
        self.bar_count = 0
        self.readings = []
        self.reported = False

    def OnData(self, data):
        current_date = self.Time.date()
        if current_date != self.day_date:
            self.day_date = current_date
            self.bar_count = 0
        self.bar_count += 1

        # Report on last day
        if (not self.reported
                and str(current_date) == "2025-12-19"
                and self.bar_count > 385):
            self.reported = True
            self._report()
            return

        # Sample bar 60 on each day (10:30 AM, good liquidity)
        if self.bar_count != 60:
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

            # For each contract, dump: symbol, strike, expiry, right, DTE
            for c in contracts[:20]:  # cap at 20 to fit in one Error
                expiry_str = str(c.Expiry.date())
                sym_str = str(c.Symbol)
                dte = (c.Expiry.date() - current_date).days
                is_0dte = 1 if c.Expiry.date() == current_date else 0
                mid = 0
                if c.BidPrice > 0 and c.AskPrice > 0:
                    mid = round((c.BidPrice + c.AskPrice) / 2.0, 2)

                right = "C" if c.Right == OptionRight.Call else "P"

                self.readings.append(
                    str(current_date)
                    + "|" + right
                    + "|K=" + str(round(float(c.Strike), 1))
                    + "|exp=" + expiry_str
                    + "|dte=" + str(dte)
                    + "|0d=" + str(is_0dte)
                    + "|mid=" + str(mid)
                    + "|sym=" + sym_str[-30:]  # last 30 chars of symbol
                )
            break

    def _report(self):
        if not self.readings:
            self.Error("NO DATA")
            return
        self.Error(" || ".join(self.readings))
