# Find SPXW 0DTE: try multiple approaches to get daily-expiry SPX options
# Paste into QC Algorithm Lab
#
# Problem: AddIndexOption("SPX") with IncludeWeeklys() only returns
# Friday-expiry SPX weeklies (sym: SPX 251219C...), not SPXW dailies.
#
# This script tries 3 approaches on a single day (Wed Dec 17, 2025)
# to find true 0DTE contracts:
#
# Approach A: Expiration(0,0) instead of (0,5)
# Approach B: OptionChainProvider to manually request the chain
# Approach C: AddIndexOption on "SPXW" ticker directly
#
# We pick Wed because SPXW 0DTE should definitely exist on Wednesdays.

from AlgorithmImports import *


class FindSPXW(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2025, 12, 17)
        self.SetEndDate(2025, 12, 17)
        self.SetCash(100000)

        self.spx = self.AddIndex("SPX", Resolution.Minute)

        # Approach A: standard AddIndexOption with Expiration(0,0)
        self.opt_a = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        self.opt_a.SetFilter(
            lambda u: u.IncludeWeeklys().Strikes(-3, 3).Expiration(0, 0)
        )

        # Approach C: try SPXW as a separate index option
        # This may fail -- that's fine, we catch it
        self.spxw_ok = False
        try:
            spxw_idx = self.AddIndex("SPXW", Resolution.Minute)
            self.opt_c = self.AddIndexOption(spxw_idx.Symbol, Resolution.Minute)
            self.opt_c.SetFilter(
                lambda u: u.IncludeWeeklys().Strikes(-3, 3).Expiration(0, 5)
            )
            self.spxw_ok = True
        except Exception as e:
            self.spxw_ok = False
            self.spxw_err = str(e)[:100]

        self.bar_count = 0
        self.reported = False
        self.results = []

    def OnData(self, data):
        self.bar_count += 1

        # Sample at bar 60 (10:30 AM)
        if self.bar_count == 60:
            self._sample_approach_a(data)
            self._sample_approach_b()
            if self.spxw_ok:
                self._sample_approach_c(data)

        # Report near close
        if not self.reported and self.bar_count > 385:
            self.reported = True
            self._report()

    def _sample_approach_a(self, data):
        """Approach A: Expiration(0,0) filter"""
        tag = "A_Exp00"
        if data.OptionChains is None or len(data.OptionChains) == 0:
            self.results.append(tag + "|NO_CHAINS")
            return

        for chain_sym, chain in data.OptionChains.items():
            contracts = list(chain)
            if not contracts:
                self.results.append(tag + "|EMPTY_CHAIN")
                continue

            # Dump first 6 contracts
            for c in contracts[:6]:
                expiry = str(c.Expiry.date())
                dte = (c.Expiry.date() - self.Time.date()).days
                sym = str(c.Symbol)[-30:]
                right = "C" if c.Right == OptionRight.Call else "P"
                mid = 0
                if c.BidPrice > 0 and c.AskPrice > 0:
                    mid = round((c.BidPrice + c.AskPrice) / 2.0, 2)
                self.results.append(
                    tag + "|" + right
                    + "|K=" + str(round(float(c.Strike), 1))
                    + "|exp=" + expiry
                    + "|dte=" + str(dte)
                    + "|mid=" + str(mid)
                    + "|" + sym
                )
            # Count total and unique expiries
            all_exp = set(str(c.Expiry.date()) for c in contracts)
            self.results.append(
                tag + "|TOTAL=" + str(len(contracts))
                + "|UNIQUE_EXP=" + ",".join(sorted(all_exp))
            )
            break

    def _sample_approach_b(self):
        """Approach B: OptionChainProvider -- request chain directly"""
        tag = "B_Provider"
        try:
            chain = self.OptionChainProvider.GetOptionContractList(
                self.spx.Symbol, self.Time
            )
            contracts = list(chain)
            if not contracts:
                self.results.append(tag + "|EMPTY")
                return

            # Find contracts expiring today
            today = self.Time.date()
            today_contracts = [
                c for c in contracts
                if c.ID.Date.date() == today
            ]
            # Find contracts expiring tomorrow
            tomorrow_contracts = [
                c for c in contracts
                if (c.ID.Date.date() - today).days == 1
            ]

            # Get all unique expiry dates
            all_exp = sorted(set(str(c.ID.Date.date()) for c in contracts))

            self.results.append(
                tag + "|TOTAL=" + str(len(contracts))
                + "|TODAY=" + str(len(today_contracts))
                + "|TOMORROW=" + str(len(tomorrow_contracts))
                + "|EXPIRIES=" + ",".join(all_exp[:10])  # first 10
            )

            # If we found today-expiry contracts, dump a few
            for c in today_contracts[:4]:
                sym = str(c)[-30:]
                self.results.append(
                    tag + "_0DTE|" + sym
                    + "|exp=" + str(c.ID.Date.date())
                    + "|K=" + str(c.ID.StrikePrice)
                    + "|R=" + str(c.ID.OptionRight)
                )

        except Exception as e:
            self.results.append(tag + "|ERROR=" + str(e)[:80])

    def _sample_approach_c(self, data):
        """Approach C: SPXW as separate index"""
        tag = "C_SPXW"
        if data.OptionChains is None:
            self.results.append(tag + "|NO_CHAINS")
            return

        found = False
        for chain_sym, chain in data.OptionChains.items():
            sym_str = str(chain_sym)
            if "SPXW" not in sym_str.upper() and "SPX" not in sym_str.upper():
                continue
            contracts = list(chain)
            if not contracts:
                self.results.append(tag + "|EMPTY for " + sym_str[:30])
                continue
            found = True
            all_exp = set(str(c.Expiry.date()) for c in contracts)
            self.results.append(
                tag + "|" + sym_str[:20]
                + "|TOTAL=" + str(len(contracts))
                + "|EXPIRIES=" + ",".join(sorted(all_exp))
            )
            for c in contracts[:4]:
                expiry = str(c.Expiry.date())
                dte = (c.Expiry.date() - self.Time.date()).days
                self.results.append(
                    tag + "|" + str(c.Symbol)[-30:]
                    + "|exp=" + expiry + "|dte=" + str(dte)
                )
            break

        if not found:
            self.results.append(tag + "|NO_MATCHING_CHAIN")

    def _report(self):
        if not self.results:
            self.Error("NO RESULTS")
            return

        # Add SPXW status
        if not self.spxw_ok:
            err = getattr(self, "spxw_err", "unknown")
            self.results.insert(0, "C_SPXW_INIT|FAILED=" + err)

        self.Error(" || ".join(self.results))
