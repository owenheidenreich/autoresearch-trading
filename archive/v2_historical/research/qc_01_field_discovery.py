# SPX Options Field Discovery - paste into QC Algorithm Lab
from AlgorithmImports import *


class SPXFieldDiscovery(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2025, 12, 8)
        self.SetEndDate(2025, 12, 19)
        self.SetCash(100000)
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-10, 10).Expiration(0, 5))
        self.option_symbol = option.Symbol
        self.checked = False
        self.day_count = 0

    def OnData(self, data):
        if self.checked:
            return

        self.day_count += 1

        # Log every 100 bars so we know it is running
        if self.day_count % 100 == 0:
            self.Log("Bar " + str(self.day_count) + " at " + str(self.Time))

        # Try OptionChains
        if data.OptionChains is not None and len(data.OptionChains) > 0:
            for sym, chain in data.OptionChains.items():
                contracts = list(chain)
                if len(contracts) == 0:
                    continue
                self.checked = True
                self.Log("=== FIELD DISCOVERY ===")
                self.Log("Found chain with symbol: " + str(sym))
                self.Log("Number of contracts: " + str(len(contracts)))
                self.Log("Time: " + str(self.Time))
                c = contracts[0]
                self.Log("Symbol: " + str(c.Symbol))
                self.Log("Strike: " + str(c.Strike))
                self.Log("Expiry: " + str(c.Expiry))
                self.Log("Right: " + str(c.Right))
                self.Log("--- PRICE FIELDS ---")
                for attr in ["LastPrice", "Volume", "OpenInterest", "BidPrice", "BidSize", "AskPrice", "AskSize", "UnderlyingLastPrice"]:
                    val = getattr(c, attr, "NOT FOUND")
                    self.Log("  " + attr + ": " + str(val))
                self.Log("--- GREEKS ---")
                greeks = getattr(c, "Greeks", None)
                if greeks is not None:
                    for attr in ["Delta", "Gamma", "Vega", "Theta", "Rho", "ImpliedVolatility", "Lambda"]:
                        val = getattr(greeks, attr, "NOT FOUND")
                        self.Log("  " + attr + ": " + str(val))
                else:
                    self.Log("  Greeks: NOT FOUND")
                self.Log("--- THEORETICAL ---")
                val = getattr(c, "TheoreticalPrice", "NOT FOUND")
                self.Log("  TheoreticalPrice: " + str(val))
                self.Log("--- ALL ATTRIBUTES ---")
                for attr in sorted(dir(c)):
                    if attr.startswith("_"):
                        continue
                    try:
                        val = getattr(c, attr)
                        if not callable(val):
                            self.Log("  " + attr + ": " + str(val))
                    except Exception as e:
                        self.Log("  " + attr + ": ERROR " + str(e))
                self.Log("--- BID/ASK ACROSS CONTRACTS ---")
                for c2 in contracts[:20]:
                    bid = getattr(c2, "BidPrice", 0)
                    ask = getattr(c2, "AskPrice", 0)
                    spread = ask - bid if (bid > 0 and ask > 0) else -1
                    right = "C" if c2.Right == OptionRight.Call else "P"
                    self.Log(str(c2.Strike) + " " + right + " bid=" + str(bid) + " ask=" + str(ask) + " spread=" + str(round(spread, 2)) + " last=" + str(c2.LastPrice) + " vol=" + str(c2.Volume))
                self.Log("=== DISCOVERY COMPLETE ===")
                self.Log("KEY: Do BidPrice/AskPrice have real values above?")
                return

    def OnEndOfAlgorithm(self):
        if not self.checked:
            self.Log("WARNING: No option chain data found in entire period")
            self.Log("Bars processed: " + str(self.day_count))
            self.Log("This may mean SPX index options are not available on the free tier")
