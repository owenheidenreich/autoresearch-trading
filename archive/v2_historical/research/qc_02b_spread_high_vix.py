# SPX Spread Model - HIGH VIX period (Aug 2024 Japan carry trade)
from AlgorithmImports import *
from collections import defaultdict


class SPXSpreadHighVix(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 8, 1)
        self.SetEndDate(2024, 8, 16)
        self.SetCash(100000)
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        self.vix = self.AddIndex("VIX", Resolution.Minute)
        option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-20, 20).Expiration(0, 5))
        self.spreads = defaultdict(list)
        self.total_obs = 0
        self.days_sampled = set()
        self.bar_count = 0
        self.day_date = None
        self.sample_bars = set(range(30, 361, 30))
        self.reported = False
        self.vix_readings = []

    def OnData(self, data):
        current_date = self.Time.date()
        if current_date != self.day_date:
            self.day_date = current_date
            self.bar_count = 0
        self.bar_count += 1
        if self.bar_count not in self.sample_bars:
            if not self.reported and str(current_date) == "2024-08-16" and self.bar_count > 350:
                self.reported = True
                self.report()
            return
        if data.OptionChains is None or len(data.OptionChains) == 0:
            return
        spx_price = self.spx.Price
        if spx_price <= 0:
            return
        vix_price = self.vix.Price if self.vix.Price > 0 else 20.0
        self.vix_readings.append(vix_price)
        if vix_price < 20:
            vix_bucket = "low_vix"
        elif vix_price < 30:
            vix_bucket = "med_vix"
        else:
            vix_bucket = "high_vix"
        if self.bar_count <= 60:
            time_bucket = "open_hour"
        elif self.bar_count <= 300:
            time_bucket = "midday"
        else:
            time_bucket = "close_hour"
        for sym, chain in data.OptionChains.items():
            for c in chain:
                bid = c.BidPrice
                ask = c.AskPrice
                if bid <= 0 or ask <= 0 or ask <= bid:
                    continue
                spread = ask - bid
                moneyness_pct = abs(c.Strike - spx_price) / spx_price * 100
                if moneyness_pct < 0.5:
                    m_bucket = "ATM"
                elif moneyness_pct < 1.5:
                    m_bucket = "OTM5"
                elif moneyness_pct < 3.0:
                    m_bucket = "OTM10"
                else:
                    m_bucket = "OTM20+"
                key = (m_bucket, time_bucket, vix_bucket)
                self.spreads[key].append(spread)
                self.total_obs += 1
        self.days_sampled.add(current_date)

    def report(self):
        m_order = ["ATM", "OTM5", "OTM10", "OTM20+"]
        t_order = ["open_hour", "midday", "close_hour"]
        v_order = ["low_vix", "med_vix", "high_vix"]
        avg_vix = round(sum(self.vix_readings) / len(self.vix_readings), 1) if self.vix_readings else 0
        parts = ["SPREAD_HIVIX days=" + str(len(self.days_sampled)) + " obs=" + str(self.total_obs) + " avgVIX=" + str(avg_vix)]
        for m in m_order:
            for t in t_order:
                for v in v_order:
                    key = (m, t, v)
                    vals = self.spreads.get(key, [])
                    if vals:
                        vals_sorted = sorted(vals)
                        n = len(vals_sorted)
                        median = vals_sorted[n // 2]
                        parts.append(m + "|" + t + "|" + v + "=$" + str(round(median, 2)) + "/n=" + str(n))
        all_vals = []
        for v in self.spreads.values():
            all_vals.extend(v)
        if all_vals:
            all_sorted = sorted(all_vals)
            n = len(all_sorted)
            parts.append("OVERALL=$" + str(round(all_sorted[n // 2], 2)))
        self.Error(" | ".join(parts))
