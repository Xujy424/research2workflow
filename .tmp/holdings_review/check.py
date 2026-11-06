import importlib.util
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v2.Monitor.v3.moneyflow import fund_holdings_similarity as calculate


class Context:
    def __init__(self, funds=8, stocks=12, days=45, lag=1):
        rng = np.random.default_rng(84)
        self.dates = pd.bdate_range('2024-01-02', periods=days)
        self.ticks = [str(i).zfill(6) for i in range(stocks)]
        self.ret = rng.normal(0, .01, (days, stocks))
        self.nv = np.cumprod(1 + rng.normal(0, .005, (days, funds)), axis=0)
        self.lag = lag
        self.positions = [0, days // 2]
        records = []
        self.snapshots = {}
        for f in range(funds):
            for r in self.positions:
                selected = rng.choice(stocks, min(6, stocks), replace=False)
                weights = rng.uniform(1, 8, len(selected))
                self.snapshots[f, r] = (selected, weights)
                for serial, (s, w) in enumerate(zip(selected, weights)):
                    records.append((f, self.dates[r], self.dates[r+3], serial,
                                    self.ticks[s], w, pd.NaT))
        self.holdings = pd.DataFrame(records, columns=['InnerCode','ReportDate','InfoPublDate',
            'SerialNumber','StockCode','RatioInNV','LastOperationDate'])
        self.nav = pd.DataFrame([(f, self.dates[d], self.dates[d+lag], self.nv[d,f])
            for f in range(funds) for d in range(days-lag)],
            columns=['InnerCode','EndDate','InfoPublDate','UnitNV'])
        self.funds = funds

    def get_position(self, date): return self.dates.get_loc(date)
    def jy(self, sql, params): return (self.holdings if 'MF_StockPortfolioDetail' in sql else self.nav).copy()
    def read(self, *args, **kwargs): return self.ret * 100

    def oracle(self):
        out = np.full(len(self.dates), np.nan)
        price = np.cumprod(1+self.ret, axis=0)
        for d, date in enumerate(self.dates):
            vectors = []
            for f in range(self.funds):
                hf = self.holdings[self.holdings.InnerCode.eq(f)]
                death = hf.LastOperationDate.dropna()
                nf = self.nav[self.nav.InnerCode.eq(f)]
                last = death.iloc[0] if len(death) else nf.InfoPublDate.max()
                if date > last: continue
                eligible = hf[hf.InfoPublDate.le(date)]
                if eligible.empty: continue
                report = eligible.ReportDate.max()
                base = nf[nf.EndDate.eq(report) & nf.InfoPublDate.le(date)]
                current = nf[nf.InfoPublDate.le(date)].sort_values('EndDate')
                if base.empty or current.empty: continue
                r = self.dates.get_loc(report)
                s, w = self.snapshots[f,r]
                v = np.zeros(len(self.ticks))
                v[s] = w*price[d,s]/price[r,s]*base.UnitNV.iloc[0]/current.UnitNV.iloc[-1]
                vectors.append(v)
            if len(vectors)>1:
                out[d] = np.mean([np.minimum(a,b).sum()/np.maximum(a,b).sum()
                    for i,a in enumerate(vectors) for b in vectors[i+1:]])
        return out


c = Context()
c.holdings.loc[c.holdings.InnerCode.eq(0),'LastOperationDate'] = c.dates[18]
c.nav = c.nav[~(c.nav.InnerCode.eq(1) & c.nav.InfoPublDate.gt(c.dates[25]))]
actual = calculate(c,c.dates[-1]).to_numpy()
np.testing.assert_allclose(actual,c.oracle(),rtol=1e-12,atol=1e-12,equal_nan=True)
print('PASS: report-date NAV with disclosure lag, unequal weights, replacement, explicit and inferred expiry')

# Later duplicate disclosure must not double holdings; old reports cannot replace a newer report.
extra = c.holdings[c.holdings.InnerCode.eq(2) & c.holdings.ReportDate.eq(c.dates[0])].copy()
extra['InfoPublDate'] = c.dates[35]
c.holdings = pd.concat([c.holdings,extra],ignore_index=True)
np.testing.assert_allclose(calculate(c,c.dates[-1]),actual,rtol=1e-12,atol=1e-12,equal_nan=True)
print('PASS: duplicate/late disclosure does not change weights')

spec = importlib.util.spec_from_file_location('before',Path(__file__).with_name('moneyflow_before.py'))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
c = Context(funds=120,stocks=100,days=40,lag=0)
timings = {}
for name, fn in [('before',old.fund_holdings_similarity),('after',calculate)]:
    t = time.perf_counter(); y = fn(c,c.dates[-1]); timings[name] = time.perf_counter()-t
    if name == 'before': expected = y
    else: np.testing.assert_allclose(y,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
print('PASS: unchanged mathematical definition on synchronized data')
print('benchmark (120 funds,100 stocks,40 dates):',timings)
