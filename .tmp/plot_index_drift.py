"""Read-only reconstruction from saved close weights and next-session returns."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2.GetData import DataPool
from v2.UpdateData.stock.update_index import _drift_weights, INDEX_CODE_TO_NAME

out = Path(__file__).resolve().parents[1] / 'artifacts/index_drift_check'
out.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
# Daily drift followed by lagged-weight returns must reproduce an unrebalanced portfolio.
r = np.array([[.10, -.10], [-.03, .08], [.02, .01]])
w = np.array([60., 40.])
growth = 1.
for row in r:
    growth *= 1 + np.dot(w / w.sum(), row)
    w = _drift_weights(w, row * 100)
np.testing.assert_allclose(growth, np.dot([.6, .4], np.prod(1+r, axis=0)), atol=1e-12)

fig, axes = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
summaries = []
with DataPool() as pool:
    dates = pd.DatetimeIndex(pool.axis.trade_dates)
    pct = np.asarray(pool.read('d_essentials/pct', dates[-1], dates[0]), dtype=float)
    returns = np.where(np.isfinite(pct), pct/100, 0.)
    daily_output = pd.DataFrame(index=dates)
    levels_output = pd.DataFrame(index=dates)
    for ax, (code, name) in zip(axes.flat, INDEX_CODE_TO_NAME.items()):
        weights = np.asarray(pool.read(f'index/weight/{name}_weight', dates[-1], dates[0]), dtype=float)
        prev = np.where(np.isfinite(weights[:-1]) & (weights[:-1] > 0), weights[:-1], 0.)
        mass = prev.sum(axis=1)
        observed_mass = np.where(np.isfinite(pct[1:]), prev, 0.).sum(axis=1)
        coverage = np.divide(observed_mass, mass, out=np.full(len(mass), np.nan), where=mass>0)
        daily = np.divide((prev*returns[1:]).sum(axis=1), mass,
                          out=np.full(len(mass), np.nan), where=(mass>0)&(observed_mass>0))
        daily_output[name] = np.r_[np.nan, daily]
        daily_output[name+'_return_coverage'] = np.r_[np.nan, coverage]
        valid = np.isfinite(daily) & (daily > -1)
        edges = np.diff(np.r_[False, valid, False].astype(int))
        starts, stops = np.flatnonzero(edges==1), np.flatnonzero(edges==-1)
        info = {'index':name, 'code':code, 'total_return_days':len(daily), 'valid_return_days':int(valid.sum())}
        if not len(starts):
            ax.set_title(name+' (no valid data)'); summaries.append(info); continue
        longest = np.argmax(stops-starts)
        a,b = starts[longest],stops[longest]
        # daily[a] is the return on dates[a+1]; dates[a] is the base close.
        level = 100 * np.r_[1., np.cumprod(1+daily[a:b])]
        levels_output.loc[dates[a:b+1],name] = level
        info.update(start=str(dates[a].date()),end=str(dates[b].date()),
            plotted_return_days=int(b-a),end_level=float(level[-1]),
            min_daily_pct=float(daily[a:b].min()*100),max_daily_pct=float(daily[a:b].max()*100),
            min_return_coverage=float(np.nanmin(coverage[a:b])),
            partial_return_days=int((coverage[a:b]<1-1e-8).sum()),
            previous_weight_sum_min=float(mass[a:b].min()),previous_weight_sum_max=float(mass[a:b].max()))
        ax.plot(dates[a:b+1],level,linewidth=1)
        ax.set_title(f'{name.upper()} ({code}) | {dates[a]:%Y-%m-%d} to {dates[b]:%Y-%m-%d}')
        ax.set_ylabel('Reconstructed level (start = 100)'); ax.grid(alpha=.25)
        summaries.append(info)
        print(json.dumps(info),flush=True)
    daily_output.to_csv(out/'daily_returns.csv',index_label='date')
    levels_output.to_csv(out/'index_levels.csv',index_label='date')
fig.suptitle('Index reconstruction: prior close weights x daily stock returns\nMissing stock returns treated as 0; longest continuous valid interval per index',fontsize=13)
fig.savefig(out/'index_drift_timeseries.png',dpi=160)
plt.close(fig)
(out/'summary.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-started,'results':summaries},indent=2),encoding='utf-8')
print('saved',out,flush=True)
