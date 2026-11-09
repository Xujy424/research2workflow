"""Read saved moneyflow indicators, report coverage and plot without filling gaps."""
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def main():
    root = Path(os.environ.get('RESEARCH_DATA_ROOT', 'Z:/'))
    output = Path(__file__).resolve().parent
    axis = np.load(root / 'axis/dates.npy', allow_pickle=False)
    positions = np.flatnonzero(~np.isnat(axis))
    dates = pd.DatetimeIndex(axis[positions])
    selected = dates >= pd.Timestamp('2024-01-01')
    names = ['active_buy_ratio', 'large_trade_share',
             'medium_trade_share', 'small_trade_share']
    frame = pd.DataFrame(index=dates[selected])
    metadata = {}
    for name in names:
        path = root / 'median_variables/monitor/moneyflow' / f'{name}.bin'
        if path.stat().st_size != len(axis) * 4:
            raise ValueError(f'Unexpected single-series file size: {path}')
        data = np.memmap(path, dtype=np.float32, mode='r', shape=(len(axis),))
        frame[name] = np.array(data[positions[selected]], dtype=float)
        metadata[name] = pd.Timestamp(path.stat().st_mtime, unit='s').isoformat()
        del data
    # A reserved trading-date axis may extend past the saved observations.
    finite = np.isfinite(frame)
    last_observed = frame.index[finite.any(axis=1)].max()
    frame = frame.loc[:last_observed]
    summaries, gaps = [], []
    for name in names:
        for year, series in frame[name].groupby(frame.index.year):
            ok = np.isfinite(series)
            observed = series[ok]
            summaries.append(dict(indicator=name, year=year, days=len(series),
                finite=int(ok.sum()), missing=int((~ok).sum()),
                coverage=float(ok.mean()), first_valid=observed.index.min(),
                last_valid=observed.index.max(), minimum=observed.min(),
                median=observed.median(), maximum=observed.max(),
                file_mtime=metadata[name]))
            gaps.extend(dict(indicator=name, date=d) for d in series.index[~ok])
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'coverage_since_2024.csv', index=False)
    pd.DataFrame(gaps, columns=['indicator', 'date']).to_csv(
        output / 'missing_since_2024.csv', index=False)
    frame.to_csv(output / 'series_since_2024.csv', index_label='date')
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for ax, name in zip(axes, names):
        ax.plot(frame.index, frame[name], linewidth=.8)
        ax.set_title(name, loc='left')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.25)
        missing = frame.index[~np.isfinite(frame[name])]
        ax.plot(missing, np.full(len(missing), .025), '|', color='red',
                transform=ax.get_xaxis_transform(), markersize=5)
    fig.suptitle(f'Moneyflow | 2024-01-01 to {last_observed.date()} | red marks: missing')
    fig.tight_layout()
    fig.savefig(output / 'moneyflow_since_2024.png', dpi=160)
    plt.close(fig)
    print(summary.drop(columns='file_mtime').to_string(index=False))
    print('Saved to', output)


if __name__ == '__main__':
    main()
