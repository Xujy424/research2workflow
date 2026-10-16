"""Run the actual update_barra entry point against dated reference snapshots.

Production inputs/outputs are read-only. Real updater output files are written
to a temporary data root, checked with check_barra, then removed. Calculated
values are retained losslessly in calculated_factors.npz with the full axes.
"""
from pathlib import Path
import argparse
import hashlib
import re
import sys
import tempfile
import warnings
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from v2.UpdateData.config import get_jy_conn
from v2.UpdateData.stock import update_barra as implementation
from v2.UpdateData.stock.check_barra import FACTOR_COLUMNS, compare


def run(root, reference_zip, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dates = np.load(root / 'axis/dates.npy', allow_pickle=False)
    ticks = np.load(root / 'axis/stock_ticks.npy', allow_pickle=False)
    references = {}
    with zipfile.ZipFile(reference_zip) as archive:
        for member in sorted(archive.namelist()):
            match = re.search(r'barra(\d{8})\.xlsx$', member)
            if match:
                date = pd.to_datetime(match[1], format='%Y%m%d').strftime('%Y-%m-%d')
                with archive.open(member) as stream:
                    ref = pd.read_excel(stream)
                codes = ref.s_info_windcode.astype(str).str.strip().str.replace(r'^CN', '', regex=True).str.split('.').str[0].str.zfill(6)
                if codes.duplicated().any():
                    raise ValueError(f'Duplicate reference tick: {member}')
                ref.index = codes
                references[date] = ref.reindex(ticks)
    arrays = {name: [] for name in FACTOR_COLUMNS}
    common_rows = []
    shape = (len(dates), len(ticks))
    old_maps = {name: np.memmap(root / 'stock/barra' / f'{name}.bin', mode='r', dtype=np.float64, shape=shape)
                for name in FACTOR_COLUMNS}
    conn = get_jy_conn()
    try:
        with tempfile.TemporaryDirectory(prefix='barra_actual_', dir=output) as temp:
            isolated = Path(temp)
            (isolated / 'axis').mkdir()
            np.save(isolated / 'axis/dates.npy', dates)
            np.save(isolated / 'axis/stock_ticks.npy', ticks)
            for date, ref in references.items():
                # No monkeypatches or audit/candidate calculation helpers.
                values = implementation.update_barra(
                    date, dates, ticks, conn, root / 'stock',
                    output_root=isolated / 'stock',
                )
                dt = int(np.flatnonzero(dates == np.datetime64(date))[0])
                for name, suffix in FACTOR_COLUMNS.items():
                    arr = np.memmap(isolated / 'stock/barra' / f'{name}.bin', mode='r', dtype=np.float64, shape=shape)
                    np.testing.assert_array_equal(arr[dt], values[name])
                    del arr
                    arrays[name].append(values[name])
                    standard = pd.to_numeric(ref['CNE5S_' + suffix], errors='coerce').to_numpy()
                    old = old_maps[name][dt]
                    new = values[name]
                    valid = np.isfinite(old) & np.isfinite(new) & np.isfinite(standard)
                    x, y, z = pd.Series(old[valid]), pd.Series(new[valid]), pd.Series(standard[valid])
                    common_rows.append(dict(date=date, factor=name, common_n=int(valid.sum()),
                        before_pearson=x.corr(z), after_pearson=y.corr(z),
                        before_spearman=x.rank().corr(z.rank()), after_spearman=y.rank().corr(z.rank())))
                np.savez_compressed(output / 'calculated_factors.npz',
                    dates=np.array(list(references)[:len(arrays['size'])], dtype='datetime64[D]'),
                    ticks=ticks, **{name: np.array(rows) for name, rows in arrays.items()})
                print(f'[CALCULATED] actual update_barra: {date}', flush=True)
            # Correlations read the actual binaries written by update_barra.
            after = compare(isolated, reference_zip, output / 'after')
        before = compare(root, reference_zip, output / 'before')
    finally:
        conn.close()
    combined = before.add_prefix('before_').join(after.add_prefix('after_'))
    combined.to_csv(output / 'comparison_summary.csv')
    common = pd.DataFrame(common_rows)
    common.to_csv(output / 'common_daily.csv', index=False)
    common.groupby('factor')[['common_n', 'before_pearson', 'after_pearson', 'before_spearman', 'after_spearman']].mean().to_csv(output / 'common_summary.csv')
    source = Path(implementation.__file__)
    (output / 'implementation_sha256.txt').write_text(
        f'{source}\n{hashlib.sha256(source.read_bytes()).hexdigest()}\n', encoding='utf-8')
    print(combined[['before_pearson_mean', 'after_pearson_mean', 'before_spearman_mean', 'after_spearman_mean', 'after_coverage_mean']].round(4).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'reference-zip', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy')
        run(args.root, args.reference_zip, args.output)
