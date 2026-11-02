"""仅负责注册表调度、结果校验和二进制保存；取数计算由各指标脚本完成。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'v2.Monitor.dailyupdate'

from .config import INDICATORS, MonitorConfig, select_indicators
from .context import MonitorContext


def calculate_daily(date, context, *, names=None, modules=None, config=None):
    """逐项调用注册函数，不包含分类汇总，也不读取指标原始数据。"""
    config = config or MonitorConfig()
    context.begin_day(date)
    frames = []
    for spec in select_indicators(names, modules):
        result = spec.load()(context, date, config=config)
        required = {'domain', 'period', *spec.outputs}
        if not isinstance(result, pd.DataFrame) or not required.issubset(result):
            raise ValueError(f'{spec.name}: invalid indicator result schema')
        result = result.reindex(columns=['domain', 'period', 'value', 'percentile']).copy()
        if result.duplicated(['domain', 'period']).any():
            raise ValueError(f'{spec.name}: duplicate domain/period outputs')
        if not result.empty:
            periods = pd.to_numeric(result.period, errors='raise')
            if ((periods < 0) | (periods % 1 != 0)).any():
                raise ValueError(f'{spec.name}: invalid period')
            result['period'] = periods.astype(int)
            if np.isinf(result[['value', 'percentile']].to_numpy(dtype=float)).any():
                raise ValueError(f'{spec.name}: infinite indicator value')
        result.insert(0, 'module', spec.module)
        result.insert(0, 'name', spec.name)
        frames.append(result)
    columns = ['name', 'module', 'domain', 'period', 'value', 'percentile']
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def metric_path(root, name, domain, period, field, config=None):
    """配置作为列定义；域名编码为安全目录，每个数值文件固定一列。"""
    from urllib.parse import quote
    config = config or MonitorConfig()
    spec = INDICATORS[name]
    if field not in ('value', 'percentile') or int(period) < 0:
        raise ValueError('invalid field or period')
    domain = str(domain)
    if not domain or domain in ('.', '..'):
        raise ValueError('invalid domain')
    relative = Path(config.output_folder)
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] != ('median_variables', 'monitor'):
        raise ValueError('output_folder 必须位于 median_variables/monitor 下，以保持轴维护兼容')
    return Path(root) / relative / spec.module / name / quote(domain, safe='-_') / f'p{int(period)}' / f'{field}.bin'


def write_daily(date, result, root, dates, *, config=None):
    """按共享日期轴覆写目标行，预留行填NaN；不自动改变或重排日期轴。"""
    from v2.UpdateData.utils import date_index, ensure_memmap
    row = date_index(date, dates)
    if np.isinf(result[['value', 'percentile']].to_numpy(dtype=float)).any():
        raise ValueError('监控结果含无穷值，未写入文件')
    for record in result.itertuples(index=False):
        # 只保存该类指标实际提供的字段；缺失观测仍写NaN，覆盖之前的旧值。
        fields = INDICATORS[record.name].outputs
        for field in fields:
            value = float(getattr(record, field))
            if np.isinf(value):
                raise ValueError(f'{record.name} 输出无穷值')
            path = metric_path(root, record.name, record.domain, record.period, field, config)
            array = ensure_memmap(path, (len(dates), 1), dtype=np.float32)
            array[row, 0] = value
            array.flush()
            del array


def update_daily(date, root, *, context=None, config=None, names=None, modules=None):
    """每日统一更新入口；可注入同一个Context供历史循环复用。"""
    own = context is None
    context = context or MonitorContext(root=root)
    try:
        # 调用前先验证日期与轴，避免计算完才发现日期不在轴上。
        from v2.UpdateData.utils import date_index
        dates = context.data.axis.full_dates
        date_index(date, dates)
        result = calculate_daily(date, context, names=names, modules=modules, config=config)
        write_daily(date, result, root, dates, config=config)
        return result
    finally:
        if own:
            context.close()


def read_indicator(root, name, domain='全市场', period=1, field='percentile', *, config=None):
    """按本地共享日期轴读取单列监控序列，不暴露预留NaT行。"""
    dates = np.load(Path(root) / 'axis' / 'dates.npy', allow_pickle=False)
    path = metric_path(root, name, domain, period, field, config)
    if path.stat().st_size != len(dates) * np.dtype(np.float32).itemsize:
        raise ValueError(f'文件大小与日期轴不一致: {path}')
    array = np.memmap(path, dtype=np.float32, mode='r', shape=(len(dates), 1))
    valid = ~np.isnat(dates)
    result = pd.Series(np.array(array[valid, 0]), index=pd.DatetimeIndex(dates[valid]), name=name)
    del array
    return result


if __name__ == '__main__':
    from v2.UpdateData.config import ROOT

    # 在IDE中直接修改参数。设为 None 可更新全部已注册指标（需要全部数据源）。
    names = ('growth_value_spread_percentile',)
    start_date = '2018-01-01'
    end_date = '2026-09-30'
    config = MonitorConfig(history=252, periods=(1, 5, 10, 20))
    context = MonitorContext(root=ROOT)
    try:
        for date in context.dates:
            if pd.Timestamp(start_date) <= date <= pd.Timestamp(end_date):
                result = update_daily(date, ROOT, context=context, config=config, names=names)
                print(f'[OK] {date.date()}: {len(result)} 条监控数值', flush=True)
    finally:
        context.close()
