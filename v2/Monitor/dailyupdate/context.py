"""市场数据访问层。DataPool 读取本地 TN；外部数据兼容 get_EDB_data / get_WSD_data。

原 get_edb_wsd.py 在模块级执行了一次 WSD 测试请求，直接 import 会发起网络访问。
因此默认通过同一 ZSFundRequestClient 接口调用相同的 EDB/WSD 服务端点，
不导入那个带副作用的脚本。也支持注入原函数：
    MarketContext(edb_reader=get_EDB_data, wsd_reader=get_WSD_data)
接口和返回格式完全按上传的 get_edb_wsd.py 处理。
"""
from __future__ import annotations

if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'v2.Monitor.dailyupdate'

from collections import OrderedDict
from pathlib import Path
import sys
from threading import RLock
from urllib.parse import urlencode
import warnings

import numpy as np
import pandas as pd

from .utils import _clean
from .config import DEFAULT_SERIES
from .snapshots import align_snapshot

try:
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
except ImportError:  # 直接以脚本路径运行时没有 package parent
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT

from v2.Monitor.dailyupdate.config import INDEX_MAP, INDUSTRY_MAP, DEFAULT_EDB_CODES, DEFAULT_STYLE_MASKS



class MonitorContext:
    """唯一监控取数上下文：本地矩阵、EDB/WSD、L2快照及共享缓存。"""
    def __init__(self, root=ROOT, data=None, *, wsd_reader=None, edb_reader=None,
                 edb_codes=None, industry_map=None, style_masks=None, client_factory=None,
                 series_map=None, series_reader=None, snapshot_root=None,
                 snapshot_pattern="proc/{date}/{exchange}_shot_1m.pq"):
        self._owns_data = data is None
        self.data = data if data is not None else DataPool(root, asset='stock')
        self.dates = pd.DatetimeIndex(self.data.axis.trade_dates)
        self.ticks = np.asarray(self.data.axis.ticks)
        self.wsd_reader, self.edb_reader = wsd_reader, edb_reader
        self.client_factory, self._client = client_factory, None
        self.edb_codes = {**DEFAULT_EDB_CODES, **(edb_codes or {})}
        self.industry_map = INDUSTRY_MAP.copy() if industry_map is None else dict(industry_map)
        self.style_masks = {**DEFAULT_STYLE_MASKS, **(style_masks or {})}
        self.series_map = {**DEFAULT_SERIES, **(series_map or {})}
        self.series_reader = series_reader
        self.snapshot_root = Path(snapshot_root or (Path(self.data.asset_root) / 'l2'))
        self.snapshot_pattern = snapshot_pattern

        # 按生命周期区分缓存：日内缓存随估值日清理，EDB原始序列跨日复用。
        self._row_cache, self._matrix_cache = OrderedDict(), {}
        self._series_cache, self._edb_raw_cache = {}, {}
        self._snapshot_cache, self._aligned_snapshot_cache = {}, {}
        self._asof = None
        self.factor_cache = OrderedDict()
        self._wsd_cache = OrderedDict()
        self._cache_lock = RLock()

    def row(self, field, index):
        key = (field, int(index))
        if key in self._row_cache:
            self._row_cache.move_to_end(key)
            return self._row_cache[key]
        value = np.asarray(self.data.read(field, self.dates[index]), dtype=float)
        if value.ndim == 2:
            value = np.nansum(value, axis=0)
        self._row_cache[key] = value
        if len(self._row_cache) > 1024:
            self._row_cache.popitem(last=False)
        return value

    def matrix(self, field, start, end):
        key = (field, int(start), int(end))
        if key not in self._matrix_cache:
            # read 支持闭区间批量读取，参数顺序为结束日、开始日。
            value = np.asarray(self.data.read(field, self.dates[end], self.dates[start]), dtype=float)
            # 分钟字段为[日期, 分钟, 股票]，保持与 row 一致的日度汇总口径。
            if value.ndim == 3:
                value = np.nansum(value, axis=1)
            self._matrix_cache[key] = value
        return self._matrix_cache[key]
    
    def stock_weights(self, universe, start, end=None):
        """指数取成分权重；行业和style域按掩码筛选后取流通市值权重。"""
        if isinstance(universe, str) and universe.startswith('barra'):
            universe = list(self.style_masks)[int(universe[5:]) - 1]

        # 先确定权重来源：全市场、行业和style用市值，指数用成分权重。
        field = 'd_essentials/circ_mv'
        mask_field, industry_id = None, None
        if universe in (None, '全市场', 'all', 'self'):
            pass
        elif universe in self.industry_map:
            mask_field = 'industry/industry'
            industry_id = self.industry_map[universe]
        elif universe in self.style_masks:
            mask_field = self.style_masks[universe]
        else:
            name = INDEX_MAP.get(universe, universe)
            field = f'index/weight/{name}_weight'

        weight = self.row(field, start) if end is None else self.matrix(field, start, end)
        valid = weight > 0
        if mask_field is not None:
            # 行业按编码匹配，style取正组；掩码值本身不作为权重。
            mask = self.row(mask_field, start) if end is None else self.matrix(mask_field, start, end)
            selected = mask == industry_id if industry_id is not None else mask > 0
            valid &= selected
        return np.where(valid, weight, np.nan)


    def _request_client(self):
        """第一次远端请求时初始化客户端。可注入 client_factory 适配项目实际包路径。"""
        if self._client is None:
            if self.client_factory is None:
                try:
                    from ...GetData.ZSFundRequestClient import RequestZSFundDateServer
                except ImportError:
                    from v2.GetData.ZSFundRequestClient import RequestZSFundDateServer
                self._client = RequestZSFundDateServer()
            else:
                self._client = self.client_factory()
        return self._client

    @staticmethod
    def _series(frame, value_col, start, end):
        """把 get_EDB_data/get_WSD_data 的 date/value 表统一转为日期索引 Series。"""
        if isinstance(frame, pd.Series):
            result = frame.copy()
        else:
            frame = pd.DataFrame(frame)
            if frame.empty:
                return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            date_col = next((x for x in ('date', 'FDate', 'Date') if x in frame), None)
            value = value_col if value_col in frame else next((x for x in ('value', 'Value') if x in frame), None)
            if date_col is None or value is None:
                return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            result = frame.set_index(date_col)[value]
        result.index = pd.to_datetime(result.index).normalize()
        result = pd.to_numeric(result, errors='coerce').sort_index()
        return result.loc[pd.Timestamp(start):pd.Timestamp(end)]

    def wsd(self, code, field, start, end, *, options=None):
        """调用 WSDQuant/New，返回单证券日期序列。"""
        start, end = pd.Timestamp(start).strftime('%Y-%m-%d'), pd.Timestamp(end).strftime('%Y-%m-%d')
        key = (code, field, start, end, options)
        with self._cache_lock:
            cached = self._wsd_cache.get(key)
        if cached is not None:
            return cached.copy()
        if self.wsd_reader is not None:
            kwargs = dict(secCodes=code, indicator=field, startDate=start, endDate=end)
            if options is not None:
                kwargs['param'] = options
            frame = self.wsd_reader(**kwargs)
        else:
            params = dict(secCodes=code, indicator=field, startDate=start,
                          endDate=end, isFromCache='false', param=options or 'Period=D')
            url = 'http://dataservice/api/WSDQuant/New?' + urlencode(params)
            response = self._request_client().request_client(url, None, 'get')
            frame = pd.DataFrame(response[0]['CodeData']).rename(columns={'FDate': 'date', 'Value': field})
        result = self._series(frame, field, start, end)
        with self._cache_lock:
            self._wsd_cache[key] = result
            if len(self._wsd_cache) > 128:
                self._wsd_cache.popitem(last=False)
        return result.copy()

    def edb(self, code, start, end):
        # EDB接口本身返回全序列，历史循环只请求一次，返回前仍严格截断到估值日。
        if code not in self._edb_raw_cache:
            if self.edb_reader is not None:
                frame = self.edb_reader(code, IDname='value')
            else:
                response = self._request_client().request_client('http://dataservice/api/EDBData/New?' + urlencode({'code': code}), None, 'get')
                frame = pd.DataFrame(response[0]['CodeData']).rename(columns={'FDate': 'date', 'Value': 'value'})
            self._edb_raw_cache[code] = self._series(frame, 'value', '1900-01-01', '2200-12-31')
        return self._edb_raw_cache[code].loc[pd.Timestamp(start):pd.Timestamp(end)].copy()

    def _clear_daily_cache(self):
        """清除只对当前估值日有效的缓存。"""
        self._row_cache.clear()
        self._matrix_cache.clear()
        self._series_cache.clear()
        self._wsd_cache.clear()
        self._snapshot_cache.clear()
        self._aligned_snapshot_cache.clear()

    def close(self):
        self._clear_daily_cache()
        self.factor_cache.clear()
        self._edb_raw_cache.clear()
        if self._owns_data:
            self.data.close()

    def snapshots(self, date, exchanges=("sh", "sz")):
        day = pd.Timestamp(date).strftime("%Y%m%d")
        if day not in self._snapshot_cache:
            frames = []
            import polars as pl
            for exchange in exchanges:
                path = self.snapshot_root / self.snapshot_pattern.format(date=day, exchange=exchange)
                if not path.exists():
                    continue
                try:
                    frame = pl.read_parquet(path).to_pandas()
                except Exception as exc:
                    warnings.warn(f"盘口快照读取失败 {path}: {exc}", RuntimeWarning)
                    continue
                if not frame.empty:
                    frame["SecurityID"] = frame["SecurityID"].astype(str).str.zfill(6)
                    frames.append(frame)
            self._snapshot_cache[day] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if len(self._snapshot_cache) > 2:
                self._snapshot_cache.pop(next(iter(self._snapshot_cache)))
        return self._snapshot_cache[day]

    def aligned_snapshot(self, date, topn=5):
        day = pd.Timestamp(date).strftime("%Y%m%d")
        key = (day, topn)
        if key not in self._aligned_snapshot_cache:
            self._aligned_snapshot_cache[key] = align_snapshot(self.snapshots(date), self.ticks, topn)
            if len(self._aligned_snapshot_cache) > 2:
                self._aligned_snapshot_cache.pop(next(iter(self._aligned_snapshot_cache)))
        return self._aligned_snapshot_cache[key]

    def l2_path(self, date, exchange, suffix):
        return self.snapshot_root / "proc" / pd.Timestamp(date).strftime("%Y%m%d") / f"{exchange}{suffix}.pq"

    def begin_day(self, date):
        # 严格要求目标日位于日期轴；不静默将周末替换成前一交易日。
        date = pd.Timestamp(date).normalize()
        if date not in self.dates:
            raise ValueError(f'目标日不在交易日期轴: {date.date()}')
        if self._asof == date:
            # 显式重跑同一天时刷新源数据，不能让旧缓存掩盖新增或修正的观测。
            self.factor_cache.clear()
            self._edb_raw_cache.clear()
        self._asof = date
        self._clear_daily_cache()

    def window(self, date, lookback):
        """返回含目标日的本地行区间；直接调用指标时也会设置估值日期。"""
        date = pd.Timestamp(date).normalize()
        if self._asof != date:
            self.begin_day(date)
        end = int(self.dates.searchsorted(date, side='right') - 1)
        return max(0, end - int(lookback)), end

    def trading_series(self, name, start, end):
        """按配置读取外部序列并对齐交易日，只前填已出现的观测。"""
        key = (name, pd.Timestamp(start), pd.Timestamp(end))
        if key not in self._series_cache:
            spec = self.series_map.get(name)
            if self.series_reader is not None:
                values = self.series_reader(name, start, end)
            elif not spec or not spec.get("code"):
                values = pd.Series(dtype=float)
            elif spec.get("source", "edb") == "wsd":
                values = self.wsd(spec["code"], spec.get("field", "close"), start, end)
            else:
                values = self.edb(spec["code"], start, end)
            self._series_cache[key] = _clean(values).loc[pd.Timestamp(start):pd.Timestamp(end)]
        values = self._series_cache[key]
        dates = self.dates[(self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))]
        return values.reindex(values.index.union(dates).sort_values()).ffill().reindex(dates)

    def cache_factor(self, key, values):
        """缓存已算好的单日个股数据，供窗口变体复用，不识别具体指标种类。"""
        self.factor_cache[key] = values
        self.factor_cache.move_to_end(key)
        if len(self.factor_cache) > 4096:
            self.factor_cache.popitem(last=False)
        return values

    def eligible_domains(self, domains, scope='all'):
        """根据配置的域范围取交集，避免把未支持的域伪装成全市场。"""
        from .config import INDEX_DOMAINS, STYLE_DOMAINS
        groups = {
            'all': ('全市场',) + INDEX_DOMAINS + STYLE_DOMAINS + tuple(self.industry_map),
            'wide': ('全市场',) + INDEX_DOMAINS + STYLE_DOMAINS,
            'index': ('全市场',) + INDEX_DOMAINS,
            'market_industry': ('全市场',) + INDEX_DOMAINS + tuple(self.industry_map),
            'market': ('全市场',),
        }
        if scope not in groups:
            raise ValueError(f'未知域范围: {scope}')
        return tuple(domain for domain in domains if domain in groups[scope])

    def domain_pair_weights(self, start, end, domain):
        """读取域权重；barra域返回高低组两份权重，其他域只返回一份。"""
        if isinstance(domain, str) and domain.startswith('barra'):
            factor = list(self.style_masks)[int(domain[5:]) - 1]
            mask = self.matrix(self.style_masks[factor], start, end)
            weights = self.stock_weights(None, start, end)
            return np.where(mask > 0, weights, np.nan), np.where(mask < 0, weights, np.nan)
        return self.stock_weights(domain, start, end), None



# 旧路径的兼容名称指向同一个类，不存在额外Context实现。
MarketContext = MonitorContext
