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

from functools import lru_cache
from urllib.parse import urlencode
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import warnings
from .utils import _clean, _time_key, _matrix_from_indexed, safe_divide, _sum_preserve_nan
from .config import DEFAULT_SERIES

try:
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
except ImportError:  # 直接以脚本路径运行时没有 package parent
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT

from v2.Monitor.dailyupdate.config import INDEX_MAP, INDUSTRY_MAP, INDEX_CODES, DEFAULT_EDB_CODES, DEFAULT_STYLE_MASKS



class MonitorContext:
    """唯一监控取数上下文：本地矩阵、EDB/WSD、L2快照及共享缓存。"""
    def __init__(self, root=ROOT, data=None, *, wsd_reader=None, edb_reader=None,
                     edb_codes=None, industry_map=None, style_masks=None, client_factory=None,
                     series_map=None, series_reader=None, snapshot_root=None,
                     snapshot_pattern="proc/{date}/{exchange}_shot_1m.pq", l2_root=None):
            self._owns_data = data is None
            self.data = data if data is not None else DataPool(root, asset='stock')
            self.dates = pd.DatetimeIndex(self.data.axis.trade_dates)
            self.ticks = np.asarray(self.data.axis.ticks)
            self._tick_positions = np.arange(len(self.ticks))
            self.wsd_reader, self.edb_reader = wsd_reader, edb_reader
            self.client_factory, self._client = client_factory, None
            self.edb_codes = {**DEFAULT_EDB_CODES, **(edb_codes or {})}
            self.industry_map = INDUSTRY_MAP.copy() if industry_map is None else dict(industry_map)
            self.style_masks = {**DEFAULT_STYLE_MASKS, **(style_masks or {})}
            self.series_map = {**DEFAULT_SERIES, **(series_map or {})}
            self.series_reader = series_reader
            self.snapshot_root = Path(snapshot_root or (Path(self.data.asset_root) / 'l2'))
            self.snapshot_pattern = snapshot_pattern
            self.l2_root = Path(l2_root or (Path(self.data.asset_root) / 'l2'))
            self._series_cache, self._matrix_cache, self._edb_raw_cache = {}, {}, {}
            self._snapshot_cache, self._aligned_snapshot_cache = {}, {}
            self._asof = None

    def position(self, date):
            """取不晚于date的最近交易日索引。"""
            return int(self.dates.searchsorted(pd.Timestamp(date), side='right') - 1)


    @lru_cache(maxsize=1024)
    def row(self, field, index):
            value = np.asarray(self.data.read(field, self.dates[index]), dtype=float)
            value = np.take(value, self._tick_positions, axis=-1)
            # Intraday fields are stored as [minute, stock].  Monitor metrics are
            # daily, so aggregate the minute axis when a 3-D field is read.
            if value.ndim == 2:
                value = np.nansum(value, axis=0)
            return value

    def matrix(self, field, start, end):
            key = (field, int(start), int(end))
            if key not in self._matrix_cache:
                # 与旧面板不同，新更新器对数据源错误直接报错，不把文件损坏吞成NaN。
                self._matrix_cache[key] = np.stack([self.row(field, j) for j in range(start, end + 1)])
            return self._matrix_cache[key]


    @staticmethod
    def wind_code(tick):
            """将本地六位证券代码转换为 Wind 代码；已有后缀的代码原样保留。"""
            code = str(tick)
            if '.' not in code and len(code) == 6 and code.isdigit():
                return f'{code}.SH' if code.startswith(('5', '6', '9')) else f'{code}.SZ'
            return code

    def stock_weights(self, index, universe=None, industry=None):
            """全A使用流通市值；指数使用历史成分权重，行业可二次过滤。"""
            name = INDEX_MAP.get(universe, universe)
            field = 'd_essentials/circ_mv' if name in (None, 'all', 'self') else f'index/weight/{name}_weight'
            weight = self.row(field, index).copy()
            if industry is not None:
                industry_id = self.industry_map.get(industry, industry)
                weight[self.row('industry/industry', index) != industry_id] = np.nan
            return np.where(weight > 0, weight, np.nan)

    def stock_weight_matrix(self, start, end, universe=None, industry=None):
            """按日期切片读取股票权重，返回形状为[日期, 股票]的矩阵。"""
            name = INDEX_MAP.get(universe, universe)
            field = 'd_essentials/circ_mv' if name in (None, 'all', 'self') else f'index/weight/{name}_weight'
            weight = self.matrix(field, start, end).copy()
            if industry is not None:
                industry_id = self.industry_map.get(industry, industry)
                codes = self.matrix('industry/industry', start, end)
                weight[codes != industry_id] = np.nan
            return np.where(weight > 0, weight, np.nan)

    def domain_weights(self, start, end, domain):
            """Return daily stock weights for index, style, or industry domains."""
            if isinstance(domain, str) and domain.startswith("barra"):
                try:
                    factor = list(self.style_masks)[int(domain[5:]) - 1]
                    domain = factor
                except (ValueError, IndexError):
                    pass
            if domain in (None, '全市场', 'all', 'self'):
                return self.stock_weight_matrix(start, end)
            if domain in self.industry_map:
                return self.stock_weight_matrix(start, end, industry=domain)
            if domain in self.style_masks:
                mask = self.matrix(self.style_masks[domain], start, end)
                weight = self.stock_weight_matrix(start, end)
                return np.where(mask > 0, weight, np.nan)
            return self.stock_weight_matrix(start, end, universe=domain)

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
                value = value_col if value_col in frame else next(
                    (x for x in ('value', 'Value') if x in frame), None
                )
                if date_col is None or value is None:
                    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
                result = frame.set_index(date_col)[value]
            result.index = pd.to_datetime(result.index).normalize()
            result = pd.to_numeric(result, errors='coerce').sort_index()
            return result.loc[pd.Timestamp(start):pd.Timestamp(end)]

    def wsd(self, code, field, start, end, *, options=None):
            """调用 WSDQuant/New，返回单证券日期序列。"""
            start, end = pd.Timestamp(start).strftime('%Y-%m-%d'), pd.Timestamp(end).strftime('%Y-%m-%d')
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
            return self._series(frame, field, start, end)

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

    def index_close(self, universe, start, end):
            """官方指数点位，避免使用成分收益重构点位。"""
            code = INDEX_CODES.get(INDEX_MAP.get(universe, universe))
            if code is None:
                return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            return self.wsd(code, 'close', start, end)

    def bond_10y(self, start, end):
            """利率单位按适配器约定输入小数，如2%=0.02。"""
            code = self.edb_codes.get('bond_10y')
            if code is None:
                return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            return self.edb(code, start, end)

    def close(self):
        self._matrix_cache.clear()
        self._edb_raw_cache.clear()
        self._series_cache.clear()
        self._snapshot_cache.clear()
        self._aligned_snapshot_cache.clear()
        self.row.cache_clear()
        if self._owns_data:
            self.data.close()

    def series(self, name, start, end):
            key = (name, pd.Timestamp(start), pd.Timestamp(end))
            if key in self._series_cache:
                return self._series_cache[key]
            spec = self.series_map.get(name)
            if self.series_reader is not None:
                value = self.series_reader(name, start, end)
            elif not spec or not spec.get("code"):
                value = pd.Series(dtype=float)
            elif spec.get("source", "edb") == "wsd":
                value = self.wsd(spec["code"], spec.get("field", "close"), start, end)
            else:
                value = self.edb(spec["code"], start, end)
            value = _clean(value).loc[pd.Timestamp(start):pd.Timestamp(end)]
            self._series_cache[key] = value
            return value

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
            self._asof = date
            self._matrix_cache.clear()
            self._series_cache.clear()
            self._snapshot_cache.clear()
            self._aligned_snapshot_cache.clear()



def align_snapshot(frame, ticks, topn=5):
    """将宽表快照对齐到 [分钟, 股票]，盘口深度保留 topn 档、OFI 仅算一档。"""
    if frame is None or frame.empty:
        return None
    source = frame.copy()
    source["_tick"] = source["SecurityID"].astype(str).str.zfill(6).str[:6]
    source["_time"] = source["BarTime"].map(_time_key)
    source = source[
        ((source["_time"] >= "09:30:00.000000") & (source["_time"] <= "11:30:00.000000"))
        | ((source["_time"] >= "13:00:00.000000") & (source["_time"] <= "14:57:00.000000"))
    ]
    if source.empty:
        return None
    source = source.sort_values(["_tick", "_time"]).drop_duplicates(["_tick", "_time"], keep="last")
    times = np.array(sorted(source["_time"].unique()), dtype=object)
    shape = (len(times), len(ticks))
    stock_codes = [str(tick).zfill(6)[:6] for tick in ticks]
    indexed = source.set_index(["_time", "_tick"]).reindex(pd.MultiIndex.from_product([times, stock_codes]))
    matrices = {name: [] for name in ("bid_px", "bid_qty", "ask_px", "ask_qty")}
    columns = {"bid_px": "BidPrice", "bid_qty": "BidQty", "ask_px": "AskPrice", "ask_qty": "AskQty"}
    for level in range(1, topn + 1):
        for name, prefix in columns.items():
            matrices[name].append(_matrix_from_indexed(indexed, f"{prefix}{level}", shape))
    bid_px, bid_qty, ask_px, ask_qty = (np.asarray(matrices[k]) for k in columns)
    midpoint = (bid_px[0] + ask_px[0]) / 2
    spread = safe_divide(ask_px[0] - bid_px[0], midpoint) * 10000
    depth = _sum_preserve_nan(bid_qty, 0) + _sum_preserve_nan(ask_qty, 0)
    best_imbalance = safe_divide(bid_qty[0] - ask_qty[0], bid_qty[0] + ask_qty[0])
    ofi = np.full_like(midpoint, np.nan)
    if len(times) > 1:
        bp, bq, ap, aq = bid_px[0], bid_qty[0], ask_px[0], ask_qty[0]
        bid_flow = np.where(bp[1:] > bp[:-1], bq[1:], np.where(bp[1:] < bp[:-1], -bq[:-1], bq[1:] - bq[:-1]))
        ask_flow = np.where(ap[1:] < ap[:-1], aq[1:], np.where(ap[1:] > ap[:-1], -aq[:-1], aq[1:] - aq[:-1]))
        valid = np.isfinite(bp[1:]) & np.isfinite(bp[:-1]) & np.isfinite(ap[1:]) & np.isfinite(ap[:-1])
        ofi[1:] = np.where(valid, bid_flow - ask_flow, np.nan)
    return {"times": times, "mid": midpoint, "spread": spread, "depth": depth,
            "best_imbalance": best_imbalance, "ofi": ofi,
            "bid_px": bid_px, "bid_qty": bid_qty, "ask_px": ask_px, "ask_qty": ask_qty}

def _daily_return(context, index):
    """读取 d_essentials/pct 并转换成收益率小数，pct 原值为百分数。"""
    return np.asarray(context.row("d_essentials/pct", index), float) / 100.0

# 旧路径的兼容名称指向同一个类，不存在额外Context实现。
MarketContext = MonitorContext
