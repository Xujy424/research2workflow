"""宏观监控数据访问：本地矩阵读取及标准化的 EDB/WSD 日期序列。"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlencode
import warnings

import numpy as np
import pandas as pd

try:
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
except ImportError:  # 支持直接从 IDE 运行本文件
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT


from v2.Monitor.v3.config import INDUSTRY_MAP


class MonitorContext:
    """提供本地 row/matrix 与 EDB/WSD 序列读取，不处理指标计算。"""

    def __init__(self, root=ROOT, data=None, *, edb_reader=None, wsd_reader=None,
                 client_factory=None, snapshot_root=None,
                 snapshot_pattern="proc/{date}/{exchange}_shot_1m.pq"):
        self._owns_data = data is None
        self.data = DataPool(root, asset="stock") if data is None else data
        self.dates = pd.DatetimeIndex(self.data.axis.trade_dates)
        self.ticks = np.asarray(self.data.axis.ticks)
        self.edb_reader, self.wsd_reader = edb_reader, wsd_reader
        self._client_factory = client_factory
        self._client = None
        self._edb_cache = {}
        self._wsd_cache = {}
        self.snapshot_root = Path(snapshot_root or (Path(self.data.asset_root) / "l2"))
        self.snapshot_pattern = snapshot_pattern
        self._snapshot_cache = OrderedDict()
        self._aligned_snapshot_cache = OrderedDict()
        self._feature_cache = OrderedDict()

    def read(self, source, key, start, end=None, *, kind="series", options=None):
        """统一读取本地矩阵或 EDB/WSD 日期序列。local 区间使用交易日位置。"""
        source, kind = source.lower(), kind.lower()
        if source == "local":
            if kind == "series" and str(key).startswith("option/"):
                path = self.data.root / f"{key}.bin"
                values = np.memmap(
                    path, dtype=np.float32, mode="r",
                    shape=(len(self.data.axis.full_dates), 1),
                )[:len(self.dates), 0]
                return pd.Series(values, index=self.dates, name=Path(key).stem)
            if kind != "matrix" or end is None:
                raise ValueError("local 数据需使用 kind='matrix' 并传入 start/end 位置")
            return self.matrix(key, int(start), int(end))
        if kind != "series" or end is None:
            raise ValueError(f"{source} 数据需使用 kind='series' 并传入日期 start/end")
        if source == "edb":
            return self.edb(key, start, end)
        if source == "wsd":
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("wsd 的 key 应为 (代码, 指标字段)")
            code, field = key
            return self.wsd(code, field, start, end, options=options)
        raise ValueError(f"未知数据源: {source}")

    def get_weight(self, domain, name, end, base=None):
        """返回域内权重；指数用成分权重，其余域用流通市值。"""
        if domain == "index":
            weight = self.read("local", f"index/weight/{name}_weight", 0, end, kind="matrix")
            return weight
        base = self.read("local", "d_essentials/circ_mv", 0, end, kind="matrix")
        if domain == "industry":
            mask = self.read("local", "industry/industry", 0, end, kind="matrix")
            valid = (mask == INDUSTRY_MAP[name]) & np.isfinite(base) & (base > 0)
            weight = np.where(valid, base, np.nan)
            total = np.nansum(weight, axis=1, keepdims=True)
            return np.divide(weight, total, out=np.full_like(weight, np.nan), where=total > 0)
        elif domain == "style":
            mask = self.read("local", f"stock/barra/{name}_mask", 0, end, kind="matrix")
            valid = np.isfinite(base) & (base > 0) & np.isfinite(mask) & (mask != 0)
            weight = np.where(valid, base * mask, np.nan)
            positive_total = np.nansum(np.where(weight > 0, weight, 0.0), axis=1, keepdims=True)
            negative_total = np.nansum(np.where(weight < 0, -weight, 0.0), axis=1, keepdims=True)
            positive = np.divide(weight, positive_total, out=np.full_like(weight, np.nan), where=(weight > 0) & (positive_total > 0))
            negative = np.divide(weight, negative_total, out=np.full_like(weight, np.nan), where=(weight < 0) & (negative_total > 0))
            return np.where(weight > 0, positive, np.where(weight < 0, negative, np.nan))
        return base
    
    def get_position(self, date):
        """返回任意日期在本地交易轴上对应的索引；非交易日映射到前一交易日。"""
        pos = int(self.dates.searchsorted(pd.Timestamp(date).normalize(), side="right") - 1)
        if pos < 0:
            raise ValueError(f"估值日早于本地交易轴: {date}")
        return pos

    def row(self, field, index):
        """按本地交易日索引读取一行；分钟字段沿分钟轴求和。"""
        value = np.asarray(self.data.read(field, self.dates[index]), dtype=float)
        return np.nansum(value, axis=0) if value.ndim == 2 else value

    def matrix(self, field, start, end):
        """按闭区间交易日索引读取矩阵；分钟字段逐日沿分钟轴求和。"""
        value = np.asarray(
            self.data.read(field, self.dates[end], self.dates[start]), dtype=float
        )
        return np.nansum(value, axis=1) if value.ndim == 3 else value

    @staticmethod
    def _to_series(frame, value_col, start=None, end=None):
        """将 EDB/WSD 响应统一为日期索引数值序列，可选按请求区间截取。"""
        if isinstance(frame, pd.Series):
            values = frame.copy()
        else:
            frame = pd.DataFrame(frame)
            if frame.empty:
                return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            date_col = next((name for name in ("date", "FDate", "Date") if name in frame), None)
            value_col = value_col if value_col in frame else next(
                (name for name in ("value", "Value") if name in frame), None
            )
            if date_col is None or value_col is None:
                return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            values = frame.set_index(date_col)[value_col]
        values.index = pd.to_datetime(values.index).normalize()
        values = pd.to_numeric(values, errors="coerce").sort_index()
        if start is not None or end is not None:
            start = pd.Timestamp(start) if start is not None else None
            end = pd.Timestamp(end) if end is not None else None
            return values.loc[start:end]
        return values

    def _request_client(self):
        if self._client is None:
            if self._client_factory is None:
                try:
                    from ...GetData.ZSFundRequestClient import RequestZSFundDateServer
                except ImportError:
                    from v2.GetData.ZSFundRequestClient import RequestZSFundDateServer
                self._client = RequestZSFundDateServer()
            else:
                self._client = self._client_factory()
        return self._client

    def edb(self, code, start, end):
        """缓存标准化后的 EDB 全序列，并返回请求区间内的数据。"""
        if code not in self._edb_cache:
            if self.edb_reader is not None:
                frame = self.edb_reader(code, IDname="value")
            else:
                url = "http://dataservice/api/EDBData/New?" + urlencode({"code": code})
                response = self._request_client().request_client(url, None, "get")
                frame = pd.DataFrame(response[0]["CodeData"])
            self._edb_cache[code] = self._to_series(frame, "value")
        return self._to_series(self._edb_cache[code], "value", start, end).copy()

    def wsd(self, code, field, start, end, *, options=None):
        """读取 WSD 区间数据并返回标准日期序列。"""
        start = pd.Timestamp(start).strftime("%Y-%m-%d")
        end = pd.Timestamp(end).strftime("%Y-%m-%d")
        key = (code, field, start, end, options)
        if key in self._wsd_cache:
            return self._wsd_cache[key].copy()
        if self.wsd_reader is not None:
            params = dict(secCodes=code, indicator=field, startDate=start, endDate=end)
            if options is not None:
                params["param"] = options
            frame = self.wsd_reader(**params)
        else:
            params = dict(secCodes=code, indicator=field, startDate=start, endDate=end,
                          isFromCache="false", param=options or "Period=D")
            url = "http://dataservice/api/WSDQuant/New?" + urlencode(params)
            response = self._request_client().request_client(url, None, "get")
            frame = pd.DataFrame(response[0]["CodeData"])
        values = self._to_series(frame, field, start, end)
        self._wsd_cache[key] = values
        if len(self._wsd_cache) > 128:
            self._wsd_cache.pop(next(iter(self._wsd_cache)))
        return values.copy()

    def close(self):
        """释放本上下文创建的本地数据池和缓存。"""
        self._edb_cache.clear()
        self._wsd_cache.clear()
        self._snapshot_cache.clear()
        self._aligned_snapshot_cache.clear()
        self._feature_cache.clear()
        if self._owns_data:
            self.data.close()

    def snapshots(self, date, exchanges=("sh", "sz")):
        """读取并合并指定日期的交易所盘口快照。"""
        day = pd.Timestamp(date).strftime("%Y%m%d")
        key = (day, tuple(exchanges))
        if key not in self._snapshot_cache:
            import polars as pl

            frames = []
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
            value = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            self._snapshot_cache[key] = value
            if len(self._snapshot_cache) > 2:
                self._snapshot_cache.popitem(last=False)
        return self._snapshot_cache[key]

    def aligned_snapshot(self, date, topn=5):
        """将盘口表对齐到本地证券轴，并缓存转换结果。"""
        from .snapshots import align_snapshot

        day = pd.Timestamp(date).strftime("%Y%m%d")
        key = (day, topn)
        if key not in self._aligned_snapshot_cache:
            self._aligned_snapshot_cache[key] = align_snapshot(
                self.snapshots(date), self.data.axis.ticks, topn
            )
            if len(self._aligned_snapshot_cache) > 2:
                self._aligned_snapshot_cache.popitem(last=False)
        return self._aligned_snapshot_cache[key]

    def l2_table(self, date, suffix, *, columns=None, lazy=False, exchanges=('sh', 'sz'),
                 securities=None):
        """合并沪深加工L2表，仅保留连续竞价；无文件时返回None。
        SecurityID统一为六位字符串，TransactTime统一为pl.Time，Exchange保留来源。
        上午[09:30,11:30]，下午[13:00,14:57)，不排序、不改变原始事件顺序。
        lazy=True返回查询计划，供大逐笔表先聚合/关联再流式执行。
        """
        from datetime import time
        import polars as pl

        folder = self.snapshot_root / 'proc' / pd.Timestamp(date).strftime('%Y%m%d')
        tables = []
        for exchange in exchanges:
            path = folder / f'{exchange}{suffix}.pq'
            if not path.exists():
                continue
            frame = pl.scan_parquet(path)
            if columns is not None:
                frame = frame.select(columns)
            stamp = pl.col('TransactTime')
            schema = frame.collect_schema()
            if securities is not None and schema['SecurityID'].is_integer():
                frame = frame.filter(pl.col('SecurityID').is_in([int(code) for code in securities]))
            dtype = schema['TransactTime']
            if dtype == pl.Time:
                clock = stamp
            elif dtype.base_type() == pl.Datetime:
                clock = stamp.dt.time()
            else:
                clock = stamp.cast(pl.String).str.extract(
                    r'(\d{2}:\d{2}:\d{2}(?:\.\d+)?)', 1).str.to_time(strict=False)
            code = pl.col('SecurityID').cast(pl.String)
            if not schema['SecurityID'].is_integer():
                code = code.str.strip_chars().str.extract(r'^(\d{1,6})', 1)
            frame = frame.with_columns(
                code.str.zfill(6).alias('SecurityID'),
                clock.alias('TransactTime'),
                pl.lit(exchange).alias('Exchange'))
            if securities is not None and not schema['SecurityID'].is_integer():
                frame = frame.filter(pl.col('SecurityID').is_in(securities))
            stamp = pl.col('TransactTime')
            tables.append(frame.filter(
                ((stamp >= time(9, 30)) & (stamp <= time(11, 30)))
                | ((stamp >= time(13)) & (stamp < time(14, 57)))))
        if not tables:
            return None
        table = pl.concat(tables, how='diagonal_relaxed')
        return table if lazy else table.collect(engine='streaming')


Context = MonitorContext
