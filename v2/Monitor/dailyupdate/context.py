"""市场数据访问层。DataPool 读取本地 TN；外部数据兼容 get_EDB_data / get_WSD_data。

原 get_edb_wsd.py 在模块级执行了一次 WSD 测试请求，直接 import 会发起网络访问。
因此默认通过同一 ZSFundRequestClient 接口调用相同的 EDB/WSD 服务端点，
不导入那个带副作用的脚本。也支持注入原函数：
    MarketContext(edb_reader=get_EDB_data, wsd_reader=get_WSD_data)
接口和返回格式完全按上传的 get_edb_wsd.py 处理。
"""
from __future__ import annotations
from functools import lru_cache
from urllib.parse import urlencode
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
except ImportError:  # 直接以脚本路径运行时没有 package parent
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT

INDEX_MAP = {
    'hs300': 'hs300', 'zz500': 'zz500', 'zz1000': 'zz1000',
    'zz2000': 'zz2000', 'zzfull': 'zzfull',
    '沪深300': 'hs300', '中证500': 'zz500', '中证1000': 'zz1000',
    '中证2000': 'zz2000', '全A': 'zzfull',
}
INDUSTRY_MAP = {
    '商贸零售': 0, '轻工制造': 1, '汽车': 2, '美容护理': 3, '房地产': 4,
    '国防军工': 5, '通信': 6, '煤炭': 7, '交通运输': 8, '公用事业': 9,
    '机械设备': 10, '电力设备': 11, '环保': 12, '食品饮料': 13, '计算机': 14,
    '纺织服饰': 15, '家用电器': 16, '医药生物': 17, '钢铁': 18, '社会服务': 19,
    '有色金属': 20, '非银金融': 21, '综合': 22, '建筑装饰': 23, '农林牧渔': 24,
    '银行': 25, '传媒': 26, '基础化工': 27, '建筑材料': 28, '石油石化': 29, '电子': 30,
}
INDEX_CODES = {
    'hs300': '000300.SH', 'zz500': '000905.SH',
    'zz1000': '000852.SH', 'zz2000': '932000.CSI',
    'zzfull': '000985.SH',
}


class MarketContext:
    """读取股票TN字段和官方时间序列。所有返回的TN行按dates、ticks严格对齐。"""

    def __init__(self, root=ROOT, data=None, *, wsd_reader=None, edb_reader=None,
                 edb_codes=None, industry_map=None, style_masks=None,
                 client_factory=None):
        self._owns_data = data is None
        self.data = data if data is not None else DataPool(root, asset='stock')
        self.dates = pd.DatetimeIndex(self.data.axis.trade_dates)
        self.ticks = np.asarray(self.data.axis.ticks)
        self._tick_positions = np.arange(len(self.ticks))
        self.wsd_reader = wsd_reader
        self.client_factory = client_factory
        self._client = None
        self.edb_reader = edb_reader
        self.edb_codes = edb_codes or {
            'bond_10y': 'S0059749',
            'margin_balance': 'M0075990',
        }
        self.industry_map = industry_map or INDUSTRY_MAP.copy()
        # 字段路径可由实际DataPool命名习惯覆盖；mask取值：高10%=1，低10%=-1。
        self.style_masks = style_masks or {
            **{name: f'barra/{name}_mask' for name in (
                'beta', 'btop', 'size', 'nonlinear_size', 'momentom',
                'residual_vol', 'liquidity', 'leverage', 'growth1',
                'earnings_yield')},
            'growth_value': 'barra/growth1_mask',
            'pe': 'valuation/pe_mask',
            'pb': 'valuation/pb_mask',
        }

    def position(self, date):
        """取不晚于date的最近交易日索引。"""
        return int(self.dates.searchsorted(pd.Timestamp(date), side='right') - 1)

    @lru_cache(maxsize=8192)
    def row(self, field, index):
        value = np.asarray(self.data.read(field, self.dates[index]), dtype=float)
        value = np.take(value, self._tick_positions, axis=-1)
        # Intraday fields are stored as [minute, stock].  Monitor metrics are
        # daily, so aggregate the minute axis when a 3-D field is read.
        if value.ndim == 2:
            value = np.nansum(value, axis=0)
        return value

    def matrix(self, field, start, end):
        """含起止索引；输出形状[日期, 股票]。"""
        try:
            return np.stack([self.row(field, j) for j in range(start, end + 1)])
        except (FileNotFoundError, PermissionError, KeyError, ValueError):
            return np.full((end - start + 1, len(self.ticks)), np.nan)

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
        """调用 EDBData/New；兼容注入的 get_EDB_data(ID, IDname='rate')。EDB全量请求后按日期截取。"""
        if self.edb_reader is not None:
            frame = self.edb_reader(code, IDname='value')
        else:
            url = 'http://dataservice/api/EDBData/New?' + urlencode({'code': code})
            response = self._request_client().request_client(url, None, 'get')
            frame = pd.DataFrame(response[0]['CodeData']).rename(columns={'FDate': 'date', 'Value': 'value'})
        return self._series(frame, 'value', start, end)

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
        self.row.cache_clear()
        if self._owns_data:
            self.data.close()
