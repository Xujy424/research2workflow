"""监控参数、数据代码与指标注册表；导入时不连接数据服务。"""
from __future__ import annotations
from dataclasses import dataclass, field
from importlib import import_module
from functools import partial
from pathlib import Path

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


PERIODS = (1, 5, 10, 20)
MACRO_PERIODS = (1, 5, 20)

RATE_METRICS = {
    "DR007", "DR007-政策利率利差", "Shibor3M", "中短期票据AA-AAA信用利差",
    "中短期票据AA-国开债信用利差", "10Y国债收益率", "国债期限利差(10Y-1Y)",
    "10Y实际利率代理(Wind一致预期)",
}

SPREAD_METRICS = {
    "DR007-政策利率利差", "中短期票据AA-AAA信用利差",
    "中短期票据AA-国开债信用利差", "国债期限利差(10Y-1Y)",
}

DEFAULT_SERIES = {
    # 2026-10-21 Wind EDB 指标表确认的代码。
    "7D政策利率": {"source": "edb", "code": "M0041653"},
    "DR007": {"source": "edb", "code": "M1006337"},
    "Shibor3M": {"source": "edb", "code": "M0017142"},
    "AAA信用收益率": {"source": "edb", "code": "S0059738"},
    "AA信用收益率": {"source": "edb", "code": "S0059717"},
    "国开3Y收益率": {"source": "edb", "code": "M1004265"},
    "10Y国债收益率": {"source": "edb", "code": "S0059749"},
    "1Y国债收益率": {"source": "edb", "code": "S0059744"},
    "M2同比": {"source": "edb", "code": "M0001385"},
    "M2余额": {"source": "edb", "code": "M0001384"},
    "社融存量": {"source": "edb", "code": "M5525755"},
    "社融存量同比": {"source": "edb", "code": "M5525763"},
    "新增社融": {"source": "edb", "code": "M5206730"},
    "名义GDP": {"source": "edb", "code": "M5567876"},
    # Wind 一致预期代码需要按实际使用的数据商序列补入；支持通过 series_map 覆盖。
    "预期通胀_Wind一致预期": {"source": "edb", "code": "M0061676"},
}

INDEX_DOMAINS = ("沪深300", "中证500", "中证1000", "中证2000")

STYLE_DOMAINS = tuple(f"barra{i}" for i in range(1, 11))

DECAY_HORIZONS = (1, 5, 15, 30)

PATH_HORIZONS = (10, 30, 60, 120)

DEFAULT_THRESHOLDS = {
    "成交额变化": (0.1, 0.9), "top10%成交额占比": (0.1, 0.9),
    "自由流通换手率": (0.1, 0.9),
}

DEFAULT_EDB_CODES = {'bond_10y': 'S0059749', 'margin_balance': 'M0075990'}
DEFAULT_STYLE_MASKS = {
    **{name: f'barra/{name}_mask' for name in ('beta', 'btop', 'size', 'nonlinear_size',
        'momentom', 'residual_vol', 'liquidity', 'leverage', 'growth1', 'earnings_yield')},
    'growth_value': 'barra/growth1_mask', 'pe': 'valuation/pe_mask', 'pb': 'valuation/pb_mask',
}

MICRO_SNAPSHOT_METRICS = ('effective_spread', 'five_level_depth_notional', 'order_flow_imbalance', 'order_flow_autocorrelation_lag1', 'order_flow_autocorrelation_lag5', 'best_level_imbalance', 'jump_frequency_30m', 'acd_jump_intensity', 'estimate_kyle_lambda', 'impact_decay_1m', 'impact_decay_5m', 'impact_decay_15m', 'impact_decay_30m', 'impact_path_10m', 'impact_path_30m', 'impact_path_60m', 'impact_path_120m')

MICRO_DOMAIN_POLICY = {'相对买卖价差': 'wide', '五档盘口深度': 'wide', '标准化OFI': 'wide', '订单流自相关-lag1': 'wide', '订单流自相关-lag5': 'wide', 'B1队列失衡': 'wide', '30分钟跳跃频率': 'wide', 'ACD预期跳跃强度': 'wide', '主动成交不平衡': 'index', 'Amihud非流动性': 'index', '跳跃-成交量偏离': 'index', '订单冲击成本': 'wide', '可成交量': 'wide', '撤单率': 'index'}

MONEYFLOW_FIELDS = {'amount': 'm_essentials/amount', 'float_mv': 'd_essentials/circ_mv', 'large_buy': 'd_moneyflow/buy_lg_amount', 'large_sell': 'd_moneyflow/sell_lg_amount', 'medium_buy': 'd_moneyflow/buy_mid_amount', 'medium_sell': 'd_moneyflow/sell_mid_amount', 'small_buy': 'd_moneyflow/buy_sm_amount', 'small_sell': 'd_moneyflow/sell_sm_amount', 'extra_buy': 'd_moneyflow/buy_elg_amount', 'extra_sell': 'd_moneyflow/sell_elg_amount', 'margin_balance': 'd_moneyflow/margin_balance'}

STYLE_UNIVERSES = ('hs300', 'zz500', 'zz1000', 'zz2000')

RISK_FIELDS = {'returns': 'd_essentials/pct', 'hit_floor': 'basic/hit_floor', 'has_trade': 'basic/hastrade', 'price_floor': 'basic/price_floor'}

@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    module: str
    label: str
    function: str
    section: str = ''
    parameters: dict = field(default_factory=dict)

    def load(self):
        module, name = self.function.split(':')
        function = getattr(import_module('.vals.' + module, __package__), name)
        return partial(function, **self.parameters) if self.parameters else function


@dataclass
class MonitorConfig:
    history: int = 252
    periods: tuple = PERIODS
    domains: tuple = ('全市场',)
    universes: tuple = STYLE_UNIVERSES
    industries: tuple | None = None
    benchmark: str = 'zzfull'
    high_lookback: int = 60
    level_history: int = 252
    rotation_window: int = 5
    output_folder: str = 'median_variables/monitor'
    module_options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.history < 1 or not self.periods or any(p < 1 for p in self.periods):
            raise ValueError('history 和 periods 必须为正整数')



INDICATORS = {
    'macro_dr007': IndicatorSpec(*('macro_dr007', 'macro', 'DR007', 'macro_level:calculate', '', {'series': 'DR007'})),
    'macro_policy_spread': IndicatorSpec(*('macro_policy_spread', 'macro', 'DR007-政策利率利差', 'macro_spread:calculate', '', {'left': 'DR007', 'right': '7D政策利率'})),
    'macro_shibor3m': IndicatorSpec(*('macro_shibor3m', 'macro', 'Shibor3M', 'macro_level:calculate', '', {'series': 'Shibor3M'})),
    'macro_credit_aa_aaa': IndicatorSpec(*('macro_credit_aa_aaa', 'macro', '中短期票据AA-AAA信用利差', 'macro_spread:calculate', '', {'left': 'AA信用收益率', 'right': 'AAA信用收益率'})),
    'macro_credit_aa_cdb': IndicatorSpec(*('macro_credit_aa_cdb', 'macro', '中短期票据AA-国开债信用利差', 'macro_spread:calculate', '', {'left': 'AA信用收益率', 'right': '国开3Y收益率'})),
    'macro_bond10y': IndicatorSpec(*('macro_bond10y', 'macro', '10Y国债收益率', 'macro_level:calculate', '', {'series': '10Y国债收益率'})),
    'macro_term_spread': IndicatorSpec(*('macro_term_spread', 'macro', '国债期限利差(10Y-1Y)', 'macro_spread:calculate', '', {'left': '10Y国债收益率', 'right': '1Y国债收益率'})),
    'macro_real_rate': IndicatorSpec(*('macro_real_rate', 'macro', '10Y实际利率代理(Wind一致预期)', 'macro_spread:calculate', '', {'left': '10Y国债收益率', 'right': '预期通胀_Wind一致预期'})),
    'macro_m2_yoy': IndicatorSpec(*('macro_m2_yoy', 'macro', 'M2同比', 'macro_yoy:calculate', '', {'yoy_name': 'M2同比', 'balance_name': 'M2余额'})),
    'macro_tsf_yoy': IndicatorSpec(*('macro_tsf_yoy', 'macro', '社融存量同比', 'macro_yoy:calculate', '', {'yoy_name': '社融存量同比', 'balance_name': '社融存量'})),
    'macro_credit_impulse': IndicatorSpec(*('macro_credit_impulse', 'macro', '信用脉冲', 'macro_credit_impulse:calculate', '', {})),
    
    'effective_spread': IndicatorSpec(*('effective_spread', 'micro', '相对买卖价差', 'effective_spread:effective_spread', '', {})),
    'five_level_depth_notional': IndicatorSpec(*('five_level_depth_notional', 'micro', '五档盘口深度', 'five_level_depth_notional:five_level_depth_notional', '', {})),
    'order_flow_imbalance': IndicatorSpec(*('order_flow_imbalance', 'micro', '标准化OFI', 'order_flow_imbalance:order_flow_imbalance', '', {})),
    'order_flow_autocorrelation_lag1': IndicatorSpec(*('order_flow_autocorrelation_lag1', 'micro', '订单流自相关-lag1', 'order_flow_autocorrelation:calculate', '', {'lag': 1})),
    'order_flow_autocorrelation_lag5': IndicatorSpec(*('order_flow_autocorrelation_lag5', 'micro', '订单流自相关-lag5', 'order_flow_autocorrelation:calculate', '', {'lag': 5})),
    'best_level_imbalance': IndicatorSpec(*('best_level_imbalance', 'micro', 'B1队列失衡', 'best_level_imbalance:best_level_imbalance', '', {})),
    'jump_frequency_30m': IndicatorSpec(*('jump_frequency_30m', 'micro', '30分钟跳跃频率', 'jump_frequency_30m:jump_frequency_30m', '', {})),
    'acd_jump_intensity': IndicatorSpec(*('acd_jump_intensity', 'micro', 'ACD预期跳跃强度', 'acd_jump_intensity:acd_jump_intensity', '', {})),
    'estimate_kyle_lambda': IndicatorSpec(*('estimate_kyle_lambda', 'micro', 'KyleLambda', 'estimate_kyle_lambda:estimate_kyle_lambda', '', {})),
    'active_trade_imbalance': IndicatorSpec(*('active_trade_imbalance', 'micro', '主动成交不平衡', 'active_trade_imbalance:active_trade_imbalance', '', {})),
    'amihud_illiquidity': IndicatorSpec(*('amihud_illiquidity', 'micro', 'Amihud非流动性', 'amihud_illiquidity:amihud_illiquidity', '', {})),
    'jump_volume_divergence': IndicatorSpec(*('jump_volume_divergence', 'micro', '跳跃-成交量偏离', 'jump_volume_divergence:jump_volume_divergence', '', {})),
    'standardized_acd_intensity': IndicatorSpec(*('standardized_acd_intensity', 'micro', '标准化跳跃强度', 'standardized_acd_intensity:standardized_acd_intensity', '', {})),
    'order_impact_cost': IndicatorSpec(*('order_impact_cost', 'micro', '订单冲击成本', 'order_impact_cost:order_impact_cost', '', {})),
    'executable_capacity': IndicatorSpec(*('executable_capacity', 'micro', '可成交量', 'executable_capacity:executable_capacity', '', {})),
    'daily_cancellation_rate': IndicatorSpec(*('daily_cancellation_rate', 'micro', '撤单率', 'daily_cancellation_rate:daily_cancellation_rate', '', {})),
    'impact_decay_1m': IndicatorSpec(*('impact_decay_1m', 'micro', '冲击保留率_1m', 'impact_decay:calculate', '', {'horizon': 1})),
    'impact_decay_5m': IndicatorSpec(*('impact_decay_5m', 'micro', '冲击保留率_5m', 'impact_decay:calculate', '', {'horizon': 5})),
    'impact_decay_15m': IndicatorSpec(*('impact_decay_15m', 'micro', '冲击保留率_15m', 'impact_decay:calculate', '', {'horizon': 15})),
    'impact_decay_30m': IndicatorSpec(*('impact_decay_30m', 'micro', '冲击保留率_30m', 'impact_decay:calculate', '', {'horizon': 30})),
    'impact_path_10m': IndicatorSpec(*('impact_path_10m', 'micro', '冲击路径_10m', 'impact_path:calculate', '', {'horizon': 10})),
    'impact_path_30m': IndicatorSpec(*('impact_path_30m', 'micro', '冲击路径_30m', 'impact_path:calculate', '', {'horizon': 30})),
    'impact_path_60m': IndicatorSpec(*('impact_path_60m', 'micro', '冲击路径_60m', 'impact_path:calculate', '', {'horizon': 60})),
    'impact_path_120m': IndicatorSpec(*('impact_path_120m', 'micro', '冲击路径_120m', 'impact_path:calculate', '', {'horizon': 120})),
    
    'amount_change': IndicatorSpec(*('amount_change', 'moneyflow', '成交额变化', 'amount_change:amount_change', '', {'baseline': 20})),
    'top_share': IndicatorSpec(*('top_share', 'moneyflow', 'top10%成交额占比', 'top_share:top_share', '', {'fraction': 0.1})),
    'turnover': IndicatorSpec(*('turnover', 'moneyflow', '自由流通换手率', 'turnover:turnover', '', {})),
    'active_buy_ratio': IndicatorSpec(*('active_buy_ratio', 'moneyflow', '主动买入占比', 'active_buy_ratio:active_buy_ratio', '', {})),
    'balance_change': IndicatorSpec(*('balance_change', 'moneyflow', '融资余额变化率', 'balance_change:balance_change', '', {})),
    'large_trade_share': IndicatorSpec(*('large_trade_share', 'moneyflow', '大单成交占比', 'category_share:calculate', '', {})),
    'medium_trade_share': IndicatorSpec(*('medium_trade_share', 'moneyflow', '中单成交占比', 'category_share:calculate', '', {})),
    'small_trade_share': IndicatorSpec(*('small_trade_share', 'moneyflow', '小单成交占比', 'category_share:calculate', '', {})),
    
    'level_percentile': IndicatorSpec(*('level_percentile', 'style', '点位252历史分位', 'level_percentile:level_percentile', 'market', {})),
    'relative_return': IndicatorSpec(*('relative_return', 'style', '相对收益', 'relative_return:relative_return', 'market', {})),
    'new_high_low_share': IndicatorSpec(*('new_high_low_share', 'style', '新高新低差占比历史分位', 'new_high_low_share:new_high_low_share', 'market', {})),
    'pe_percentile': IndicatorSpec(*('pe_percentile', 'style', 'PE_TTM历史分位', 'valuation_percentile:calculate', 'market', {})),
    'pb_percentile': IndicatorSpec(*('pb_percentile', 'style', 'PB_LF历史分位', 'valuation_percentile:calculate', 'market', {})),
    'earnings_bond_spread': IndicatorSpec(*('earnings_bond_spread', 'style', '盈利-10年国债收益率', 'earnings_bond_spread:earnings_bond_spread', 'market', {})),
    'amount_percentile': IndicatorSpec(*('amount_percentile', 'style', '成交额历史分位', 'amount_percentile:amount_percentile', 'market', {})),
    'amount_share_percentile': IndicatorSpec(*('amount_share_percentile', 'style', '成交额占比历史分位', 'amount_share_percentile:amount_share_percentile', 'market', {})),
    'industry_relative_return': IndicatorSpec(*('industry_relative_return', 'style', '行业相对收益', 'industry_relative_return:industry_relative_return', 'industry', {})),
    'size_spread_percentile': IndicatorSpec(*('size_spread_percentile', 'style', '大小盘收益差分位', 'size_spread_percentile:size_spread_percentile', 'style', {})),
    'growth_value_spread_percentile': IndicatorSpec(*('growth_value_spread_percentile', 'style', '成长价值收益差分位', 'growth_value_spread_percentile:growth_value_spread_percentile', 'style', {})),
    'beta_spread_percentile': IndicatorSpec(*('beta_spread_percentile', 'style', '高低Beta收益差分位', 'beta_spread_percentile:beta_spread_percentile', 'style', {})),
    'rotation_speed_percentile': IndicatorSpec(*('rotation_speed_percentile', 'style', '行业轮动速度分位', 'rotation_speed_percentile:rotation_speed_percentile', 'style', {})),
    'pe_return_spread': IndicatorSpec(*('pe_return_spread', 'style', 'PE收益差分位', 'valuation_spread_percentile:calculate', 'style', {})),
    'pb_return_spread': IndicatorSpec(*('pb_return_spread', 'style', 'PB收益差分位', 'valuation_spread_percentile:calculate', 'style', {})),
    
    'cross_section_dispersion': IndicatorSpec(*('cross_section_dispersion', 'risk', '横截面收益离散度', 'cross_section_dispersion:cross_section_dispersion', '', {})),
    'average_correlation': IndicatorSpec(*('average_correlation', 'risk', '平均收益相关性', 'average_correlation:average_correlation', '', {'window': 60})),
    'downside_correlation': IndicatorSpec(*('downside_correlation', 'risk', '下跌条件相关性', 'downside_correlation:downside_correlation', '', {'window': 120})),
    'pca_explained_ratio': IndicatorSpec(*('pca_explained_ratio', 'risk', '首主成分解释率', 'pca_explained_ratio:pca_explained_ratio', '', {'window': 60})),
    'downside_resonance': IndicatorSpec(*('downside_resonance', 'risk', '市场下行共振', 'downside_resonance:downside_resonance', '', {})),
    'limit_down_ratio': IndicatorSpec(*('limit_down_ratio', 'risk', '跌停扩散比例', 'limit_down_ratio:limit_down_ratio', '', {})),
    'worst_rolling_loss': IndicatorSpec(*('worst_rolling_loss', 'risk', '最差滚动5日收益', 'worst_rolling_loss:worst_rolling_loss', '', {})),
    'liquidity_pressure_percentile': IndicatorSpec(*('liquidity_pressure_percentile', 'risk', '流动性压力历史分位', 'liquidity_pressure_percentile:liquidity_pressure_percentile', '', {})),
    'rv5': IndicatorSpec(*('rv5', 'risk', '实现波动率RV5', 'realized_volatility:calculate', '', {'window': 5})),
    'rv20': IndicatorSpec(*('rv20', 'risk', '实现波动率RV20', 'realized_volatility:calculate', '', {'window': 20})),
    'rv60': IndicatorSpec(*('rv60', 'risk', '实现波动率RV60', 'realized_volatility:calculate', '', {'window': 60})),
    'rv_ratio_5_20': IndicatorSpec(*('rv_ratio_5_20', 'risk', '波动率期限结构RV5/RV20', 'rv_ratio:calculate', '', {'short_window': 5, 'long_window': 20})),
    'rv_ratio_20_60': IndicatorSpec(*('rv_ratio_20_60', 'risk', '波动率期限结构RV20/RV60', 'rv_ratio:calculate', '', {'short_window': 20, 'long_window': 60})),
    'index_iv30': IndicatorSpec(*('index_iv30', 'risk', '指数隐含波动率IV30', 'index_iv30:calculate', '', {})),
    'variance_risk_premium': IndicatorSpec(*('variance_risk_premium', 'risk', '波动率风险溢价代理IV30²-RV30²', 'variance_risk_premium:calculate', '', {})),
}


def select_indicators(names=None, modules=None):
    selected = tuple(INDICATORS) if names is None else ((names,) if isinstance(names, str) else tuple(names))
    unknown = set(selected) - INDICATORS.keys()
    if unknown:
        raise KeyError(f'未注册的监控指标: {sorted(unknown)}')
    if isinstance(modules, str):
        modules = (modules,)
    if modules is not None and set(modules) - {'macro', 'micro', 'moneyflow', 'style', 'risk'}:
        raise KeyError(f'未知监控模块: {modules}')
    return tuple(INDICATORS[n] for n in dict.fromkeys(selected)
                 if modules is None or INDICATORS[n].module in modules)

MACRO_INPUTS = {'macro_bond10y': ('10Y国债收益率',), 'macro_credit_aa_aaa': ('AAA信用收益率', 'AA信用收益率'), 'macro_credit_aa_cdb': ('AA信用收益率', '国开3Y收益率'), 'macro_credit_impulse': ('名义GDP', '新增社融'), 'macro_dr007': ('DR007',), 'macro_m2_yoy': ('M2余额', 'M2同比'), 'macro_policy_spread': ('7D政策利率', 'DR007'), 'macro_real_rate': ('10Y国债收益率', '预期通胀_Wind一致预期'), 'macro_shibor3m': ('Shibor3M',), 'macro_term_spread': ('10Y国债收益率', '1Y国债收益率'), 'macro_tsf_yoy': ('社融存量', '社融存量同比')}
