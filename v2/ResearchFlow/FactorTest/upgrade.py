"""中文说明：本脚本提供当前模块的量化研究或生产能力。"""


from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any


# 中文说明：定义 `UpgradeAdvice`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class UpgradeAdvice:
    issues: tuple[str, ...]
    causes: tuple[str, ...]
    sources: tuple[str, ...]
    methods: tuple[str, ...]
    validation: tuple[str, ...]


# Each entry is (metric-level inference, direct adjustment). Do not infer
# causes that are not supported by the corresponding score output.
CAUSE_ACTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "table_PRF_stats": (
        ("触发信号相对基准无明显增益", "对因子做截面rank，强化尾部信号"),
        ("Precision与Recall未有效兼顾", "压缩中部因子值，保留两端区分度"),
    ),
    "table_winrate_scan": (
        ("方向命中率不足", "校正因子方向，并增强高分位信号"),
        ("平均收益的统计显著性不足", "对因子做时序平滑，降低日间噪声"),
        ("收益方向可能不稳定", "统一因子方向，避免正负含义漂移"),
        ("部分检验结果未达到显著水平", "降低弱信号权重，突出稳定分位"),
    ),
    "table_monthly_ret": (
        ("正收益月份覆盖不足", "对因子做滚动标准化，降低尺度漂移"),
        ("亏损持续性偏强", "缩短因子衰减周期，降低旧信号权重"),
        ("月均收益未转正", "校正因子方向，并强化尾部信号"),
        ("月度收益稳定性不足", "平滑因子时序，减少信号跳变"),
    ),
    "table_annual_stats": (
        ("单位波动收益不足", "对因子去极值并平滑，降低噪声"),
        ("年化收益不足", "增强头尾分位差，压低中部弱信号"),
        ("收益波动偏高", "对因子做滚动标准化，稳定信号尺度"),
        ("回撤控制不足", "降低因子突变值权重，增加时序衰减"),
    ),
    "plot_basic_performance": (
        ("有效因子值覆盖不足", "统一缺失值处理，扩大有效因子覆盖"),
        ("持仓变化过于频繁", "降频处理"),
        ("有效覆盖包含不可用样本", "统一有效样本过滤口径"),
        ("累计收益未形成正向趋势", "校正因子方向，重构头尾分位"),
    ),
    "table_alpha_annual_stats": (
        ("因子中心位置跨年漂移", "对因子做逐日截面中心化"),
        ("因子离散度跨年不稳定", "采用中位数和IQR稳健标准化"),
        ("因子整体分布跨年变化较大", "统一因子标准化与变换口径"),
        ("因子尾部结构跨年不稳定", "采用固定分位数截尾"),
    ),
    "plot_alpha_distribution": (
        ("极端值占比偏高", "使用MAD去极值"),
        ("分布偏度或峰度偏高", "使用截面rank或分位数变换"),
        ("尾部相对IQR过宽", "对因子两端做分位数截尾"),
        ("因子与收益分布差异较大", "使用分位数映射校准因子尺度"),
    ),
    "table_ic_annual_stats": (
        ("线性相关方向错误或强度不足", "校正因子符号，并做截面标准化"),
        ("排序相关强度不足", "使用截面rank强化排序信息"),
        ("IC稳定性不足", "对因子做时序平滑，降低短期噪声"),
        ("IC与RankIC方向不一致", "改用单调分位数变换统一方向"),
    ),
    "plot_ic_distribution": (
        ("IC平均方向偏弱", "校正因子符号，强化尾部排序"),
        ("IC波动可能偏大", "对因子做轻度时序平滑"),
        ("正IC日期覆盖不足", "降低中部弱信号权重"),
        ("负IC尾部可能偏重", "对因子极端值做稳健截尾"),
    ),
    "plot_ic_contribution": (
        ("累计IC趋势或稳定性不足", "对因子做时序衰减，降低短期反转信号"),
        ("上涨收益识别弱于下跌收益识别", "提高正收益样本权重，增强上涨股票区分度"),
        ("下跌收益识别弱于上涨收益识别", "提高负收益样本权重，增强抗跌股票区分度"),
        ("IC与RankIC结构差异较大", "对因子做MAD去极值，并用截面rank降低异常幅度干扰"),
    ),
    "table_group_stats": (
        ("头尾组收益方向相反", "反转因子符号"),
        ("头尾组收益差不足", "使用非线性分位数映射强化两端"),
        ("头尾组收益可能受极端值影响", "对因子做稳健截尾"),
        ("分组收益区分度不足", "压缩中部因子值，突出头尾信号"),
    ),
    "plot_group_cumret": (
        ("分组收益单调性不足", "将原始因子转换为截面rank"),
        ("组间收益差异不足", "使用非线性映射拉开头尾因子值"),
        ("中间组区分度不足", "将中部因子值收缩至零附近"),
        ("线性分组不适配当前信号", "改用分位数映射构造因子"),
    ),
    "table_industry_annual_stats": (
        ("Sharpe为正的行业覆盖不足", "改用行业内rank构造因子"),
        ("行业间表现差异较大", "按行业统一因子方向与尺度"),
        ("多数行业风险调整收益不足", "对行业均值残差化"),
        ("行业有效性不够普遍", "降低跨行业公共成分权重"),
    ),
    "plot_industry_performance": (
        ("年化收益为正的行业覆盖不足", "改用行业内标准化因子"),
        ("部分行业收益为负", "按行业校正因子方向"),
        ("行业收益一致性不足", "剔除因子中的行业均值成分"),
        ("行业有效范围偏窄", "增强行业内个股排序信息"),
    ),
    "table_industry_exposure_stats": (
        ("单行业平均暴露偏高", "对因子做行业中性化"),
        ("行业暴露偏离零值", "使用行业哑变量回归残差"),
        ("多空行业权重可能不平衡", "在行业内分别标准化因子"),
        ("行业偏离控制不足", "剔除因子的行业均值成分"),
    ),
    "plot_industry_exposure_ret": (
        ("单行业暴露收益贡献偏高", "对因子做行业收益残差化"),
        ("行业收益贡献不够分散", "降低行业公共成分权重"),
        ("总收益对单行业较敏感", "剔除最高贡献行业的均值成分"),
        ("行业中性收益尚未验证", "使用行业中性残差因子"),
    ),
    "plot_industry_component": (
        ("前五行业持仓集中", "改用行业内rank降低行业聚集"),
        ("单行业持仓集中", "对该行业因子值做截面中心化"),
        ("行业持仓分散度不足", "压缩行业间因子均值差"),
        ("行业结构约束不足", "采用行业中性因子"),
    ),
    "table_sector_annual_stats": (
        ("Sharpe为正的板块覆盖不足", "改用板块内rank构造因子"),
        ("板块间表现差异较大", "按板块统一因子方向与尺度"),
        ("多数板块风险调整收益不足", "对板块均值残差化"),
        ("板块有效范围偏窄", "降低跨板块公共成分权重"),
    ),
    "plot_sector_performance": (
        ("年化收益为正的板块覆盖不足", "改用板块内标准化因子"),
        ("部分板块收益为负", "按板块校正因子方向"),
        ("板块收益一致性不足", "剔除因子中的板块均值成分"),
        ("板块有效范围偏窄", "增强板块内个股排序信息"),
    ),
    "table_sector_exposure_stats": (
        ("单板块平均暴露偏高", "对因子做板块中性化"),
        ("板块暴露偏离零值", "使用板块哑变量回归残差"),
        ("多空板块权重可能不平衡", "在板块内分别标准化因子"),
        ("板块偏离控制不足", "剔除因子的板块均值成分"),
    ),
    "plot_sector_exposure_ret": (
        ("单板块暴露收益贡献偏高", "对因子做板块收益残差化"),
        ("板块收益贡献不够分散", "降低板块公共成分权重"),
        ("总收益对单板块较敏感", "剔除最高贡献板块的均值成分"),
        ("板块中性收益尚未验证", "使用板块中性残差因子"),
    ),
    "plot_sector_component": (
        ("前三板块持仓集中", "改用板块内rank降低板块聚集"),
        ("板块持仓分散度不足", "压缩板块间因子均值差"),
        ("板块结构约束不足", "采用板块中性因子"),
        ("单板块权重可能偏高", "对该板块因子值做截面中心化"),
    ),
    "table_barra_exposure_stats": (
        ("单一Barra暴露偏高", "对最高暴露因子做中性化"),
        ("风格暴露偏离零值", "使用Barra回归残差重构因子"),
        ("风格暴露控制不足", "逐项剔除高暴露风格成分"),
        ("中性化效果尚未验证", "采用Barra中性残差因子"),
    ),
    "plot_barra_exposure": (
        ("平均风格暴露偏高", "对全部Barra风格做残差化"),
        ("最大风格暴露偏高", "对最大暴露风格中性化"),
        ("风格暴露分散度不足", "剔除主导风格成分"),
        ("暴露稳定性尚未验证", "使用滚动Barra残差化"),
    ),
    "plot_barra_exposure_ret": (
        ("单一Barra收益贡献偏高", "剔除最高贡献风格成分"),
        ("风格收益贡献不够分散", "对Barra风格整体残差化"),
        ("总收益对单风格较敏感", "降低该风格在因子中的载荷"),
        ("风格中性收益尚未验证", "使用风格中性残差因子"),
    ),
    "plot_corr_redundancy": (
        ("与存量因子最大相关偏高", "对最高相关因子残差化"),
        ("整体平均相关偏高", "对高相关因子逐一残差化"),
        ("独立信息占比不足", "仅保留对存量因子的正交残差"),
        ("相关性控制不足", "用正交化重构因子"),
    ),
    "table_regime_stats": (
        ("Sharpe为正的状态覆盖不足", "按状态标准化因子尺度"),
        ("状态间收益稳定性不足", "加入状态条件的因子权重"),
        ("IC为正的状态覆盖不足", "按状态校正因子方向"),
        ("最差状态出现负Sharpe", "在该状态衰减因子值"),
    ),
    "plot_regime_cumret": (
        ("正年化状态覆盖不足", "按状态校正因子方向"),
        ("最差状态回撤偏大", "在高回撤状态衰减因子值"),
        ("状态收益一致性不足", "按状态标准化因子尺度"),
        ("状态回撤控制不足", "增强因子的状态衰减项"),
    ),
    "table_shadow_capacity_test": (
        ("成交填充率不足", "加入流动性权重，压低难成交股票信号"),
        ("成本后风险调整收益不足", "平滑因子并惩罚高换手信号"),
        ("通过容量范围偏窄", "降低小成交额股票的因子权重"),
        ("容量约束未满足", "用成交额缩放因子强度"),
    ),
    "plot_shadow_capacity_curve": (
        ("部分容量下净年化为负", "提高流动性因子权重"),
        ("高容量下Sharpe不足", "压低高冲击股票的因子值"),
        ("容量曲线变化不稳定", "用成交额平滑缩放因子"),
        ("容量放大后收益恶化", "增强因子的流动性约束项"),
    ),
}


SOURCE_INFERENCES: dict[str, tuple[str, ...]] = {
    "table_PRF_stats": (
        "收益缺少高置信度信号贡献",
        "收益可能由少量命中样本贡献",
    ),
    "table_winrate_scan": (
        "收益缺少稳定的方向性贡献",
        "收益均值可能由噪声主导",
        "收益方向尚未稳定",
        "收益仅在部分检验区间显现",
    ),
    "table_monthly_ret": (
        "收益月份分布偏稀疏",
        "收益存在持续失效阶段",
        "整体收益贡献不足",
        "收益时序稳定性较弱",
    ),
    "table_annual_stats": (
        "收益以波动换取，Alpha效率偏低",
        "因子绝对收益贡献不足",
        "收益波动成分偏高",
        "收益可能受少数回撤阶段拖累",
    ),
    "plot_basic_performance": (
        "",
        "收益可能依赖高频换仓",
        "",
        "因子尚未形成稳定正收益来源",
    ),
    "table_alpha_annual_stats": (
        "收益可能受因子位置漂移影响",
        "收益可能受因子尺度变化影响",
        "收益来源跨年不稳定",
        "收益可能依赖不稳定尾部",
    ),
    "plot_alpha_distribution": (
        "收益可能由少数极端样本贡献",
        "收益可能依赖偏态或肥尾",
        "收益可能集中于因子尾部",
        "",
    ),
    "table_ic_annual_stats": (
        "线性选股Alpha较弱或方向相反",
        "排序型选股Alpha不足",
        "选股Alpha时序稳定性不足",
        "线性与排序收益来源不一致",
    ),
    "plot_ic_distribution": (
        "选股Alpha平均贡献偏弱",
        "选股Alpha日间波动较大",
        "选股Alpha有效日期偏少",
        "选股Alpha可能受负向尾部拖累",
    ),
    "plot_ic_contribution": (
        "选股Alpha缺少稳定时序贡献",
        "收益主要来自下跌期抗跌识别",
        "收益主要来自上涨期选股识别",
        "收益对因子幅度或异常值较敏感",
    ),
    "table_group_stats": (
        "收益方向与因子排序相反",
        "收益缺少头尾分层贡献",
        "收益可能由头尾极端样本贡献",
        "排序型收益来源偏弱",
    ),
    "plot_group_cumret": (
        "收益缺少稳定的排序来源",
        "收益缺少清晰的头尾价差",
        "中部信号未提供有效收益",
        "收益关系可能偏非线性",
    ),
    "table_industry_annual_stats": (
        "收益集中于少数行业",
        "收益来源存在行业差异",
        "多数行业选股Alpha偏弱",
        "行业普适性不足",
    ),
    "plot_industry_performance": (
        "收益集中于少数行业",
        "部分行业贡献为负",
        "行业收益来源不一致",
        "行业覆盖范围偏窄",
    ),
    "table_industry_exposure_stats": (
        "收益可能夹带行业Beta",
        "收益可能来自行业方向暴露",
        "多空收益可能受行业结构影响",
        "行业成分可能替代个股Alpha",
    ),
    "plot_industry_exposure_ret": (
        "收益较多来自单一行业Beta",
        "收益来源偏行业集中",
        "总收益对单行业依赖较高",
        "行业中性Alpha尚不明确",
    ),
    "plot_industry_component": (
        "收益可能依赖头部行业持仓",
        "收益可能依赖单一行业",
        "收益来源行业分散度不足",
        "行业结构可能主导收益",
    ),
    "table_sector_annual_stats": (
        "收益集中于少数板块",
        "收益来源存在板块差异",
        "多数板块选股Alpha偏弱",
        "板块普适性不足",
    ),
    "plot_sector_performance": (
        "收益集中于少数板块",
        "部分板块贡献为负",
        "板块收益来源不一致",
        "板块覆盖范围偏窄",
    ),
    "table_sector_exposure_stats": (
        "收益可能夹带板块Beta",
        "收益可能来自板块方向暴露",
        "多空收益可能受板块结构影响",
        "板块成分可能替代个股Alpha",
    ),
    "plot_sector_exposure_ret": (
        "收益较多来自单一板块Beta",
        "收益来源偏板块集中",
        "总收益对单板块依赖较高",
        "板块中性Alpha尚不明确",
    ),
    "plot_sector_component": (
        "收益可能依赖头部板块持仓",
        "收益来源板块分散度不足",
        "板块结构可能主导收益",
        "收益可能依赖单一板块",
    ),
    "table_barra_exposure_stats": (
        "收益可能夹带单一风格Beta",
        "收益可能来自系统性风格暴露",
        "风格成分可能替代个股Alpha",
        "风格中性Alpha尚不明确",
    ),
    "plot_barra_exposure": (
        "收益可能依赖整体风格暴露",
        "收益可能依赖单一风格",
        "收益来源风格分散度不足",
        "风格暴露可能持续贡献收益",
    ),
    "plot_barra_exposure_ret": (
        "收益较多来自单一风格溢价",
        "收益来源偏风格集中",
        "总收益对单风格依赖较高",
        "风格中性Alpha尚不明确",
    ),
    "plot_corr_redundancy": (
        "收益可能来自存量因子的共性信息",
        "收益独立来源占比偏低",
        "新增Alpha信息有限",
        "收益来源与存量因子重叠",
    ),
    "table_regime_stats": (
        "收益集中于少数市场状态",
        "收益来源具有状态依赖",
        "选股Alpha集中于少数状态",
        "最差状态显著拖累收益",
    ),
    "plot_regime_cumret": (
        "收益集中于正年化状态",
        "收益受最差状态回撤拖累",
        "收益来源具有状态依赖",
        "状态回撤侵蚀收益",
    ),
    "table_shadow_capacity_test": (
        "收益可能依赖低流动性标的",
        "毛收益被换手与成本侵蚀",
        "收益来源容量较窄",
        "因子收益可交易性不足",
    ),
    "plot_shadow_capacity_curve": (
        "收益可能依赖小容量标的",
        "高容量下Alpha被冲击成本稀释",
        "收益来源对容量较敏感",
        "收益可能依赖低流动性溢价",
    ),
}


_METRIC_PATTERN = re.compile(
    r"([^;；=]+?)\s*=\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(%)?"
)


# 中文说明：`_parse_metrics`：内部辅助步骤，不作为稳定公共接口。
def _parse_metrics(actual: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in _METRIC_PATTERN.finditer(actual):
        name = match.group(1).strip().lower()
        value = float(match.group(2))
        if match.group(3):
            value /= 100.0
        metrics[name] = value
    return metrics


# 中文说明：`_metric`：内部辅助步骤，不作为稳定公共接口。
def _metric(metrics: dict[str, float], *names: str) -> float:
    for name in names:
        target = name.lower()
        for key, value in metrics.items():
            if target == key or target in key:
                return value
    return float("nan")


# 中文说明：`_text_metric`：内部辅助步骤，不作为稳定公共接口。
def _text_metric(actual: str, name: str) -> str:
    match = re.search(rf"(?:^|[;；])\s*{re.escape(name)}\s*=\s*([^;；]+)", actual)
    return match.group(1).strip() if match else ""


# 中文说明：`_range_metric`：内部辅助步骤，不作为稳定公共接口。
def _range_metric(actual: str, name: str) -> tuple[float, float] | None:
    match = re.search(
        rf"{re.escape(name)}\s*=\s*([+-]?\d+(?:\.\d+)?)\s*[~～]\s*([+-]?\d+(?:\.\d+)?)",
        actual,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


# 中文说明：`_add_issue`：内部辅助步骤，不作为稳定公共接口。
def _add_issue(
    found: list[tuple[str, int]],
    condition: bool,
    issue: str,
    method_index: int,
) -> None:
    if condition:
        found.append((issue, method_index))


# 中文说明：`_diagnose_metrics`：内部辅助步骤，不作为稳定公共接口。
def _diagnose_metrics(func_name: str, actual: str) -> list[tuple[str, int]]:
    metrics = _parse_metrics(actual)
    found: list[tuple[str, int]] = []

    if func_name == "table_PRF_stats":
        precision = _metric(metrics, "precision")
        baseline = _metric(metrics, "baseline")
        lift = _metric(metrics, "lift")
        f1 = _metric(metrics, "f1")
        _add_issue(found, lift is not None and lift < 0.05, f"Lift仅为{lift:.2%}，信号增益不足", 0)
        _add_issue(found, math.isfinite(precision) and precision < 0.55, f"Precision仅为{precision:.2%}", 0)
        _add_issue(
            found,
            precision is not None and baseline is not None and precision <= baseline,
            "Precision未超过baseline，当前触发信号没有有效增益",
            0,
        )
        _add_issue(found, f1 is not None and f1 < 0.15, f"F1仅为{f1:.3f}，精确率与召回率平衡较弱", 1)
    elif func_name == "table_winrate_scan":
        win_rate = _metric(metrics, "mean_winrate")
        mean_t = _metric(metrics, "mean_t")
        max_p = _metric(metrics, "max_p")
        _add_issue(found, win_rate is not None and win_rate < 0.52, f"平均胜率仅为{win_rate:.2%}", 0)
        _add_issue(found, mean_t is not None and mean_t < 2.0, f"平均t值仅为{mean_t:.2f}，统计显著性不足", 1)
        _add_issue(found, max_p is not None and max_p > 0.05, f"最大p值为{max_p:.4f}，部分周期不显著", 3)
    elif func_name == "table_monthly_ret":
        positive = _metric(metrics, "正收益月")
        streak = _metric(metrics, "最大连续亏损")
        _add_issue(found, positive is not None and positive < 0.60, f"正收益月占比仅为{positive:.2%}", 0)
        streak_text = str(int(streak)) if math.isfinite(streak) else ""
        _add_issue(found, math.isfinite(streak) and streak > 3, f"最大连续亏损达到{streak_text}个月", 1)
    elif func_name == "table_annual_stats":
        sharpe = _metric(metrics, "sharpe")
        annret = _metric(metrics, "annret")
        maxdd = _metric(metrics, "maxdd")
        worst_year_sharpe = _metric(metrics, "最差年度sharpe")
        _add_issue(found, sharpe is not None and sharpe < 1.0, f"Sharpe仅为{sharpe:.2f}", 0)
        _add_issue(found, annret is not None and annret < 0.10, f"年化收益仅为{annret:.2%}", 1)
        _add_issue(found, maxdd is not None and maxdd > 0.20, f"最大回撤达到{maxdd:.2%}", 3)
        _add_issue(found, math.isfinite(worst_year_sharpe) and worst_year_sharpe < 0,
                   f"最差年度Sharpe为{worst_year_sharpe:.2f}", 0)
    elif func_name == "plot_basic_performance":
        coverage = _metric(metrics, "coverage")
        turnover = _metric(metrics, "日均换手")
        _add_issue(found, coverage is not None and coverage < 0.75, f"覆盖率仅为{coverage:.2%}", 0)
        _add_issue(found, turnover is not None and turnover > 0.50, f"日均换手达到{turnover:.2%}", 1)
        _add_issue(found, "未向上" in actual, "累计收益趋势未向上", 3)
    elif func_name == "table_alpha_annual_stats":
        drift = _metric(metrics, "历年最大中位数漂移/iqr")
        iqr_range = _range_metric(actual, "IQR比率区间")
        jsd = _metric(metrics, "最大jsd距离")
        wasserstein = _metric(metrics, "标准化wasserstein")
        skew = _metric(metrics, "偏度跨年变化")
        tail = _metric(metrics, "尾部比率最大变化")
        _add_issue(found, drift is not None and drift > 0.50, f"中位数漂移/IQR达到{drift:.2f}", 0)
        _add_issue(
            found,
            iqr_range is not None and (iqr_range[0] < 0.50 or iqr_range[1] > 2.00),
            (
                f"IQR比率区间为{iqr_range[0]:.2f}~{iqr_range[1]:.2f}"
                if iqr_range is not None
                else "历年IQR稳定性需要复核"
            ),
            1,
        )
        _add_issue(found, jsd is not None and jsd > 0.20, f"JSD距离达到{jsd:.3f}", 2)
        _add_issue(found, wasserstein is not None and wasserstein > 0.50, f"标准化Wasserstein达到{wasserstein:.2f}", 2)
        _add_issue(found, skew is not None and skew > 2.0, f"偏度跨年变化达到{skew:.2f}", 3)
        _add_issue(found, tail is not None and tail > 1.00, f"尾部比率最大变化达到{tail:.1%}", 3)
    elif func_name == "plot_alpha_distribution":
        skew = _metric(metrics, "|偏度|")
        kurtosis = _metric(metrics, "峰度")
        tail = _metric(metrics, "尾部占比")
        extreme = _metric(metrics, "5mad外样本占比")
        shape_js = _metric(metrics, "比收益分布的jsd")
        _add_issue(found, extreme is not None and extreme > 0.05, f"5MAD外样本占比达到{extreme:.2%}", 0)
        _add_issue(found, tail is not None and tail > 20, f"尾部/IQR达到{tail:.2f}", 2)
        _add_issue(found, (skew is not None and skew > 3) or (kurtosis is not None and kurtosis > 20), "偏度或峰度偏高", 1)
        _add_issue(found, shape_js is not None and shape_js > 0.55, f"因子与收益分布JSD达到{shape_js:.3f}", 3)
    elif func_name == "table_ic_annual_stats":
        avg_ic = _metric(metrics, "avgic")
        icir = _metric(metrics, "icir")
        rank_ic = _metric(metrics, "rankic")
        _add_issue(found, avg_ic is not None and avg_ic < 0.08, f"AvgIC仅为{avg_ic:.4f}", 0 if avg_ic <= 0 else 1)
        _add_issue(found, rank_ic is not None and rank_ic < 0.10, f"RankIC仅为{rank_ic:.4f}", 0 if rank_ic <= 0 else 1)
        _add_issue(found, icir is not None and icir < 0.50, f"ICIR仅为{icir:.2f}", 2)
    elif func_name == "plot_ic_distribution":
        mean_ic = _metric(metrics, "ic均值")
        positive = _metric(metrics, "正ic占比")
        _add_issue(found, mean_ic is not None and mean_ic < 0.01, f"IC均值仅为{mean_ic:.4f}", 0)
        _add_issue(found, positive is not None and positive < 0.55, f"正IC占比仅为{positive:.2%}", 2)
    elif func_name == "plot_ic_contribution":
        trend = _metric(metrics, "总体趋势")
        drawdown = _metric(metrics, "最大回撤")
        side, structure = _text_metric(actual, "强侧"), _text_metric(actual, "结构")
        weak_side = _metric(metrics, "弱侧强度")
        side_gap, structure_gap = _metric(metrics, "侧别差"), _metric(metrics, "结构差")
        _add_issue(found, math.isfinite(trend) and trend < 0.01, f"累计IC总体趋势仅为{trend:.4f}", 0)
        _add_issue(found, math.isfinite(drawdown) and drawdown > 0.30, f"累计IC最大回撤达{drawdown:.2%}", 0)
        side_problem = math.isfinite(side_gap) and (side_gap > 0.40 or weak_side < 0.005)
        _add_issue(found, side_problem and side == "负收益侧", f"上涨收益识别偏弱，侧别差{side_gap:.2%}", 1)
        _add_issue(found, side_problem and side == "正收益侧", f"下跌收益识别偏弱，侧别差{side_gap:.2%}", 2)
        _add_issue(found, math.isfinite(weak_side) and weak_side < 0.005 and side == "均衡",
                   f"正负收益侧识别均弱，弱侧强度仅为{weak_side:.4f}", 1)
        _add_issue(
            found,
            bool(structure) and math.isfinite(structure_gap)
            and (structure != "基本一致" or structure_gap > 0.50),
            f"IC与RankIC结构差异较大（{structure}，结构差{structure_gap:.2%}）",
            3,
        )
    elif func_name == "table_group_stats":
        spread = _metric(metrics, "g1-g10 spread")
        _add_issue(found, spread is not None and spread <= 0, f"G1-G10收益差为{spread:.2%}，方向可能反转", 0)
        _add_issue(found, spread is not None and 0 < spread < 0.10, f"G1-G10收益差仅为{spread:.2%}", 1)
    elif func_name == "plot_group_cumret":
        ratio = _metric(metrics, "期末分组单调邻接比例")
        _add_issue(found, ratio is not None and ratio < 0.60, f"分组单调邻接比例仅为{ratio:.2%}", 0)
    elif func_name in {"table_industry_annual_stats", "plot_industry_performance"}:
        ratio = _metric(metrics, "占比")
        threshold = 0.70 if func_name == "table_industry_annual_stats" else 0.60
        _add_issue(found, ratio is not None and ratio < threshold, f"行业有效占比仅为{ratio:.2%}", 0)
    elif func_name in {"table_sector_annual_stats", "plot_sector_performance"}:
        ratio = _metric(metrics, "占比")
        _add_issue(found, ratio is not None and ratio < 0.60, f"板块有效占比仅为{ratio:.2%}", 0)
    elif func_name in {"table_industry_exposure_stats", "table_sector_exposure_stats"}:
        exposure = _metric(metrics, "最大|行业平均暴露|", "最大|板块平均暴露|")
        threshold = 0.10 if "industry" in func_name else 0.15
        _add_issue(found, exposure is not None and exposure > threshold, f"最大平均暴露达到{exposure:.2%}", 0)
    elif func_name in {"plot_industry_exposure_ret", "plot_sector_exposure_ret"}:
        contribution = _metric(metrics, "贡献")
        _add_issue(found, contribution is not None and contribution > 0.30, f"最大单域收益贡献达到{contribution:.2%}", 0)
    elif func_name == "plot_industry_component":
        cr5 = _metric(metrics, "行业多头cr5")
        single = _metric(metrics, "最大单行业多头")
        _add_issue(found, cr5 is not None and cr5 > 0.50, f"行业CR5达到{cr5:.2%}", 0)
        _add_issue(found, single is not None and single > 0.20, f"最大单行业持仓达到{single:.2%}", 1)
    elif func_name == "plot_sector_component":
        cr3 = _metric(metrics, "板块多头cr3")
        _add_issue(found, cr3 is not None and cr3 > 0.50, f"板块CR3达到{cr3:.2%}", 0)
    elif func_name in {"table_barra_exposure_stats", "plot_barra_exposure"}:
        exposure = _metric(metrics, "最大|barra平均暴露|", "最大|暴露|")
        method_index = 0 if func_name == "table_barra_exposure_stats" else 1
        _add_issue(found, exposure is not None and exposure > 0.20, f"最大Barra暴露达到{exposure:.3f}", method_index)
    elif func_name == "plot_barra_exposure_ret":
        contribution = _metric(metrics, "最大单barra收益贡献")
        _add_issue(found, contribution is not None and contribution > 0.30, f"最大单Barra收益贡献达到{contribution:.2%}", 0)
    elif func_name == "plot_corr_redundancy":
        max_corr = _metric(metrics, "最大相关")
        avg_corr = _metric(metrics, "平均相关")
        _add_issue(found, max_corr is not None and max_corr >= 0.60, f"最大相关达到{max_corr:.3f}", 0)
        _add_issue(found, avg_corr is not None and avg_corr > 0.30, f"平均相关达到{avg_corr:.3f}", 1)
    elif func_name == "table_regime_stats":
        sharpe_ratio = _metric(metrics, "sharpe>0占比")
        ic_ratio = _metric(metrics, "ic>0占比")
        worst_sharpe = _metric(metrics, "最差sharpe")
        _add_issue(found, sharpe_ratio is not None and sharpe_ratio < 0.60, f"Sharpe为正的状态仅占{sharpe_ratio:.2%}", 0)
        _add_issue(found, ic_ratio is not None and ic_ratio < 0.60, f"IC为正的状态仅占{ic_ratio:.2%}", 2)
        _add_issue(found, worst_sharpe is not None and worst_sharpe < 0, f"最差状态Sharpe为{worst_sharpe:.2f}", 3)
    elif func_name == "plot_regime_cumret":
        positive = _metric(metrics, "状态年化收益为正占比")
        worst_dd = _metric(metrics, "最差状态回撤")
        _add_issue(found, positive is not None and positive < 0.60, f"状态年化收益为正占比仅为{positive:.2%}", 0)
        _add_issue(found, worst_dd is not None and worst_dd > 0.30, f"最差状态回撤达到{worst_dd:.2%}", 1)
    elif func_name == "table_shadow_capacity_test":
        fill = _metric(metrics, "最小成交填充率")
        sharpe = _metric(metrics, "最低sharpe")
        pass_ratio = _metric(metrics, "通过容量占比")
        maxdd = _metric(metrics, "最大回撤")
        _add_issue(found, fill is not None and fill < 0.80, f"最低成交填充率仅为{fill:.2%}", 0)
        _add_issue(found, sharpe is not None and sharpe < 1.0, f"成本后最低Sharpe仅为{sharpe:.2f}", 1)
        _add_issue(found, pass_ratio is not None and pass_ratio < 0.50, f"通过容量占比仅为{pass_ratio:.2%}", 2)
        _add_issue(found, maxdd is not None and maxdd > 0.30, f"容量测试最大回撤达到{maxdd:.2%}", 3)
    elif func_name == "plot_shadow_capacity_curve":
        positive = _metric(metrics, "正年化收益占比")
        sharpe = _metric(metrics, "最低sharpe")
        _add_issue(found, positive is not None and positive < 0.60, f"正年化收益容量占比仅为{positive:.2%}", 0)
        _add_issue(found, sharpe is not None and sharpe < 1.0, f"高容量下最低Sharpe仅为{sharpe:.2f}", 1)
        _add_issue(found, "需复核" in actual, "容量曲线未呈现可解释的单调变化", 2)

    return found


# 中文说明：`diagnose_upgrades`：执行该名称对应的业务计算，并返回调用方所需结果。
def diagnose_upgrades(
    func_name: str,
    score_result: Any | None = None,
    max_items: int = 3,
) -> UpgradeAdvice:
    if func_name not in CAUSE_ACTIONS:
        return UpgradeAdvice((), (), (), (), ())

    level = str(getattr(score_result, "level", "")).lower() if score_result is not None else ""
    actual = str(getattr(score_result, "actual", "")) if score_result is not None else ""
    hard_fail = bool(getattr(score_result, "hard_fail", False)) if score_result is not None else False
    found = _diagnose_metrics(func_name, actual)
    needs_improvement = hard_fail or "fail" in level or "watch" in level or "待" in level
    if not needs_improvement:
        return UpgradeAdvice((), (), (), (), ())

    issues: list[str] = []
    method_indexes: list[int] = []
    item_limit = max_items
    for issue, method_index in found:
        if method_index in method_indexes:
            continue
        issues.append(issue)
        method_indexes.append(method_index)
        if len(issues) >= item_limit:
            break

    if not issues:
        return UpgradeAdvice((), (), (), (), ())

    cause_actions = CAUSE_ACTIONS.get(func_name, ())
    causes = [
        cause_actions[index][0]
        for index in method_indexes
        if index < len(cause_actions)
    ]
    methods = [
        cause_actions[index][1]
        for index in method_indexes
        if index < len(cause_actions)
    ]
    source_hints = SOURCE_INFERENCES.get(func_name, ())
    sources = []
    for index in method_indexes:
        if index >= len(source_hints):
            continue
        source = source_hints[index]
        if source and source not in sources:
            sources.append(source)
    return UpgradeAdvice(
        tuple(issues[:item_limit]),
        tuple(causes[:item_limit]),
        tuple(sources[:item_limit]),
        tuple(methods[:item_limit]),
        (),
    )


# 中文说明：`suggest_upgrades`：执行该名称对应的业务计算，并返回调用方所需结果。
def suggest_upgrades(func_name: str, score_result: Any | None = None, max_items: int = 3) -> list[str]:
    return list(diagnose_upgrades(func_name, score_result, max_items).methods)
