"""中文说明：本脚本提供当前模块的量化研究或生产能力。"""


from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


DEFAULT_TEMPLATE_PATH = Path(r"C:\Users\12404\Desktop\私募级单因子验收标准_完整整合修正版.xlsx")
DEFAULT_SHEET = "02_自动评分模板"
PASS_LINE = 75.0


# 中文说明：定义 `ScoreResult`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class ScoreResult:
    actual: str   # 实际值/观察结论
    level: str    # 评分挡位
    score: float  # 单项得分
    opinion: str
    hard_fail: bool = False
    gate_note: str = ""


# 中文说明：`main`：执行该名称对应的业务计算，并返回调用方所需结果。
def main() -> None:
    args = parse_args()
    analyzer = build_analyzer(args)
    score_df, summary = score_analyzer(analyzer, args.template)
    output_path = resolve_output_path(args.output, analyzer)
    write_scorebook(score_df, summary, args.template, output_path)
    print(f"评分完成: {output_path}")
    print(f"总分: {summary['total_score']:.2f}; 结论: {summary['conclusion']}")
    if summary["hard_gate_notes"]:
        print("硬Gate: " + "；".join(summary["hard_gate_notes"]))


# 中文说明：`parse_args`：解析外部输入。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 SingleTest analyzer 结果生成私募级单因子验收评分表。")
    parser.add_argument("--factor", required=True, help="因子文件路径，支持 csv/parquet/pkl/feather。")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE_PATH), help="验收标准 Excel 模板路径。")
    parser.add_argument("--output", default="", help="评分结果输出 Excel 路径。")
    parser.add_argument("--name", default="MinuteGRU", help="因子名称。")
    parser.add_argument("--factor-type", default="longshort", choices=["longshort", "long"], help="因子方向。")
    parser.add_argument("--alpha-type", default="深度学习", help="因子类型。")
    parser.add_argument("--usage", default="日频选股", help="数据用途。")
    parser.add_argument("--universe", default="范围池", help="universe，例如 universe/hs300/zz500/zz1000/zz2000/a500。")
    parser.add_argument("--start-date", default="2021-01-01", help="评分起始日期。")
    parser.add_argument("--end-date", default="2025-12-16", help="评分结束日期。")
    parser.add_argument("--summary", default="", help="因子描述。")
    return parser.parse_args()


# 中文说明：`build_analyzer`：构建下游所需对象。
def build_analyzer(args: argparse.Namespace) -> Any:
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent
    for path in (str(package_dir), str(project_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        from papertest import FactorAnalyzer  # type: ignore
    except ImportError:
        from .analyzer import FactorAnalyzer

    factor_df = read_factor_file(Path(args.factor).expanduser())
    info = {
        "name": args.name,
        "factor_type": args.factor_type,
        "alpha_type": args.alpha_type,
        "usage": args.usage,
        "universe": args.universe,
        "start_date": str(factor_df.index.min().date()),
        "end_date": str(factor_df.index.max().date()),
        "summary": args.summary,
    }
    analyzer = FactorAnalyzer(info, factor_df)
    analyzer.reset_axis(args.start_date, args.end_date, args.universe)
    return analyzer


# 中文说明：`read_factor_file`：读取持久化数据。
def read_factor_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif suffix == ".feather":
        df = pd.read_feather(path)
        df = df.set_index(df.columns[0])
    else:
        raise ValueError(f"不支持的因子文件类型: {suffix}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df.sort_index()


# 中文说明：`score_analyzer`：计算评分或监控指标。
def score_analyzer(analyzer: Any, template_path: str | Path = DEFAULT_TEMPLATE_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    template = pd.read_excel(template_path, sheet_name=DEFAULT_SHEET)
    scorers: dict[str, Callable[[Any], ScoreResult]] = {
        "PRF检验": score_prf,
        "多期胜率检验": score_winrate,
        "月度收益": score_monthly_return,
        "年度收益指标表现": score_annual_stats,
        "基本表现图": score_basic_performance,
        "因子值分布表": score_alpha_distribution_table,
        "因子值分布图": score_alpha_distribution_shape,
        "IC年度统计指标": score_ic_annual_stats,
        "各回报期IC分布图": score_ic_distribution,
        "IC指标累计时序图": score_ic_curve,
        "分组收益表现": score_group_stats,
        "分组收益累计图": score_group_curve,
        "行业分域指标": score_industry_annual,
        "行业分域表现": score_industry_curve,
        "行业暴露表现": score_industry_exposure,
        "行业暴露收益": score_industry_exposure_return,
        "行业持仓结构": score_industry_component,
        "板块分域指标": score_sector_annual,
        "板块分域表现": score_sector_curve,
        "板块暴露表现": score_sector_exposure,
        "板块暴露收益": score_sector_exposure_return,
        "板块持仓结构": score_sector_component,
        "Barra因子暴露表现": score_barra_exposure_stats,
        "Barra因子暴露": score_barra_exposure_bar,
        "Barra因子暴露收益时序图": score_barra_exposure_return,
        "Spearman秩相关冗余检验": score_redundancy,
        "上下线模型检验": score_regime_stats,
        "上下线状态检验": score_regime_stats,
        "Regime检验": score_regime_stats,
        "市场状态检验": score_regime_stats,
        "市场状态累计收益": score_regime_curve,
        "上下线模型累计收益": score_regime_curve,
        "影子盘容量检验": score_shadow_capacity,
        "小资金影子盘容量检验": score_shadow_capacity,
        "容量检验": score_shadow_capacity,
        "上线容量检验": score_shadow_capacity,
        "影子盘容量曲线": score_shadow_capacity_curve,
        "容量曲线": score_shadow_capacity_curve,
    }

    rows = []
    hard_gate_notes: list[str] = []
    for _, row in template.iterrows():
        item = str(row["分析项"])
        result = safe_score(item, analyzer, scorers)
        weight = parse_weight(row.get("权重", 0))
        weighted_score = result.score * weight
        if result.hard_fail:
            hard_gate_notes.append(f"{item}: {result.gate_note}")
        rows.append(
            {
                "一级模块": row.get("一级模块"),
                "分析项": item,
                "实际值/观察结论": result.actual,
                "评分档位": result.level,
                "单项得分(0-100)": round(result.score, 2),
                "权重": row.get("权重"),
                "加权得分": round(weighted_score, 4),
                "评审意见": result.opinion,
                "责任人": row.get("责任人", ""),
                "复测状态": "已评分",
            }
        )

    score_df = pd.DataFrame(rows)
    weight_sum = sum(parse_weight(x) for x in score_df["权重"])
    raw_weighted = float(score_df["加权得分"].sum())
    total_score = raw_weighted / weight_sum if weight_sum > 0 else 0.0
    hard_fail = len(hard_gate_notes) > 0
    conclusion = conclusion_from_score(total_score, hard_fail)
    summary = {
        "total_score": total_score,
        "raw_weighted_score": raw_weighted,
        "weight_sum": weight_sum,
        "hard_fail": hard_fail,
        "hard_gate_notes": hard_gate_notes,
        "conclusion": conclusion,
    }
    return score_df, summary


# 中文说明：`safe_score`：执行该名称对应的业务计算，并返回调用方所需结果。
def safe_score(item: str, analyzer: Any, scorers: dict[str, Callable[[Any], ScoreResult]]) -> ScoreResult:
    scorer = scorers.get(item)
    if scorer is None:
        scorer = scorer_by_alias(item, scorers)
    if scorer is None:
        return ScoreResult("未配置自动评分规则", "待人工复核", 50.0, "该项需人工补充。")
    try:
        return scorer(analyzer)
    except Exception as exc:
        return ScoreResult(f"计算失败: {exc}", "待复核", 40.0, "自动评分失败，请检查 analyzer 数据或人工复核。")


# 中文说明：`scorer_by_alias`：计算评分或监控指标。
def scorer_by_alias(item: str, scorers: dict[str, Callable[[Any], ScoreResult]]) -> Callable[[Any], ScoreResult] | None:
    compact = item.replace(" ", "").replace("_", "").lower()
    if any(keyword.lower() in compact for keyword in ("上下线", "regime", "市场状态", "牛熊", "牛市", "熊市", "震荡", "高波动", "风格切换")):
        return lambda analyzer: score_regime_item(analyzer, item)
    if any(keyword.lower() in compact for keyword in ("影子盘", "容量", "capacity", "冲击成本", "上线资金", "1000万", "5000万", "1亿", "成交满足")):
        return lambda analyzer: score_capacity_item(analyzer, item)
    return None


# 中文说明：`score_prf`：计算评分或监控指标。
def score_prf(analyzer: Any) -> ScoreResult:
    df = analyzer.table_PRF_stats()
    precision = float(df["precision"].mean())
    baseline = float(df["baseline"].mean())
    lift = float(df["lift"].mean())
    f1 = float(df["f1"].mean())
    hard = lift <= 0 or precision < baseline
    score, level = grade_all([(precision, 0.55, 0.58, 0.60), (lift, 0.05, 0.08, 0.10), (f1, 0.15, 0.20, 0.25)])
    return result(
        f"Precision={precision:.2%}; baseline={baseline:.2%}; Lift={lift:.2%}; F1={f1:.3f}",
        level,
        0 if hard else score,
        hard,
        "平均 Lift<=0 ",
    )   


# 中文说明：`score_winrate`：计算评分或监控指标。
def score_winrate(analyzer: Any) -> ScoreResult:
    df = analyzer.table_winrate_scan()
    win_rate = float(df["win_rate"].mean())
    t_stat = float(df["t_stat"].mean())
    p_value = float(df["p_value"].max())
    mean_ret = float(df["mean_ret"].mean())
    hard = win_rate <= 0.50 and t_stat <= 1
    score, level = grade_all([(win_rate, 0.52, 0.54, 0.55), (t_stat, 2.0, 2.5, 3.0)])
    if p_value > 0.05:
        score = min(score, 65)
        level = "Watch"
    return result(
        f"mean_winrate={win_rate:.2%}; mean_ret={mean_ret:.4%}; mean_t={t_stat:.2f}; max_p={p_value:.4f}",
        level,
        0 if hard else score,
        hard,
        "平均 胜率<=50% 且 t_stat<=1",
    )


# 中文说明：`score_monthly_return`：计算评分或监控指标。
def score_monthly_return(analyzer: Any) -> ScoreResult:
    monthly = monthly_returns(analyzer)
    pos_ratio = float((monthly > 0).mean())
    max_loss_streak = max_consecutive(monthly < 0)
    hard = max_loss_streak > 4 or pos_ratio < 0.50
    score, level = grade_all([(pos_ratio, 0.60, 0.63, 0.65)], lower_better=[(max_loss_streak, 3, 2, 2)])
    return result(
        f"正收益月={pos_ratio:.2%}; 最大连续亏损={max_loss_streak}个月; 月均={monthly.mean():.2%}",
        level,
        0 if hard else score,
        hard,
        "连续亏损>4个月或正收益月份占比<50%",
    )


# 中文说明：`score_annual_stats`：计算评分或监控指标。
def score_annual_stats(analyzer: Any) -> ScoreResult:
    ret = as_series(analyzer.cache["ret_df"])
    annret = calc_annret(ret)
    annvol = calc_annvol(ret)
    sharpe = annret / annvol if annvol > 0 else 0.0
    maxdd = abs(calc_maxdrawdown(ret))
    year_sharpes = []
    years = pd.DatetimeIndex(ret.index).year
    for _, sub in ret.groupby(years):
        vol = calc_annvol(sub)
        year_sharpes.append(calc_annret(sub) / vol if vol > 0 else 0.0)
    worst_year_sharpe = min(year_sharpes, default=0.0)
    hard = sharpe < 0.5 or worst_year_sharpe < 0
    score, level = grade_all([(sharpe, 1.0, 1.25, 1.5), (annret, 0.10, 0.15, 0.20)], lower_better=[(maxdd, 0.20, 0.15, 0.10)])
    return result(
        f"AnnRet={annret:.2%}; AnnVol={annvol:.2%}; Sharpe={sharpe:.2f}; "
        f"MaxDD={maxdd:.2%}; 最差年度Sharpe={worst_year_sharpe:.2f}",
        level,
        0 if hard else score,
        hard,
        "Overall Sharpe<0.5 或任一年 Sharpe<0",
    )


# 中文说明：`score_basic_performance`：计算评分或监控指标。
def score_basic_performance(analyzer: Any) -> ScoreResult:
    alpha_values = analyzer.cache["alpha_df"].values
    coverage_numer = np.sum(~np.isnan(alpha_values), axis=1)
    coverage_denom = np.nansum(analyzer.cache["pool_mask"], axis=1)
    coverage = np.divide(
        coverage_numer,
        coverage_denom,
        out=np.zeros_like(coverage_numer, dtype=float),
        where=coverage_denom != 0,
    )
    turnover = as_series(analyzer.cache["turnover_df"]).dropna()
    ret_values = as_series(analyzer.cache["ret_df"]).to_numpy(dtype=float)
    cumret = np.nancumsum(ret_values)
    avg_coverage = float(np.nanmean(coverage))
    avg_turnover = float(turnover.mean()) if len(turnover) else np.nan
    trend_up = bool(len(cumret) > 1 and cumret[-1] > cumret[0])
    hard = avg_coverage < 0.50 or not trend_up
    score, level = grade_all([(avg_coverage, 0.75, 0.85, 0.90)], lower_better=[(avg_turnover, 0.50, 0.35, 0.25)])
    return result(
        f"Coverage={avg_coverage:.2%}; 日均换手={avg_turnover:.2%}; 累计收益={'向上' if trend_up else '未向上'}",
        level,
        0 if hard else score,
        hard,
        "Coverage<50% 或累计收益长期向下",
    )


# 中文说明：`score_alpha_distribution_table`：计算评分或监控指标。
def score_alpha_distribution_table(analyzer: Any) -> ScoreResult:
    '''
        中位数漂移 / 参考 IQR：衡量位置漂移
        历年 IQR 比率：衡量横截面区分度是否扩张或坍缩
        Jensen-Shannon 距离：衡量整体分布结构变化
        标准化 Wasserstein 距离：衡量位置和尺度漂移
        偏度跨年变化：只关注稳定性，不区分左偏或右偏
        尾部比率跨年变化：检查肥尾结构是否突然改变
    '''
    alpha_df = analyzer.cache["alpha_df"]
    yearly = {
        int(year): sample_finite_values(group.to_numpy(dtype=float))
        for year, group in alpha_df.groupby(alpha_df.index.year)
    }
    yearly = {year: values for year, values in yearly.items() if len(values) >= 100}
    if len(yearly) < 2:
        return ScoreResult(
            "有效年度少于2年，无法评价历年分布稳定性",
            "待复核",
            45.0,
            "至少需要两个完整年度的有效因子值。",
        )

    reference = sample_finite_values(alpha_df.to_numpy(dtype=float))
    ref_median, ref_iqr = robust_location_scale(reference)
    ref_tail = distribution_tail_ratio(reference)
    bins = stable_histogram_bins(reference)

    # Columns: median drift, IQR ratio, JSD, normalized Wasserstein, skew, tail ratio.
    yearly_metrics = np.empty((len(yearly), 6), dtype=float)
    for row, values in enumerate(yearly.values()):
        median, iqr = robust_location_scale(values)
        yearly_metrics[row] = (
            abs(median - ref_median) / ref_iqr,
            iqr / ref_iqr,
            histogram_js_distance(values, reference, bins),
            wasserstein_distance(values, reference) / ref_iqr,
            pd.Series(values).skew(),
            distribution_tail_ratio(values),
        )

    max_median_drift = float(np.nanmax(yearly_metrics[:, 0]))
    min_iqr_ratio = float(np.nanmin(yearly_metrics[:, 1]))
    max_iqr_ratio = float(np.nanmax(yearly_metrics[:, 1]))
    max_js = float(np.nanmax(yearly_metrics[:, 2]))
    max_wasserstein = float(np.nanmax(yearly_metrics[:, 3]))
    skew_drift = float(np.nanmax(yearly_metrics[:, 4]) - np.nanmin(yearly_metrics[:, 4]))
    tail_change = float(
        np.nanmax(np.abs(yearly_metrics[:, 5] / max(ref_tail, 1e-12) - 1.0))
    )

    lower_score_specs = (
        (max_median_drift, 0.50, 0.25, 0.10),
        (max_js, 0.20, 0.10, 0.05),
        (max_wasserstein, 0.50, 0.25, 0.10),
        (skew_drift, 2.00, 1.00, 0.50),
        (tail_change, 1.00, 0.50, 0.20),
    )
    component_scores = [
        grade_lower(*lower_score_specs[0])[0],
        range_stability_score(min_iqr_ratio, max_iqr_ratio),
        *(grade_lower(*spec)[0] for spec in lower_score_specs[1:]),
    ]
    weights = np.asarray([0.22, 0.22, 0.20, 0.16, 0.08, 0.12], dtype=float)
    score = float(np.dot(component_scores, weights))
    hard = (
        not np.isfinite(ref_iqr)
        or ref_iqr <= 1e-12
        or max_median_drift > 1.0
        or min_iqr_ratio < 0.40
        or max_iqr_ratio > 2.50
        or max_js > 0.30
    )
    level = name_from_level(level_high(score, 65.0, 80.0, 92.0))
    return result(
        (
            f"历年最大中位数漂移/IQR={max_median_drift:.2f}; "
            f"IQR比率区间={min_iqr_ratio:.2f}~{max_iqr_ratio:.2f}; "
            f"最大JSD距离={max_js:.3f}; 标准化Wasserstein={max_wasserstein:.2f}; "
            f"偏度跨年变化={skew_drift:.2f}; 尾部比率最大变化={tail_change:.1%}"
        ),
        level,
        0.0 if hard else score,
        hard,
        "因子历年位置、离散度或分布结构发生明显漂移",
    )


# 中文说明：`score_alpha_distribution_shape`：计算评分或监控指标。
def score_alpha_distribution_shape(analyzer: Any) -> ScoreResult:
    alpha = sample_finite_values(analyzer.cache["alpha_df"].to_numpy(dtype=float))
    if len(alpha) < 100:
        return ScoreResult("无有效因子值", "Fail", 0, "因子值为空。", True, "无有效因子值")

    alpha_series = pd.Series(alpha)
    _, alpha_iqr = robust_location_scale(alpha)
    skew_abs = abs(float(alpha_series.skew()))
    kurtosis = float(alpha_series.kurtosis())
    tail_ratio = distribution_tail_ratio(alpha)
    extreme_ratio = robust_extreme_ratio(alpha, mad_multiple=5.0)

    shape_js = np.nan
    tail_gap = np.nan
    future_ret = sample_finite_values(np.asarray(analyzer.cache["label_arr"], dtype=float))
    if len(future_ret) >= 100:
        alpha_scaled = robust_standardize(alpha)
        return_scaled = robust_standardize(future_ret)
        comparison = np.concatenate([alpha_scaled, return_scaled])
        bins = stable_histogram_bins(comparison)
        shape_js = histogram_js_distance(alpha_scaled, return_scaled, bins)
        tail_gap = abs(distribution_tail_ratio(alpha_scaled) - distribution_tail_ratio(return_scaled))

    lower_score_specs = [
        (skew_abs, 3.0, 2.0, 1.0),
        (max(kurtosis, 0.0), 20.0, 10.0, 5.0),
        (tail_ratio, 20.0, 12.0, 8.0),
        (extreme_ratio, 0.05, 0.02, 0.005),
    ]
    weights = [0.22, 0.20, 0.25, 0.23]
    if np.isfinite(shape_js):
        lower_score_specs.append((shape_js, 0.55, 0.35, 0.20))
        weights.append(0.10)
    component_scores = [grade_lower(*spec)[0] for spec in lower_score_specs]
    score = float(np.average(component_scores, weights=weights))
    hard = (
        not np.isfinite(alpha_iqr)
        or alpha_iqr <= 1e-12
        or skew_abs > 5.0
        or kurtosis > 50.0
        or tail_ratio > 35.0
        or extreme_ratio > 0.10
    )
    level = name_from_level(level_high(score, 65.0, 80.0, 92.0))
    comparison_text = (
        f"; 比收益分布的JSD={shape_js:.3f}; 尾部占比差={tail_gap:.2f}"
        if np.isfinite(shape_js)
        else ""
    )
    return result(
        (
            f"|偏度|={skew_abs:.2f}; 峰度={kurtosis:.2f}; "
            f"尾部占比={tail_ratio:.2f}; 5MAD外样本占比={extreme_ratio:.2%}"
            f"{comparison_text}"
        ),
        level,
        0.0 if hard else score,
        hard,
        "因子分布退化或极端样本占比过高，线性持仓容易被尾部主导",
    )


# 中文说明：`score_ic_annual_stats`：计算评分或监控指标。
def score_ic_annual_stats(analyzer: Any) -> ScoreResult:
    df = analyzer.table_ic_annual_stats()
    overall = select_overall(df, "year")
    avg_ic = float(overall["avg_ic"])
    icir = float(overall["ic_ir"])
    rankic = float(overall["avg_rank_ic"])
    hard = avg_ic <= 0 or rankic <= 0 or np.sign(avg_ic) != np.sign(rankic)
    score, level = grade_all([(avg_ic, 0.08, 0.12, 0.14), (icir, 0.5, 0.75, 1.0), (rankic, 0.10, 0.14, 0.16)])
    return result(
        f"AvgIC={avg_ic:.4f}; ICIR={icir:.2f}; RankIC={rankic:.4f}",
        level,
        0 if hard else score,
        hard,
        "Avg IC<=0、RankIC<=0 或方向不一致",
    )


# 中文说明：`score_ic_distribution`：计算评分或监控指标。
def score_ic_distribution(analyzer: Any) -> ScoreResult:
    ics = as_series(analyzer.cache["ics_df"]).dropna()
    mean_ic = float(ics.mean())
    positive_ratio = float((ics > 0).mean())
    hard = mean_ic <= 0 or positive_ratio<0.5
    score, level = grade_all([(positive_ratio, 0.55, 0.60, 0.65), (mean_ic, 0.01, 0.03, 0.05)])
    return result(
        f"IC均值={mean_ic:.4f}; 正IC占比={positive_ratio:.2%}", 
        level, 0 if hard else score, 
        hard, 
        "IC均值<=0 或 正IC占比<50%"
    )


# 中文说明：`_ic_curve_features`：内部辅助步骤，不作为稳定公共接口。
def _ic_curve_features(values: Any) -> tuple[np.ndarray, float, np.ndarray, float, bool]:
    curves = np.nancumsum(np.asarray(values, dtype=float), axis=1)
    edge = max(1, curves.shape[1] // 10)
    trend = (np.nanmedian(curves[:, -edge:], axis=1) - np.nanmedian(curves[:, :edge], axis=1))
    trend /= max(curves.shape[1] - edge, 1)
    decline = np.max(np.maximum.accumulate(curves[:2], axis=1) - curves[:2], axis=1)
    drawdown = float(np.max(decline / np.maximum(curves[:2, -1] - curves[:2, 0], 1e-6)))
    linear, ranked = trend[[0, 2, 3]], trend[[1, 4, 5]]
    relative_gap = (ranked - linear) / np.maximum(np.abs(ranked) + np.abs(linear), 1e-6)
    return trend, drawdown, relative_gap, float(np.median(np.abs(relative_gap))), bool(np.any(linear * ranked < 0))


# 中文说明：`score_ic_curve`：计算评分或监控指标。
def score_ic_curve(analyzer: Any) -> ScoreResult:
    from metrics import calc_sign_IC  # type: ignore

    cache = analyzer.cache
    values = (as_series(cache["ics_df"]), as_series(cache["rankics_df"]),
              *calc_sign_IC(cache["label_arr"], cache["alpha_df"].values))
    if len(values[0]) < 2:
        return result("IC累计曲线样本不足", "Watch", 45, True, "有效时序样本不足")

    trend, drawdown, rank_gap, structure_gap, conflict = _ic_curve_features(values)
    side = np.array([trend[[2, 4]].mean(), trend[[3, 5]].mean()])
    weak_side_strength = min(side[0] + .005, side[1])
    side_gap = abs(side[0] - side[1]) / max(np.abs(side).sum(), 1e-6)
    strong_side = "均衡" if side_gap < .2 else ("正收益侧" if side[0] > side[1] else "负收益侧")
    structure = ("方向冲突" if conflict else "RankIC明显更强" if np.median(rank_gap) > .3
                 else "IC明显更强" if np.median(rank_gap) < -.3 else "基本一致")

    hard = min(trend[:2]) <= 0 or side[0] < -.005 or side[1] <= 0 or drawdown > .35 or conflict
    score, level = grade_all(
        [(min(trend[:2]), .01, .03, .05), (weak_side_strength, .005, .02, .04)],
        lower_better=[(drawdown, .30, .20, .10), (structure_gap, .50, .35, .20)],
    )
    if structure != "基本一致" or side_gap > .40:
        score, level = min(score, 45), "Watch"
    actual = (f"总体趋势={min(trend[:2]):.4f}; 最大回撤={drawdown:.2%}; "
              f"强侧={strong_side}; 弱侧强度={weak_side_strength:.4f}; 侧别差={side_gap:.2%}; "
              f"结构={structure}; 结构差={structure_gap:.2%}")
    return result(actual, level, 0 if hard else score, hard,
                  "总体趋势非正、累计曲线回撤>35%、收益侧识别失效、IC/RankIC结构差异较大")


# 中文说明：`score_group_stats`：计算评分或监控指标。
def score_group_stats(analyzer: Any) -> ScoreResult:
    df = analyzer.table_group_stats()
    g10 = float(df.loc["G10", "AnnRet"])
    g1 = float(df.loc["G1", "AnnRet"])
    spread = g1 - g10
    hard = g1 <= g10
    score, level = grade_high(spread, 0.10, 0.15, 0.20)
    return result(f"G1 AnnRet={g1:.2%}; G10 AnnRet={g10:.2%}; G1-G10 Spread={spread:.2%}", level, 0 if hard else score, hard, "G1<=G10 or factor direction reversed")


# 中文说明：`score_group_curve`：计算评分或监控指标。
def score_group_curve(analyzer: Any) -> ScoreResult:
    groupret = analyzer.cache["groupret_df"]
    end_values = groupret.cumsum().iloc[-1]
    monotonic_ratio = monotonic_descending_adjacent_ratio(end_values.sort_index())
    score, level = grade_high(monotonic_ratio, 0.60, 0.75, 0.90)
    hard = monotonic_ratio < 0.45
    return result(f"期末分组单调邻接比例={monotonic_ratio:.2%}", level, 0 if hard else score, hard, "组间曲线交叉严重")


# 中文说明：`score_industry_annual`：计算评分或监控指标。
def score_industry_annual(analyzer: Any) -> ScoreResult:
    df = analyzer.table_industry_annual_stats()
    sharpe_col = find_col(df, ["Sharpe", "夏普"])
    positive_ratio = float((df[sharpe_col] > 0).mean())
    score, level = grade_high(positive_ratio, 0.70, 0.75, 0.80)
    hard = positive_ratio < 0.60
    return result(f"行业 Sharpe>0 占比={positive_ratio:.2%}", level, 0 if hard else score, hard, "少数行业贡献过高，多数行业无效")


# 中文说明：`score_industry_curve`：计算评分或监控指标。
def score_industry_curve(analyzer: Any) -> ScoreResult:
    df = analyzer.table_industry_annual_stats()
    ann_col = find_col(df, ["AnnRet", "年化收益"])
    positive_ratio = float((df[ann_col] > 0).mean())
    score, level = grade_high(positive_ratio, 0.60, 0.70, 0.80)
    hard = positive_ratio <= 0.50
    return result(f"行业年化收益为正占比={positive_ratio:.2%}", level, 0 if hard else score, hard, "超过半数行业长期下行")


# 中文说明：`score_industry_exposure`：计算评分或监控指标。
def score_industry_exposure(analyzer: Any) -> ScoreResult:
    df = analyzer.table_industry_exposure_stats()
    max_abs = float(df["Avg Exposure"].abs().max())
    hard = max_abs > 0.20
    score, level = grade_lower(max_abs, 0.10, 0.07, 0.05)
    return result(f"最大|行业平均暴露|={max_abs:.2%}", level, 0 if hard else score, hard, "单行业平均暴露>20%")


# 中文说明：`score_industry_exposure_return`：计算评分或监控指标。
def score_industry_exposure_return(analyzer: Any) -> ScoreResult:
    ensure_industry_cache(analyzer)
    total = abs(float(calc_annret(as_series(analyzer.cache["ret_df"]))))
    contrib = exposure_contribution(analyzer.cache["long_ind_pct_df"], analyzer.cache["short_ind_pct_df"], analyzer.cache["ind_ret_df"], analyzer.factor_type, total)
    hard = contrib > 0.50
    score, level = grade_lower(contrib, 0.30, 0.20, 0.15)
    return result(f"最大单行业平均暴露收益贡献={contrib:.2%}", level, 0 if hard else score, hard, "单行业平均暴露收益贡献最大>50%")


# 中文说明：`score_industry_component`：计算评分或监控指标。
def score_industry_component(analyzer: Any) -> ScoreResult:
    long_pct, short_pct = analyzer.calc_ind_exposure()
    cr5 = float(long_pct.mean().sort_values(ascending=False).head(5).sum())
    max_single = float(long_pct.mean().max())
    hard = max_single > 0.20 or cr5 > 0.5
    score, level = grade_lower(cr5, 0.60, 0.50, 0.40)
    return result(f"行业多头CR5={cr5:.2%}; 最大单行业多头={max_single:.2%}", level, 0 if hard else score, hard, "前5大持仓行业占比>50% 或 单行业持仓占比>20%")


# 中文说明：`score_sector_annual`：计算评分或监控指标。
def score_sector_annual(analyzer: Any) -> ScoreResult:
    df = analyzer.table_sector_annual_stats()
    sharpe_col = find_col(df, ["Sharpe", "夏普"])
    positive_ratio = float((df[sharpe_col] > 0).mean())
    score, level = grade_high(positive_ratio, 0.60, 0.75, 0.95)
    hard = positive_ratio <= 0.20
    return result(f"板块 Sharpe>0 占比={positive_ratio:.2%}", level, 0 if hard else score, hard, "仅单一板块有效")


# 中文说明：`score_sector_curve`：计算评分或监控指标。
def score_sector_curve(analyzer: Any) -> ScoreResult:
    df = analyzer.table_sector_annual_stats()
    ann_col = find_col(df, ["AnnRet", "年化收益"])
    positive_ratio = float((df[ann_col] > 0).mean())
    score, level = grade_high(positive_ratio, 0.60, 0.75, 0.95)
    hard = positive_ratio <= 0.50
    return result(f"板块年化收益为正占比={positive_ratio:.2%}", level, 0 if hard else score, hard, "超过半数板块向下")


# 中文说明：`score_sector_exposure`：计算评分或监控指标。
def score_sector_exposure(analyzer: Any) -> ScoreResult:
    df = analyzer.table_sector_exposure_stats()
    max_abs = float(df["Avg Exposure"].abs().max())
    hard = max_abs > 0.30
    score, level = grade_lower(max_abs, 0.15, 0.12, 0.10)
    return result(f"最大|板块平均暴露|={max_abs:.2%}", level, 0 if hard else score, hard, "单板块长期暴露>30%")


# 中文说明：`score_sector_exposure_return`：计算评分或监控指标。
def score_sector_exposure_return(analyzer: Any) -> ScoreResult:
    ensure_sector_cache(analyzer)
    total = abs(float(calc_annret(as_series(analyzer.cache["ret_df"]))))
    contrib = exposure_contribution(analyzer.cache["long_sec_pct_df"], analyzer.cache["short_sec_pct_df"], analyzer.cache["sec_ret_df"], analyzer.factor_type, total)
    hard = contrib > 0.50
    score, level = grade_lower(contrib, 0.30, 0.20, 0.15)
    return result(f"最大单板块平均暴露收益贡献={contrib:.2%}", level, 0 if hard else score, hard, "单板块平均暴露收益贡献最大>50%")


# 中文说明：`score_sector_component`：计算评分或监控指标。
def score_sector_component(analyzer: Any) -> ScoreResult:
    long_pct, _ = analyzer.calc_sec_exposure()
    cr3 = float(long_pct.mean().sort_values(ascending=False).head(3).sum())
    hard = cr3 > 0.70
    score, level = grade_lower(cr3, 0.50, 0.40, 0.35)
    return result(f"板块多头CR3={cr3:.2%}", level, 0 if hard else score, hard, "CR3>70%")


# 中文说明：`score_barra_exposure_stats`：计算评分或监控指标。
def score_barra_exposure_stats(analyzer: Any) -> ScoreResult:
    df = analyzer.table_barra_exposure_stats()
    max_abs = float(df["Avg Exposure"].abs().max())
    hard = max_abs > 0.35
    score, level = grade_lower(max_abs, 0.20, 0.15, 0.10)
    return result(f"最大|Barra平均暴露|={max_abs:.3f}", level, 0 if hard else score, hard, "最大|Barra平均暴露|>0.35")


# 中文说明：`score_barra_exposure_bar`：计算评分或监控指标。
def score_barra_exposure_bar(analyzer: Any) -> ScoreResult:
    if "barra_exposure_df" not in analyzer.cache:
        analyzer.table_barra_exposure_stats()
    mean_abs = float(analyzer.cache["barra_exposure_df"].mean().abs().mean())
    max_abs = float(analyzer.cache["barra_exposure_df"].mean().abs().max())
    hard = max_abs > 0.35
    score, level = grade_lower(max_abs, 0.20, 0.15, 0.10)
    return result(f"平均|暴露|={mean_abs:.3f}; 最大|暴露|={max_abs:.3f}", level, 0 if hard else score, hard, "最大|平均暴露|>0.35")


# 中文说明：`score_barra_exposure_return`：计算评分或监控指标。
def score_barra_exposure_return(analyzer: Any) -> ScoreResult:
    if "barra_exposure_df" not in analyzer.cache or "barra_ret_df" not in analyzer.cache:
        analyzer.table_barra_exposure_stats()
    total = abs(float(calc_annret(as_series(analyzer.cache["ret_df"]))))
    contribs = []
    for col in analyzer.cache["barra_exposure_df"].columns:
        contrib_ret = analyzer.cache["barra_exposure_df"][col] * analyzer.cache["barra_ret_df"][col]
        contribs.append(abs(float(calc_annret(contrib_ret))) / max(total, 1e-12))
    max_contrib = float(np.nanmax(contribs)) if contribs else np.nan
    hard = max_contrib > 0.50
    score, level = grade_lower(max_contrib, 0.30, 0.20, 0.15)
    return result(f"最大单Barra收益贡献={max_contrib:.2%}", level, 0 if hard else score, hard, "最大单Barra收益贡献>50%")


# 中文说明：`score_redundancy`：计算评分或监控指标。
def score_redundancy(analyzer: Any) -> ScoreResult:
    try:
        corr_df = calc_pool_corr_summary(analyzer)
    except Exception as exc:
        return ScoreResult(f"冗余检验依赖因子池，自动计算失败: {exc}", "待复核", 50, "请根据 plot_corr_redundancy 热图人工复核。")
    max_corr = float(corr_df["abs_corr"].max())
    avg_corr = float(corr_df["abs_corr"].mean())
    hard = max_corr >= 0.60
    score, level = grade_all([], lower_better=[(max_corr, 0.70, 0.60, 0.50), (avg_corr, 0.30, 0.25, 0.20)])
    return result(f"最大相关={max_corr:.3f}; 平均相关={avg_corr:.3f}", level, 0 if hard else score, hard, "最大相关>=0.60")


# 中文说明：`score_regime_stats`：计算评分或监控指标。
def score_regime_stats(analyzer: Any) -> ScoreResult:
    df = analyzer.table_regime_stats()
    if df.empty:
        return ScoreResult("Regime样本不足，未生成有效统计", "待复核", 45.0, "请延长样本或补充benchmark后复核。")

    sharpe = pd.to_numeric(df.get("Sharpe"), errors="coerce")
    annret = pd.to_numeric(df.get("AnnRet"), errors="coerce")
    avg_ic = pd.to_numeric(df.get("Avg_IC"), errors="coerce")
    win_rate = pd.to_numeric(df.get("WinRate"), errors="coerce")

    min_sharpe = float(sharpe.min())
    positive_sharpe_ratio = float((sharpe > 0).mean())
    positive_ic_ratio = float((avg_ic > 0).mean()) if avg_ic.notna().any() else 0.0
    positive_annret_ratio = float((annret > 0).mean())
    avg_win_rate = float(win_rate.mean()) if win_rate.notna().any() else np.nan

    hard = min_sharpe < 0 or positive_sharpe_ratio < 0.50 or positive_ic_ratio < 0.50
    score, level = grade_all(
        [(positive_sharpe_ratio, 0.60, 0.75, 0.85), (positive_ic_ratio, 0.60, 0.75, 0.85)],
        lower_better=[(-min_sharpe, 0.50, 0.20, 0.00)],
    )
    return result(
        (
            f"Regime数={len(df)}; Sharpe>0占比={positive_sharpe_ratio:.2%}; "
            f"AnnRet>0占比={positive_annret_ratio:.2%}; IC>0占比={positive_ic_ratio:.2%}; "
            f"最差Sharpe={min_sharpe:.2f}; 平均胜率={avg_win_rate:.2%}"
        ),
        level,
        0 if hard else score,
        hard,
        "某一市场状态下显著失效，暂不满足上线稳定性",
    )


# 中文说明：`score_regime_curve`：计算评分或监控指标。
def score_regime_curve(analyzer: Any) -> ScoreResult:
    df = analyzer.table_regime_stats()
    if df.empty:
        return ScoreResult("Regime累计收益无法评分", "待复核", 45.0, "请确认样本覆盖牛/熊/震荡或高低波动阶段。")
    annret = pd.to_numeric(df.get("AnnRet"), errors="coerce")
    maxdd = pd.to_numeric(df.get("MaxDrawdown"), errors="coerce").abs()
    positive_annret_ratio = float((annret > 0).mean())
    worst_dd = float(maxdd.max()) if maxdd.notna().any() else np.nan
    hard = positive_annret_ratio < 0.50 or (not np.isnan(worst_dd) and worst_dd > 0.35)
    score, level = grade_all([(positive_annret_ratio, 0.60, 0.75, 0.85)], lower_better=[(worst_dd, 0.30, 0.22, 0.15)])
    return result(
        f"状态年化收益为正占比={positive_annret_ratio:.2%}; 最差状态回撤={worst_dd:.2%}",
        level,
        0 if hard else score,
        hard,
        "多数状态累计收益不向上或某一状态内回撤过大",
    )


# 中文说明：`score_regime_item`：计算评分或监控指标。
def score_regime_item(analyzer: Any, item: str) -> ScoreResult:
    compact = item.replace(" ", "").lower()
    df = analyzer.table_regime_stats()
    if df.empty:
        return ScoreResult("Regime样本不足", "待复核", 45.0, "请补充更长样本或benchmark后复核。")

    if "风格" in compact or "style" in compact:
        avg_ic = pd.to_numeric(df.get("Avg_IC"), errors="coerce").dropna()
        if len(avg_ic) < 2:
            return ScoreResult("风格/状态IC样本不足", "待复核", 45.0, "当前Regime统计不足以计算稳定性。")
        instability = float(avg_ic.std() / max(abs(avg_ic.mean()), 1e-12))
        hard = instability > 0.80
        score, level = grade_lower(instability, 0.50, 0.40, 0.30)
        return result(
            f"状态Avg_IC波动/均值={instability:.2f}",
            level,
            0 if hard else score,
            hard,
            "风格/状态切换下IC稳定性不足",
        )

    target = regime_target_name(item, df.index)
    if target is None:
        return score_regime_stats(analyzer)
    row = df.loc[target]
    sharpe = float(row.get("Sharpe", np.nan))
    avg_ic = float(row.get("Avg_IC", np.nan))
    annret = float(row.get("AnnRet", np.nan))
    maxdd = abs(float(row.get("MaxDrawdown", np.nan)))
    win_rate = float(row.get("WinRate", np.nan))

    if "熊" in compact or "bear" in compact:
        pass_v, good_v, excellent_v = 0.50, 0.80, 1.00
    elif "高波" in compact or "highvol" in compact or "vol" in compact:
        pass_v, good_v, excellent_v = 0.00, 0.80, 1.00
    else:
        pass_v, good_v, excellent_v = 0.80, 1.00, 1.20
    hard = (not np.isnan(avg_ic) and avg_ic <= 0) or (not np.isnan(sharpe) and sharpe < 0)
    score, level = grade_high(sharpe, pass_v, good_v, excellent_v)
    return result(
        f"{target}: Sharpe={sharpe:.2f}; Avg_IC={avg_ic:.4f}; AnnRet={annret:.2%}; MaxDD={maxdd:.2%}; WinRate={win_rate:.2%}",
        level,
        0 if hard else score,
        hard,
        f"{target}状态IC<=0或Sharpe失效",
    )


# 中文说明：`regime_target_name`：执行该名称对应的业务计算，并返回调用方所需结果。
def regime_target_name(item: str, index: pd.Index) -> Any | None:
    candidates = list(index)
    item_text = item.lower()
    rules = [
        (("牛", "bull"), ("牛", "bull", "鐗")),
        (("熊", "bear"), ("熊", "bear", "鐔")),
        (("震", "side"), ("震", "side", "闇")),
        (("高波", "high"), ("高", "high", "楂")),
        (("低波", "low"), ("低", "low", "浣")),
    ]
    for item_keys, row_keys in rules:
        if any(key in item_text for key in item_keys):
            for candidate in candidates:
                text = str(candidate).lower()
                if any(key.lower() in text for key in row_keys):
                    return candidate
    return None


# 中文说明：`score_shadow_capacity`：计算评分或监控指标。
def score_shadow_capacity(analyzer: Any) -> ScoreResult:
    df = analyzer.table_shadow_capacity_test()
    if df.empty:
        return ScoreResult("容量测试无有效结果", "待复核", 45.0, "请检查持仓、换手和流动性数据。")

    fill = pd.to_numeric(df.get("AvgFillRatio"), errors="coerce")
    sharpe = pd.to_numeric(df.get("Sharpe_Net"), errors="coerce")
    annret = pd.to_numeric(df.get("AnnRet_Net"), errors="coerce")
    maxdd = pd.to_numeric(df.get("MaxDrawdown_Net"), errors="coerce").abs()
    decision = df.get("OnlineDecision", pd.Series(index=df.index, dtype=object)).astype(str)

    min_fill = float(fill.min())
    min_sharpe = float(sharpe.min())
    pass_ratio = float(decision.str.lower().eq("pass").mean()) if len(decision) else 0.0
    max_capacity = df.index[decision.str.lower().eq("pass")].max() if pass_ratio > 0 else np.nan
    min_annret = float(annret.min())
    worst_dd = float(maxdd.max()) if maxdd.notna().any() else np.nan

    hard = min_fill < 0.60 or min_sharpe < 0.50 or pass_ratio == 0
    score, level = grade_all(
        [(min_fill, 0.80, 0.90, 0.95), (min_sharpe, 1.00, 1.25, 1.50), (pass_ratio, 0.50, 0.75, 1.00)],
        lower_better=[(worst_dd, 0.30, 0.22, 0.15)],
    )
    capacity_text = "无" if pd.isna(max_capacity) else f"{float(max_capacity)/1e8:.1f}亿"
    return result(
        (
            f"最小成交填充率={min_fill:.2%}; 最低Sharpe={min_sharpe:.2f}; "
            f"最小年化={min_annret:.2%}; 最大回撤={worst_dd:.2%}; "
            f"通过容量占比={pass_ratio:.2%}; 建议上线容量={capacity_text}"
        ),
        level,
        0 if hard else score,
        hard,
        "给定下线容量尚且不满足要求",
    )


# 中文说明：`score_capacity_item`：计算评分或监控指标。
def score_capacity_item(analyzer: Any, item: str) -> ScoreResult:
    compact = item.replace(" ", "").lower()
    df = analyzer.table_shadow_capacity_test()
    if df.empty:
        return ScoreResult("容量测试无有效结果", "待复核", 45.0, "请检查持仓、换手和流动性数据。")

    if "成交" in compact or "fill" in compact:
        fill = pd.to_numeric(df.get("AvgFillRatio"), errors="coerce")
        min_fill = float(fill.min())
        hard = min_fill < 0.60
        score, level = grade_high(min_fill, 0.70, 0.80, 0.90)
        return result(f"全容量最小AvgFillRatio={min_fill:.2%}", level, 0 if hard else score, hard, "成交满足率<60%")

    if "冲击" in compact or "impact" in compact:
        impact = pd.to_numeric(df.get("AvgImpactCost"), errors="coerce")
        turnover = pd.to_numeric(df.get("AvgTurnover"), errors="coerce")
        impact_proxy = float((impact / np.maximum(turnover.abs(), 1e-12)).max())
        hard = impact_proxy > 0.50
        score, level = grade_lower(impact_proxy, 0.40, 0.30, 0.20)
        return result(f"冲击成本/换手代理={impact_proxy:.2%}", level, 0 if hard else score, hard, "冲击成本占收益即交易强度过高")

    capital = target_capital_from_item(item)
    if capital is None:
        return score_shadow_capacity(analyzer)
    row = nearest_capacity_row(df, capital)
    fill = float(row.get("AvgFillRatio", np.nan))
    sharpe = float(row.get("Sharpe_Net", np.nan))
    annret = float(row.get("AnnRet_Net", np.nan))
    maxdd = abs(float(row.get("MaxDrawdown_Net", np.nan)))
    decision = str(row.get("OnlineDecision", ""))

    if capital <= 1e7:
        hard = sharpe < 0.80
        score, level = grade_all([(sharpe, 1.00, 1.20, 1.50), (fill, 0.80, 0.85, 0.90)])
        gate_note = "1000万成本后Sharpe<0.8"
    elif capital <= 5e7:
        hard = fill < 0.60 or sharpe < 0.50
        score, level = grade_all([(fill, 0.70, 0.80, 0.90), (sharpe, 0.70, 1.00, 1.20)])
        gate_note = "5000万收益保留或成交满足率不达标"
    else:
        hard = fill < 0 or sharpe < 0.40
        score, level = grade_all([(sharpe, 0.50, 0.70, 0.90), (fill, 0.60, 0.75, 0.85)])
        gate_note = "1亿收益保留或成交满足率不达标"
    return result(
        f"{capital/1e8:.2f}亿: AvgFillRatio={fill:.2%}; Sharpe_Net={sharpe:.2f}; AnnRet_Net={annret:.2%}; MaxDD_Net={maxdd:.2%}; OnlineDecision={decision}",
        level,
        0 if hard else score,
        hard,
        gate_note,
    )


# 中文说明：`target_capital_from_item`：执行该名称对应的业务计算，并返回调用方所需结果。
def target_capital_from_item(item: str) -> float | None:
    text = item.replace(" ", "").lower()
    if "1000万" in text or "0.1亿" in text:
        return 1e7
    if "5000万" in text or "0.5亿" in text:
        return 5e7
    if "1亿" in text or "一亿" in text:
        return 1e8
    return None


# 中文说明：`nearest_capacity_row`：执行该名称对应的业务计算，并返回调用方所需结果。
def nearest_capacity_row(df: pd.DataFrame, capital: float) -> pd.Series:
    numeric_index = pd.to_numeric(pd.Index(df.index), errors="coerce")
    if np.isnan(numeric_index).all():
        return df.iloc[0]
    idx = int(np.nanargmin(np.abs(numeric_index - capital)))
    return df.iloc[idx]


# 中文说明：`score_shadow_capacity_curve`：计算评分或监控指标。
def score_shadow_capacity_curve(analyzer: Any) -> ScoreResult:
    df = analyzer.table_shadow_capacity_test()
    if df.empty:
        return ScoreResult("容量曲线无法评分", "待复核", 45.0, "请检查容量测试输入。")
    annret = pd.to_numeric(df.get("AnnRet_Net"), errors="coerce")
    sharpe = pd.to_numeric(df.get("Sharpe_Net"), errors="coerce")
    monotonic_ok = bool(annret.is_monotonic_decreasing or len(annret.dropna()) <= 1)
    positive_ratio = float((annret > 0).mean())
    min_sharpe = float(sharpe.min())
    hard = positive_ratio < 0.50 or min_sharpe < 0.50
    score, level = grade_all([(positive_ratio, 0.60, 0.75, 1.00), (min_sharpe, 1.00, 1.25, 1.50)])
    if not monotonic_ok:
        score = min(score, 70.0)
        level = "Pass" if score >= 70 else "Watch"
    return result(
        f"正年化收益占比={positive_ratio:.2%}; 最低Sharpe={min_sharpe:.2f}; 容量曲线单调性={'正常' if monotonic_ok else '需复核'}",
        level,
        0 if hard else score,
        hard,
        "容量扩大后净值曲线失效",
    )


# 中文说明：`result`：执行该名称对应的业务计算，并返回调用方所需结果。
def result(actual: str, level: str, score: float, hard: bool, hard_note: str) -> ScoreResult:
    if hard:
        return ScoreResult(actual, "Fail", 0.0, "未通过自动评分标准", True, hard_note)
    opinions = {
        "Excellent": "达到优秀上线候选标准。",
        "Good": "指标较好，可进入复测或组合候选。",
        "Pass": "达到基础通过线，建议继续观察稳定性。",
        "Watch": "未完全达标，建议复核并优化后再评审。",
    }
    return ScoreResult(actual, level, float(score), opinions.get(level, "请人工复核。"))


# 中文说明：`grade_all`：执行该名称对应的业务计算，并返回调用方所需结果。
def grade_all(
    higher_better: list[tuple[float, float, float, float]],
    lower_better: list[tuple[float, float, float, float]] | None = None,
) -> tuple[float, str]:
    levels = []
    for value, pass_v, good_v, excellent_v in higher_better:
        levels.append(level_high(value, pass_v, good_v, excellent_v))
    for value, pass_v, good_v, excellent_v in (lower_better or []):
        levels.append(level_lower(value, pass_v, good_v, excellent_v))
    if not levels:
        return 50.0, "待复核"
    level_value = min(levels)
    return score_from_level(level_value), name_from_level(level_value)


# 中文说明：`grade_high`：执行该名称对应的业务计算，并返回调用方所需结果。
def grade_high(value: float, pass_v: float, good_v: float, excellent_v: float) -> tuple[float, str]:
    level = level_high(value, pass_v, good_v, excellent_v)
    return score_from_level(level), name_from_level(level)


# 中文说明：`grade_lower`：执行该名称对应的业务计算，并返回调用方所需结果。
def grade_lower(value: float, pass_v: float, good_v: float, excellent_v: float) -> tuple[float, str]:
    level = level_lower(value, pass_v, good_v, excellent_v)
    return score_from_level(level), name_from_level(level)


# 中文说明：`level_high`：执行该名称对应的业务计算，并返回调用方所需结果。
def level_high(value: float, pass_v: float, good_v: float, excellent_v: float) -> int:
    if value >= excellent_v:
        return 3
    if value >= good_v:
        return 2
    if value >= pass_v:
        return 1
    return 0


# 中文说明：`level_lower`：执行该名称对应的业务计算，并返回调用方所需结果。
def level_lower(value: float, pass_v: float, good_v: float, excellent_v: float) -> int:
    if value <= excellent_v:
        return 3
    if value <= good_v:
        return 2
    if value <= pass_v:
        return 1
    return 0


# 中文说明：`score_from_level`：计算评分或监控指标。
def score_from_level(level: int) -> float:
    return {3: 100.0, 2: 85.0, 1: 70.0, 0: 45.0}.get(level, 50.0)


# 中文说明：`name_from_level`：执行该名称对应的业务计算，并返回调用方所需结果。
def name_from_level(level: int) -> str:
    return {3: "Excellent", 2: "Good", 1: "Pass", 0: "Watch"}.get(level, "待复核")


# 中文说明：`parse_weight`：解析外部输入。
def parse_weight(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        return float(text)
    return float(value)


# 中文说明：`conclusion_from_score`：执行该名称对应的业务计算，并返回调用方所需结果。
def conclusion_from_score(score: float, hard_fail: bool) -> str:
    if hard_fail:
        return "Fail / 硬Gate不通过"
    if score >= 85:
        return "Excellent / 可重点推进"
    if score >= PASS_LINE:
        return "Pass / 通过"
    if score >= 65:
        return "Watch / 复测观察"
    return "Fail / 不通过"


# 中文说明：`write_scorebook`：执行该名称对应的业务计算，并返回调用方所需结果。
def write_scorebook(score_df: pd.DataFrame, summary: dict[str, Any], template_path: str | Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        try:
            for sheet_name in pd.ExcelFile(template_path).sheet_names:
                if sheet_name != DEFAULT_SHEET:
                    pd.read_excel(template_path, sheet_name=sheet_name).to_excel(writer, sheet_name=sheet_name, index=False)
        except Exception:
            pass
        score_df.to_excel(writer, sheet_name=DEFAULT_SHEET, index=False)
        pd.DataFrame(
            [
                ("总分", round(summary["total_score"], 2)),
                ("原始加权分", round(summary["raw_weighted_score"], 4)),
                ("权重合计", round(summary["weight_sum"], 4)),
                ("结论", summary["conclusion"]),
                ("硬Gate", "；".join(summary["hard_gate_notes"]) if summary["hard_gate_notes"] else "无"),
            ],
            columns=["项目", "结果"],
        ).to_excel(writer, sheet_name="评分汇总", index=False)


# 中文说明：`resolve_output_path`：解析配置或数据映射。
def resolve_output_path(output: str, analyzer: Any) -> Path:
    if output:
        return Path(output).expanduser()
    name = str(analyzer.info_table.get("name", "factor"))
    return Path.cwd() / f"{name}_score.xlsx"


# 中文说明：`as_series`：执行该名称对应的业务计算，并返回调用方所需结果。
def as_series(value: Any) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 1:
            return value.iloc[:, 0]
        return value.mean(axis=1)
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value)


# 中文说明：`sample_finite_values`：执行该名称对应的业务计算，并返回调用方所需结果。
def sample_finite_values(value: Any, max_points: int = 200_000) -> np.ndarray:
    values = np.asarray(value, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) <= max_points:
        return values
    indices = np.linspace(0, len(values) - 1, max_points, dtype=int)
    return values[indices]


# 中文说明：`robust_location_scale`：执行该名称对应的业务计算，并返回调用方所需结果。
def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    q25, median, q75 = np.nanquantile(values, [0.25, 0.50, 0.75])
    return float(median), float(max(q75 - q25, 1e-12))


# 中文说明：`robust_standardize`：执行该名称对应的业务计算，并返回调用方所需结果。
def robust_standardize(values: np.ndarray) -> np.ndarray:
    median, iqr = robust_location_scale(values)
    return np.clip((values - median) / iqr, -20.0, 20.0)


# 中文说明：`distribution_tail_ratio`：执行该名称对应的业务计算，并返回调用方所需结果。
def distribution_tail_ratio(values: np.ndarray) -> float:
    q01, q25, q75, q99 = np.nanquantile(values, [0.01, 0.25, 0.75, 0.99])
    return float((q99 - q01) / max(q75 - q25, 1e-12))


# 中文说明：`robust_extreme_ratio`：执行该名称对应的业务计算，并返回调用方所需结果。
def robust_extreme_ratio(values: np.ndarray, mad_multiple: float = 5.0) -> float:
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    if mad <= 1e-12:
        return 1.0 if np.any(np.abs(values - median) > 1e-12) else 0.0
    return float(np.mean(np.abs(values - median) > mad_multiple * mad))


# 中文说明：`stable_histogram_bins`：执行该名称对应的业务计算，并返回调用方所需结果。
def stable_histogram_bins(values: np.ndarray, n_bins: int = 30) -> np.ndarray:
    lower, upper = np.nanquantile(values, [0.001, 0.999])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
    if upper <= lower:
        upper = lower + 1.0
    return np.linspace(lower, upper, n_bins + 1)


# 中文说明：`histogram_js_distance`：执行该名称对应的业务计算，并返回调用方所需结果。
def histogram_js_distance(values: np.ndarray, reference: np.ndarray, bins: np.ndarray) -> float:
    lower, upper = float(bins[0]), float(bins[-1])
    values_hist, _ = np.histogram(np.clip(values, lower, upper), bins=bins)
    reference_hist, _ = np.histogram(np.clip(reference, lower, upper), bins=bins)
    epsilon = 1e-12
    p = values_hist.astype(float) + epsilon
    q = reference_hist.astype(float) + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q, base=2.0))


# 中文说明：`range_stability_score`：执行该名称对应的业务计算，并返回调用方所需结果。
def range_stability_score(min_ratio: float, max_ratio: float) -> float:
    if min_ratio >= 0.80 and max_ratio <= 1.25:
        return score_from_level(3)
    if min_ratio >= 0.67 and max_ratio <= 1.50:
        return score_from_level(2)
    if min_ratio >= 0.50 and max_ratio <= 2.00:
        return score_from_level(1)
    return score_from_level(0)


# 中文说明：`monthly_returns`：执行该名称对应的业务计算，并返回调用方所需结果。
def monthly_returns(analyzer: Any) -> pd.Series:
    ret = as_series(analyzer.cache["ret_df"])
    try:
        return ret.resample("ME").apply(lambda x: (1 + x).prod() - 1).dropna()
    except ValueError:
        return ret.resample("M").apply(lambda x: (1 + x).prod() - 1).dropna()


# 中文说明：`max_consecutive`：执行该名称对应的业务计算，并返回调用方所需结果。
def max_consecutive(mask: pd.Series | np.ndarray) -> int:
    max_run = run = 0
    for flag in np.asarray(mask, dtype=bool):
        run = run + 1 if flag else 0
        max_run = max(max_run, run)
    return max_run


# 中文说明：`calc_annret`：计算研究或生产指标。
def calc_annret(ret: pd.Series) -> float:
    ret = as_series(ret).dropna()
    if len(ret) < 2:
        return 0.0
    nav = np.nanprod(1 + ret.values)
    years = max((ret.index[-1] - ret.index[0]).days / 365.25, 1 / 365.25)
    return float(nav ** (1 / years) - 1)


# 中文说明：`calc_annvol`：计算研究或生产指标。
def calc_annvol(ret: pd.Series) -> float:
    ret = as_series(ret).dropna()
    return float(np.nanstd(ret.values) * np.sqrt(242))


# 中文说明：`calc_maxdrawdown`：计算研究或生产指标。
def calc_maxdrawdown(ret: pd.Series) -> float:
    ret = as_series(ret).dropna()
    if len(ret) == 0:
        return 0.0
    nav = np.nancumprod(1 + ret.values)
    return max_drawdown_from_nav(nav)


# 中文说明：`max_drawdown_from_nav`：执行该名称对应的业务计算，并返回调用方所需结果。
def max_drawdown_from_nav(nav: pd.Series | np.ndarray) -> float:
    arr = np.asarray(nav, dtype=float)
    if len(arr) == 0:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    return float(np.nanmin((arr - running_max) / np.maximum(running_max, 1e-12)))


# 中文说明：`select_overall`：执行该名称对应的业务计算，并返回调用方所需结果。
def select_overall(df: pd.DataFrame, year_col: str) -> pd.Series:
    if year_col in df.columns:
        mask = df[year_col].astype(str).str.lower() == "overall"
        if mask.any():
            return df.loc[mask].iloc[0]
    return df.iloc[-1]


# 中文说明：`find_col`：执行该名称对应的业务计算，并返回调用方所需结果。
def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in df.columns:
        col_text = str(col)
        if any(candidate.lower() in col_text.lower() for candidate in candidates):
            return col
    raise KeyError(f"找不到列: {candidates}")


# 中文说明：`monotonic_adjacent_ratio`：执行该名称对应的业务计算，并返回调用方所需结果。
def monotonic_adjacent_ratio(series: pd.Series) -> float:
    values = series.values.astype(float)
    if len(values) < 2:
        return 0.0
    return float(np.mean(np.diff(values) >= 0))


# 中文说明：`monotonic_descending_adjacent_ratio`：执行该名称对应的业务计算，并返回调用方所需结果。
def monotonic_descending_adjacent_ratio(series: pd.Series) -> float:
    values = series.values.astype(float)
    if len(values) < 2:
        return 0.0
    return float(np.mean(np.diff(values) <= 0))


# 中文说明：`ensure_industry_cache`：执行该名称对应的业务计算，并返回调用方所需结果。
def ensure_industry_cache(analyzer: Any) -> None:
    if not {"long_ind_pct_df", "short_ind_pct_df", "ind_ret_df"}.issubset(analyzer.cache):
        analyzer.table_industry_exposure_stats()


# 中文说明：`ensure_sector_cache`：执行该名称对应的业务计算，并返回调用方所需结果。
def ensure_sector_cache(analyzer: Any) -> None:
    if not {"long_sec_pct_df", "short_sec_pct_df", "sec_ret_df"}.issubset(analyzer.cache):
        analyzer.table_sector_exposure_stats()


# 中文说明：`exposure_contribution`：执行该名称对应的业务计算，并返回调用方所需结果。
def exposure_contribution(long_pct: pd.DataFrame, short_pct: pd.DataFrame, ret_df: pd.DataFrame, factor_type: str, total_annret: float) -> float:
    contributions = []
    for col in ret_df.columns:
        weight = long_pct[col] - short_pct[col] if factor_type == "longshort" else long_pct[col]
        contributions.append(abs(calc_annret(weight * ret_df[col])) / max(total_annret, 1e-12))
    return float(np.nanmax(contributions)) if contributions else np.nan


# 中文说明：`calc_pool_corr_summary`：计算研究或生产指标。
def calc_pool_corr_summary(analyzer: Any) -> pd.DataFrame:
    from metrics import IC, rankIC  # type: ignore

    valid_dates = np.intersect1d(pd.to_datetime(analyzer.poolfactor_dates), analyzer.cache["dates"])
    valid_pooldates_idx = np.searchsorted(pd.to_datetime(analyzer.poolfactor_dates), valid_dates)
    valid_alphadates_idx = np.searchsorted(analyzer.cache["dates"], valid_dates)
    alpha_arr = analyzer.cache["alpha_df"].values[valid_alphadates_idx]
    poolfactors = analyzer.poolfactors[valid_pooldates_idx]
    valid_pool = analyzer.cache["pool_mask"][valid_alphadates_idx]
    rows, cols = np.nonzero(valid_pool)
    _, valid_poolticks_idx, sub_idx = np.intersect1d(analyzer.poolfactor_ticks, analyzer.ticks[cols], return_indices=True)
    valid_alphaticks_idx = cols[sub_idx]
    alpha_arr = alpha_arr[:, valid_alphaticks_idx]
    poolfactors = poolfactors.transpose(2, 0, 1)[valid_poolticks_idx].transpose(1, 0, 2)

    records = []
    for i, name in enumerate(analyzer.poolfactor_names):
        ic = float(np.nanmean(IC(poolfactors[:, :, i], alpha_arr)))
        ric = float(np.nanmean(rankIC(poolfactors[:, :, i], alpha_arr)))
        records.append({"factor": name, "ic": ic, "rankic": ric, "abs_corr": max(abs(ic), abs(ric))})
    return pd.DataFrame(records).sort_values("abs_corr", ascending=False)


if __name__ == "__main__":
    main()
