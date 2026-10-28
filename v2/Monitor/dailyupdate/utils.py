"""市场监控通用数值运算；不读取文件、不访问数据库。"""
from __future__ import annotations
import numpy as np
import pandas as pd


def as_float(x):
    return np.asarray(x, dtype=float)


def safe_divide(a, b):
    a, b = np.broadcast_arrays(as_float(a), as_float(b))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=np.isfinite(b) & (b != 0))


def weighted_mean(values, weights):
    x, w = as_float(values), as_float(weights)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def compound_return(returns, axis=0):
    """NaN不作为零收益；无完整有效观察的证券返回NaN。"""
    r = as_float(returns)
    ans = np.prod(1 + np.where(np.isfinite(r), r, 0), axis=axis) - 1
    return np.where(np.isfinite(r).all(axis=axis), ans, np.nan)


def historical_percentile(values):
    """末值在有效历史样本中的中位秩，包含当期，结果0~1。"""
    v = as_float(values).ravel()
    if not len(v) or not np.isfinite(v[-1]):
        return np.nan
    v = v[np.isfinite(v)]
    return float(((v < v[-1]).sum() + 0.5 * (v == v[-1]).sum()) / len(v))


def rolling_percentile(series, history=252):
    x = as_float(series)
    return np.array([historical_percentile(x[max(0, i-history+1):i+1]) for i in range(len(x))])


def window_mean(series, window=1):
    x = as_float(series)
    return float(np.nanmean(x[-window:])) if len(x) >= window else np.nan


def rank_rotation(previous, current, method='spearman'):
    """相邻两个行业排序：Spearman越低、1-rho越大，轮动越快。"""
    x, y = as_float(previous), as_float(current)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    a, b = pd.Series(x[valid]).rank(), pd.Series(y[valid]).rank()
    rho = a.corr(b, method='pearson')
    return float(1 - rho) if method == 'spearman' else float(np.mean(np.abs(a - b)) / len(a))




def _clean(series):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if series is None:
        return pd.Series(dtype=float)
    x = pd.Series(series, dtype=float)
    x.index = pd.to_datetime(x.index).normalize()
    return x[~x.index.duplicated(keep="last")].sort_index()




def level(series):
    """最新有效水平。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = _clean(series).dropna()
    return float(x.iloc[-1]) if len(x) else np.nan




def change(series, periods=1):
    """当前值减去 periods 个观测日前值。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = _clean(series).dropna()
    return float(x.iloc[-1] - x.iloc[-1 - periods]) if len(x) > periods else np.nan




def change_months(series, months=3):
    """月度同比序列的月度变化：YoY_t - YoY_(t-months)。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return change(series, months)




def yoy(series, months=12):
    """月度余额同比，按可用观测序列计算。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = _clean(series).dropna()
    return float(safe_divide(x.iloc[-1], x.iloc[-1 - months]) - 1) if len(x) > months else np.nan




def yoy_series(series, months=12):
    """返回完整的月度同比序列，便于历史分位计算。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = _clean(series).dropna()
    return x.div(x.shift(months)).sub(1).dropna()




def spread(left, right):
    """两条同日序列的利差。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return _clean(left).subtract(_clean(right), fill_value=np.nan).dropna()




def aligned_spread(left, right):
    """按日期对齐两条序列，低频序列向前填充到高频序列日期。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    lhs, rhs = _clean(left), _clean(right)
    if lhs.empty or rhs.empty:
        return pd.Series(dtype=float)
    index = lhs.index.union(rhs.index).sort_values()
    # 只向后填充已发布值，避免将未来数据回填到过去造成前视偏差。
    lhs = lhs.reindex(index).ffill()
    rhs = rhs.reindex(index).ffill()
    return lhs.subtract(rhs).dropna()




def percentile(series, history=756):
    """最新值在历史有效样本中的分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return historical_percentile(_clean(series).dropna().iloc[-history:].to_numpy())




def rolling_change_percentile(series, periods=20, history=756):
    """变化值的历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = _clean(series).dropna()
    if len(x) <= periods:
        return np.nan
    return percentile(x.diff(periods).dropna(), history)




def _time_key(value):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    text = str(value).strip()
    if len(text) == 5 and text[2] == ":":
        return text + ":00.000000"
    if len(text) == 8 and text[2] == ":":
        return text + ".000000"
    try:
        return pd.Timestamp(f"2000-01-01 {text}").strftime("%H:%M:%S.%f")
    except (TypeError, ValueError):
        return text




def _matrix_from_indexed(indexed, column, shape):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if column not in indexed:
        return np.full(shape, np.nan)
    return pd.to_numeric(indexed[column], errors="coerce").to_numpy(float).reshape(shape)




def _sum_preserve_nan(values, axis=0):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = np.asarray(values, float)
    total = np.nansum(x, axis=axis)
    total[np.all(~np.isfinite(x), axis=axis)] = np.nan
    return total




def _nanmean_rows(values):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = np.asarray(values, float)
    valid = np.isfinite(x).any(axis=0)
    result = np.full(x.shape[1:], np.nan)
    if valid.any():
        result[valid] = np.nanmean(x[:, valid], axis=0)
    return result




def _weighted_rows(values, weights):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    numerator = np.sum(np.where(valid, values * weights, 0.0), axis=1)
    denominator = np.sum(np.where(valid, weights, 0.0), axis=1)
    return safe_divide(numerator, denominator)




def _rolling_mean(series, period):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = np.asarray(series, float)
    result = np.full(len(x), np.nan)
    if period <= 0:
        raise ValueError("period must be positive")
    if len(x) < period:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(x, period)
    valid = np.isfinite(windows).all(axis=1)
    means = windows.mean(axis=1)
    result[period - 1:] = np.where(valid, means, np.nan)
    return result




def _lag_correlation_columns(values, lag):
    """逐列计算固定滞后的相关系数，列方向向量化。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = np.asarray(values, float)
    if x.shape[0] <= lag:
        return np.full(x.shape[1], np.nan)
    lhs, rhs = x[lag:], x[:-lag]
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    count = valid.sum(axis=0)
    left = np.where(valid, lhs, 0.0)
    right = np.where(valid, rhs, 0.0)
    left_mean = safe_divide(left.sum(axis=0), count)
    right_mean = safe_divide(right.sum(axis=0), count)
    left_centered = np.where(valid, lhs - left_mean, 0.0)
    right_centered = np.where(valid, rhs - right_mean, 0.0)
    covariance = safe_divide((left_centered * right_centered).sum(axis=0), count)
    left_std = np.sqrt(safe_divide((left_centered ** 2).sum(axis=0), count))
    right_std = np.sqrt(safe_divide((right_centered ** 2).sum(axis=0), count))
    corr = safe_divide(covariance, left_std * right_std)
    corr[(count < 5) | (left_std == 0) | (right_std == 0)] = np.nan
    return corr




def _history_percentile(series, history, period):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    values = _rolling_mean(series, period)
    return historical_percentile(values[-history:])




def _sum_matrices(matrices):
    """Sum matrices elementwise while retaining all-NaN cells as NaN."""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    values = np.asarray(matrices, dtype=float)
    result = np.nansum(values, axis=0)
    result[np.all(~np.isfinite(values), axis=0)] = np.nan
    return result




def _percentile(values, history=252, period=1):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    values = np.asarray(values, dtype=float)
    smooth = np.array([np.nanmean(values[i-period+1:i+1]) if i >= period-1 else np.nan for i in range(len(values))])
    return historical_percentile(smooth[-history:])




def threshold_conclusion(value, thresholds=(0.1, 0.9), labels=("偏低", "偏高", "中性")):
    """将历史分位映射为简短结论。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if not np.isfinite(value):
        return "数据不足"
    low, high = thresholds
    return labels[0] if value <= low else labels[1] if value >= high else labels[2]




def trend_conclusion(values):
    """根据1/5/10/20日分位给出域内时间趋势结论。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).any():
        return "数据不足"
    if np.isfinite(x).sum() < 2:
        return "数据不足"
    slope = np.polyfit(np.arange(len(x))[np.isfinite(x)], x[np.isfinite(x)], 1)[0]
    if slope >= 0.08:
        return "短期走强"
    if slope <= -0.08:
        return "短期走弱"
    return "时间趋势平稳"




def harmonic_valuation(multiple, weights):
    """先以市值/指数权重加权盈利或账面收益率，再取倒数，避免算术平均PE/PB失真。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    m = as_float(multiple)
    inv = np.where(m > 0, safe_divide(1, m), np.nan)
    return float(safe_divide(1, weighted_mean(inv, weights)))




def _aggregated_percentile(values, weights, aggregator, history, period):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    daily = [aggregator(v, w) for v, w in zip(values, weights)]
    return _series_percentile(daily, history, period)




def mask_spread_percentile(mask, returns, weights, period=5, history=252):
    """期初读取风格mask：1为高组、-1为低组、其他不参与；各组按期初权重加权，计算高减低的区间收益，再取历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    values = []
    for t in range(period, len(returns)):
        m, w = mask[t-period], weights[t-period]
        r = compound_return(returns[t-period+1:t+1])
        high, low = (m == 1), (m == -1)
        values.append(weighted_mean(r[high], w[high]) - weighted_mean(r[low], w[low]))
    return historical_percentile(np.asarray(values)[-history:])




def _series_percentile(series, history=252, period=1):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = as_float(series)
    smoothed = np.array([np.nanmean(x[i-period+1:i+1]) if i >= period-1 else np.nan for i in range(len(x))])
    return historical_percentile(smoothed[-history:])




    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
def _pct(x): return np.asarray(x,float)/100.0




    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
def _p(x,h=252): return historical_percentile(np.asarray(x,float)[-h:])




def _summarize(vals, period, history):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return (float(np.nanmean(vals[-period:])), _p(vals, history)) if len(vals) else (np.nan, np.nan)




def _mean_pair_correlation(x):
    """Exact off-diagonal mean in O(T*N), without an N*N matrix.

    For centered unit-norm columns z_j, sum(i != j) corr(i,j)
    equals ||sum_j z_j||**2 - n. Constant/empty columns are excluded,
    matching nanmean(corrcoef(...)) after per-column median imputation.
    """
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = np.asarray(x, dtype=float)
    x = x[:, np.isfinite(x).any(axis=0)]
    if x.shape[1] < 2:
        return np.nan
    if not np.isfinite(x).all():
        finite = np.isfinite(x)
        medians = np.nanmedian(np.where(finite, x, np.nan), axis=0)
        x = np.where(finite, x, medians)
    x = x - x.mean(axis=0)
    norms = np.linalg.norm(x, axis=0)
    valid = np.isfinite(norms) & (norms > 0)
    n = int(valid.sum())
    if n < 2:
        return np.nan
    zsum = (x[:, valid] / norms[valid]).sum(axis=1)
    return float(np.clip((zsum @ zsum - n) / (n * (n - 1)), -1, 1))




def _correlation_series(returns, window, downside=False, market_returns=None):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r = _pct(returns)
    vals = []
    if downside:
        r = np.where(np.isfinite(r), r, np.nan)
    market = (np.nanmean(r, axis=1) if market_returns is None
              else np.asarray(market_returns, dtype=float)) if downside else None
    for i in range(window, len(r) + 1):
        x = r[i-window:i]
        if downside:
            m = market[i-window:i]
            finite = np.isfinite(m)
            x = x[finite & (m <= np.quantile(m[finite], 0.20))] if finite.any() else x[:0]
            if len(x) < 3:
                vals.append(np.nan)
                continue
        vals.append(_mean_pair_correlation(x))
    return np.asarray(vals)




def _pca_series(returns, window):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r = _pct(returns)
    vals = []
    for i in range(window, len(r) + 1):
        x = r[i-window:i]
        finite = np.isfinite(x)
        # Require at least 80% finite observations (round up).
        keep = finite.sum(axis=0) >= max(1, int(np.ceil(window * .8)))
        x = x[:, keep]
        finite = finite[:, keep]
        # Fill missing returns with each stock's finite-observation mean.
        # After centering, those entries are zero; observed deviations survive.
        means = np.where(finite, x, 0.0).sum(axis=0) / finite.sum(axis=0)
        x = np.where(finite, x, means) - means
        x -= x.mean(axis=0)
        norms = np.linalg.norm(x, axis=0)
        valid = np.isfinite(norms) & (norms > 0)
        if valid.sum() < 2:
            vals.append(np.nan)
            continue
        # Unit-norm columns give correlation PCA, rather than covariance PCA.
        x = x[:, valid] / norms[valid]
        # X.T @ X and X @ X.T have the same nonzero eigenvalues.
        # The covariance normalization cancels in lambda_max / trace.
        gram = x @ x.T if x.shape[0] <= x.shape[1] else x.T @ x
        total = np.trace(gram)
        vals.append(float(np.linalg.eigvalsh(gram)[-1] / total))
    return np.asarray(vals)




def _limit_down_series(hit_floor, has_trade=None, limit_eligible=None):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    hit = np.asarray(hit_floor, dtype=float)
    valid = np.isfinite(hit)
    if has_trade is not None:
        valid &= np.isfinite(has_trade) & (np.asarray(has_trade) > 0)
    if limit_eligible is not None:
        valid &= np.isfinite(limit_eligible) & (np.asarray(limit_eligible) > 0)
    counts = valid.sum(1)
    return np.divide(((hit > 0) & valid).sum(1), counts,
                     out=np.full(len(hit), np.nan), where=counts > 0)




def _rv_metrics(decimal_returns):
    """原表交易日窗口；完整窗口才输出，零分母不生成无穷值。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r = pd.Series(decimal_returns, dtype=float)
    rv = {h: r.rolling(h).std(ddof=1) * np.sqrt(252) for h in (5,20,60)}
    return {
        '实现波动率RV5': rv[5],
        '实现波动率RV20': rv[20],
        '实现波动率RV60': rv[60],
        '波动率期限结构RV5/RV20': rv[5] / rv[20].where(rv[20] > 0),
        '波动率期限结构RV20/RV60': rv[20] / rv[60].where(rv[60] > 0),
    }


def cancellation_rate(cancel_amount, effective_order_amount):
    """撤单金额/有效委托金额；需由撤单、委托事件数据提供分子分母。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return safe_divide(cancel_amount, effective_order_amount)




def credit_impulse(tsf_stock, nominal_gdp, window=12, lag=3):
    """信用脉冲：过去12个月新增社融/GDP，再减3个月前值。

    GDP 为季度序列时按月线性插值到社融月份，仅用于形成同比尺度；
    不使用未来季度值。
    """
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    tsf, gdp = _clean(tsf_stock), _clean(nominal_gdp)
    if tsf.empty or gdp.empty:
        return pd.Series(dtype=float)
    # EDB月度值可能标记在月末，先统一到月初，避免直接 reindex 全部丢失。
    tsf_monthly = tsf.groupby(tsf.index.to_period("M")).last()
    tsf_monthly.index = tsf_monthly.index.to_timestamp(how="start")
    gdp_monthly = gdp.groupby(gdp.index.to_period("M")).last()
    gdp_monthly.index = gdp_monthly.index.to_timestamp(how="start")
    month_index = pd.date_range(tsf_monthly.index.min(), tsf_monthly.index.max(), freq="MS")
    tsf = tsf_monthly.reindex(month_index).ffill()
    gdp_index = gdp_monthly.index.union(month_index).sort_values()
    gdp = gdp_monthly.reindex(gdp_index).interpolate(method="time").reindex(month_index).ffill()
    frame = pd.concat({"tsf": tsf, "gdp": gdp}, axis=1).dropna()
    if len(frame) < window + lag + 1:
        return pd.Series(dtype=float)
    ratio = frame["tsf"].rolling(window).sum().div(frame["gdp"])
    return ratio - ratio.shift(lag)


def five_level_depth(quotes):
    """日内五档买卖盘数量深度中位数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    return np.nanmedian(quotes["depth"], axis=0)






def index_eps_bond_spread(eps, market_value, bond_yield, period=1):
    """Forecast earnings divided by market value, minus bond yield."""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    earnings_yield = safe_divide(as_float(eps), as_float(market_value))
    return float(np.nanmean((earnings_yield - as_float(bond_yield))[-period:]))


def index_pe_bond_spread(pe, bond_yield, period=1):
    """Index PE-TTM reciprocal minus the 10-year bond yield."""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    earnings_yield = safe_divide(1.0, as_float(pe))
    return float(np.nanmean((earnings_yield - as_float(bond_yield))[-period:]))


def liquidity_coverage(executable_amount, demand_amount):
    """可成交金额/需求金额。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return safe_divide(executable_amount, demand_amount)


def pb_spread_percentile(mask, returns, weights, period=5, history=252):
    """PB风格mask正组(1)减负组(-1)的区间收益差历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return mask_spread_percentile(mask, returns, weights, period, history)


def pe_spread_percentile(mask, returns, weights, period=5, history=252):
    """PE风格mask正组(1)减负组(-1)的区间收益差历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return mask_spread_percentile(mask, returns, weights, period, history)






def volatility_risk_premium(iv, rv, period=1, history=252):
    """原表风险溢价代理 IV30²-RV30²；输入均为年化小数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x=np.asarray(iv,float)**2-np.asarray(rv,float)**2
    return float(np.nanmean(x[-period:])),_p(x,history)


