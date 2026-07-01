"""中文说明：本脚本提供当前模块的量化研究或生产能力。"""


import numpy as np
import bottleneck as bn
import pandas as pd



# 中文说明：`corr`：执行该名称对应的业务计算，并返回调用方所需结果。
def corr(a, b, axis):
    b[np.isnan(a)] = np.nan
    a[np.isnan(b)] = np.nan
    arr = (
            (bn.nanmean(a * b, axis=axis) - bn.nanmean(a, axis=axis) * bn.nanmean(b, axis=axis))
            / (bn.nanstd(a, axis=axis) + 1e-6)
            / (bn.nanstd(b, axis=axis) + 1e-6)
    )
    bn.replace(arr, np.nan, 0)
    arr[np.isinf(arr)] = 0
    return arr

# 中文说明：`IC`：执行该名称对应的业务计算，并返回调用方所需结果。
def IC(y_, y):
    ics = corr(y_.copy(), y.copy(), axis=-1)
    return ics

# 中文说明：`rankIC`：执行该名称对应的业务计算，并返回调用方所需结果。
def rankIC(y_, y):
    rank_ics = corr(bn.nanrankdata(y_.copy(), axis=-1), bn.nanrankdata(y.copy(), axis=-1), axis=-1)
    return rank_ics

# 中文说明：`calc_group_ret`：计算研究或生产指标。
def calc_group_ret(alpha, label, num_group=10):
    rank = bn.nanrankdata(alpha, axis=-1)
    num_signal = np.nanmax(rank, axis=-1)
    stock_each_group = num_signal // num_group
    group_ret = np.full((num_group, num_signal.shape[0]), np.nan)
    for i in range(num_group):
        if i==num_group-1:
            group_ix = (rank.T > stock_each_group * i) & (rank.T <= num_signal)
        else:
            group_ix = (rank.T > stock_each_group * i) & (rank.T <= stock_each_group * (i + 1)) # n_stock, n_date
        temp_ret = label.copy()
        temp_ret[~group_ix.T] = np.nan
        group_ret[i] = np.nanmean(temp_ret, axis=-1)
    group_ret = group_ret - np.nanmean(group_ret, axis=0)
    col_list = list(range(1, num_group + 1))[::-1]
    group_ret = pd.DataFrame(
        group_ret.T,
        columns=col_list,
        index=alpha.index,
    )
    return group_ret

# 中文说明：`calc_annret`：计算研究或生产指标。
def calc_annret(ret_df):
    nav = np.nancumprod(1+ret_df.values)
    years = (ret_df.index[-1] - ret_df.index[0]).days / 242
    total_ret = nav[-1]/nav[0]-1
    annret = (1+total_ret)**(1/years) - 1
    return annret

# 中文说明：`calc_annvol`：计算研究或生产指标。
def calc_annvol(ret_df):
    annvol = np.nanstd(ret_df.values) * np.sqrt(242)
    return annvol

# 中文说明：`calc_sharpe`：计算研究或生产指标。
def calc_sharpe(ret_df):
    annret = calc_annret(ret_df)
    annvol = calc_annvol(ret_df)
    sharpe = annret / annvol if annvol>0 else 0
    return sharpe

# 中文说明：`calc_maxdrawdown`：计算研究或生产指标。
def calc_maxdrawdown(ret_df):
    nav = np.nancumprod(1+ret_df.values)
    return ((nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav)).min()

# 中文说明：`calc_calmar`：计算研究或生产指标。
def calc_calmar(ret_df):
    annret = calc_annret(ret_df)
    max_dd = calc_maxdrawdown(ret_df)
    calmar = annret / abs(max_dd) if max_dd<0 else np.nan
    return calmar

# 中文说明：`calc_weekly_bps`：计算研究或生产指标。
def calc_weekly_bps(ret_df):
    weekly_rets = ret_df.resample('W').apply(lambda x: (1+x).prod()-1).dropna()
    weekly_avg_bps = weekly_rets.mean() * 10000
    return weekly_avg_bps

# 中文说明：`calc_holdings`：计算研究或生产指标。
def calc_holdings(alpha: pd.DataFrame, num_group: int = 10) -> pd.DataFrame:
    rank = bn.nanrankdata(alpha.values, axis=-1)   # (n_dates, n_stocks)
    n_valid = np.nansum(~np.isnan(alpha.values), axis=-1)
    stock_each_group = n_valid // num_group
    topgroup_ix = (rank.T > stock_each_group * (num_group-1) ) & (rank.T <= n_valid) # (n_stocks, n_dates)
    bottomgroup_ix = (rank.T > 0) & (rank.T <= stock_each_group)
    long_holds = pd.DataFrame(topgroup_ix.T, index=alpha.index, columns=alpha.columns)
    short_holds= pd.DataFrame(bottomgroup_ix.T, index=alpha.index, columns=alpha.columns)
    holds = pd.DataFrame(0, index=alpha.index, columns=alpha.columns, dtype=int)
    holds[long_holds.values] = 1
    holds[short_holds.values] = -1
    return holds

# 中文说明：`calc_turnover`：计算研究或生产指标。
def calc_turnover(holds,  freq: str = 'D') -> pd.Series:
    # 1. 生成调仓日期列表
    if freq == 'D':
        rebalance_dates = holds.index.sort_values()  # 所有交易日
    elif freq == 'W':
        rebalance_dates = pd.Series(holds.index).groupby(holds.index.to_period('W-MON')).first().values
    elif freq == 'M':
        rebalance_dates = pd.Series(holds.index).groupby(holds.index.to_period('M')).first().values
    else:
        raise ValueError("freq 必须是 'D', 'W', 'M'")
    
    #holdings = calc_top_holdings(alpha)
    curr = holds.loc[rebalance_dates].astype(int)
    prev = curr.shift(1).fillna(0).astype(int)
    
    # 双边平均换手率 = (买入+卖出) / (前总持仓+现总持仓)
    change = (curr - prev).abs().sum(axis=1)          # 操作次数加权和
    total = prev.abs().sum(axis=1) + curr.abs().sum(axis=1)
    turnover = change / total
    turnover = turnover.fillna(0.0)
    turnover.iloc[0] = np.nan
    return turnover.rename('turnover').to_frame(name='turnover')

# 中文说明：`calc_alpha_distribution`：计算研究或生产指标。
def calc_alpha_distribution(pred_df, year='Overall'):
    arr = pred_df.values if hasattr(pred_df, "values") else np.asarray(pred_df)
    vals = arr.flatten()
    vals = vals[~np.isnan(vals)]
    daily_mean = np.nanmean(arr, axis=-1)
    return {
        'year': year,
        'max': np.max(vals),
        'min': np.min(vals),
        'mean': np.mean(vals),
        'std': np.std(vals, ddof=1),
        'skew': pd.Series(vals).skew(),
        'kurtosis': pd.Series(vals).kurtosis(),
        'max_avg': np.nanmax(daily_mean),
        'min_avg': np.nanmin(daily_mean)
    }

# 中文说明：`calc_IC_stats`：计算研究或生产指标。
def calc_IC_stats(ics, rankics, pos_ics, neg_ics, yr='Overall'):
    avg_ic = ics.mean()
    ic_ir = avg_ic / ics.std() if ics.std()>0 else np.nan
    avg_pos_ic = pos_ics.mean()
    pos_ic_ir = avg_pos_ic / pos_ics.std() if pos_ics.std()>0 else np.nan
    avg_neg_ic = neg_ics.mean()
    neg_ic_ir = avg_neg_ic / neg_ics.std() if neg_ics.std()>0 else np.nan
    avg_rank_ic = rankics.mean()
    rank_ic_ir = avg_rank_ic / rankics.std() if rankics.std()>0 else np.nan
    
    return{
        'year': yr,
        'avg_ic': avg_ic,
        'ic_ir': ic_ir,
        'avg_pos_ic': avg_pos_ic,
        'pos_ic_ir': pos_ic_ir,
        'avg_neg_ic': avg_neg_ic,
        'neg_ic_ir': neg_ic_ir,
        'avg_rank_ic': avg_rank_ic,
        'rank_ic_ir': rank_ic_ir
    }

# 中文说明：`calc_sign_IC`：计算研究或生产指标。
def calc_sign_IC(label_arr, alpha_arr):
    pos_ret_mask = label_arr>0
    neg_ret_mask = label_arr<0

    pos_ret = np.where(pos_ret_mask, label_arr, np.nan)
    pos_alpha = np.where(pos_ret_mask, alpha_arr, np.nan)
    neg_ret = np.where(neg_ret_mask, label_arr, np.nan)
    neg_alpha = np.where(neg_ret_mask, alpha_arr, np.nan)

    pos_ics = IC(pos_alpha, pos_ret)
    neg_ics = IC(neg_alpha, neg_ret)
    pos_rankics = rankIC(pos_alpha, pos_ret)
    neg_rankics = rankIC(neg_alpha, neg_ret)
    return pos_ics, neg_ics, pos_rankics, neg_rankics


# 中文说明：`calc_beta`：计算研究或生产指标。
def calc_beta(x_t, y_t):
    var_x = np.var(x_t, ddof=0)
    if var_x < 1e-12: return 0.0
    cov_xy = np.cov(x_t, y_t, ddof=0)[0, 1]
    return cov_xy / var_x



if __name__ == "__main__":

    alpha_df = pd.read_csv('/home/xujiayi/PycharmProjects/Models/XJY_end2end/0_result/gru/rolling/alpha_merge_20210104_20251231.csv', index_col=0, parse_dates=True)

    holds = calc_holdings(alpha_df)
