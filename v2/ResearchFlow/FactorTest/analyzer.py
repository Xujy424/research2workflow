"""中文说明：本脚本提供当前模块的量化研究或生产能力。"""


from typing import Optional, Sequence, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.stats import norm, ttest_1samp, spearmanr
from statsmodels.tsa.stattools import adfuller
from scipy.linalg import solve_triangular
from joblib import Parallel, delayed
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

from .utils import Loader, Calculator, Processor
from .metrics import *


DEFAULT_FIGSIZE = (8, 5)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


# 中文说明：定义 `FactorAnalyzer`，封装本模块对应的数据、配置与行为。
class FactorAnalyzer:

    root = Path("D:/data")
    loader = Loader(root)

    direct_variables = [
        'start_date', 'end_date', 'dates', 'dates_idx', 'pool_mask',
        'alpha_df', 'alpha_arr', 'label_arr', 'mv_arr',
        'ind_code_arr', 'sec_code_arr', 'barras_arr',
    ]
    indirect_variables = [
        'ics_df', 'rankics_df', 'groupret_df', 'longshort_holds', 'ret_df', 'hold_df', 'turnover_df'
    ]

    # Spearman redundancy check is disabled in FactorTest UI; avoid loading external factorpool at import time.
    # poolfactor_dates, poolfactor_names, poolfactor_ticks, poolfactors = loader.load_factorpool()  # T,K,N
    poolfactor_dates, poolfactor_names, poolfactor_ticks, poolfactors = None, None, None, None

    pool_regime_specs = (
        ("HS300", "hs300_mask"),
        ("ZZ500", "zz500_mask"),
        ("A500", "a500_mask"),
        ("ZZ1000", "zz1000_mask"),
        ("ZZ2000", "zz2000_mask"),
    )

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, info, alpha_df):
        self._info(info)
        self._universe_key = self.universe +'_mask' if self.universe!='universe' else 'tradable'

        self.alpha_df = alpha_df
        self.prepare_data()

        self._tables = {'info': self.info_table}
        self._figures = {}
    
    # 中文说明：`_info`：内部辅助步骤，不作为稳定公共接口。
    def _info(self, info: dict):
        self.name = info.get('name')
        self.factor_type = info.get('factor_type')
        self.alpha_type = info.get('alpha_type')
        self.data_usage = info.get('usage')
        self.universe = info.get('universe')
        self.start_date = info.get('start_date')
        self.end_date = info.get('end_date')
        self.summary = info.get('summary')
        self.info_table = pd.Series(info, dtype='object')
    
    # 中文说明：`prepare_data`：执行该名称对应的业务计算，并返回调用方所需结果。
    def prepare_data(self):
        self.start_date = self.info_table['start_date']
        self.end_date =  self.info_table['end_date']

        date_mask = (self.alpha_df.index >= self.start_date) & (self.alpha_df.index <= self.end_date)
        self.dates = self.alpha_df.index[date_mask]
        self.dates_idx = np.searchsorted(self.loader.dates, self.dates)

        self.ticks = self.alpha_df.columns
        
        self.alpha_df = self.alpha_df.loc[self.dates]
        self.alpha_arr = self.alpha_df.values
        self.label_arr = np.memmap(self.root/'label'/'Y.1D.bin', shape=(self.loader.T, self.loader.N), dtype=float, mode='r')[self.dates_idx]
        self.mv_arr = np.memmap(self.root/'d_field'/'mv.bin', shape=(self.loader.T, self.loader.N), dtype=float, mode='r')[self.dates_idx]

        masks = self.loader.load_masks()
        self.ind_code_arr = masks['industry'][self.dates_idx]
        self.sec_code_arr = masks['sector'][self.dates_idx]
        self.tradable = masks['tradable'][self.dates_idx]
        self.pool_mask = masks[self._universe_key][self.dates_idx]

        barras = self.loader.load_barras()
        self.barras_arr = np.stack(list(barras.values()),axis=-1)[self.dates_idx]
        
        indirect_vals = self.calc_basic_df(self.alpha_df, self.label_arr, self.dates)
        for i, name in enumerate(self.indirect_variables):
            setattr(self, name, indirect_vals[i])

        self.cache = {var_name: getattr(self, var_name) for var_name in self.direct_variables+self.indirect_variables}
    
    # 中文说明：`calc_basic_df`：计算研究或生产指标。
    def calc_basic_df(self, alpha_df, label_arr, dates):
        ics_df = pd.Series(IC(alpha_df.values, label_arr), index=dates)
        rankics_df = pd.Series(rankIC(alpha_df.values, label_arr), index=dates)
        groupret_df = calc_group_ret(alpha_df, label_arr)
        longshort_holds = calc_holdings(alpha_df)
        hold_df = longshort_holds.where(longshort_holds==1, 0).astype('int') if self.factor_type == 'long' else longshort_holds.astype('int') 
        ret_df = pd.DataFrame(np.nanmean(label_arr*hold_df,axis=-1), index=dates, columns=['ret'])
        turnover_df = calc_turnover(hold_df)
        return ics_df, rankics_df, groupret_df, longshort_holds, ret_df, hold_df, turnover_df
    
    # 中文说明：`reset_cache`：重置会话内状态。
    def reset_cache(self):
        self.cache = {var_name: getattr(self, var_name) for var_name in self.direct_variables+self.indirect_variables}
    
    # 中文说明：`reset_axis`：重置会话内状态。
    def reset_axis(self, start_date, end_date, pool_name):
        self.reset_cache()

        start_date = pd.to_datetime(start_date) if start_date else self.start_date
        end_date = pd.to_datetime(end_date) if end_date else self.end_date

        date_mask = (self.dates >= start_date) & (self.dates <= end_date)
        start_date, end_date = self.dates[date_mask][0], self.dates[date_mask][-1]
        self.cache['start_date'] = self.info_table['start_date'] = start_date
        self.cache['end_date'] = self.info_table['end_date'] = end_date

        self.cache['dates'] = self.cache['dates'][date_mask]
        self.cache['dates_idx'] = self.cache['dates_idx'][date_mask]

        self.info_table['universe'] = pool_name if pool_name else self.universe
        universe_key = self.info_table['universe'] +'_mask' if self.info_table['universe']!='universe' else 'tradable'
        pool_mask = self.loader.load_masks()[universe_key][self.cache['dates_idx']]
        self.cache['pool_mask'] = pool_mask
        
        self.cache['alpha_df'] = self.cache['alpha_df'].loc[start_date:end_date].where(pool_mask, np.nan)


        for var_name in self.direct_variables[6:]:
            value = self.cache[var_name]
            if isinstance(value, (pd.DataFrame, pd.Series)):
                self.cache[var_name] = value.loc[start_date:end_date]
            else:
                self.cache[var_name] = value[date_mask]
            if var_name!='barras_arr':
                self.cache[var_name] = np.where(pool_mask, self.cache[var_name], np.nan)
            else:
                self.cache[var_name] = np.where(pool_mask[...,np.newaxis], self.cache[var_name], np.nan)
        
        indirect_vals = self.calc_basic_df(self.cache['alpha_df'], self.cache['label_arr'], self.cache['dates'])
        for i, name in enumerate(self.indirect_variables):
            self.cache[name] = indirect_vals[i]


    # 中文说明：`calc_ind_exposure`：计算研究或生产指标。
    def calc_ind_exposure(self):
        dates = self.cache['dates']

        long_mask = (self.cache['longshort_holds'].values == 1)
        short_mask = (self.cache['longshort_holds'].values == -1)

        long_total = np.nansum(long_mask, axis=1, keepdims=True)
        short_total = np.nansum(short_mask, axis=1, keepdims=True)

        J = len(self.loader.id_to_industry)
        onehot_ind = (self.cache['ind_code_arr'][..., None] == np.arange(J)).astype(float)

        long_ind_pct = np.divide(np.nansum(onehot_ind * long_mask[:, :, None], axis=1), long_total, out=np.zeros((len(dates), J), dtype=float), where=long_total != 0)
        short_ind_pct = np.divide(np.nansum(onehot_ind * short_mask[:, :, None], axis=1), short_total, out=np.zeros((len(dates), J), dtype=float), where=short_total != 0)

        long_ind_pct_df = pd.DataFrame(long_ind_pct, index=dates, columns=self.loader.id_to_industry.values())
        short_ind_pct_df = pd.DataFrame(short_ind_pct, index=dates, columns=self.loader.id_to_industry.values())
        return long_ind_pct_df, short_ind_pct_df
    
    # 中文说明：`calc_sec_exposure`：计算研究或生产指标。
    def calc_sec_exposure(self):
        dates = self.cache['dates']

        long_mask = (self.cache['longshort_holds'].values == 1)
        short_mask = (self.cache['longshort_holds'].values == -1)

        long_total = np.nansum(long_mask, axis=1, keepdims=True)
        short_total = np.nansum(short_mask, axis=1, keepdims=True)

        K = len(self.loader.id_to_sector)
        onehot_sec = (self.cache['sec_code_arr'][..., None] == np.arange(K)).astype(float)

        long_sec_pct = np.divide(np.nansum(onehot_sec * long_mask[:, :, None], axis=1), long_total, out=np.zeros((len(dates), K), dtype=float), where=long_total != 0)
        short_sec_pct = np.divide(np.nansum(onehot_sec * short_mask[:, :, None], axis=1), short_total, out=np.zeros((len(dates), K), dtype=float), where=short_total != 0)

        long_sec_pct_df = pd.DataFrame(long_sec_pct, index=dates, columns=self.loader.id_to_sector.values())
        short_sec_pct_df = pd.DataFrame(short_sec_pct, index=dates, columns=self.loader.id_to_sector.values())
        return long_sec_pct_df, short_sec_pct_df

    # 中文说明：`calc_ind_ret`：计算研究或生产指标。
    def calc_ind_ret(self):
        dates = self.cache['dates']
        label_arr = self.cache['label_arr']

        J = len(self.loader.id_to_industry)
        ind_ret = np.full((len(dates), J), np.nan)
        for i in range(J):
            mask = self.cache['ind_code_arr'] == i
            ind_ret[:, i] = np.nanmean(np.where(mask, label_arr, np.nan), axis=1)
        ind_ret_df = pd.DataFrame(ind_ret, index=dates, columns=self.loader.id_to_industry.values())
        return ind_ret_df

    # 中文说明：`calc_sec_ret`：计算研究或生产指标。
    def calc_sec_ret(self):
        dates = self.cache['dates']
        label_arr = self.cache['label_arr']

        K = len(self.loader.id_to_sector)
        sec_ret = np.full((len(dates), K), np.nan)
        for i in range(K):
            mask = self.cache['sec_code_arr'] == i
            sec_ret[:, i] = np.nanmean(np.where(mask, label_arr, np.nan), axis=1)
        sec_ret_df = pd.DataFrame(sec_ret, index=dates, columns=self.loader.id_to_sector.values())
        return sec_ret_df

    # 中文说明：`calc_barra_exposure`：计算研究或生产指标。
    def calc_barra_exposure(self):
        long_mask = (self.cache['longshort_holds'].values == 1).astype(float)
        if self.factor_type == 'longshort':
            short_mask = (self.cache['longshort_holds'].values == -1).astype(float)
            exposure_weight = long_mask - short_mask
        else:
            exposure_weight = long_mask

        barra_exposures = {
            self.loader.id_to_barra[i]: np.nanmean(self.cache['barras_arr'][:,:,i] * exposure_weight, axis=1)
            for i in range(self.cache['barras_arr'].shape[2])
        }
        barra_exposure_df = pd.DataFrame(barra_exposures, index=self.cache['dates'])
        return barra_exposure_df

    # 中文说明：`calc_barra_ret`：计算研究或生产指标。
    def calc_barra_ret(self):
        barras_arr = self.cache['barras_arr']
        barras_id = list(range(self.cache['barras_arr'].shape[2]))
        ret = self.cache['label_arr']
        K = len(barras_id)
        T = len(self.cache['dates'])
        J = len(self.loader.id_to_industry)
        sqrt_mv = np.sqrt(np.clip(self.cache['mv_arr'], 1e-8, None))

        style_ret = np.full((T, K), np.nan)
        ind_ret = np.full((T, J), np.nan)

        onehot_ind = np.eye(J, dtype=float)
        for t in range(T):
            barra_t = np.stack([barras_arr[:,:,i][t] for i in barras_id], axis=1)

            valid = ~(
                np.isnan(ret[t]) |
                np.isnan(self.cache['ind_code_arr'][t]) |
                np.isnan(sqrt_mv[t]) |
                np.isnan(self.cache['mv_arr'][t]) |
                np.isnan(barra_t).any(axis=1)
            )
            if valid.sum() < K + J + 1:
                continue

            X = barra_t[valid]
            y = ret[t][valid]
            w = sqrt_mv[t][valid]
            ind_code = self.cache['ind_code_arr'][t][valid].astype(int)

            if ind_code.size == 0:
                continue

            present_ind = np.flatnonzero(np.bincount(ind_code, minlength=J) > 0)
            if len(present_ind) < 2:
                Z = X
                beta = np.linalg.lstsq(Z * w[:, None], w * y, rcond=None)[0]
                style_ret[t] = beta[:K]
                continue

            base_ind = present_ind[0]
            D = onehot_ind[ind_code][:, np.arange(J) != base_ind]
            Z = np.concatenate([X, D], axis=1)

            WZ = Z * w[:, None]
            ZTWZ = WZ.T @ Z
            ZTWy = Z.T @ (w * y)
            try:
                beta = np.linalg.solve(ZTWZ, ZTWy)
            except np.linalg.LinAlgError:
                ZTWZ_reg = ZTWZ + 1e-8 * np.eye(ZTWZ.shape[0])
                beta = np.linalg.lstsq(ZTWZ_reg, ZTWy, rcond=None)[0]

            style_ret[t] = beta[:K]
            g = np.zeros(J, dtype=float)
            g_rest = beta[K:]
            non_base = np.setdiff1d(np.arange(J), base_ind)
            g[non_base] = g_rest
            g[base_ind] = -g_rest.sum()
            ind_ret[t] = g

        return pd.DataFrame(style_ret, index=self.cache['dates'], columns=self.loader.id_to_barra.values()), pd.DataFrame(ind_ret, index=self.cache['dates'], columns=self.loader.id_to_industry.values())

    # 中文说明：`cal_break_signal`：执行该名称对应的业务计算，并返回调用方所需结果。
    def cal_break_signal(self, window=3):
        rolling_mean = self.cache['alpha_df'].rolling(window=window, min_periods=1).mean().shift(1)
        signal = (self.cache['alpha_df'] > rolling_mean).astype(int)
        return signal
    

    # 中文说明：`table_PRF_stats`：生成诊断表格。
    def table_PRF_stats(self, horizons=[1,3,5,10]):
        signal = self.cal_break_signal().values
        res_rows = []
        for h in horizons:
            fut = Calculator.rolling_retprod(self.cache['label_arr'], window=h, future=True)
            valid = ~np.isnan(fut)
            sig_valid = signal.copy()
            sig_valid[~valid] = 0
            pos_mask = fut>0

            TP = np.nansum((sig_valid==1) & pos_mask)
            FP = np.nansum((sig_valid==1) & (~pos_mask))
            FN = np.nansum((sig_valid==0) & pos_mask)
            N_signal = int(np.nansum(sig_valid==1))

            precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            f1 = 2 * precision * recall / (precision + recall) if (precision is not np.nan and recall is not np.nan and (precision + recall) > 0) else np.nan
            baseline = float(np.nansum(pos_mask) / np.nansum(valid)) if np.nansum(valid) > 0 else np.nan
            lift = precision - baseline if not np.isnan(precision) and not np.isnan(baseline) else np.nan
            
            res_rows.append({'horizon': h, 'precision': precision, 'baseline': baseline, 'lift': lift, 'recall': recall, 'f1': f1, 'n_signal': N_signal})
        
        return pd.DataFrame(res_rows).set_index('horizon').round(3)
    
    # 中文说明：`table_winrate_scan`：生成诊断表格。
    def table_winrate_scan(self, horizons=[1,3,5,10]):
        rows = []
        signal = self.cal_break_signal().values
        for h in horizons:
            fut = Calculator.rolling_retprod(self.cache['label_arr'], window=h, future=True)
            vals = fut[signal==1]
            vals = vals[~np.isnan(vals)]
            mean_ret = np.mean(vals)
            win_rate = np.mean(vals>0)
            t_stat, p_value = ttest_1samp(vals, 0.0)
            rows.append({'horizon': h, 'mean_ret': mean_ret, 'win_rate': win_rate, 't_stat': float(t_stat), 'p_value': float(p_value), 'n': len(vals)})
        return pd.DataFrame(rows).set_index('horizon').round(3)


    # 中文说明：`table_monthly_ret`：生成诊断表格。
    def table_monthly_ret(self):
        '''月度收益'''
        df = self.cache['ret_df'].copy()
        idx = pd.to_datetime(df.index, errors='coerce')
        df['year'] = idx.year
        df['month'] = idx.month
        
        month_labels = {1:'1月',2:'2月',3:'3月',4:'4月',5:'5月',6:'6月',7:'7月',8:'8月',9:'9月',10:'10月',11:'11月',12:'12月'}
        monthly = df.groupby(['year', 'month']).apply(lambda x: (1+x).prod()-1).unstack(level='month')
        monthly = monthly.rename(columns=month_labels)
        monthly.columns = monthly.columns.droplevel(0)

        annual = df.groupby(['year']).apply(lambda x: (1+x).prod()-1)['ret']
        monthly['年度'] = annual

        return (monthly*100).round(2)
    
    # 中文说明：`table_annual_stats`：生成诊断表格。
    def table_annual_stats(self):
        '''年度收益指标表现'''
        dates, ret_df, hold_df = self.cache['dates'], self.cache['ret_df'], self.cache['hold_df']

        records = []
        years = dates.year.unique()
        for y in years:
            ret_y = ret_df.loc[ret_df.index.year==y]
            hold_y = hold_df[hold_df.index.year==y]
            records.append({
                '年份': y,
                '年化收益': calc_annret(ret_y),
                '年化波动率': calc_annvol(ret_y),
                '夏普比': calc_sharpe(ret_y),
                '最大回撤': calc_maxdrawdown(ret_y),
                '换手率': calc_turnover(hold_y,freq='D').mean().values[0]
            })
        records.append({
            '年份': 'Overall',
            '年化收益': calc_annret(ret_df),
            '年化波动率': calc_annvol(ret_df),
            '夏普比': calc_sharpe(ret_df),
            '最大回撤': calc_maxdrawdown(ret_df),
            '换手率': calc_turnover(hold_df,freq='D').mean().values[0]
        })
        return pd.DataFrame(records).set_index('年份').round(3)
    
    # 中文说明：`plot_basic_performance`：绘制诊断图表。
    def plot_basic_performance(self):
        '''基本表现'''
        dates, ret_df, hold_df = self.cache['dates'], self.cache['ret_df'], self.cache['hold_df']
        
        #coverage_numer = np.nansum(np.abs(hold_df.values), axis=1)
        coverage_numer = np.sum(~np.isnan(self.cache['alpha_df'].values), axis=1)
        coverage_denom = np.nansum(self.cache['pool_mask'],axis=1)
        coverage = np.divide(
            coverage_numer,
            coverage_denom,
            out=np.zeros_like(coverage_numer, dtype=float),
            where=coverage_denom != 0,
        )       
        cumret = np.nancumsum(ret_df.values.reshape(-1))
        turnover = self.cache['turnover_df'].values.reshape(-1)

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.plot(dates, cumret, label='Cumulative Return (%)')
        ax.plot(dates, coverage, label='Coverage (%)')
        ax.plot(dates, turnover, label='Turnover (%)')
        
        ax.set_title('Factor Basic Performance', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('absolute value', fontsize=12)
        ax.legend(loc='best')
        return fig
    

    # 中文说明：`table_alpha_annual_stats`：生成诊断表格。
    def table_alpha_annual_stats(self):
        '''因子值分布表，确保因子分布不漂移，比ADF更具实践性的平稳性检验'''
        alpha_df = self.cache['alpha_df']

        results = []
        years = alpha_df.index.year.unique()
        for yr in years:
            sub = alpha_df[alpha_df.index.year == yr]
            stats = calc_alpha_distribution(sub.values, year=yr)
            results.append(stats)
        overall = calc_alpha_distribution(alpha_df.values, year='Overall')
        results.append(overall)
        
        result = pd.DataFrame(results)
        result['year'] = result['year'].astype(str)
        return result[['year', 'max', 'min', 'max_avg', 'min_avg', 'mean', 'std', 'skew', 'kurtosis']].round(3).set_index('year')
    
    # 中文说明：`plot_alpha_distribution`：绘制诊断图表。
    def plot_alpha_distribution(self, year='Overall', show_stats=True, bins=100):
        '''因子值分布图'''
        alpha_df = self.cache['alpha_df']
        label_df = pd.DataFrame(self.cache['label_arr'], index=alpha_df.index, columns=alpha_df.columns)

        if year=='Overall':
            sub, sub_ret = alpha_df[:], label_df[:]
            title = f"Overall Factor Distribution (All Years)"
        else:
            sub, sub_ret = alpha_df[alpha_df.index.year==year], label_df[label_df.index.year==year]
            title = f"Factor Distribution for {year}"
        vals, vals_ret = sub.values.flatten(), sub_ret.values.flatten()
        vals, vals_ret = vals[~np.isnan(vals)], vals_ret[~np.isnan(vals)]
        
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        if len(vals) == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No valid factor values", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        sns.histplot(vals, bins=bins, stat='density', alpha=0.35, color='#88CCEE', edgecolor='white', label='Actual Data', ax=ax)
        sns.kdeplot(vals, color='#345995', linestyle='-', linewidth=2.2, label='KDE (Alpha)', ax=ax)
        # 参考分布线
        x_range = np.linspace(np.min(vals)-0.5, np.max(vals)+0.5, 500)
        y_norm = norm.pdf(x_range, 0, 1)
        ax.plot(x_range, y_norm, color='#287C71', linestyle='--', linewidth=2, label='N(0,1) Reference')
        #sns.kdeplot(vals_ret, color='#E07A5F', linestyle='--', linewidth=2, label='KDE (Return)', ax=ax)
        
        # 统计量文本框
        if show_stats:
            stats = calc_alpha_distribution(sub.values, year)
            stats_text = (f"Mean = {stats['mean']:.3f}\nStd = {stats['std']:.3f}\n"
                        f"Skew = {stats['skew']:.3f}\nKurtosis = {stats['mean']:.3f}\n"
                        f"Min = {stats['min']:.3f}\nMax = {stats['max']:.3f}")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title)
        ax.set_xlabel('Factor Value Distribution')
        ax.set_ylabel('Density')
        ax.legend()
        return fig
    

    # 中文说明：`table_ic_annual_stats`：生成诊断表格。
    def table_ic_annual_stats(self):
        '''IC年度统计指标'''
        ics = self.cache['ics_df']
        rankics = self.cache['rankics_df']
        pos_ics, neg_ics, _,_ = calc_sign_IC(self.cache['label_arr'], self.cache['alpha_df'].values)
        pos_ics, neg_ics = pd.Series(pos_ics,index=ics.index), pd.Series(neg_ics,index=ics.index)

        results = []
        years = ics.index.year.unique()
        for yr in years:
            sub_ics, sub_rankics, sub_pos_ics, sub_neg_ics = ics[ics.index.year==yr], rankics[rankics.index.year==yr], pos_ics[pos_ics.index.year==yr], neg_ics[neg_ics.index.year==yr], 
            stats = calc_IC_stats(sub_ics, sub_rankics, sub_pos_ics, sub_neg_ics, yr)
            results.append(stats)
        overall = calc_IC_stats(ics, rankics, sub_pos_ics, sub_neg_ics, 'Overall')
        results.append(overall) 

        result = pd.DataFrame(results)
        result['year'] = result['year'].astype(str)
        return result[['year', 'avg_ic', 'ic_ir', 'avg_pos_ic', 'pos_ic_ir', 'avg_neg_ic', 'neg_ic_ir', 'avg_rank_ic', 'rank_ic_ir']].set_index('year').round(3)
    
    # 中文说明：`plot_ic_contribution`：绘制诊断图表。
    def plot_ic_contribution(self):
        '''IC指标累计时序图'''
        ics = self.cache['ics_df']
        rankics = self.cache['rankics_df']
        pos_ics, neg_ics, pos_rankics, neg_rankics = calc_sign_IC(self.cache['label_arr'], self.cache['alpha_df'].values)

        ic_df = pd.DataFrame({
            'ic': np.cumsum(ics),
            'pos_ic': np.cumsum(pos_ics),
            'neg_ic': np.cumsum(neg_ics),
            'rank_ic': np.cumsum(rankics),
            'rank_pos_ic': np.cumsum(pos_rankics),
            'rank_neg_ic': np.cumsum(neg_rankics)
        },index=ics.index)
        style_dict = {
            'ic': {'color': 'darkblue', 'linestyle': '-', 'label': 'ic', 'alpha': 0.8},
            'pos_ic': {'color': 'darkred', 'linestyle': '-', 'label': 'pos_ic', 'alpha': 0.7},
            'neg_ic': {'color': 'darkgreen', 'linestyle': '-', 'label': 'neg_ic', 'alpha': 0.7},
            'rank_ic': {'color': 'blue', 'linestyle': '--', 'label': 'rank_ic', 'alpha': 0.8},
            'rank_pos_ic': {'color': 'red', 'linestyle': '--', 'label': 'rank_pos_ic', 'alpha': 0.7},
            'rank_neg_ic': {'color': 'green', 'linestyle': '--', 'label': 'rank_neg_ic', 'alpha': 0.7}
        }
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        for col, style in style_dict.items():
            ax.plot(ic_df.index, ic_df[col], 
                    color=style['color'], 
                    linestyle=style['linestyle'], 
                    label=style['label'],
                    alpha=style['alpha'],
                    linewidth=1.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # 格式化 x 轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.YearLocator())   # 每年一个主刻度
        ax.xaxis.set_minor_locator(mdates.MonthLocator())  # 每月一个次刻度
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 标签、标题、图例、网格
        ax.set_xlabel('Date')
        ax.set_ylabel('IC Value')
        ax.set_title('IC and Rank IC Contribution Over Time')
        ax.legend(loc='best', fontsize=10)
        return fig
    
    # 中文说明：`plot_ic_distribution`：绘制诊断图表。
    def plot_ic_distribution(self, horizons=[1,3,5,10]):
        '''各回报期IC分布图'''
        alpha_df = self.cache['alpha_df']
        label_df = pd.DataFrame(self.cache['label_arr'], index=alpha_df.index, columns=alpha_df.columns)

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        for i in horizons:
            if i == 1: future_ret = label_df.values
            else: future_ret = Calculator.rolling_retprod(label_df.values, window=i, future=True)
            ic_vals = IC(alpha_df.values, future_ret)  # 形状 (T,)
            ic_valid = ic_vals[~np.isnan(ic_vals)]
            if len(ic_valid) > 1:
                sns.kdeplot(ic_valid, label=f'IC ({i} Day{"s" if i>1 else ""})', linewidth=2, ax=ax)
        ax.set_xlabel('Information Coefficient (IC)')
        ax.set_ylabel('Density')
        ax.set_title('IC Distribution for Different Horizons')
        ax.legend()
        return fig


    # 中文说明：`table_group_stats`：生成诊断表格。
    def table_group_stats(self):
        '''分组收益表现'''
        records = []
        for i in range(1,11):
            group_ret = self.cache['groupret_df'][i]
            records.append({
                'Group': f'G{i}',
                'AnnRet': calc_annret(group_ret),
                'AnnVol': calc_annvol(group_ret),
                'Sharpe': calc_sharpe(group_ret),
                'MaxDrawdown': calc_maxdrawdown(group_ret),
                'Calmar': calc_calmar(group_ret)
            })
        df = pd.DataFrame(records).set_index('Group').sort_values('Group', key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int), ascending=False)
        return df.round(3)
    
    # 中文说明：`plot_group_cumret`：绘制诊断图表。
    def plot_group_cumret(self):
        '''分组收益累计图'''
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        for i in range(1,11):
            cumret = np.nancumsum(self.cache['groupret_df'][i])
            ax.plot(self.cache['dates'], cumret, linewidth=1.8, label=f'Group {i}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Return by Group')
        ax.legend()
        return fig


    # 中文说明：`table_industry_annual_stats`：生成诊断表格。
    def table_industry_annual_stats(self):
        '''用各行业超额收益才有可比性'''
        records = []
        for ind_id, ind in self.loader.id_to_industry.items():
            ind_mask = np.where(self.cache['ind_code_arr']==ind_id, 1, 0).astype(bool)
            alpha_df = self.cache['alpha_df'].where(ind_mask, np.nan)
            label_arr = np.where(ind_mask, self.cache['label_arr'], np.nan)

            ind_ret = np.nanmean(label_arr,axis=-1)

            longshort_holds = calc_holdings(alpha_df)
            hold_df = longshort_holds.where(longshort_holds==1, 0).astype('int') if self.factor_type == 'long' else longshort_holds.astype('int')
            ret = np.nanmean(label_arr*hold_df,axis=-1)

            excess_ret = pd.Series(ret-ind_ret, index=self.cache['dates'])
            records.append({
                '行业名称': ind,
                '年化收益': calc_annret(excess_ret),
                '年化波动率': calc_annvol(excess_ret),
                '夏普比': calc_sharpe(excess_ret),
                '最大回撤': calc_maxdrawdown(excess_ret),
                '换手率': calc_turnover(hold_df,freq='D').mean().values[0]
            })
        return pd.DataFrame(records).set_index('行业名称').round(3)
   
    # 中文说明：`plot_industry_performance`：绘制诊断图表。
    def plot_industry_performance(self):
        '''用各行业超额收益才有可比性'''
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        for ind_id, ind in self.loader.id_to_industry.items():
            ind_mask = np.where(self.cache['ind_code_arr']==ind_id, 1, 0).astype(bool)
            alpha_df = self.cache['alpha_df'].where(ind_mask, np.nan)
            label_arr = np.where(ind_mask, self.cache['label_arr'], np.nan)

            ind_ret = np.nanmean(label_arr,axis=-1)

            longshort_holds = calc_holdings(alpha_df)
            hold_df = longshort_holds.where(longshort_holds==1, 0).astype('int') if self.factor_type == 'long' else longshort_holds.astype('int')
            ret = np.nanmean(label_arr*hold_df,axis=-1)

            excess_ret = ret - ind_ret
            ax.plot(self.cache['dates'], np.nancumsum(excess_ret), linewidth=1.8, label=f'{ind}')

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Excess Return')
        ax.set_title('Cumulative Excess Return in Industries')
        ax.legend()
        return fig
    
    # 中文说明：`table_industry_exposure_stats`：生成诊断表格。
    def table_industry_exposure_stats(self):
        '''行业暴露表现'''
        long_ind_pct_df, short_ind_pct_df = self.calc_ind_exposure()
        self.cache['long_ind_pct_df'], self.cache['short_ind_pct_df'] = long_ind_pct_df, short_ind_pct_df
        self.cache['ind_ret_df'] = ind_ret_df = self.calc_ind_ret()
        
        ind_stats = []
        for ind in self.loader.id_to_industry.values():
            ind_weight = self.cache['long_ind_pct_df'][ind] - self.cache['short_ind_pct_df'][ind] if self.factor_type=='longshort' else self.cache['long_ind_pct_df'][ind]
            ind_exposure_ret = ind_weight * ind_ret_df[ind]
            ind_stats.append({
                'Industry': ind,
                'Avg Exposure': ind_weight.mean(),
                'Ind Return': calc_annret(ind_ret_df[ind]),
                'AnnRet': calc_annret(ind_exposure_ret),
                'AnnVol': calc_annvol(ind_exposure_ret),
                'Sharpe': calc_sharpe(ind_exposure_ret),
                'MaxDrawdown': calc_maxdrawdown(ind_exposure_ret),
                'Calmar': calc_calmar(ind_exposure_ret)
            })
        return pd.DataFrame(ind_stats).set_index('Industry').sort_values('Sharpe', ascending=False).round(3)
    
    # 中文说明：`plot_industry_exposure_ret`：绘制诊断图表。
    def plot_industry_exposure_ret(self):
        '''行业暴露收益时序图'''
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        for i, ind in enumerate(self.loader.id_to_industry.values(), start=1):
            ind_weight = self.cache['long_ind_pct_df'][ind] - self.cache['short_ind_pct_df'][ind] if self.factor_type=='longshort' else self.cache['long_ind_pct_df'][ind]
            ind_exposure_ret = ind_weight * self.cache['ind_ret_df'][ind]
            ax.plot(self.cache['dates'], np.nancumprod(ind_exposure_ret+1)-1, linewidth=1.5, label=str(ind))
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Return by Industry')
        return fig
    
    # 中文说明：`plot_industry_component`：绘制诊断图表。
    def plot_industry_component(self):
        '''行业持仓结构'''
        long_ind_pct_df, short_ind_pct_df = self.cache['long_ind_pct_df'], self.cache['short_ind_pct_df']

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        n_ind = len(self.loader.id_to_industry)
        colors_ind = plt.cm.tab20(np.linspace(0, 1, n_ind))
        self._plot_bidirectional(ax, long_ind_pct_df, short_ind_pct_df, 'Industry Holdings (Long / Short)', colors_ind)
        fig.suptitle('Industry Long-Short Holding Structure', fontsize=16, fontweight='bold')
        return fig
    

    # 中文说明：`table_sector_annual_stats`：生成诊断表格。
    def table_sector_annual_stats(self):
        '''用各板块超额收益才有可比性'''
        records = []
        for sec_id, sec in self.loader.id_to_sector.items():
            sec_mask = np.where(self.cache['sec_code_arr']==sec_id, 1, 0).astype(bool)
            alpha_df = self.cache['alpha_df'].where(sec_mask, np.nan)
            label_arr = np.where(sec_mask, self.cache['label_arr'], np.nan)

            sec_ret = np.nanmean(label_arr,axis=-1)

            longshort_holds = calc_holdings(alpha_df)
            hold_df = longshort_holds.where(longshort_holds==1, 0).astype('int') if self.factor_type == 'long' else longshort_holds.astype('int')
            ret = np.nanmean(label_arr*hold_df,axis=-1)

            excess_ret = pd.Series(ret-sec_ret, index=self.cache['dates'])
            records.append({
                '板块名称': sec,
                '年化收益': calc_annret(excess_ret),
                '年化波动率': calc_annvol(excess_ret),
                '夏普比': calc_sharpe(excess_ret),
                '最大回撤': calc_maxdrawdown(excess_ret),
                '换手率': calc_turnover(hold_df,freq='D').mean().values[0]
            })
        return pd.DataFrame(records).set_index('板块名称').round(3)
    
    # 中文说明：`plot_sector_performance`：绘制诊断图表。
    def plot_sector_performance(self):
        '''用各板块超额收益才有可比性'''
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        for sec_id, sec in self.loader.id_to_sector.items():
            sec_mask = np.where(self.cache['sec_code_arr']==sec_id, 1, 0).astype(bool)
            alpha_df = self.cache['alpha_df'].where(sec_mask, np.nan)
            label_arr = np.where(sec_mask, self.cache['label_arr'], np.nan)

            ind_ret = np.nanmean(label_arr,axis=-1)

            longshort_holds = calc_holdings(alpha_df)
            hold_df = longshort_holds.where(longshort_holds==1, 0).astype('int') if self.factor_type == 'long' else longshort_holds.astype('int')
            ret = np.nanmean(label_arr*hold_df,axis=-1)

            excess_ret = ret - ind_ret
            ax.plot(self.cache['dates'], np.nancumsum(excess_ret), linewidth=1.8, label=f'{sec}')

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Excess Return')
        ax.set_title('Cumulative Excess Return in Sectors')
        ax.legend()
        return fig
            
    # 中文说明：`table_sector_exposure_stats`：生成诊断表格。
    def table_sector_exposure_stats(self):
        '''板块暴露表现'''
        long_sec_pct_df, short_sec_pct_df = self.calc_sec_exposure()
        self.cache['long_sec_pct_df'], self.cache['short_sec_pct_df'] = long_sec_pct_df, short_sec_pct_df
        self.cache['sec_ret_df'] = sec_ret_df = self.calc_sec_ret()

        sec_stats = []
        for sec in self.loader.id_to_sector.values():
            sec_weight = self.cache['long_sec_pct_df'][sec] - self.cache['short_sec_pct_df'][sec] if self.factor_type=='longshort' else self.cache['long_sec_pct_df'][sec]
            sec_exposure_ret = sec_weight * sec_ret_df[sec]
            sec_stats.append({
                'Sector': sec,
                'Avg Exposure': sec_weight.mean(),
                'Sec Return': calc_annret(sec_ret_df[sec]),
                'AnnRet': calc_annret(sec_exposure_ret),
                'AnnVol': calc_annvol(sec_exposure_ret),
                'Sharpe': calc_sharpe(sec_exposure_ret),
                'MaxDrawdown': calc_maxdrawdown(sec_exposure_ret),
                'Calmar': calc_calmar(sec_exposure_ret)
            })
        return pd.DataFrame(sec_stats).set_index('Sector').sort_values('Sharpe', ascending=False).round(3)

    # 中文说明：`plot_sector_exposure_ret`：绘制诊断图表。
    def plot_sector_exposure_ret(self):
        '''板块暴露收益时序图'''
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        for i, sec in enumerate(self.loader.id_to_sector.values(), start=1):
            sec_weight = self.cache['long_sec_pct_df'][sec] - self.cache['short_sec_pct_df'][sec] if self.factor_type=='longshort' else self.cache['long_sec_pct_df'][sec]
            sec_exposure_ret = sec_weight * self.cache['sec_ret_df'][sec]
            ax.plot(self.cache['dates'], np.nancumsum(sec_exposure_ret), linewidth=1.7, label=str(sec))
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Return by Sector')
        return fig

    # 中文说明：`plot_sector_component`：绘制诊断图表。
    def plot_sector_component(self):
        '''板块持仓结构'''
        long_sec_pct_df, short_sec_pct_df = self.cache['long_sec_pct_df'], self.cache['short_sec_pct_df']

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        n_sec = len(self.loader.id_to_sector)
        colors_sec = plt.cm.Set3(np.linspace(0, 1, n_sec))
        self._plot_bidirectional(ax, long_sec_pct_df, short_sec_pct_df, 'Sector Holdings (Long / Short)', colors_sec)
        fig.suptitle('Sector Long-Short Holding Structure', fontsize=16, fontweight='bold', y=1.01)
        return fig

    # 中文说明：`_plot_bidirectional`：内部辅助步骤，不作为稳定公共接口。
    def _plot_bidirectional(self, ax, long_pct, short_pct, title, colors):
        """在同一个轴上画多头（正）和空头（负）的百分比堆积面积图"""
        # 统一列顺序，按字母或自定义顺序排列以保证颜色固定
        all_cats = sorted(list(long_pct.columns))
        long_pct = long_pct[all_cats]
        short_pct = short_pct[all_cats]

        x = long_pct.index
        # 多头向上
        display_labels = [str(cat) for cat in all_cats]
        ax.stackplot(x, long_pct.values.T, labels=display_labels, colors=colors[:len(all_cats)], alpha=0.85)
        # 空头向下
        ax.stackplot(x, (-short_pct).values.T, colors=colors[:len(all_cats)], alpha=0.85)

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Holding Weight')
        ax.set_ylim(-1.05, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False, ncol=1)
        # X轴日期格式
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


    # 中文说明：`table_barra_exposure_stats`：生成诊断表格。
    def table_barra_exposure_stats(self):
        '''Barra因子暴露表现'''
        self.cache['barra_exposure_df'] = barra_exposure_df = self.calc_barra_exposure()
        self.cache['barra_ret_df'] = barra_ret_df = self.calc_barra_ret()[0]

        barra_stats = []
        for barra_name in self.loader.id_to_barra.values():
            barra_weight = barra_exposure_df[barra_name]
            barra_exposure_ret = barra_exposure_df[barra_name] * barra_ret_df[barra_name]
            barra_stats.append({
                'Barra': barra_name,
                'Avg Exposure': barra_weight.mean(),
                'Exposure Return': calc_annret(barra_ret_df[barra_name]),
                'AnnRet': calc_annret(barra_exposure_ret),
                'AnnVol': calc_annvol(barra_exposure_ret),
                'Sharpe': calc_sharpe(barra_exposure_ret),
                'MaxDrawdown': calc_maxdrawdown(barra_exposure_ret),
                'Calmar': calc_calmar(barra_exposure_ret)
            })
        return pd.DataFrame(barra_stats).set_index('Barra').sort_values('Sharpe', ascending=False).round(3)
    
    # 中文说明：`plot_barra_exposure`：绘制诊断图表。
    def plot_barra_exposure(self):
        '''Barra因子暴露'''
        barra_exposure_df = self.cache['barra_exposure_df']

        avg_barra_exposure = barra_exposure_df.mean()
        s = avg_barra_exposure.sort_values(ascending=False)
        colors = plt.cm.tab20(np.linspace(0, 1, len(s)))

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.bar(s.index, s.values, color=colors, edgecolor='white', linewidth=0.7)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # 旋转45度避免重叠
        ax.set_ylabel('Exposure')
        ax.set_title('Avg Barra Factor Exposure')
        return fig

    # 中文说明：`plot_barra_exposure_ret`：绘制诊断图表。
    def plot_barra_exposure_ret(self):
        '''Barra因子暴露收益时序图'''
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        for barra_name in self.loader.id_to_barra.values():
            barra_exposure_ret = self.cache['barra_exposure_df'][barra_name] * self.cache['barra_ret_df'][barra_name]
            ax.plot(self.cache['dates'], np.nancumprod(barra_exposure_ret+1)-1, linewidth=1.6, label=barra_name)
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Return by Barra')
        return fig
    

    # Spearman redundancy analyzer methods disabled per current FactorTest workflow; kept for future restoration.
#     # 中文说明：`calc_pair_corr`：计算研究或生产指标。
#     def calc_pair_corr(self, X, y, i):
#         ic = np.nanmean(IC(X[:,:,i], y))
#         rankic = np.nanmean(rankIC(X[:,:,i], y))
#         return i, ic, rankic
#
#     # 中文说明：`plot_corr_redundancy`：绘制诊断图表。
#     def plot_corr_redundancy(self, threshold=0.7):
#         # 数据对齐
#         K = len(self.poolfactor_names)
#
#         valid_dates = np.intersect1d(pd.to_datetime(self.poolfactor_dates), self.cache['dates'])
#         valid_pooldates_idx = np.searchsorted(pd.to_datetime(self.poolfactor_dates), valid_dates)
#         valid_alphadates_idx = np.searchsorted(self.cache['dates'], valid_dates)
#
#         alpha_arr = self.cache['alpha_df'].values[valid_alphadates_idx]
#         poolfactors = self.poolfactors[valid_pooldates_idx]  # T,K,N
#         valid_pool = self.cache['pool_mask'][valid_alphadates_idx]
#
#         rows, cols = np.nonzero(valid_pool)
#         valid_ticks, valid_poolticks_idx, sub_idx = np.intersect1d(
#             self.poolfactor_ticks, self.ticks[cols], return_indices=True
#         )
#         valid_alphaticks_idx = cols[sub_idx]
#
#         # 因子
#         alpha_arr = alpha_arr[:,valid_alphaticks_idx]
#         poolfactors = poolfactors.transpose(2,0,1)[valid_poolticks_idx,...].transpose(1,0,2)  # N,T,K -> T,N,K
#
#         T, N, K = poolfactors.shape
#         corr_res = Parallel(n_jobs=-1, batch_size=100)(
#             delayed(self.calc_pair_corr)(poolfactors, alpha_arr, i) for i in range(K)
#         )
#         ic_vals = np.array([r[1] for r in corr_res])
#         rankic_vals = np.array([r[2] for r in corr_res])
#         names = self.poolfactor_names
#
#         # -------------------- 取出绝对值最大的10个因子（分别对IC和RankIC）--------------------
#         top10_ic_idx = np.argsort(np.abs(ic_vals))[-10:][::-1]
#         top10_rankic_idx = np.argsort(np.abs(rankic_vals))[-10:][::-1]
#
#         # 构建 2×10 热力图数据
#         heatmap_data = np.array([
#             ic_vals[top10_ic_idx],
#             rankic_vals[top10_rankic_idx]
#         ])
#         xlabels_ic = [names[i] for i in top10_ic_idx]
#         xlabels_rankic = [names[i] for i in top10_rankic_idx]
#
#         # -------------------- 绘图 --------------------
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=DEFAULT_FIGSIZE)
#         cmap = plt.cm.RdBu_r
#         norm = plt.Normalize(-1, 1)
#
#         # 第一行：IC
#         im1 = ax1.imshow(heatmap_data[0:1, :], cmap=cmap, norm=norm, aspect='auto')
#         ax1.set_yticks([0])
#         ax1.set_yticklabels(['IC'])
#         ax1.set_xticks(range(10))
#         ax1.set_xticklabels(xlabels_ic, rotation=45, ha='right', fontsize=9)
#         for j, val in enumerate(heatmap_data[0]):
#             color = 'white' if abs(val) > 0.5 else 'black'
#             ax1.text(j, 0, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)
#
#         # 第二行：RankIC
#         ax2.imshow(heatmap_data[1:2, :], cmap=cmap, norm=norm, aspect='auto')
#         ax2.set_yticks([0])
#         ax2.set_yticklabels(['RankIC'])
#         ax2.set_xticks(range(10))
#         ax2.set_xticklabels(xlabels_rankic, rotation=45, ha='right', fontsize=9)
#         for j, val in enumerate(heatmap_data[1]):
#             color = 'white' if abs(val) > 0.5 else 'black'
#             ax2.text(j, 0, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)
#
#         fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', pad=0.15, fraction=0.02, label='Correlation')
#         fig.suptitle('Top Redundant Factors', fontsize=13, fontweight='bold')
#         return fig
#
#
    # 中文说明：`table_regime_stats`：生成诊断表格。
    def table_regime_stats(self, benchmark_ret=None, vol_window=20):
        dates = self.cache['dates']
        ret_df = self.cache['ret_df']
        ics = self.cache['ics_df']
        rankics = self.cache['rankics_df']

        if benchmark_ret is None:
            benchmark_ret = pd.Series(np.nanmean(self.cache['label_arr'], axis=1), index=dates)
        else:
            benchmark_ret = benchmark_ret.reindex(dates)

        bench_cum = (1 + benchmark_ret.fillna(0)).cumprod()
        bench_ma = bench_cum.rolling(60, min_periods=20).mean()
        regime = pd.Series("震荡市", index=dates)
        regime[bench_cum > bench_ma * 1.05] = "牛市"
        regime[bench_cum < bench_ma * 0.95] = "熊市"

        vol = benchmark_ret.rolling(vol_window).std()
        vol_regime = pd.Series("低波动", index=dates)
        vol_regime[vol > vol.median()] = "高波动"

        mask_data = self.loader.load_masks()
        pool_masks = {
            name: mask_data[key][self.cache['dates_idx']].astype(bool)
            for name, key in self.pool_regime_specs
            if key in mask_data
        }
        pool_ic_dict = {name: [] for name in pool_masks}
        pool_rankic_dict = {name: [] for name in pool_masks}
        self.pool_excessret_dict = {name: [] for name in pool_masks}

        alpha_df = self.cache['alpha_df']
        label_arr = self.cache['label_arr']

        for t, dt in enumerate(dates):
            a = alpha_df.iloc[t].values
            y = label_arr[t]
            finite = np.isfinite(a) & np.isfinite(y)

            for name, pool_mask in pool_masks.items():
                valid = finite & pool_mask[t]
                ic_val = rankic_val = excess_ret = np.nan
                if valid.sum() >= 30:
                    xx = a[valid]
                    yy = y[valid]
                    bench_ret = float(np.nanmean(yy))
                    one_row_alpha = pd.DataFrame([xx], index=[dt], columns=np.asarray(alpha_df.columns)[valid])
                    holds = calc_holdings(one_row_alpha)
                    hold_arr = holds.where(holds == 1, 0).values if self.factor_type == 'long' else holds.values
                    ret = float(np.nanmean(yy * hold_arr.reshape(-1)))
                    ic_val = self._safe_pearson(xx, yy)
                    rankic_val = self._safe_rank_corr(xx, yy)
                    excess_ret = ret - bench_ret
                pool_ic_dict[name].append(ic_val)
                pool_rankic_dict[name].append(rankic_val)
                self.pool_excessret_dict[name].append(excess_ret)

        # 中文说明：`calc_sub_stats`：计算研究或生产指标。
        def calc_sub_stats(mask, name):
            sub_ret = ret_df.loc[mask]
            sub_ic = ics.loc[mask]
            sub_rankic = rankics.loc[mask]
            return {
                "Regime": name,
                "Days": int(mask.sum()),
                "AnnRet": calc_annret(sub_ret),
                "AnnVol": calc_annvol(sub_ret),
                "Sharpe": calc_sharpe(sub_ret),
                "MaxDrawdown": calc_maxdrawdown(sub_ret),
                "Avg_IC": sub_ic.mean(),
                "ICIR": sub_ic.mean() / sub_ic.std() * np.sqrt(252) if sub_ic.std() != 0 else np.nan,
                "Avg_RankIC": sub_rankic.mean(),
                "RankICIR": sub_rankic.mean() / sub_rankic.std() * np.sqrt(252) if sub_rankic.std() != 0 else np.nan,
                "WinRate": (sub_ret["ret"] > 0).mean(),
            }

        rows = []
        for name in ["牛市", "熊市", "震荡市"]:
            mask = regime == name
            if mask.sum() > 20:
                rows.append(calc_sub_stats(mask, name))
        for name in ["高波动", "低波动"]:
            mask = vol_regime == name
            if mask.sum() > 20:
                rows.append(calc_sub_stats(mask, name))

        for name in pool_masks:
            sub_ic = pd.Series(pool_ic_dict[name], index=dates)
            sub_rankic = pd.Series(pool_rankic_dict[name], index=dates)
            excess_ret = pd.Series(self.pool_excessret_dict[name], index=dates)
            rows.append({
                "Regime": name,
                "Days": int(excess_ret.notna().sum()),
                "AnnRet": calc_annret(excess_ret.fillna(0)),
                "AnnVol": calc_annvol(excess_ret),
                "Sharpe": calc_sharpe(excess_ret.fillna(0)),
                "MaxDrawdown": calc_maxdrawdown(excess_ret.fillna(0)),
                "Avg_IC": sub_ic.mean(),
                "ICIR": sub_ic.mean() / sub_ic.std() * np.sqrt(252) if sub_ic.std() != 0 else np.nan,
                "Avg_RankIC": sub_rankic.mean(),
                "RankICIR": sub_rankic.mean() / sub_rankic.std() * np.sqrt(252) if sub_rankic.std() != 0 else np.nan,
                "WinRate": (excess_ret > 0).mean(),
            })

        return pd.DataFrame(rows).set_index("Regime").round(3)
    
    # 中文说明：`plot_regime_cumret`：绘制诊断图表。
    def plot_regime_cumret(self, benchmark_ret=None):
        """Cumulative return split by market regime; safe to call before table_regime_stats."""
        dates = self.cache['dates']
        ret = self.cache['ret_df']['ret']

        if not hasattr(self, 'pool_excessret_dict'):
            self.table_regime_stats(benchmark_ret=benchmark_ret)

        if benchmark_ret is None:
            benchmark_ret = pd.Series(np.nanmean(self.cache['label_arr'], axis=1), index=dates)
        else:
            benchmark_ret = benchmark_ret.reindex(dates)

        bench_cum = (1 + benchmark_ret.fillna(0)).cumprod()
        bench_ma = bench_cum.rolling(60, min_periods=20).mean()
        regime = pd.Series("震荡市", index=dates)
        regime[bench_cum > bench_ma * 1.05] = "牛市"
        regime[bench_cum < bench_ma * 0.95] = "熊市"

        vol = benchmark_ret.rolling(20).std()
        vol_median = vol.median()
        vol_regime = pd.Series("低波动", index=dates)
        vol_regime[vol > vol_median] = "高波动"

        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        for name in ["牛市", "熊市", "震荡市"]:
            sub_ret = ret.where(regime == name, 0)
            ax.plot(dates, np.nancumsum(sub_ret), label=name, linestyle='solid', linewidth=1.8)

        for name in ["高波动", "低波动"]:
            sub_ret = ret.where(vol_regime == name, 0)
            ax.plot(dates, np.nancumsum(sub_ret), label=name, linestyle='dashed', linewidth=1.8)

        for pool_name, values in getattr(self, 'pool_excessret_dict', {}).items():
            if len(values) == len(dates):
                ax.plot(dates, np.nancumsum(values), label=pool_name, linestyle='dotted', linewidth=1.8)

        ax.set_title("Factor Cumulative Return by Market Regime")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        return fig


    # 中文说明：`_safe_pearson`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _safe_pearson(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return np.nan
        return float(np.corrcoef(x[mask], y[mask])[0, 1])

    # 中文说明：`_safe_rank_corr`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _safe_rank_corr(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return np.nan
        corr, _ = spearmanr(x[mask], y[mask])
        return float(corr)

    # 中文说明：`table_shadow_capacity_test`：生成诊断表格。
    def table_shadow_capacity_test(
        self,
        capital_list=[1e7, 5e7, 1e8, 5e8],
        max_participation=0.1,
        cost_rate=0.001,
        impact_coef=0.001,
        min_lot=100,
    ):
        dates = self.cache['dates']
        hold_df = self.cache['hold_df']

        date_idx1 = np.searchsorted(self.dates, dates)
        pool_mask = self.pool_mask[date_idx1+1]

        date_idx2 = np.searchsorted(self.loader.dates, dates)
        next_open = np.memmap(self.root/'d_field/open.bin', shape=(self.loader.T, self.loader.N), mode='r', dtype=float)[date_idx2+1]
        next_open = np.where(pool_mask, next_open, np.nan)
        next_amount = np.memmap(self.root/'d_field/amount.bin', shape=(self.loader.T, self.loader.N), mode='r', dtype=float)[date_idx2+1]
        next_amount = np.where(pool_mask, next_amount, np.nan)

        n_days, n_stocks = hold_df.shape
        results = []
        self.net_ret = {}
        self.shadow_equity_curve = {}

        for capital in capital_list:
            equity = float(capital)
            cash = float(capital)
            shares = np.zeros(n_stocks, dtype=float)
            equity_curve = np.zeros(n_days, dtype=float)
            daily_turnover_vals = np.zeros(n_days, dtype=float)
            daily_fill_vals = np.ones(n_days, dtype=float)
            trans_cost_ratio = np.zeros(n_days, dtype=float)
            impact_cost_ratio = np.zeros(n_days, dtype=float)

            for i in range(n_days):
                price = next_open[i]
                amount_i = np.nan_to_num(next_amount[i], nan=0.0, posinf=0.0, neginf=0.0)
                if np.all(np.isnan(price)):
                    equity_curve[i] = equity
                    continue

                signal = hold_df.iloc[i].values
                long_mask = (signal == 1)
                short_mask = (signal == -1)
                n_long, n_short = int(np.sum(long_mask)), int(np.sum(short_mask))
                if self.factor_type == 'longshort' and n_long > 0 and n_short > 0:
                    long_budget, short_budget = equity * 0.5, equity * 0.5
                else:
                    long_budget, short_budget = equity, equity

                target_shares = shares.copy()
                if n_long > 0:
                    target_shares[long_mask] = ( (long_budget / n_long) / price[long_mask] ) // min_lot
                if n_short > 0 and self.factor_type == 'longshort':
                    target_shares[short_mask] = ( -(short_budget / n_short) / price[short_mask] ) // min_lot

                trade_shares = np.nan_to_num(target_shares - shares, nan=0.0, posinf=0.0, neginf=0.0)
                max_tradable_value = amount_i * max_participation

                sell_shares = np.where(trade_shares < 0, -trade_shares, 0.0)
                sell_value = sell_shares * price
                fill_sell = np.clip(np.divide(max_tradable_value, sell_value, out=np.ones_like(sell_value), where=sell_value > 0), 0, 1)
                actual_sell_shares = np.where(sell_shares * fill_sell >= 1, sell_shares * fill_sell, 0.0)
                actual_sell_value = np.nan_to_num(actual_sell_shares * price, nan=0.0, posinf=0.0, neginf=0.0)
                sell_proceeds = float(np.nansum(actual_sell_value))
                part_sell = np.divide(actual_sell_value, amount_i, out=np.zeros_like(actual_sell_value), where=amount_i > 0)
                impact_sell = float(np.nansum(actual_sell_value * np.sqrt(np.clip(part_sell, 0, None)) * impact_coef / 100))
                trans_sell = sell_proceeds * cost_rate
                shares -= actual_sell_shares
                cash += sell_proceeds - impact_sell - trans_sell

                buy_shares = np.where(trade_shares > 0, trade_shares, 0.0)
                buy_value = buy_shares * price
                fill_buy = np.clip(np.divide(max_tradable_value, buy_value, out=np.ones_like(buy_value), where=buy_value > 0), 0, 1)
                raw_buy_shares = buy_shares * fill_buy
                raw_buy_cost = float(np.nansum(raw_buy_shares * price))
                if raw_buy_cost > cash and raw_buy_cost > 0:
                    raw_buy_shares *= max(cash, 0.0) / raw_buy_cost
                actual_buy_shares = np.where(raw_buy_shares >= 1, raw_buy_shares, 0.0)
                actual_buy_value = np.nan_to_num(actual_buy_shares * price, nan=0.0, posinf=0.0, neginf=0.0)
                buy_proceeds = float(np.nansum(actual_buy_value))
                part_buy = np.divide(actual_buy_value, amount_i, out=np.zeros_like(actual_buy_value), where=amount_i > 0)
                impact_buy = float(np.nansum(actual_buy_value * np.sqrt(np.clip(part_buy, 0, None)) * impact_coef / 100))
                trans_buy = buy_proceeds * cost_rate
                shares += actual_buy_shares
                cash -= buy_proceeds + impact_buy + trans_buy

                total_cost = trans_sell + trans_buy
                total_impact = impact_sell + impact_buy
                position_value = float(np.nansum(shares * price))
                equity = cash + position_value
                equity_curve[i] = equity
                base_equity = max(abs(equity), 1e-12)
                trans_cost_ratio[i] = total_cost / base_equity
                impact_cost_ratio[i] = total_impact / base_equity
                total_trade_value = sell_proceeds + buy_proceeds
                daily_turnover_vals[i] = 0.5 * total_trade_value / base_equity
                sell_fills = actual_sell_shares[sell_shares > 0] / sell_shares[sell_shares > 0] if np.any(sell_shares > 0) else np.array([1.0])
                buy_fills = actual_buy_shares[buy_shares > 0] / buy_shares[buy_shares > 0] if np.any(buy_shares > 0) else np.array([1.0])
                daily_fill_vals[i] = float(np.nanmean(np.concatenate([sell_fills, buy_fills])))

            equity_series = pd.Series(equity_curve, index=dates).replace([np.inf, -np.inf], np.nan).ffill().fillna(capital)
            ret_series = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).fillna(0.0)
            self.net_ret[capital] = ret_series.values
            self.shadow_equity_curve[capital] = equity_series
            sharpe = calc_sharpe(ret_series)
            avg_fill = float(np.nanmean(daily_fill_vals))
            results.append({
                "Capital": capital,
                "AvgFillRatio": avg_fill,
                "AnnRet_Net": calc_annret(ret_series),
                "AnnVol_Net": calc_annvol(ret_series),
                "Sharpe_Net": sharpe,
                "MaxDrawdown_Net": calc_maxdrawdown(ret_series),
                "AvgTurnover_oneside": float(np.nanmean(daily_turnover_vals)),
                "AvgTransCost": float(np.nanmean(trans_cost_ratio)),
                "AvgImpactCost": float(np.nanmean(impact_cost_ratio)),
                "OnlineDecision": "Pass" if avg_fill > 0.8 and sharpe > 1 else "Fail",
            })
        return pd.DataFrame(results).set_index("Capital").round(4)

    # 中文说明：`plot_shadow_capacity_curve`：绘制诊断图表。
    def plot_shadow_capacity_curve(self, capital_list=[1e7, 5e7, 1e8, 5e8]):
        if not hasattr(self, 'net_ret') or any(capital not in self.net_ret for capital in capital_list):
            self.table_shadow_capacity_test(capital_list=capital_list)
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        for capital in capital_list:
            curve = getattr(self, 'shadow_equity_curve', {}).get(capital)
            if curve is None:
                curve = pd.Series(np.nancumprod(1 + self.net_ret[capital]), index=self.cache['dates'])
            curve = pd.Series(curve, index=self.cache['dates'])
            y = curve / float(curve.iloc[0])
            ax.plot(self.cache['dates'], y, label=f"{capital / 1e8:.1f}亿元", linewidth=1.8)
        ax.set_title("Shadow Portfolio Capacity Test")
        ax.set_xlabel("Date")
        ax.set_ylabel("Net Cumulative Return")
        ax.legend()
        return fig



if __name__ == '__main__':
    # 示例用法
    info = {
        'name': 'MinuteGRU',
        'factor_type': 'longshort',
        'alpha_type': '深度学习',
        'usage': '日频选股',
        'universe': 'universe',
        'start_date': '2020-01-01',
        'end_date': '2025-12-16',
        'summary': 'This is an example factor for demonstration purposes.'
    }
    pred_df = pd.read_csv('MinuteGRU.csv', index_col=0, parse_dates=True)

    analyzer = FactorAnalyzer(info, pred_df)
    analyzer.reset_axis('2021-01-01','2025-11-30','universe')

    # prf_stats = analyzer.table_PRF_stats()
    #winrate_scan = analyzer.table_winrate_scan()

    # monthly_ret = analyzer.table_monthly_ret()
    # annual_stats = analyzer.table_annual_stats()
    # basic_performance = analyzer.plot_basic_performance()
    # alpha_annual_stats = analyzer.table_alpha_annual_stats()
    # alpha_distribution = analyzer.plot_alpha_distribution()

    # ic_annual_stats = analyzer.table_ic_annual_stats()
    # ic_distribution = analyzer.plot_ic_distribution()
    # ic_contribution = analyzer.plot_ic_contribution()

    # group_stats = analyzer.table_group_stats()
    # group_cumret = analyzer.plot_group_cumret()

    #industry_excess_cumret = analyzer.plot_industry_performance()
    # industry_exposure_stats = analyzer.table_industry_exposure_stats()
    # industry_component = analyzer.plot_industry_component()
    # industry_exposure_ret = analyzer.plot_industry_exposure_ret()

    # sector_excess_stats = analyzer.table_sector_annual_stats()
    # sector_excess_cumret = analyzer.plot_sector_performance()
    # sector_exposure_stats = analyzer.table_sector_exposure_stats()
    # sector_component = analyzer.plot_sector_component()
    # sector_exposure_ret = analyzer.plot_sector_exposure_ret()

    # barra_exposure_stats = analyzer.table_barra_exposure_stats()
    # barra_exposure = analyzer.plot_barra_exposure()
    # barra_exposure_ret = analyzer.plot_barra_exposure_ret()

    # redundancy_hotmap = analyzer.plot_corr_redundancy()
    # redundancy_hotmap.savefig('debug.png')

    #regime_stats = analyzer.table_regime_stats()
    #regime_fig = analyzer.plot_regime_cumret()

    capacity_stats = analyzer.table_shadow_capacity_test()
    #capacity_fig = analyzer.plot_shadow_capacity_curve()

