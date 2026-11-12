# 单因子检验

作者：xujiayi；创建日期：2026-11-11。

## 原链路梳理

`ResearchFlow/FactorTest/web.py → FactorAnalyzer → prepare_data → Loader`
读取因子、标签、市值、行业、Barra及股票池，`metrics.py` 计算IC、分组和持仓。
分析器按cache切换区间/域，输出表现、分布、IC、域诊断及影子容量。
原影子容量独立读取次日open/amount；其整手换算、非选中持仓退出、缺失价格和
收益裁剪口径不宜直接沿用。新模块不导入旧analyzer，避免其模块级环境变量和绘图配置副作用。

`Backtest/singleFactor/example_usage.py → FactorData/adapters → BacktestConfig →
SingleFactorBacktester._targets → builders → run → BacktestResult.report`。
该run主要为目标权重乘未来收益、扣换手和借券费用；容量另经
`CapacitySimulator.run → CapacityResult.report`，以价格、成交额、整手和冲击模拟。
它们对资金成交的约束不同。新模块保留“研究收益/实际成交收益”两层，实际组合与容量共用execution，
不直接复用含事件/配对逻辑的目标构建器，也不改旧模块行为。

## 文件职责

| 文件 | 对应需求与入口 |
|---|---|
| factorbase.py | 显式配置、DataPool/注入数据、process_alpha、cal_labels、cal_benchmark、cal_groups、cal_target_holding、cal_portfolio |
| overall_cross_section.py | 整体表现（截面）：plot、yearly_table、monthly_table |
| overall_time_series.py | 整体表现（时序）：classification_table、returns_table |
| distribution.py | 因子值分布：value_table、plot_values、group_table、plot_groups |
| ic_test.py | IC检验：table、plot_distribution、cumulative、decay_table、plot_decay |
| domains.py | 分域表现：八类域函数、domain_returns、table、plot |
| portfolio_tracking.py | 组合跟踪：plot_performance、plot_holdings、plot_execution、plot_costs |
| capacity.py | 容量测试：table、plot |
| maths.py | 无状态数学函数：变换、残差化、相关、标签、权重、收益、HAC、绩效 |
| execution.py | 共享成交与自融资账本，返回PortfolioResult |
| example.py | 本地全链路run示例 |

```python
from v2.SingleFactorTest import FactorBase, overall_cross_section, ic_test, capacity

f = FactorBase("factor_pool/your_factor", construction="longshort", root="Z:/")
f.load()  # 至少加载6年；默认验证至少5年有效日期
f.process_alpha()
f.cal_labels()
stats = overall_cross_section.yearly_table(f)
fig = overall_cross_section.plot(f)
ic_stats = ic_test.table(f, horizon=5, kind="rankIC")
p = f.cal_portfolio(construction="long", turnover_limit=.3, stock_cap=.03,
                    commission=.0003, stamp_tax=.0005, execution_price="vwap")
cap = capacity.table(f, capitals=(1e7, 5e7, 1e8), construction="long", stock_cap=.03)
```

`load(alpha=df, pct=decimal_returns, fields=..., strict_history=False)` 可做短样本独立实验。
默认本地pct按百分数除100，注入pct必须已经是小数；市值、amount、价格须由调用方保证单位一致
（成交额元、股价元、volume股）。文件路径可通过`field_paths`覆盖；BP字段可能随本地存储版本不同，
默认`d_financial/bp_lf`，缺失时相关表列为NaN，请覆盖为实际路径。
所有DataFrame按时间标签、股票代码对齐，不按列位置直接相乘。缺失状态禁止交易。
调用 `load_domains(indices=(...), styles=(...))` 按需读取指数权重和Barra暴露/分组mask；
也可在load时注入`domain_masks/domain_weights/barra`。默认不一次性读入所有风格，以控制内存。

## 统一口径与设计决策

* 标签：t行对应t+1开始的1/3/5/10/20交易日复合收益；未知尾端和途中缺失保留NaN。
* 分组：处理后因子在当日域内分位分组，G1最小、G10最大，并列值不强拆；常数截面不分组。
  组内等权，组外为0，研究收益日期为收益实际发生日，使用前日权重。
* 预处理：当日中位数/MAD缩放后asinh（可替换），OLS去行业与log流通市值，再做等权Z-score。
  回归不使用未来统计；不对全样本拟合Yeo-Johnson参数。标准化用于可比性，持仓另行归一化。
* 权重：多空每腿分别归一化后默认各占NAV的50%，总敞口100%；单腿占100%。
  `active_gross=2` 可让多空每腿100%。因子加权使用腿方向分数减去当日腿内最小值，保证非负、单调。
  `benchmark_enhanced=True` 为域内等权基准+主动权重，单股及行业偏离约束围绕该基准。
* 整体研究：多空收益差为每腿100%；多头为多头组-域内等权；空头为域内等权-空头组。
  空头方向明确为有符号收益。研究指标不含成本，不能与总敞口100%的实盘模拟直接等同。
* 年表：数据末日所在年及之前5个自然年，比如2026年末日展示2021—2026和Overall。
  Overall使用全部载入样本；绘图和大部分近期诊断用report_start至数据末日。
  月表为当月日超额收益复合，尚未发生或无有效数据的月份为空。
* IC IR是日均/日标准差，未乘sqrt252。正负IC沿用旧calc_sign_IC：按未来收益正负分股票子样本，
  在该子样本内计算IC/rankIC，不是按IC时序自身的正负切片。该子样本用于事后诊断，不能据此选股。
  累积IC求和。衰减使用未来第k个独立交易日收益，不用重叠的k日累计收益冒充衰减；
  同时提供因子rank跨k日截面自相关、IC_k与IC_1的时序相关。
* 择时必须显式传一维信号；大于threshold触发，side决定多/空收益方向。
  分类基准胜率取所有有效日期的方向收益胜率；收益超额取触发收益减全样本平均收益，
  避免用同一个基准减自身得到恒零。HAC滞后h-1处理重叠收益。
  需求中的边际价格比按复合窗口实现：2/1、3/2、5/3、10/5、20/10。
* BP为账面市值比。组内IC样本不足为NaN。拥挤度仅为指定三指标相对平均组的代理值，
  缺一指标不强行补齐；不将其当作真实交易拥挤识别模型。
* 域beta/收益贡献来自单变量时间回归；域交叠，贡献不能求和。Barra柱图为组合风格暴露，
  其他柱图为域内标准因子均值减全市场均值。regime沿用60日均线±5%，不使用全样本阈值。

## 实际成交与容量的边界

次日open/VWAP成交、当日close估值，现金与逐股股数维护自融资账本。
使用raw价格和`ratio_adjusting_factor`处理复权变化，禁止用未复权跳空直接污染组合PnL。
成交额上限、整手、佣金、卖出印花税、平方根冲击、年化借券费在同一引擎核算。
费用参数由调用方按回测期间确定，默认值不是历史税率时间表。
所有成本保留当日盘前NAV比例，`pnl`为元，`net_return`包括全部成本；初始日不开仓。

涨跌停默认双向禁交易，停牌禁止交易；ST目标归零并持续请求退出，若跌停/停牌导致无法卖出则
保留真实持仓并报告`st_remaining_value`，不能伪造强制成交。新ST退出优先于自主换手约束。
本地`tradable_t`仅用于当日研究池；执行卖出不直接套用该池，否则ST会永远无法清仓。
换手/单股/行业约束先投影目标，再应用成交约束；无法成交和收盘价格漂移可能造成实际约束偏离，
在diagnostics中如实保留，优化不可行直接报错。偏离约束启用时用SLSQP，股票多时建议只启用必要约束。

这是日频影子成交模型：VWAP/当日总成交额隐含全天参与近似，未还原盘口、集合竞价和订单排队；
open成交额约束也只能作为日频容量近似。卖空假设存在券源，未模拟保证金、逐日可借券和召回。
容量表固定构建参数，只改变资金规模，保留每档全部持仓/成交/成本；汇总成交率为金额加权，
另提供低分位成交率、跟踪差、现金比例和ST残余，准入阈值可配置，不能单凭夏普判断容量。

## tradable_t维护

`v2.UpdateData.stock.tradable_t.calculate_tradable_t` 保留旧tradable的所有当日条件，
完全去除次日条件。`update_basic`、`update_tradable`均自动维护；也有独立`update_tradable_t`入口。

```powershell
python -m v2.UpdateData.stock.tradable_t --root Z:/
python -m unittest discover -s v2/SingleFactorTest/tests -v
```

回填从现有8个basic布尔矩阵生成，完整预留轴为bool，每次252行，未来预留区为False。
原始历史状态未维护时不能凭回填恢复真实历史状态。
