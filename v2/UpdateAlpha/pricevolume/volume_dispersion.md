# 新增成交异常与分歧因子的统一入口

共 11 个独立因子脚本，没有因子公共计算模块。
每个因子脚本都提供 Config / Context / Factor；统一由 `get_factor_spec(name)`
发现，无需在 backtest 中手动导入或更换类。
仅线性分位数回归残差、有效样本相关系数等通用数学运算放在 `operators.py`；
稀疏分歧识别、事件窗口、贝叶斯收缩和 L2 统计均留在各自因子内部。

| backtest 因子名称 | 方向 | 额外输入 |
| --- | --- | --- |
| vr_quantile_resid_corr | +1 | 分钟收盘价、成交量 |
| vr_q80_ret_corr | -1 | 分钟收盘价、成交量 |
| lr_score_mor | -1 | 分钟收盘价、成交量 |
| lr_beta2_aft_minus_mor | +1 | 分钟收盘价、成交量 |
| sr_dispersion_stdnumtrades | -1 | 分钟收盘价、成交量，以及 L2 成交笔数 |
| sr_dispersion_mean_vwap | -1 | 分钟收盘价、成交量、成交额 |
| sr_dispersion_volume_absret_corr | -1 | 分钟收盘价、成交量 |
| highvol_onset | +1 | 分钟收盘价、成交量 |
| lowvol_onset | +1 | 分钟收盘价、成交量 |
| bayeshighvol_onset | +1 | 分钟收盘价、成交量 |
| bayeslowvol_onset | +1 | 分钟收盘价、成交量 |

名称不区分大小写，如 `bayesHighVol_onset` 也可使用。矩阵文件名统一为小写。
上述输入均由 Context 读取；共用预处理还需要 `basic/tradable`、
`industry/industry`、`d_essentials/total_mv`，回测需要 `d_essentials/pct`。

## 在 IDE 中使用

在 `v2/UpdateAlpha/backtest.py` 底部只替换 `FACTORS` 中的名称，保持
`CALCULATE = True`，设置同一组数据根目录、日期和股票池，即先计算再画分层图。
默认输出到 `v2/UpdateAlpha/output/<小写因子名>/`，同时保存 PNG 和 IC 统计 CSV。

也可以直接调用：

```python
from v2.UpdateAlpha.backtest import run_from_ide

results = run_from_ide(
    "bayesHighVol_onset",  # 或包含多个名称的 tuple/list
    root="Z:/",
    start_date="2023-01-01",
    end_date="2026-06-30",
    calculate=True,
)
```

如使用 `v2/Backtest/singleFactor/example_usage.py`，替换 `FACTOR_NAME`，
`CALCULATE_FACTOR=True` 会先计算并生成分层图，再执行该示例原有的组合回测和容量模拟。
该入口的输出位于 `v2/Backtest/output/<FACTOR_NAME>/`。

## 命令行

```powershell
python -m v2.UpdateAlpha.backtest vr_quantile_resid_corr --start 2023-01-01 --end 2026-06-30
python -m v2.UpdateAlpha.backtest bayesHighVol_onset --plot-only --start 2023-01-01 --end 2026-06-30
```

## 参数和时间对齐

- 11 个因子默认统一为 10 日均值，最少 10 个有效日，`lag_days=1`。
  即 d 日保存的原始因子来自截至 d-1 的窗口；预处理沿用 AlphaBase 的 d 日口径。
- `return_offset=2` 是回测入口的统一默认值，收益从因子日期 d+2 开始。
  它与因子内部 `lag_days` 各自生效，不能把两者当作同一个参数。
- 稀疏度默认 `sparse_minutes=24`；这是实现假设，报告未指定该值。
- 参数可用字典传入，不必导入专用 Config 类：
  `context_kwargs={"config": {"sparse_minutes": 12}}`。
- L2 默认读取 `<root>/stock/l2/proc/YYYYMMDD/{sh,sz}cj.pq`，也可通过
  `context_kwargs={"l2_root": "其他路径"}` 覆盖。成交笔数按真实成交记录数汇总，
  没有 L2 的股票保留缺失值。
- 因子矩阵保留原始符号，回测统一按 `meta.direction` 调整方向。
- 分层图为不同持有期的累计分组超额收益诊断；多日收益窗口会重叠，
  不等同于组合回测的可交易净值。
- 请求区间没有足够有效因子值时明确报错；缺少完整远期收益窗口的子图会标注无数据。

## 验证范围

`python -m unittest discover -s v2/UpdateAlpha/tests -v`

覆盖 11 个新增因子的自动发现、默认配置、模拟数据计算、矩阵写入、分层 PNG/CSV，
以及 L2 笔数时间对齐、稀疏回归全局解、贝叶斯公式和日内未来窗口。
单元测试使用临时模拟数据；另有只读真实数据测速，不代表已完成绩效回测。

## 计算速度与进度

- 放量命中因子只拟合 `tau=0.8`，缩量命中因子只拟合 `tau=0.2`。
  两个分位数的有效输入样本相同，未使用的另一侧残差仅参与有效性判断，
  因此省略它不会改变事件定义；每股 LP 数量从 2 次减为 1 次。
- 命中窗口改用有效分钟数和分歧次数的前缀和，保留缺失分钟位置，
  计算复杂度从 `O(T * horizon)` 降为 `O(T)`。
- CSAD 完整样本的股票共享市场设计矩阵，每个时段只进行一次多右端项
  最小二乘分解；缺失样本股票仍按自身有效分钟拟合。早盘 R² 不再拟合午后数据。
- 稀疏模型仍在各自因子脚本内；通用一元截尾最小二乘求解器放在
  `operators.least_trimmed_squares_slope`。它以区间下界校验全局最优目标，
  默认相对目标误差 `1e-13`，超过搜索工作量上限会明确报错。
- 9 个含分位数回归或稀疏模型的因子支持 `workers`：默认 `0` 自动选择，
  少于 256 只可交易股票时串行，否则最多使用 8 个进程；`1` 强制串行。
  同一 Context 中复用进程池，`close()` 会关闭进程池。
- `show_progress=True` 默认显示读取阶段及逐股处理数量，初始进程启动时不显示
  容易误导的剩余时间。可通过 Config 设置为 `False`。
- 相邻日期复用 9 个已计算日的日特征，第二个日期只需补算一个新交易日；
  首个日期需要准备完整的 10 日窗口。
- 分位数回归保留原 LP 求解口径，单次求解设有 30 秒时限，超时会报错。
- Windows 多进程调用需置于 `if __name__ == "__main__":` 保护下；
  项目的两个 backtest 入口已具备这个保护。

只读测速（不向 `Z:/` 写因子矩阵）：

```powershell
python -m v2.UpdateAlpha.tests.benchmark_volume_dispersion --factors bayesHighVol_onset --dates 2026-06-29 2026-06-30 --mode rolling
```

JSON 结果区分单个原始日特征（`daily`）和完整 10 日滚动因子（`rolling`），
同时记录分钟数据读取耗时。远程盘读取和首次窗口准备包含在耗时内；
预处理、矩阵写盘和绘图不包含在该计算测速内。
