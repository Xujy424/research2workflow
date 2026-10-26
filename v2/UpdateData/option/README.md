# IO / MO 日更新

在 IDE 中直接运行文件，参数在底部 `main(...)` 中设置，不使用命令行。单日设置 `date`；历史回补改为 `start` 和 `end`。默认采用 close 和固定利率近似，不依赖外部合约或利率 CSV。

未传 contracts 时，以标的指数当日收盘价的65%—135%、50点步长生成候选行权价，覆盖近三个月及随后三个季月，通过 WSD 查询实际可用报价。候选代码不保证完整挂牌覆盖，到期时间暂按第三个周五15:00估计，未核实节假日顺延。诊断明确标记 `wsd_probed_grid_proxy` 和 `unverified_third_friday`，不能视为官方 VIX。探测范围较宽，可能发出较多 WSD 请求。

不修改 stock、GetData 或现有 daily_update。入口与 stock 更新器一样显式传入日期、共享日期轴、数据读取器和输出目录。

```python
import numpy as np
from v2.UpdateData.option import update_iv30, update_history

# ctx 为已有 DataContext；也可传入其他兼容的 dated Series/DataFrame reader。
dates = np.load('D:/data/axis/dates.npy', allow_pickle=False)
current, diagnostics = update_iv30(
    '2026-10-26', dates, ctx.wsd, 'D:/data/option',
    price_field='close', flat_rate=0.02,
)
# update_history(start, end, dates, ctx.wsd, root, **同上参数)
```

可选传入内存合约表或提供器以替换候选探测，列为 `code,prefix,cp,strike,listed,expiry`。
prefix 为 IO/MO，cp 为 C/P，listed 为实际挂牌日，expiry 为实际到期结算时间
（上海本地无时区时间戳，必须包含时分秒）。支持传入 `contracts(day)` 回调，便于每日同步合约清单。
实际合约表模式使用给定的到期结算时间。WSD 候选探测模式不能证明完整历史挂牌覆盖。

利率表列为 `date,tenor_days,rate`：rate 为连续复利年利率小数（2% 写 0.02）。
仅使用当日曲线，期限必须落在节点范围内；不自动前填或外推。
也支持 `rate_curve(day)` 回调。若明确接受固定利率近似，传入 `flat_rate=0.02`，省略 rate_curve。

买卖字段需在实际 WSD 服务确认可获取历史日终同步报价后填写，代码不猜字段名。
缺少真实 bid/ask 时可显式指定 `price_field='close'` 或已确认的结算价字段，省略 bid/ask 字段；
诊断会标记代理口径，不使用零买价截断规则，也不自动从中间价降级。

输出：`daily/YYYY-MM-DD/quotes.csv` 保存当日候选或输入合约及原始报价、失败原因；
同目录 `diagnostics.json` 保存计算口径、各期限 F/K0/利率/方差、选定期限及缺失原因；
`index_iv30.csv` 保存 HS300_IV30、CSI1000_IV30、大小盘差、5交易日变化、756交易日分位数
（至少126个有效值，分位数含当日）。波动率单位为年化百分数，变化和差值为百分点。
历史重跑覆盖当日并重新计算衍生序列；不足30天两侧期限时保存 NaN，不外推。
报价文件与序列分别原子替换，不支持并发写同一目录。

序列采用 CSV，而非 stock 的 bin：现有轴维护将 bin 解释为股票矩阵，期权不应混入。
交易日期仍通过共享 dates 校验，未更新的轴内交易日保留缺失，防止 diff(5) 跨越缺口压缩。
本入口不初始化或修改日期轴，需在日期轴更新后调用。未接入 Monitor；IV-RV 需明确 RV 口径后添加。
当前实现使用每日15:00估值时间，不处理盘中计算。
