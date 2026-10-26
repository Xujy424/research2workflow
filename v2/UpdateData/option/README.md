# IO / MO 日度 IV30

在 IDE 直接运行 `update_iv30.py`，修改底部 `if __name__ == "__main__"` 中的日期和调用参数即可：

```python
value = update_iv30(date, dates, context.wsd, Path(ROOT) / "option",
                    prefix="MO", price_field="close", rate=0.02)
# 主入口循环 IO、MO；只计算中证1000时将循环改为 ("MO",)。
```

仅输出 `ROOT/option/HS300_IV30.bin` 和 `CSI1000_IV30.bin`，不再生成报价、合约、诊断或衍生指标文件。
每个文件均为 float32、形状 `(len(dates), 1)`，行对应 `ROOT/axis/dates.npy`（含预留行），
列数固定为1；单位为年化百分数，缺失和未写入位置为 NaN，重复更新覆盖同一行。
此前版本产生的 CSV/JSON 不再读取或更新，也不会自动删除。

```python
dates = np.load(ROOT / "axis/dates.npy", allow_pickle=False)
iv = np.memmap(ROOT / "option/CSI1000_IV30.bin", dtype=np.float32,
               mode="r", shape=(len(dates), 1))
```

`UpdateData.axis` 已支持 option 固定单列矩阵：日期扩容、历史插行、历史裁剪同步维护，股票列扩容不影响它。
应先维护共享日期轴，再加载 dates 并更新 IV；不要手工替换日期轴或保持旧 memmap 跨越轴维护。
单日接口为 `update_iv30(date, dates, wsd, option_root, prefix="IO", **参数)`，返回一个浮点数。
计算与存储解耦：`calculate_iv30` 获取报价并返回 IV，`maturity_variance` 是纯 NumPy 方差计算。

WSD 候选探测以指数收盘价65%—135%、50点步长生成行权价，按近三个月及随后三季月查找。
优先查询30天两侧最近期限，仅当报价不足时尝试其他候选期限；缺少任一侧时返回 NaN，不外推。
到期时间暂按第三个周五15:00、估值时间15:00，未验证节假日顺延；候选网格不保证完整合约覆盖。
`close`/结算价、固定连续复利利率 `rate` 均属近似，因此结果是类 VIX 代理值。
中间价模式请设置 `price_field=None, bid_field=已验证字段, ask_field=已验证字段`，应用连续两档零买价截断。
WSD 缺失候选应返回空序列或 NaN；读取异常直接抛出，避免把服务故障吞成正常更新。

本地模拟 WSD 测速（5次中位数，4000/6000点，正常两期限）：IO 0.110秒/229次请求，MO 0.151秒/341次请求。
纯单期限方差约145微秒。这些是模拟数据耗时，真实耗时主要取决于逐合约 WSD 请求；
主入口打印每个指数的实际总耗时。默认不并发访问共享请求客户端。
