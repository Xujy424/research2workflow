# 每日监控指标

目录分工：

```text
dailyupdate/
  context.py       唯一的 MonitorContext：本地轴/矩阵、EDB、WSD、L2及缓存
  utils.py         共用数学运算，无文件读取和网络请求
  config.py        参数、字段映射、数据代码及指标注册表
  dailyupdate.py   取数编排、统一结果、按日保存、历史循环主入口
  vals/            52个参数化指标脚本，对应73个注册输出
  tests/           计算、存储和轴维护验证
```

旧的 `macro/micro/moneyflow/style/risk.py` 仅保留兼容导入，不再存放公式。
`MarketContext` 以及旧模块导出的 Context 名称均指向同一个 `MonitorContext`，没有分模块子类。

## 参数化指标与注册表

同一个指标的窗口变体共用一个函数，例如：

```python
from v2.Monitor.dailyupdate import INDICATORS

spec = INDICATORS['impact_path_120m']
print(spec.module, spec.function, spec.parameters)
# micro, impact_path:calculate, {'horizon': 120}
values = spec.load()(quotes)  # 自动绑定 horizon=120
```

冲击路径10/30/60/120分钟、冲击保留率1/5/15/30分钟、相关性lag1/lag5、RV5/20/60及RV期限比
均采用参数绑定；不再复制窗口专属函数。PE/PB调和估值分位、宏观水平/利差/同比也分别共用函数。
修改默认窗口或数据字段，在 `config.py` 中调整；输出ID保留，避免已有文件路径变化。
核心指标函数中保留中文公式、单位、窗口及缺失处理说明。

## 单日计算和保存

```python
from v2.UpdateData.config import ROOT
from v2.Monitor.dailyupdate import MonitorContext, MonitorConfig, update_daily

context = MonitorContext(root=ROOT)
try:
    result = update_daily(
        '2026-09-30', ROOT, context=context,
        names=('growth_value_spread_percentile', 'size_spread_percentile', 'rv20'),
        config=MonitorConfig(history=252, periods=(1, 5, 10, 20)),
    )
    print(result)
finally:
    context.close()
```

`names=None` 表示全部已注册指标；`modules=('style', 'risk')` 可按模块筛选。
原面板输出的注册覆盖：macro 11、micro 24、moneyflow 8、style 15、risk 15，共73项。
这里统计指标及窗口变体，不是将每个域、每个周期展开后的文件数量。
`domains` 默认只计算全市场；如需原面板支持的全部域，可从config导入 `ALL_DOMAINS`，
使用 `MonitorConfig(domains=ALL_DOMAINS)`。各指标仍按其原有域限制筛选，不会为不适用域构造数值。
风格模块的指数池由universes控制，行业池由industries控制。
`calculate_daily(...)` 只计算、不保存。结果列统一为：
`name, module, domain, period, value, percentile`。
仅提供分位的指标，其 `value` 为NaN；这不代表额外计算了未定义的原始值。
缺少原始文件或读取失败直接报错；原始观测不足时按原公式返回NaN。
同日先计算完所选模块再写文件，单个文件flush后结束；跨文件没有事务，不支持并发写同一输出。

## 本地存储

路径：

```text
ROOT/median_variables/monitor/<module>/<name>/<domain>/p<period>/<field>.bin
```

每个文件都是 **float32、形状 `(len(axis/dates.npy), 1)`**，含预留行，默认填NaN。
`domain` 使用URL编码生成安全目录，建议通过辅助函数取路径和读数据。
`field` 为 `value` 或 `percentile`，仅保存指标实际定义的字段；重复更新覆盖同一日。
分位为0—1，收益率为小数；宏观数值单位沿用原面板（利率水平为%、利差与利率变化为bp）。
宏观 `p0/p5/p20` 分别是当前值/5日变化/20日变化；其他模块 `p1/p5/p10/p20` 是原定义的周期。
RV和固定期限IV当前观测保存到 `p1`，其计算窗口由注册参数决定，不能把 `p1` 理解为RV窗口。

```python
from v2.Monitor.dailyupdate import read_indicator, metric_path

series = read_indicator(ROOT, 'growth_value_spread_percentile', period=20)
path = metric_path(ROOT, 'rv20', '全市场', 1, 'value')
```

轴维护已识别 `median_variables/monitor` 下的固定单列文件：随日期扩容、插行、裁剪，
不随股票列扩容。请先维护共享日期轴，更新器本身不创建或修改日期轴。
维护轴时不要同时保留外部打开的memmap。

## 历史循环

直接在IDE运行 `dailyupdate.py`，底部主入口已给出历史循环；修改开始/结束日期和 `names` 即可。
整个循环复用一个Context，按日期轴交易日依次计算。
默认示例只计算本地成长价值指标，避免首次运行就触发所有远端和L2依赖。
模块选项通过 `MonitorConfig.module_options` 传入对应面板编排器；域、周期、指数池等集中在Config。

## 数据依赖和口径边界

- style：本地收益、估值、权重与风格mask；仅点位和股债收益差需要外部序列。
- risk：本地收益及交易状态；仅IV/方差溢价请求原有WSD曲面。流动性压力需通过
  `module_options={'risk': {'micro_result': ...}}` 注入三个压力分位分量，否则显式保存NaN。
- moneyflow：本地资金流矩阵；仅融资余额指标请求EDB。
- macro：按注册指标依赖请求EDB/WSD，同一Context对EDB全序列只请求一次，返回前截断到估值日。
- micro：读取L2分钟快照/委托成交文件；报价缓存最多保留两天，完整历史回补仍可能耗时较长。

本次是结构重构，不改变原公式：低growth组仍是原来的“价值代理”，没有额外引入btop；
资金流大小单占比、信用脉冲GDP插值等也保留旧口径。
宏观数据按观测日期截断不能消除发布时间差异和历史修订，尚不等于严格的PIT回测数据。
风险IV接口保留原字段与参数，需由实际数据服务支持，不会静默替换为其他价格口径。
原代码中的 `liquidity_coverage`（需要委托需求额）、`index_eps_bond_spread` 等独立工具函数
没有在原snapshot中形成独立输出，本次仍作为utils兼容函数保留，不计入上述73项每日指标。

## 验证

```text
python -m unittest v2.Monitor.dailyupdate.tests.test_workflow
```

另已对重构前后的五类面板使用同一模拟输入逐表验证（含缺失值），计算一致；
实际本地成长价值、大小盘、RV20的两个交易日更新与读取也已验证。
覆盖测试还逐项调用全部73项指标，检查全量结果和二进制输出；模拟测试允许依赖缺失时输出NaN，
不代表外部EDB/WSD及L2源已全部通过真实数据验证。
