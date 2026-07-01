# Quant Workflow

这是一个从单因子研究、多因子组合、风险模型、组合优化，到逐笔回测和模拟盘验证的量化投研框架。

## 主要入口

- `FactorToPortfolioWorkflow.run()`：从因子数据生成单策略目标权重；
- `FactorToPortfolioWorkflow.run_strategies()`：共享同一套 alpha 和风险模型，生成多头、指数增强、市场中性等不同策略权重；
- `PortfolioTradingBridge`：把组合优化结果接入逐笔回测或模拟盘；
- `HistoricalReplayEngine`：按历史逐笔事件快速回放；
- `PaperTradingEngine`：使用相同 OMS/账户/风控栈做加速或近实时模拟；
- `CanonicalL2Preprocessor`：把每日四张逐笔表预处理为两个交易所事件流；
- `ProductionMonitoringLoop`：把对账、漂移、风险和拥挤度转成运行决策。

## 推荐阅读

先看这几份文档：

- [项目事件流思维导图](docs/PROJECT_MIND_MAP.md)：按事件流理解整个项目；
- [逐笔回测与模拟盘架构](docs/TRADING_ARCHITECTURE.md)：重点解释限价单后续撮合、延迟、maker/taker；
- [研究与生产边界](docs/RESEARCH_PRODUCTION_BOUNDARY.md)：说明研究产物如何进入生产侧。

## 安装

```powershell
py -m pip install -e ..\shared
py -m pip install -e .
```

如果要使用 Parquet 预处理逐笔数据：

```powershell
py -m pip install -e ".[l2]"
```

## 测试

```powershell
$env:PYTHONPATH = "..\shared\src;src"
py -m unittest discover -s tests -v
```

## 限价单后续链路

限价单不会停在 `LimitOrderBook`。实际链路是：

```text
策略下单 -> OMS风控 -> 订单延迟 -> 到达交易所 -> 判断 taker/maker
       -> 主动成交或挂单排队 -> 后续逐笔成交触发部分/全部成交
       -> 更新账户/费用/持仓 -> 回调策略 -> 写入回测结果
```

更详细的图在 `docs/TRADING_ARCHITECTURE.md`。
