# 项目事件流思维导图

这份图是给学习代码用的。你可以先看“总览”，再沿着“研究侧 -> 组合权重 -> 交易侧 -> 监控反馈”的路径读代码。

## 总览

```mermaid
mindmap
  root((Quant Workflow))
    研究侧
      数据预处理
      单因子检验
      因子组合
      风险模型
      组合优化
    生产侧
      每日任务图
      目标权重发布
      执行计划
      风控监控
    交易侧
      L2四表预处理
      事件流回放
      订单簿重建
      OMS
      撮合器
      账户持仓
    结果侧
      订单表
      成交表
      净值曲线
      持仓表
      统计指标
```

## 从单因子到权重

```mermaid
flowchart LR
    A[PanelData<br/>因子/收益/行业/市值/基准]
    B[Preprocessor<br/>去极值/标准化/中性化]
    C[FactorAnalyzer<br/>IC/RankIC/分组收益/换手]
    D[FactorRegistry<br/>准入/版本/审计]
    E[AlphaModel<br/>线性/岭回归/Fama-MacBeth/机器学习]
    F[RiskModel<br/>协方差/因子暴露/特异风险]
    G[CostModel<br/>交易成本/冲击成本]
    H[PortfolioOptimizer<br/>多头/指增/市场中性]
    I[OptimizationResult<br/>目标权重/交易/主动权重]

    A --> B --> C --> D --> E
    A --> F
    G --> H
    E --> H
    F --> H
    H --> I
```

## 指数增强权重逻辑

指数增强不是简单做多高分股，而是：

```text
最终权重 w = 基准权重 b + 主动权重 a
```

其中：

- `b` 是指数成分股权重；
- `a` 是因子模型给出的主动偏离；
- 看空的因子不一定意味着卖空，而是在股票只能做多时形成“低配/剔除”作用；
- 对公募指增或严格成分股内增强，非成分股默认不能进入组合。

## L2 四表到事件流

```mermaid
flowchart TD
    A1[SSE 委托表]
    A2[SSE 成交表]
    A3[SZSE 委托表]
    A4[SZSE 成交表]
    B[L2ColumnMap<br/>字段映射]
    C[L2TableGateway<br/>读 CSV/Parquet]
    D[CanonicalL2Preprocessor<br/>清洗/排序/压缩]
    E1[SSE_events.parquet]
    E2[SZSE_events.parquet]
    F[CanonicalL2Gateway<br/>按时间顺序输出 MarketEvent]

    A1 --> B
    A2 --> B
    A3 --> B
    A4 --> B
    B --> C --> D
    D --> E1
    D --> E2
    E1 --> F
    E2 --> F
```

## 交易回测主链路

```mermaid
sequenceDiagram
    participant Data as MarketEvent流
    participant Engine as HistoricalReplayEngine
    participant Book as LimitOrderBook
    participant Strategy as TradingStrategy
    participant OMS as SimulationOms
    participant Matcher as QueueAwareMatcher
    participant Account as ChinaEquityAccount

    Data->>Engine: 下一条逐笔事件
    Engine->>OMS: advance_time(event.timestamp)
    OMS->>Matcher: 激活已到达订单
    Engine->>OMS: on_market_event(event)
    OMS->>Matcher: 更新撮合状态
    Matcher->>Book: apply(event)
    Engine->>Strategy: on_market_event(event)
    Strategy->>OMS: send_order(OrderRequest)
    OMS->>OMS: 风控 + 延迟 + 状态记录
    OMS->>Matcher: 到达后 submit(order)
    Matcher->>Account: 成交后 apply_fill()
    OMS->>Strategy: on_order/on_trade
```

## 限价单后续状态

```mermaid
stateDiagram-v2
    [*] --> SUBMITTING: 有延迟
    [*] --> ACTIVE: 无延迟且未立即全成
    SUBMITTING --> ACTIVE: 到达交易所
    ACTIVE --> PARTIALLY_FILLED: 部分成交
    PARTIALLY_FILLED --> FILLED: 剩余成交
    ACTIVE --> FILLED: 全部成交
    SUBMITTING --> CANCELLED: 到达前撤单
    ACTIVE --> CANCELLED: 到达后撤单/收盘撤单
    [*] --> REJECTED: 风控拒单/市价无深度
```

## 文件学习路径

建议按下面顺序阅读：

1. `trading/events.py`：所有事件和订单状态的数据结构；
2. `trading/data.py`：原始四表如何读入并变成标准事件；
3. `trading/l2_preprocess.py`：如何离线预处理为可复用事件流；
4. `trading/book.py`：如何从逐笔委托/成交重建订单簿；
5. `trading/oms.py`：订单生命周期、延迟、风控和成交回报；
6. `trading/matching.py`：限价单 taker/maker 撮合逻辑；
7. `trading/account.py`：现金、费用、T+1 和持仓；
8. `trading/engine.py`：历史回放和模拟盘如何驱动整条链；
9. `trading/integration.py`：组合权重如何接入交易回测；
10. `tests/test_trading_engine.py`：用小样本验证上述逻辑。

## 关键理解

- `LimitOrderBook` 不是最终节点，它只是“当前盘口状态”；
- 真正的订单生命周期在 `SimulationOms` 和 `QueueAwareMatcher`；
- 限价单要么到达时主动成交，要么挂入队列等待后续逐笔成交；
- 回测结果不是只看订单簿，而是看 `ReplayResult.orders`、`ReplayResult.trades`、`ReplayResult.equity_curve` 和 `ReplayResult.positions`。
