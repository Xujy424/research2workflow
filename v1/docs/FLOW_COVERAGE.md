# 流程图功能覆盖审计

## 图纸融合规则

本审计不把八张图当作八套流程。实际关系如下：

1. `研究侧.png` 定义研究主链和发布边界。
2. `生产侧_error.png` 是每日生产总图。
3. `修改.png` 覆盖总图中 Alpha、组合库、组合完成通知和沪深逐笔数据的错误箭头。
4. Airflow 图是生产总图的数据任务和组合任务展开。
5. 多策略图展开组合优化节点。
6. 权重到交易图展开目标权重之后的执行节点。
7. 回测流是交易执行链在历史事件源下的运行方式。
8. 监控闭环是交易结果之后的异步治理反馈，不反向修改当日模型。

状态说明：

- **闭环**：主入口能够实际调用，且有测试。
- **组件**：已有实现，但需要调用方提供数据或显式编排。
- **外部**：属于数据供应商、Airflow、券商或真实交易基础设施。
- **受限**：数学层已实现，但真实交易资源尚未接入。

## 研究侧

| 图示功能 | 状态 | 代码入口 | 说明 |
|---|---|---|---|
| PIT 面板契约与索引校验 | 闭环 | `PanelData.validate` | 校验日期、股票、重复行和字段对齐；供应商修订历史真实性仍属外部数据责任 |
| 去极值、标准化、行业和市值中性化 | 闭环 | `CrossSectionalPreprocessor` | 在单因子分析和聚类之前统一执行 |
| IC、RankIC、分组收益、稳定性 | 闭环 | `FactorResearchEngine`、`legacy_template.FactorAnalyzer` | SingleTest 原功能由覆盖测试守护 |
| 衰减、成本、容量、Regime | 组件 | `FactorRobustnessValidator`、legacy score、`CapacityAnalyzer` | 基础检验齐全；真实成交数据校准需要外部输入 |
| 经济逻辑分类和风险暴露诊断 | 组件 | `factor_families`、legacy analyzer | 分类由研究员提交，行业、板块和 Barra 暴露可诊断 |
| 因子值相关与聚类 | 闭环 | `FactorClusterer` | 聚类输入明确为统一预处理后的原语义因子 |
| 因子收益相关、IC 相关 | 组件 | analyzer 指标和报告数据 | 可从已有时间序列计算，当前标准聚类只使用因子值相关 |
| 聚类代表因子选择 | 闭环 | `FactorClusterer.cluster` | 按质量评分选代表因子 |
| 边际 IC / 增量价值 | 闭环 | `incremental_value`、`ResearchFlowResult.incremental_ic` | 相对其余入选因子组合计算残差 RankIC |
| 正交、残差、PCA、PLS | 闭环 | `FactorTransformer` | 强制位于聚类和代表选择之后 |
| 正则化和机器学习 Alpha | 闭环 | `WalkForwardSklearnAlpha` | Elastic Net、Bayesian Ridge、PLS、RF、GBDT、MLP 等 |
| Walk-forward 样本外预测 | 闭环 | 各 Alpha 模型、`WalkForwardSplitter` | 模型仅使用预测日前的训练样本 |
| 风险、成本和组合优化 | 组件 | `quant_workflow.risk/costs/portfolio` | 研究发布后可复用生产模型验证；不反向塞入因子准入主链 |
| 组合归因和压力测试 | 组件 | `risk_attribution`、`StressTester` | 已实现，需要研究脚本显式提供组合权重和场景 |
| 发布工件、参数和诊断 | 闭环 | `ResearchArtifact`、`FactorResearchWorkflow` | 包含生效日、因子、模型、风险和多策略配置 |

## 每日生产与多策略

| 图示功能 | 状态 | 代码入口 | 说明 |
|---|---|---|---|
| 交易日和数据更新任务 | 外部 | `DailyProductionGraph` | 依赖关系已表达，真实传感器和下载脚本由部署环境提供 |
| 修改图中的箭头方向 | 闭环 | `DailyProductionGraph` | Alpha 向组合库单向传播，三个组合产物汇入完成通知 |
| 工件版本和生效日保护 | 闭环 | `ResearchArtifact.effective_from`、`DailyProductionWorkflow.run` | 防止未来版本进入历史或尚未生效的每日运行 |
| Alpha 预测和统一风险快照 | 闭环 | `DailyProductionWorkflow` | 三策略共享一次 Alpha 和风险模型结果 |
| 因子协方差和特异风险 | 闭环 | `EquityFactorRiskModel` | Newey-West、收缩、PSD 修复和特异风险收缩 |
| 佣金、税费、价差和冲击 | 闭环 | `TransactionCostModel` | 进入优化目标函数 |
| 融券费和期货持有成本 | 组件 | `HoldingCostModel` | 可估计但尚未并入现金股票执行账户 |
| 量化多头 | 闭环 | `PortfolioOptimizer` | 权重和、非负、单票、暴露、换手、ADV 约束 |
| 指数增强 | 闭环 | `PortfolioOptimizer` | 指数底仓、主动权重、TE 和主动暴露约束 |
| 市场中性数学优化 | 闭环 | `PortfolioOptimizer` | 净敞口、毛敞口、单票和风险暴露约束 |
| 市场中性真实执行 | 受限 | `PortfolioTradingBridge` | 负权重会明确拒绝；必须接融券池或期货对冲适配器 |
| 组合库和模型快照落库 | 组件 | DAG 节点、`ResearchArtifact`、结果对象 | 内存契约齐全，机构数据库写入器属于部署适配 |
| 目标权重发布 | 组件 | `OptimizationResult` | 文件、消息队列或数据库发布器需按机构环境接入 |

## 权重、交易与回测

| 图示功能 | 状态 | 代码入口 | 说明 |
|---|---|---|---|
| 权重转目标股数和买卖差额 | 闭环 | `TargetWeightExecutionStrategy`、`PositionPostProcessor` |
| 整手、停牌和可交易处理 | 闭环 | `PositionPostProcessor`、`PreTradeRiskEngine` |
| 盘前和盘中风控 | 闭环 | `PreTradeRiskEngine` | 单笔、单票、总敞口、现金、T+1、流量、撤单和 kill switch |
| OMS 和订单状态机 | 闭环 | `SimulationOms` |
| 沪深四表规范化 | 闭环 | `CanonicalL2Preprocessor`、`CanonicalL2Gateway` |
| 订单簿和队列撮合 | 闭环 | `LimitOrderBook`、`QueueAwareMatcher` |
| 延迟、部分成交和撤单 | 闭环 | `SimulationOms`、`HistoricalReplayEngine` |
| A 股现金账户和费用 | 闭环 | `ChinaEquityAccount` |
| 日志、原子状态和对账 | 闭环 | `TradingJournal`、`AtomicStateStore`、`AccountReconciler` |
| 历史回放和加速模拟盘 | 闭环 | `HistoricalReplayEngine`、`PaperTradingEngine` |
| 真实行情和券商柜台 | 外部 | 需实现 Gateway、Broker 和账户查询适配器 |

## 监控闭环

| 图示功能 | 状态 | 代码入口 | 说明 |
|---|---|---|---|
| 对账失败禁止启动 | 闭环 | `ProductionMonitoringLoop` |
| 因子漂移和降权请求 | 闭环 | `LiveDriftMonitor` |
| 风险预测校准 | 闭环 | `RiskForecastMonitor` |
| 资金容量上限 | 闭环 | `CapacityAnalyzer` |
| 拥挤监控 | 闭环 | `CrowdingMonitor` |
| 每日统一追踪快照 | 闭环 | `DailyProductionTracker` |
| 自动修改因子和参数 | 明确禁止 | 监控只生成治理动作 | 下一版本经研究审批后发布，不热改当日运行 |

## 可用性结论

当前代码适合：

- Point-in-Time 面板上的因子研究、SingleTest 验收和模型发布。
- 多头与指数增强的每日权重生成。
- 市场中性组合的数学研究和约束验证。
- 沪深逐笔历史回放、模拟交易、账户状态和每日监控追踪。

当前代码不能直接宣称可用于真实资金全自动交易，缺少的不是量化公式，而是部署适配：

- 数据供应商 PIT 修订审计、交易日传感器和 Airflow Operator。
- 机构数据库、消息队列、权限、重试、SLA 和告警通道。
- 融券券源、借券费率、召回规则及股指期货保证金和展期。
- 券商柜台、实时行情、订单回报乱序、灾备和人工接管。

因此准确定位是：**研究、组合构建、历史回放和每日模拟追踪框架已经可运行；接真实资金前仍需机构级数据与交易适配和灰度验证。**
