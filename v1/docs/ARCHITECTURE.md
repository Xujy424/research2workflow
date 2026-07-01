# 量化投研与生产融合架构

本架构融合研究侧、Airflow 生产 DAG、箭头修正图、多策略分支、回测流、权重到交易流
和监控闭环。各图不是并列主流程，而是不同层级的视图。

## 图纸优先级

1. 研究侧图定义研究发布边界。
2. `修改.png` 覆盖生产总图中相冲突的箭头。
3. Airflow 图描述外部调度任务，不决定算法模块归属。
4. 多策略图展开组合优化分支。
5. 权重到交易图展开执行链。
6. 回测图是执行链在历史事件网关下的运行方式，不是另一套撮合系统。
7. 监控图是交易结果产生后的异步反馈环，不反向插入当日主链。

## 三个代码边界

```text
shared/quant_shared
  数据契约、配置、预处理、组合变换、Alpha 模型、研究发布工件

researchflow/researchflow
  单因子 analyzer/score/upgrade、稳健性、聚类、增量检验、状态模型、
  sleeve、因子治理、研究发布

workflow/quant_workflow
  日更 DAG、风险模型、成本模型、多策略优化、权重执行、L2 回测/模拟盘、
  OMS、撮合、账户、状态存储、对账和生产监控
```

依赖方向固定为：

```text
researchflow -> quant_shared
workflow     -> quant_shared
researchflow -X-> workflow
workflow     -X-> researchflow
```

旧的研究与生产混合入口仅保存在 `researchflow.legacy_pipeline`，生产包不提供反向桥接。

## 研究主链

```text
PIT 原始数据
-> 契约与时点校验
-> 统一预处理
-> 完整单因子诊断、评分和改进建议
-> 经济逻辑与风险暴露诊断
-> 原语义因子相关性与聚类
-> 聚类代表选择
-> 增量 IC、稳健性、衰减、成本、容量
-> 可选正交/残差/PCA/PLS/正则化/机器学习
-> Walk-forward
-> 组合回测、归因和压力测试
-> ResearchArtifact
```

正交化和残差化不允许位于聚类之前。残差 IC 只用于增量价值检验。

## 每日生产主链

```text
交易日判断
-> 行情/基本面/指数/L2 数据更新
-> data_ready
-> 风险库与已批准因子并行更新
-> alpha_ready
-> 组合因子更新
-> comb_bank / factor_comb_bank / model_snapshot 并行落库
-> combination_ready
-> LONG_ONLY / INDEX_ENHANCED / MARKET_NEUTRAL 并行优化
-> 目标权重发布
-> 执行或历史回放
-> 账户对账
-> 监控反馈
```

`combination_ready` 只能汇合上游结果，不能反向触发 Alpha 或风险库。

## 执行与回测

实时、模拟和历史回测共享：

```text
OptimizationResult.weights
-> PortfolioTradingBridge
-> TargetWeightExecutionStrategy
-> PreTradeRiskEngine
-> SimulationOms
-> QueueAwareMatcher
-> SimTrade
-> ChinaEquityAccount
-> TradingJournal / AtomicStateStore / AccountReconciler
```

差异只在事件来源和时钟：

- 回测：`CanonicalL2Gateway + HistoricalReplayEngine`
- 模拟盘：实时或加速网关 + `PaperTradingEngine`
- 实盘：机构行情/券商适配器，复用策略、风控和状态契约

## 监控闭环

对账失败直接阻止交易。因子漂移、风险预测偏差、容量和拥挤问题生成明确治理请求：

- 因子动作提交研究治理审批，不由生产进程直接篡改因子注册表。
- 风险校准更新进入下一版本 `RiskConfig`。
- 容量和拥挤更新进入下一版本 `OptimizerConfig`。
- 已开始的当日运行只允许阻断或降级，不允许无审计热改模型。

## 旧方法覆盖

原 `research_template/SingleTest` 的 analyzer、metrics、score、upgrade 和 Web 工作台已完整迁入
`researchflow.legacy_template`，旧 `research_template` 目录已删除。方法覆盖测试通过 AST
清单守护，避免后续重构误删报告、评分、Regime 或容量检验入口。

原 `Combination` 和 `Portfolio` 中的示例算法分别由 `quant_shared.transforms`、
`quant_shared.combination`、`quant_workflow.risk`、`quant_workflow.costs` 和
`quant_workflow.portfolio` 的可测试实现覆盖；示例文件不进入任何生产 DAG。
