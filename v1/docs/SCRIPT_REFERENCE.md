# 脚本职责索引

## Shared

- `contracts.py`：`PanelData`、研究报告、风险输出和优化结果的数据契约。
- `config.py`：预处理、组合、Alpha、风险、优化器和策略类型配置。
- `preprocessing.py`：截面去极值、标准化、行业和市值中性化。
- `transforms.py`：聚类之后可选的正交、残差、PCA 和 PLS。
- `combination.py`：简单加权和滞后滚动 ICIR 因子组合。
- `alpha.py`：Walk-forward Ridge、Fama-MacBeth、动态线性和机器学习 Alpha。
- `artifacts.py`：研究侧向生产侧发布的不可变 `ResearchArtifact`。

## Researchflow

- `pipeline.py`：标准研究主入口和阶段顺序。
- `research.py`：IC、RankIC、分组收益和相关性等单因子统计。
- `adapters.py`：准入评分及新旧 analyzer 结果适配。
- `clustering.py`：原语义因子相关性聚类和代表因子组合。
- `validation.py`：子区间、状态、衰减和参数平台稳健性检验。
- `registry.py`：因子元数据、状态和生产准入清单。
- `regime.py`：市场状态识别和状态条件组合。
- `sleeves.py`：因子类别子组合和类别间资金分配。
- `stress.py`：组合压力测试。
- `legacy_pipeline.py`：旧研究加生产混合入口，仅供迁移，不用于每日生产。
- `legacy_template/analyzer.py`：原 SingleTest 完整诊断表和图。
- `legacy_template/metrics.py`：原 SingleTest 基础收益、风险、IC 和换手指标。
- `legacy_template/score.py`：验收评分、硬 Gate、Regime 和容量评分。
- `legacy_template/upgrade.py`：根据评分结果生成升级建议。
- `legacy_template/utils.py`：旧数据加载、滚动计算和截面处理工具。
- `legacy_template/web.py`：SingleTest Streamlit 报告工作台。
- `legacy_template/combination.py`：旧组合脚本命名接口的兼容实现。
- `legacy_template/portfolio.py`：旧 Newey-West 协方差入口。

## Workflow

- `production.py`：每日生产主入口和多策略并行优化。
- `risk.py`：因子收益、因子协方差、特异风险和股票协方差。
- `costs.py`：佣金、税费、价差、冲击和换手成本。
- `portfolio.py`：多头、指数增强和市场中性约束优化。
- `execution.py`：目标权重后处理、订单生成、参与率计划和成交模拟。
- `monitoring.py`：对账、因子漂移、风险预测、容量和拥挤监控。
  `DailyProductionTracker` 将上述结果汇总为每日追踪快照和运行决策。
- `orchestration/daily_graph.py`：修正箭头后的每日任务依赖图。
- `trading/data.py`：沪深逐笔数据映射、流式读取和质量检查。
- `trading/l2_preprocess.py`：四表预处理为规范事件流。
- `trading/book.py`：逐笔限价订单簿。
- `trading/matching.py`：队列感知撮合。
- `trading/oms.py`：订单生命周期和成交回报。
- `trading/account.py`：A 股现金、持仓、费用和 T+1 账户。
- `trading/risk.py`：盘前交易风控。
- `trading/strategy.py`：目标权重执行策略。
- `trading/engine.py`：历史回放和模拟盘引擎。
- `trading/persistence.py`：日志、原子状态和账户对账。
- `trading/integration.py`：组合权重到交易引擎的桥接。

共享算法必须从 `quant_shared` 导入，研究能力必须从 `researchflow` 导入；
`quant_workflow` 不再保留同名转发模块。
