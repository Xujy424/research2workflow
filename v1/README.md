# FactorTest

项目已拆分为三个互不反向依赖的边界：

- `shared/`：研究端和生产端共享的配置、契约与纯算法。
- `researchflow/`：单因子检验、准入、聚类、稳健性验证和研究发布。
- `workflow/`：每日数据编排、风险成本、多策略优化、执行回测和监控闭环。

第一次进入项目，请按以下顺序阅读：

1. [项目使用指南](docs/PROJECT_GUIDE.md)：先看什么、如何研究、如何发布、生产每天怎么跑。
2. [融合架构](docs/ARCHITECTURE.md)：理解研究侧、生产侧、执行侧和监控侧边界。
3. [脚本职责索引](docs/SCRIPT_REFERENCE.md)：按文件查找入口和功能。
4. [流程覆盖审计](docs/FLOW_COVERAGE.md)：查看图中每个功能的闭环、组件和外部边界。
5. [研究侧 README](researchflow/README.md)：开始做因子研究。
6. [生产侧 README](workflow/README.md)：消费研究工件并生成组合和交易。

原 `research_template` 已删除。其 `SingleTest` 完整实现位于
`researchflow/src/researchflow/legacy_template/`，并由方法覆盖测试防止重构时误删。
