# 项目使用指南

## 先看什么

1. 根目录 `README.md`：确认三个包的职责。
2. `docs/ARCHITECTURE.md`：理解研究、生产、执行和监控边界。
3. `researchflow/README.md`：了解因子从候选到发布的顺序。
4. `workflow/docs/RESEARCH_PRODUCTION_BOUNDARY.md`：理解研究工件如何进入每日生产。
5. `workflow/docs/TRADING_ARCHITECTURE.md`：需要逐笔回测或模拟交易时再阅读。

## 环境准备

在项目根目录执行：

```powershell
$env:PYTHONPATH = "shared\src;researchflow\src;workflow\src;workflow"
```

也可以按 `shared -> researchflow -> workflow` 的顺序以 editable 模式安装三个包。

## 研究员的工作顺序

1. 准备历史 Point-in-Time 数据，保证日期、股票、收益、行业、市值和可交易状态对齐。
2. 构造 `PanelData` 并运行 `validate()`，失败时停止研究，不带病进入统计分析。
3. 运行统一预处理：去极值、标准化、行业和市值中性化。
4. 使用 `legacy_template.FactorAnalyzer` 或 `FactorResearchEngine` 完成单因子检验。
5. 使用 `score_analyzer()` 和 upgrade 诊断决定淘汰、观察、影子或可发布状态。
6. 在保留经济含义的因子上做相关性分析和聚类。
7. 选择聚类代表因子，再做增量 IC、衰减、稳定性、成本和容量检验。
8. 只有进入模型构建后，才可选择正交、残差、PCA、PLS、正则化或机器学习。
9. 使用 Walk-forward 样本外预测、组合回测、归因和压力测试验证完整策略。
10. 通过 `FactorResearchWorkflow.run()` 发布不可变的 `ResearchArtifact`。

研究阶段不要直接调用生产优化器，也不要为了改善聚类结果而提前正交化。

## 投资经理的每日生产顺序

1. 判断交易日并完成行情、基本面、指数、L2 和风险数据更新。
2. 加载已批准的 `ResearchArtifact`，校验版本、日期、字段和因子覆盖率。
3. 按工件配置执行预处理、模型变换、因子组合和 Alpha 预期收益计算。
4. 估计因子协方差、特异风险和完整股票协方差矩阵。
5. 建模佣金、税费、价差、冲击、换手和容量约束。
6. 共享同一 Alpha 与风险快照，分别运行多头、指数增强和市场中性优化器。
7. 将目标权重转换为目标持仓、买卖差额和订单。
8. 通过盘前风控、OMS、撮合或券商适配器执行。
9. 更新账户、持仓、交易日志和原子状态，并完成收盘对账。
10. 运行漂移、风险预测、容量和拥挤监控；异常只生成下一版本治理请求，不热改当日模型。

## 常用验证命令

```powershell
python -m compileall -q shared\src researchflow\src workflow\src
python -m unittest discover -s researchflow\tests -v
python -m unittest discover -s workflow\tests -v
python researchflow\examples\basic_research.py
python workflow\examples\daily_production.py
```

## 修改代码时的边界检查

- 改共享契约：同时运行研究和生产测试。
- 改 SingleTest：必须通过 `test_legacy_method_coverage.py`。
- 改研究流程：确认聚类位于任何正交、残差、PCA、PLS 之前。
- 改生产流程：确认不新增因子准入、评分或聚类逻辑。
- 改交易状态机：运行订单簿、撮合、T+1、账户和对账测试。
- 删除脚本：先在 `docs/SCRIPT_REFERENCE.md` 和全仓引用中确认没有入口依赖。
