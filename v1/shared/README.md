# Quant Shared

研究端与生产端共同依赖的最小核心：

- 数据契约与只读配置
- 因子统一预处理
- 正交、残差、PCA、PLS
- Alpha 模型
- 因子组合
- `ResearchArtifact`

本包不包含研究准入流程、组合优化、交易、监控或外部数据适配器。

## 使用原则

- 数据进入任一流程前先通过 `PanelData.validate()`。
- 研究侧和生产侧必须使用同一份不可变配置和 `ResearchArtifact`。
- `FactorTransformer` 中的正交、残差、PCA、PLS 是模型构建工具，不是单因子准入前置步骤。
- 本包不得导入 `researchflow` 或 `quant_workflow`。

脚本职责见 [脚本职责索引](../docs/SCRIPT_REFERENCE.md)。
