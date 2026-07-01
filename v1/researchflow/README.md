# Research Flow

研究侧只负责回答“这个因子和模型是否值得发布”，不承担每日组合运行。

```text
历史 Point-in-Time 数据
  -> 数据契约校验
  -> 统一预处理（去极值 / 标准化 / 行业与市值中性化）
  -> 单因子 analyzer 诊断
  -> score 准入评分与 upgrade 改进建议
  -> 经济逻辑分类与风险暴露诊断
  -> 原始语义因子相关性分析与聚类
  -> 聚类内代表因子选择 / 类别因子构建
  -> 增量有效性、稳健性、衰减、成本和容量验证
  -> 可选组合变换（正交 / 残差 / PCA / PLS / 正则化 / 机器学习）
  -> Walk-forward 样本外预测、风险模型、组合回测和压力测试
  -> 发布 ResearchArtifact
```

关键约束：正交化、残差化、PCA 和 PLS 不得作为单因子评价或相关性聚类的前置步骤。
聚类使用统一预处理后、仍保留经济语义的因子相关矩阵。每日生产侧只读取
`ResearchArtifact`，不重新进行因子准入或聚类。

## 主要入口

- `FactorResearchWorkflow.run()`：执行标准研究流程并发布 `ResearchArtifact`。
- `FactorResearchEngine.analyze()`：执行标准化的 IC、RankIC、分组收益和相关性分析。
- `legacy_template.FactorAnalyzer`：保留原 SingleTest 的完整诊断表和图。
- `legacy_template.score_analyzer()`：执行单因子评分和硬 Gate 判断。
- `legacy_template.diagnose_upgrades()`：根据诊断结果生成改进建议。

`legacy_template` 只是原 SingleTest 能力的归档名称，不表示这些功能已经废弃。
新研究流程仍可通过适配器读取其诊断结果，但不会在聚类前做正交化或残差化。

## 运行

```powershell
$env:PYTHONPATH = "..\shared\src;src;..\workflow"
python examples\basic_research.py
python -m unittest discover -s tests -v
```

完整操作顺序见 [项目使用指南](../docs/PROJECT_GUIDE.md)。
