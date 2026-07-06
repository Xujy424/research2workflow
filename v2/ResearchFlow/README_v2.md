# ResearchFlow v2 结构说明

v2 是按“单因子检验 -> 因子治理 -> 多因子组合 -> 股票权重”的顺序重写的版本，不依赖、不 import v1。v1 中有价值的方法会迁移为 v2 自己的矩阵实现，但按职责拆分，避免把正交、聚类、组合、风险、交易成本和持仓权重都堆在同一个脚本里。

## 1. 总体边界

`FactorTest`：单因子人工检验。负责网页报告、缺失/极值检查、去极值、横截面标准化、行业市值中性化等。这里是研究员人工判断入口，不自动进入组合。

`FactorRegistry`：因子生命周期治理。负责因子版本、family、owner、状态、路径和上线/下线记录。状态包括 `research`、`candidate`、`shadow`、`production`、`retired`。

`FactorComb`：多因子组合层。只负责从已入库因子生成 family composite、统一 alpha 或 sleeve 信号，不负责最终股票约束和持仓权重。

`Portfolio`：股票权重层。负责把 alpha/sleeve 输出转成股票目标权重，并处理风险、成本、换手、ADV、行业约束等。

`workflow.py`：顶层编排。串联 `FactorRegistry -> FactorComb -> Portfolio`，自己不放具体模型算法。

## 2. 顶层文件

| 文件 | 作用 |
| --- | --- |
| `matrix_store.py` | 读取/写入 `D:/data` 下的 axis + bin 矩阵数据 |
| `matrix_math.py` | nan-safe 数学工具、winsorize、zscore、中性化、rankIC、权重 cap |
| `config.py` | v2 主流程参数 |
| `workflow.py` | 全流程编排入口 |
| `README_v2.md` | 当前说明文档 |

## 3. FactorComb：多因子组合层

FactorComb 只处理“多个因子如何变成组合信号”。运行顺序如下：

| 顺序 | 文件 | 职责 |
| --- | --- | --- |
| 0 | `combination.py` | 通用组合加权方法，如等权、滚动 ICIR |
| 0 | `orthogonal.py` | 单因子对因子池残差化、整池顺序正交、WLS 残差化 |
| 0 | `clustering.py` | greedy 去重、hierarchical 聚类、代表因子选择 |
| 0 | `transforms.py` | PCA、PLS 等降维/信号变换 |
| 1 | `preprocess.py` | 对因子池统一 winsorize、zscore、行业市值中性化 |
| 2 | `family.py` | 按 family 分组，类内去重，选代表因子，生成类内复合因子 |
| 3A | `sleeve.py` | 分支 A：每个因子类生成 sleeve，再做 sleeve 资金配置 |
| 3B | `alpha.py` | 分支 B：各因子类复合为统一 alpha |

FactorComb 不再包含 `portfolio.py`、`risk.py`、`cost.py`、`regime.py`，这些属于股票权重层。

## 4. Portfolio：股票权重层

Portfolio 只处理“组合信号如何变成股票目标权重”。v1 中属于股票权重层的完整风险、成本、优化器能力已经迁移到这里，和 `FactorComb` 解耦。

| 文件 | 职责 |
| --- | --- |
| `portfolio.py` | `StockWeightProjector` 快速投影；`CvxPortfolioOptimizer` 支持 long-only、指数增强、市场中性 convex 优化 |
| `risk.py` | `MatrixRiskModel` 轻量协方差；`MatrixFactorRiskModel` Barra 风格因子收益、因子协方差、特异风险和股票协方差；`risk_attribution` 风险归因 |
| `cost.py` | 线性交易成本、冲击成本估计；`HoldingCostModel` 支持 borrow/carry 持仓成本 |
| `regime.py` | 根据外部状态概率对权重做 bounded tilt |
| `stress.py` | 组合压力测试和场景冲击 |

说明：`CvxPortfolioOptimizer` 对 `cvxpy` 使用 lazy import。没有安装 `cvxpy` 时，不影响普通 `StockWeightProjector` 和主流程；只有实际调用 convex 优化器时才需要安装。

## 5. 从单因子到最终持仓的运行顺序

1. `FactorTest` 做单因子检验和网页报告。
2. 研究员人工判断因子是否进入候选、模拟盘观察、上线或下线。
3. `FactorRegistry` 登记因子版本、family、owner、路径和状态。
4. 顶层 `ResearchToPortfolioWorkflow` 读取 registry 中的 `shadow` 和 `production` 因子。
5. `FactorComb/preprocess.py` 对因子池做统一预处理。
6. `FactorComb/family.py` 在每个 family 内做去重、代表因子选择和类内复合。
7. 选择组合分支：
   - `PortfolioRoute.SLEEVE`：走 `FactorComb/sleeve.py`，得到 sleeve 合成信号。
   - `PortfolioRoute.UNIFIED_ALPHA`：走 `FactorComb/alpha.py`，得到统一 alpha。
8. `Portfolio/portfolio.py` 把信号转成股票目标权重。
9. 默认输出：
   - `D:/data/factorpool/composite_alpha.bin`
   - `D:/data/position/target_weight.bin`

## 6. v1 功能迁移状态

已经迁移到主流程的能力：

- 单因子检验网页和指标分析：`FactorTest`
- 去极值、标准化、中性化：`FactorTest/preprocessing.py`、`FactorComb/preprocess.py`
- 因子版本和生命周期：`FactorRegistry/registry.py`
- 类内相关性去重、IC 相关去重、代表因子选择：`FactorComb/family.py`
- 类内复合、类间等权/ICIR：`FactorComb/combination.py`、`FactorComb/alpha.py`
- 正交/残差化：`FactorComb/orthogonal.py`
- 聚类去重/代表选择：`FactorComb/clustering.py`
- PCA/PLS：`FactorComb/transforms.py`
- sleeve 构建和 sleeve 资金配置：`FactorComb/sleeve.py`
- 股票权重生成：`Portfolio/portfolio.py`
- 交易成本估计：`Portfolio/cost.py`
- 风险估计：`Portfolio/risk.py`
- 状态 tilt：`Portfolio/regime.py`

目前已显式包含但仍可继续扩展的能力：

- `FactorComb/orthogonal.py` 已包含残差化和顺序正交，但还没有做批量并行/缓存。
- `FactorComb/clustering.py` 已包含 greedy 和 hierarchical 两种去重/聚类方法。
- `FactorComb/alpha.py` 默认提供透明的 equal/ICIR 类间 alpha；walk-forward ridge、Fama-MacBeth 和 sklearn 类模型放在可选方法模块里，主流程通过 method 配置切换。
- `Portfolio/portfolio.py` 同时保留快速权重投影和从 v1 迁移来的 cvxpy convex 优化器；未安装 `cvxpy` 时只有调用优化器会报依赖缺失。
- `Portfolio/risk.py` 同时保留轻量协方差 shrinkage 和从 v1 迁移来的 Barra 风格因子收益率风险模型。
- `Portfolio/regime.py` 使用外部传入状态概率，不负责估计 HMM/状态模型。
- `Portfolio/cost.py` 是成本估计，不是交易撮合或执行模拟。

## 7. 最小运行示例

```python
import sys
sys.path.insert(0, "v2")

from ResearchFlow import ResearchToPortfolioWorkflow
from ResearchFlow.config import PortfolioRoute, ResearchFlowV2Config

cfg = ResearchFlowV2Config(
    data_root="D:/data",
    registry_path="D:/data/factorpool/registry.json",
    route=PortfolioRoute.UNIFIED_ALPHA,
)

result = ResearchToPortfolioWorkflow(cfg).run(save=True)
print(result.stock_weights.shape)
```

切换到 sleeve 路径：

```python
cfg = ResearchFlowV2Config(route=PortfolioRoute.SLEEVE)
```

## 8. 数据约定

所有生产侧矩阵使用 `T x N` 的二进制 `.bin` 文件，date/tick 单独存放：

| 类型 | 路径 |
| --- | --- |
| 日期轴 | `D:/data/axis/date.npy` 或 `dates.npy` |
| 股票轴 | `D:/data/axis/tick.npy` 或 `ticks.npy` |
| 因子池 | `D:/data/factorpool/{field}.bin` |
| 标签 | `D:/data/label/Y.1D.bin` |
| 可交易 mask | `D:/data/mask/tradable.bin` |
| 行业 | `D:/data/mask/industry.bin` |
| 市值 | `D:/data/d_field/mv.bin` |
| 输出权重 | `D:/data/position/target_weight.bin` |

## 9. 后续开发规则

1. `workflow.py` 只负责编排，不写模型细节。
2. 多因子组合相关方法放在 `FactorComb`。
3. 股票权重、风险、成本、交易约束相关方法放在 `Portfolio`。
4. 新增因子生命周期字段放在 `FactorRegistry`。
5. 单因子检验和网页展示放在 `FactorTest`。
6. 不要在一个脚本里同时实现 FactorComb 和 Portfolio 两层逻辑。
## 10. 方法对比配置

研究员做方法对比时，不需要改流程代码，只改 `ResearchFlowV2Config` 里的子配置。

```python
from ResearchFlow.config import (
    AlphaConfig,
    FamilyConfig,
    PortfolioRoute,
    PreprocessConfig,
    ResearchFlowV2Config,
    SleeveConfig,
)

cfg = ResearchFlowV2Config(
    route=PortfolioRoute.UNIFIED_ALPHA,
    preprocess=PreprocessConfig(winsor_method="mad", standardize=True, neutralize=True),
    family=FamilyConfig(clustering_method="hierarchical", composite_method="icir"),
    alpha=AlphaConfig(method="ridge", lookback=252, min_periods=60),
)
```

当前可选方法：

| 步骤 | 配置字段 | 可选方法 |
| --- | --- | --- |
| 因子池预处理 | `preprocess.winsor_method` | `mad`, `sigma`, `quantile` |
| 类内去重 | `family.clustering_method` | `greedy`, `hierarchical` |
| 类内组合 | `family.composite_method` | `equal`, `icir` |
| 统一 Alpha | `alpha.method` | `equal`, `icir`, `correlation_adjusted`, `ridge`, `fama_macbeth`, `score_slope`, `dynamic_linear`, `elastic_net`, `lasso`, `bayesian_ridge`, `pls`, `random_forest`, `gbdt`, `hist_gbdt`, `rank_gbdt`, `mlp` |
| Sleeve 配置 | `sleeve.allocation_method` | `equal`, `icir` |
| 股票权重 | `optimizer.*` | 单票上限、换手、ADV、行业约束等参数化控制 |

说明：sklearn 系列 Alpha 方法已经迁移为 lazy import。只有实际选择 `elastic_net/random_forest/gbdt/mlp` 等方法时才需要当前 Python 环境安装 `scikit-learn`。


## 11. 单因子报告网页

`FactorTest` 里新增了一个可执行启动脚本，用来自动拉起现有 Streamlit 因子报告页面：

```powershell
python v2/ResearchFlow/FactorTest/run_report.py --port 8501
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--port` | Streamlit 端口，默认 `8501` |
| `--host` | 监听地址，默认 `localhost` |
| `--no-browser` | 只启动服务，不自动打开浏览器 |

启动后页面仍使用 `FactorTest/web.py` 的分析逻辑，支持上传因子文件或输入文件路径，并在页面里导出 PDF 报告。

示例启动脚本：

```powershell
python v2/ResearchFlow/examples/run_factor_report_demo.py
```


## 12. FactorRegistry 生命周期与监控

`FactorRegistry` 现在分成三层职责：

1. `registry.py`：保存因子版本元数据、人工状态迁移、监控快照和决策日志。
2. `monitoring.py`：基于矩阵 `T x N` 计算监控指标，并给出生命周期建议。
3. 后续 `workflow.py`：可以读取建议，但是否上线、暂停或下线仍通过 Registry 的人工审批接口完成。

生命周期状态保持私募投研常用的五档：

| 状态 | 含义 |
| --- | --- |
| `research` | 研究中，可能还没有稳定生产更新 |
| `candidate` | 候选池，单因子报告已经初步通过，等待持续观察 |
| `shadow` | 模拟盘/影子观察，参与监控但不直接进入生产仓位 |
| `production` | 上线因子，可被主流程读取参与组合 |
| `retired` | 下线归档，不再被主流程使用 |

监控模块会计算：覆盖率、缺失率、极值率、IC、RankIC、多空收益、分组单调性、滚动 ICIR、滚动胜率、滚动多空 Sharpe 和回撤。示例：

```python
from ResearchFlow.FactorRegistry import FactorMonitor, FactorRegistry

registry = FactorRegistry("D:/data/factorpool/registry.json")
monitor = FactorMonitor()
rolling, summary, decision = monitor.evaluate(
    factor_id="my_factor",
    version="v1",
    current_status="candidate",
    factor_values=factor_matrix,
    forward_returns=label_matrix,
    dates=dates,
    mask=tradable_mask,
)

registry.record_monitoring("my_factor", "v1", summary, alert_level="info", message=decision.reason)
registry.append_decision(
    "my_factor",
    "v1",
    decision.current_status,
    decision.suggested_status,
    decision.action,
    decision.reason,
    metrics_snapshot=decision.metrics_snapshot,
    operator="system",
)
registry.save()
```

注意：`FactorMonitor.decide()` 只产生建议，不直接改状态。真正状态迁移仍用：

```python
registry.promote("my_factor", "v1", "shadow", approved_by="researcher", reason="人工复核通过")
registry.retire("my_factor", "v1", notes="持续衰减，停止观察", decision_by="researcher")
```

## 13. 矩阵增量更新

`matrix_store.py` 里保留四类核心 I/O：

| 方法 | 用途 |
| --- | --- |
| `open_matrix(category, field)` | 打开完整 memmap 矩阵 |
| `ensure_matrix(category, field)` | 初始化或校验 `.bin` 文件 |
| `write_matrix(category, field, values)` | 全量覆盖写完整 `T x N` 矩阵 |
| `read_slice/update_slice` | 按日期和股票标签读写行、列、块或配对点 |

`dates=None` 表示全部日期，`ticks=None` 表示全部股票。只传 `dates` 是行，只传 `ticks` 是列，两个都传是块；`paired=True` 表示 `(date_i, tick_i)` 一一配对的稀疏点。

完整可运行示例见：`v2/ResearchFlow/examples/matrix_store_demo.py`。

运行：

```powershell
$env:PYTHONPATH="v2"; python v2/ResearchFlow/examples/matrix_store_demo.py
```

```python
from ResearchFlow.matrix_store import MatrixStore

store = MatrixStore("D:/data")

store.update_slice("factorpool", "my_factor", today_values, dates="2026-07-06")
store.update_slice("factorpool", "my_factor", stock_history, ticks="000001.SZ")
store.update_slice("factorpool", "my_factor", block_values, dates=[...], ticks=[...])
store.update_slice("factorpool", "my_factor", cell_values, dates=[...], ticks=[...], paired=True)
```

`write_matrix()` 用于历史回灌或全量重算；日常生产更新统一走 `update_slice()`。如果新股导致 tick axis 变化，应先做全库 axis reindex/扩列，再对新列写数据。
