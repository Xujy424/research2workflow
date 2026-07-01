"""Minimal research-to-artifact example."""

from __future__ import annotations

from examples.basic_workflow import make_demo_data
from quant_workflow import PreprocessConfig, TransformConfig
from researchflow import FactorResearchWorkflow, ResearchFlowConfig


# 中文说明：`main` 是本示例的函数入口。
def main() -> None:
    panel = make_demo_data(seed=42)
    workflow = FactorResearchWorkflow(
        ResearchFlowConfig(
            preprocess=PreprocessConfig(neutralize=True, min_observations=15),
            # Optional model transform; clustering has already completed.
            transform=TransformConfig(method="orthogonal"),
            run_robustness=True,
        )
    )
    result = workflow.run(panel)
    print(result.selected_factors)
    print(result.artifact.artifact_id)


if __name__ == "__main__":
    main()
