"""Research-only factor validation, clustering, and artifact publishing."""

from .adapters import AcceptanceRule, LegacyAnalyzerAdapter, ResearchScorecard
from .clustering import ClusterResult, FactorClusterer, HierarchicalFactorComposite
from .pipeline import (
    FactorResearchWorkflow,
    ResearchFlowConfig,
    ResearchFlowResult,
)
from .regime import (
    MixtureOfExperts,
    ObservableRegimeModel,
    RegimeResult,
    RegimeWeightController,
)
from .registry import (
    FactorMetadata,
    FactorRegistry,
    FactorStatus,
    metadata_from_dict,
    metadata_to_dict,
)
from .research import FactorResearchEngine, ResearchConfig
from .sleeves import FactorSleeveBuilder, SleeveAllocationResult, SleeveAllocator
from .stress import StressResult, StressTester
from .validation import (
    FactorRobustnessValidator,
    RobustnessReport,
    WalkForwardSplitter,
    incremental_value,
)

__all__ = [
    "AcceptanceRule",
    "ClusterResult",
    "FactorClusterer",
    "FactorMetadata",
    "FactorRegistry",
    "FactorResearchWorkflow",
    "FactorResearchEngine",
    "FactorRobustnessValidator",
    "FactorSleeveBuilder",
    "FactorStatus",
    "metadata_from_dict",
    "metadata_to_dict",
    "HierarchicalFactorComposite",
    "LegacyAnalyzerAdapter",
    "MixtureOfExperts",
    "ObservableRegimeModel",
    "RegimeResult",
    "RegimeWeightController",
    "ResearchConfig",
    "ResearchFlowConfig",
    "ResearchFlowResult",
    "ResearchScorecard",
    "RobustnessReport",
    "SleeveAllocationResult",
    "SleeveAllocator",
    "StressResult",
    "StressTester",
    "WalkForwardSplitter",
    "incremental_value",
]
