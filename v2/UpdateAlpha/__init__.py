"""Alpha calculation, discovery, and matrix-maintenance interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from .alphabase import AlphaBase, AlphaContext, AlphaMeta

from .analyst_forecast.afr import (
    AFRConfig,
    AFRContext,
    AFRFactor,
    ExpectedInertiaFactor,
    ExpectedVolatilityFactor,
    PAFRFactor,
    calculate_afr_family,
    update_afr_family,
)
from .analyst_forecast.sue import (
    SUEConfig,
    SUEContext,
    SUE0Factor,
    SUE1Factor,
    SUR0Factor,
    SUR1Factor,
    calculate_sue_family,
    update_sue_family,
)

from .analyst_forecast.discard.score import (
    ScoreConfig,
    ScoreContext,
    ScoreLevelFactor,
    ScoreAdjustment30Factor,
    ScoreAdjustment60Factor,
    ScoreAdjustment90Factor,
    calculate_score_family,
    update_score_family,
)
from .analyst_forecast.discard.tper import TPERFactor

from .analyst_forecast.cov import (
    COVContext,
    COVFactor,
    COVAuthorFactor,
    COVOrganFactor,
    COVAuthorOverlapFactor,
    COVCurrentCoverageFactor,
    COVCoverageGrowthFactor,
    COVRetentionFactor,
    COVNewRatioFactor,
    COVNewIntensityFactor,
    COVExitRatioFactor,
    COVStableExpansionFactor,
    COVCoverageDecayFactor,
    COVExpansionEventFactor,
    COVDecayEventFactor,
)
from .analyst_forecast.disp import (
    DISPContext,
    DISPFreshnessFactor,
    DISPInstitutionFactor,
    DISPMidFactor,
    DISPSeqFactor,
    DISPEqualFactor,
)
from .analyst_forecast.suef_surf import (
    SUEFSURFContext,
    SUEFFactor,
    SURFFactor,
    SUEFReportFactor,
    SURFReportFactor,
    SUEFSimpleFactor,
    SURFSimpleFactor,
)
from .pricevolume.w_cut_reversal import WCutContext, WCutReversalFactor
from .pricevolume.smart_money import SmartMoneyContext, SmartMoneyFactor
from .pricevolume.apm import APMContext, APMFactor
from .pricevolume.qua import QUAContext, QUAFactor, MTSFactor, MTEFactor
from .pricevolume.split_momentum import (
    IntradayOvernightMomentumContext,
    IntradayOvernightMomentumFactor,
)
from .pricevolume.active_trade import (
    ACTContext,
    ACTPositiveFactor,
    ACTNegativeFactor,
)


@dataclass(frozen=True)
class AlphaSpec:
    """Explicit pairing of a factor with the Context that constructs it."""

    factor_class: type[AlphaBase]
    context_class: type[AlphaContext]
    category: str

    @property
    def name(self) -> str:
        return self.factor_class.meta.name

    def create_context(self, *, root=None, **context_kwargs) -> AlphaContext:
        if root is not None:
            context_kwargs["root"] = root
        return self.context_class(**context_kwargs)

    def create_factor(self, context: AlphaContext) -> AlphaBase:
        return self.factor_class(context)


def _specs(context_class, factor_classes, category):
    return (
        AlphaSpec(factor_class, context_class, category)
        for factor_class in factor_classes
    )


FACTOR_REGISTRY = {
    spec.name: spec
    for spec in (
        *_specs(
            AFRContext,
            (
                AFRFactor, PAFRFactor, ExpectedInertiaFactor,
                ExpectedVolatilityFactor,
            ),
            "analyst_forecast",
        ),
        *_specs(
            SUEContext,
            (SUE0Factor, SUE1Factor, SUR0Factor, SUR1Factor),
            "analyst_forecast",
        ),
        *_specs(
            SUEFSURFContext,
            (
                SUEFFactor, SURFFactor, SUEFReportFactor, SURFReportFactor,
                SUEFSimpleFactor, SURFSimpleFactor,
            ),
            "analyst_forecast",
        ),
        *_specs(
            COVContext,
            (
                COVFactor, COVAuthorFactor, COVOrganFactor,
                COVAuthorOverlapFactor, COVCurrentCoverageFactor,
                COVCoverageGrowthFactor, COVRetentionFactor,
                COVNewRatioFactor, COVNewIntensityFactor,
                COVExitRatioFactor, COVStableExpansionFactor,
                COVCoverageDecayFactor, COVExpansionEventFactor,
                COVDecayEventFactor,
            ),
            "analyst_forecast",
        ),
        *_specs(
            DISPContext,
            (
                DISPFreshnessFactor, DISPInstitutionFactor, DISPMidFactor,
                DISPSeqFactor, DISPEqualFactor,
            ),
            "analyst_forecast",
        ),
        *_specs(WCutContext, (WCutReversalFactor,), "pricevolume"),
        *_specs(SmartMoneyContext, (SmartMoneyFactor,), "pricevolume"),
        *_specs(APMContext, (APMFactor,), "pricevolume"),
        *_specs(QUAContext, (QUAFactor, MTSFactor, MTEFactor), "pricevolume"),
        *_specs(
            IntradayOvernightMomentumContext,
            (IntradayOvernightMomentumFactor,),
            "pricevolume",
        ),
        *_specs(
            ACTContext,
            (ACTPositiveFactor, ACTNegativeFactor),
            "pricevolume",
        ),
    )
}


def get_factor_spec(
    name: str,
    *,
    factor_class=None,
    context_class=None,
    category="custom",
) -> AlphaSpec:
    """Return a registered spec, creating and caching one when classes are given."""

    key = name.strip().lower()
    spec = FACTOR_REGISTRY.get(key)
    if spec is not None:
        return spec

    if factor_class is not None and context_class is not None:
        spec = AlphaSpec(factor_class, context_class, category)
        if spec.name != key:
            raise ValueError(
                f"factor name {name!r} does not match meta.name {spec.name!r}"
            )
        FACTOR_REGISTRY[key] = spec
        return spec

    available = ", ".join(sorted(FACTOR_REGISTRY))
    raise KeyError(
        f"unknown factor {name!r}; pass factor_class and context_class "
        f"to register it, or choose from: {available}"
    )


def create_factor(name: str, *, root=None, **context_kwargs):
    """Create a context and factor pair; the caller closes the Context."""
    spec = get_factor_spec(name)
    context = spec.create_context(root=root, **context_kwargs)
    return context, spec.create_factor(context)


__all__ = [
    "AlphaBase",
    "AlphaContext",
    "AlphaMeta",
    "AlphaSpec",
    "FACTOR_REGISTRY",
    "get_factor_spec",
    "create_factor",
    "AFRConfig",
    "AFRContext",
    "AFRFactor",
    "PAFRFactor",
    "ExpectedInertiaFactor",
    "ExpectedVolatilityFactor",
    "calculate_afr_family",
    "update_afr_family",
    "SUEConfig",
    "SUEContext",
    "SUE0Factor",
    "SUE1Factor",
    "SUR0Factor",
    "SUR1Factor",
    "calculate_sue_family",
    "update_sue_family",
    "TPERFactor",
    "COVFactor",
    "COVAuthorFactor",
    "COVOrganFactor",
    "COVAuthorOverlapFactor",
    "COVCurrentCoverageFactor",
    "COVCoverageGrowthFactor",
    "COVRetentionFactor",
    "COVNewRatioFactor",
    "COVNewIntensityFactor",
    "COVExitRatioFactor",
    "COVStableExpansionFactor",
    "COVCoverageDecayFactor",
    "COVExpansionEventFactor",
    "COVDecayEventFactor",
    "ScoreConfig",
    "ScoreContext",
    "ScoreLevelFactor",
    "ScoreAdjustment30Factor",
    "ScoreAdjustment60Factor",
    "ScoreAdjustment90Factor",
    "calculate_score_family",
    "update_score_family",
    "SUEFSURFContext",
    "SUEFFactor",
    "SURFFactor",
    "SUEFReportFactor",
    "SURFReportFactor",
    "SUEFSimpleFactor",
    "SURFSimpleFactor",
    "DISPContext",
    "DISPFreshnessFactor",
    "DISPInstitutionFactor",
    "DISPMidFactor",
    "DISPSeqFactor",
    "DISPEqualFactor",
    "WCutContext",
    "WCutReversalFactor",
    "SmartMoneyContext",
    "SmartMoneyFactor",
    "APMContext",
    "APMFactor",
    "QUAContext",
    "QUAFactor",
    "MTSFactor",
    "MTEFactor",
    "IntradayOvernightMomentumContext",
    "IntradayOvernightMomentumFactor",
    "ACTContext",
    "ACTPositiveFactor",
    "ACTNegativeFactor",
]
