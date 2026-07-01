"""Guard the complete legacy analyzer/score/upgrade surface during refactors."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "src" / "researchflow" / "legacy_template"


# 中文说明：`public_functions` 验证该场景的预期行为。
def public_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


# 中文说明：`class_methods` 验证该场景的预期行为。
def class_methods(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not item.name.startswith("_")
            }
    raise AssertionError(f"class not found: {class_name}")


# 中文说明：`LegacyMethodCoverageTest` 验证该场景的预期行为。
class LegacyMethodCoverageTest(unittest.TestCase):
    # 中文说明：`test_factor_analyzer_methods_are_preserved` 验证该场景的预期行为。
    def test_factor_analyzer_methods_are_preserved(self) -> None:
        expected = {
            "prepare_data",
            "reset_cache",
            "reset_axis",
            "calc_ind_exposure",
            "calc_sec_exposure",
            "calc_ind_ret",
            "calc_sec_ret",
            "calc_barra_exposure",
            "calc_barra_ret",
            "table_PRF_stats",
            "table_winrate_scan",
            "table_monthly_ret",
            "table_annual_stats",
            "plot_basic_performance",
            "table_alpha_annual_stats",
            "plot_alpha_distribution",
            "table_ic_annual_stats",
            "plot_ic_contribution",
            "plot_ic_distribution",
            "table_group_stats",
            "plot_group_cumret",
            "table_industry_annual_stats",
            "plot_industry_performance",
            "table_industry_exposure_stats",
            "plot_industry_exposure_ret",
            "plot_industry_component",
            "table_sector_annual_stats",
            "plot_sector_performance",
            "table_sector_exposure_stats",
            "plot_sector_exposure_ret",
            "plot_sector_component",
            "table_barra_exposure_stats",
            "plot_barra_exposure",
            "plot_barra_exposure_ret",
            "plot_corr_redundancy",
            "table_regime_stats",
            "plot_regime_cumret",
            "table_shadow_capacity_test",
            "plot_shadow_capacity_curve",
        }
        self.assertTrue(
            expected.issubset(class_methods(ROOT / "analyzer.py", "FactorAnalyzer"))
        )

    # 中文说明：`test_score_and_upgrade_entry_points_are_preserved` 验证该场景的预期行为。
    def test_score_and_upgrade_entry_points_are_preserved(self) -> None:
        scorers = public_functions(ROOT / "score.py")
        self.assertIn("score_analyzer", scorers)
        self.assertIn("score_redundancy", scorers)
        self.assertIn("score_regime_stats", scorers)
        self.assertIn("score_shadow_capacity", scorers)
        upgrades = public_functions(ROOT / "upgrade.py")
        self.assertIn("diagnose_upgrades", upgrades)
        self.assertIn("suggest_upgrades", upgrades)

    # 中文说明：`test_combination_and_portfolio_named_methods_are_preserved` 验证该场景的预期行为。
    def test_combination_and_portfolio_named_methods_are_preserved(self) -> None:
        methods = class_methods(ROOT / "combination.py", "Orthogonalization")
        self.assertTrue(
            {
                "align_index",
                "ortho_for_t",
                "ortho_newalpha_parallel",
                "run",
            }.issubset(methods)
        )
        self.assertIn("rolling_icir", public_functions(ROOT / "combination.py"))
        self.assertIn("newey_west_cov", public_functions(ROOT / "portfolio.py"))


if __name__ == "__main__":
    unittest.main()
