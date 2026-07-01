from __future__ import annotations

import unittest

import numpy as np

from quant_shared.local_data import LocalMarketDataStore
from researchflow.registry import FactorRegistry, FactorStatus


class FactorRegistryTest(unittest.TestCase):
    def test_scan_save_load_and_build_panel_spec(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            store = LocalMarketDataStore(tmp)
            store.save_axis(
                dates=np.array(["20260102", "20260105"]),
                ticks=np.array(["000001", "600000"]),
            )
            store.ensure_matrix("research_factors", "momentum", fill_value=1.0)
            store.ensure_matrix("research_factors", "quality", fill_value=2.0)

            registry = FactorRegistry.for_store(store)
            discovered = registry.scan_local_factors(
                store,
                categories=("research_factors",),
                default_owner="research",
                default_family="technical",
            )

            self.assertEqual([item.factor_id for item in discovered], ["momentum", "quality"])
            self.assertEqual(discovered[0].shape, (2, 2))
            self.assertEqual(discovered[0].status, FactorStatus.RESEARCH)

            registry.update_status(
                "momentum",
                "v1",
                FactorStatus.PRODUCTION,
                validation={"ic_mean": 0.04, "approved_by": "committee"},
                update_enabled=True,
            )
            path = registry.save()
            loaded = FactorRegistry(path)

            production = loaded.production_factors()
            self.assertEqual([item.factor_id for item in production], ["momentum"])
            self.assertEqual(production[0].validation["approved_by"], "committee")
            self.assertEqual(production[0].storage_category, "research_factors")
            self.assertTrue(production[0].update_enabled)

            workflow_updates = loaded.workflow_update_factors()
            self.assertEqual([item.field_name for item in workflow_updates], ["momentum"])

            spec = loaded.to_panel_spec(
                exposure_fields=("beta",),
                statuses=(FactorStatus.PRODUCTION,),
            )
            self.assertEqual(spec.factor_category, "research_factors")
            self.assertEqual(spec.factor_fields, ("momentum",))
            self.assertEqual(spec.exposure_fields, ("beta",))


if __name__ == "__main__":
    unittest.main()