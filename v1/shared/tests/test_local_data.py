from __future__ import annotations

import numpy as np
import pandas as pd

from quant_shared.local_data import (
    LocalMarketDataStore,
    format_trading_date,
    normalize_exchange,
)


def test_daily_memmap_uses_axis_shape(tmp_path):
    store = LocalMarketDataStore(tmp_path)
    store.save_axis(
        dates=np.array(["20260102", "20260105", "20270104"]),
        ticks=np.array(["000001", "600000"]),
    )

    path = store.ensure_matrix("d_field", "close", fill_value=0.0)
    matrix = store.open_daily("d_field", "close", mode="r+")
    matrix[0, 1] = 12.5
    matrix.flush()

    reopened = store.open_daily("d_field", "close")
    assert path.name == "close.bin"
    assert reopened.shape == (3, 2)
    assert reopened[0, 1] == 12.5


def test_minute_memmap_uses_t_n_241_shape(tmp_path):
    store = LocalMarketDataStore(tmp_path)
    store.save_axis(
        dates=np.array(["20260102", "20260105"]),
        ticks=np.array(["000001", "600000", "000002"]),
    )

    store.ensure_matrix("ignored_for_minute", "volume", frequency="minute", fill_value=0.0)
    values = np.ones((3, 241))
    store.write_minute_slice("volume", "2026-01-05", values)

    matrix = store.open_minute("volume")
    assert matrix.shape == (2, 3, 241)
    np.testing.assert_array_equal(matrix[1], values)


def test_write_daily_frame_from_sql_style_long_data(tmp_path):
    store = LocalMarketDataStore(tmp_path)
    store.save_axis(
        dates=np.array(["20260102", "20260105"]),
        ticks=np.array(["000001", "600000"]),
    )
    store.ensure_matrix("online_factors", "alpha_1", fill_value=np.nan)
    frame = pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-05"],
            "asset": ["000001", "600000"],
            "value": [0.1, 0.2],
        }
    )

    store.write_daily_frame("online_factors", "alpha_1", frame)

    matrix = store.open_daily("online_factors", "alpha_1")
    assert np.isnan(matrix[0, 0])
    assert matrix[1, 0] == 0.1
    assert matrix[1, 1] == 0.2


def test_read_panel_selects_dates_ticks_and_fields(tmp_path):
    store = LocalMarketDataStore(tmp_path)
    store.save_axis(
        dates=np.array(["20260102", "20260105"]),
        ticks=np.array(["000001", "600000"]),
    )
    store.ensure_matrix("d_field", "open", fill_value=1.0)
    store.ensure_matrix("d_field", "close", fill_value=2.0)

    panel = store.read_panel("d_field", ["open", "close"], dates=["20260105"], ticks=["600000"])

    assert panel.index.names == ["date", "asset"]
    assert panel.iloc[0].to_dict() == {"open": 1.0, "close": 2.0}


def test_resolve_and_read_l2_table(tmp_path):
    store = LocalMarketDataStore(tmp_path)
    day_dir = tmp_path / "L2" / "20260105" / "SSE"
    day_dir.mkdir(parents=True)
    expected = pd.DataFrame({"symbol": ["600000"], "price": [10.0]})
    expected.to_csv(day_dir / "orders.csv", index=False)

    paths = store.resolve_l2_tables("2026-01-05", exchanges=("SH",), tables=("orders",))
    loaded = store.read_l2_table("20260105", "SSE", "orders")

    assert paths.require("SSE", "orders").name == "orders.csv"
    pd.testing.assert_frame_equal(loaded, expected)


def test_normalizers():
    assert format_trading_date(pd.Timestamp("2026-01-05")) == "20260105"
    assert format_trading_date(np.datetime64("2026-01-05")) == "20260105"
    assert normalize_exchange("sh") == "SSE"
    assert normalize_exchange("XSHE") == "SZSE"

class FakeSqlReader:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def read_sql(self, sql, params=None):
        self.calls.append((sql, params))
        return self.frames[sql]


def test_research_panel_loader_builds_panel_data(tmp_path):
    from quant_shared.local_data import LocalPanelSpec, LocalResearchPanelLoader

    store = LocalMarketDataStore(tmp_path)
    store.save_axis(
        dates=np.array(["20260102", "20260105"]),
        ticks=np.array(["000001", "600000"]),
    )
    for category, field, value in [
        ("research_factors", "mom", 1.0),
        ("research_factors", "value", 2.0),
        ("label", "forward_return", 0.01),
        ("barra", "beta", 0.5),
        ("d_field", "market_cap", 100.0),
        ("mask", "tradable", 1.0),
    ]:
        store.ensure_matrix(category, field, fill_value=value)

    panel = LocalResearchPanelLoader(store).load(
        LocalPanelSpec(
            factor_fields=("mom", "value"),
            exposure_fields=("beta",),
        )
    )

    assert panel.factors.columns.tolist() == ["mom", "value"]
    assert panel.exposures.columns.tolist() == ["beta"]
    assert panel.forward_returns.name == "forward_return"
    assert panel.metadata["source"] == "local_binary_store"


def test_workflow_updater_runs_sql_then_online_factor_then_l2_check(tmp_path):
    from quant_shared.local_data import (
        DailyMatrixRef,
        LocalWorkflowDataUpdater,
        OnlineFactorSpec,
        SqlDailyUpdateSpec,
    )

    store = LocalMarketDataStore(tmp_path)
    store.save_axis(
        dates=np.array(["20260102", "20260105"]),
        ticks=np.array(["000001", "600000"]),
    )
    sql_reader = FakeSqlReader(
        {
            "select close": pd.DataFrame(
                {
                    "date": ["2026-01-05", "2026-01-05"],
                    "asset": ["000001", "600000"],
                    "value": [10.0, 20.0],
                }
            )
        }
    )
    l2_dir = tmp_path / "L2" / "20260105" / "SSE"
    l2_dir.mkdir(parents=True)
    pd.DataFrame({"symbol": ["600000"]}).to_csv(l2_dir / "orders.csv", index=False)

    def double_close(inputs, axis, row):
        return inputs["close"][row, :] * 2.0

    updater = LocalWorkflowDataUpdater(store, sql_reader)
    result = updater.run_daily_update(
        as_of="2026-01-05",
        sql_updates=(
            SqlDailyUpdateSpec(
                name="close_update",
                category="d_field",
                field="close",
                sql="select close",
            ),
        ),
        online_factors=(
            OnlineFactorSpec(
                name="double_close",
                output_field="double_close",
                inputs={"close": DailyMatrixRef("d_field", "close")},
                compute=double_close,
            ),
        ),
        require_l2=True,
        l2_exchanges=("SSE",),
        l2_tables=("orders",),
    )

    factor = store.open_daily("online_factors", "double_close")
    assert result.sql_updates == ("close_update",)
    assert result.online_factors == ("double_close",)
    assert result.l2_tables.require("SSE", "orders").name == "orders.csv"
    np.testing.assert_array_equal(factor[1, :], np.array([20.0, 40.0]))
    assert sql_reader.calls[0][1]["as_of"] == "20260105"