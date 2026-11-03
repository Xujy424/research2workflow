"""聚源查询接口的参数绑定、连接复用和资源释放。"""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import pandas as pd

from v2.Monitor.v3.context import MonitorContext


class JyTests(unittest.TestCase):
    def setUp(self):
        self.connection = MagicMock()
        self.cursor = self.connection.cursor.return_value.__enter__.return_value
        self.cursor.description = [('InnerCode',), ('UnitNV',)]
        self.cursor.fetchall.return_value = [(123, 1.5)]
        self.factory = MagicMock(return_value=self.connection)
        data = SimpleNamespace(asset_root=Path('.'), axis=SimpleNamespace(
            trade_dates=pd.bdate_range('2026-01-01', periods=2), ticks=['600000']))
        self.context = MonitorContext(data=data, jy_factory=self.factory)

    def tearDown(self):
        self.context.close()

    def test_lazy_connection_parameter_binding_and_read_dispatch(self):
        self.factory.assert_not_called()
        sql = 'SELECT InnerCode, UnitNV FROM MF_NetValue WHERE InnerCode=%s'
        values = self.context.jy(sql, (123,))
        self.cursor.execute.assert_called_once_with(sql, (123,))
        pd.testing.assert_frame_equal(values, pd.DataFrame({'InnerCode': [123], 'UnitNV': [1.5]}))
        self.context.read('jy', sql, options=(123,))
        self.factory.assert_called_once()
        self.context.close()
        self.connection.close.assert_called_once()

    def test_empty_result_keeps_columns_and_errors_release_cursor(self):
        self.cursor.fetchall.return_value = []
        result = self.context.jy('SELECT InnerCode, UnitNV FROM MF_NetValue')
        self.assertEqual(result.columns.tolist(), ['InnerCode', 'UnitNV'])
        self.assertTrue(result.empty)
        self.cursor.execute.side_effect = RuntimeError('query failed')
        with self.assertRaisesRegex(RuntimeError, 'query failed'):
            self.context.jy('SELECT missing_column FROM MF_NetValue')
        self.assertEqual(self.connection.cursor.return_value.__exit__.call_count, 2)


if __name__ == '__main__':
    unittest.main()
