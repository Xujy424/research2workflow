"""作者：xujiayi | 创建日期：2026-11-11
基于DataPool字段/轴契约的连续块读取，避免SMB网络盘逐页memmap读取延迟。
"""
import numpy as np
from v2.GetData import DataPool


class ReportDataPool(DataPool):
    """研究报告数据源；Params: 同DataPool；Return: 支持原read接口的数据源。"""
    def read(self, field, end_date, start_date=None, *, ticks=None, copy=True):
        """连续读取指定日期块，保留DataPool轴/类型检查。

        Params: 同DataPool.read；Return: ndarray，独立内存；copy=False亦不保留网络memmap。
        """
        spec = self.field_spec(field)
        if spec.middle != 1:
            return super().read(field, end_date, start_date, ticks=ticks, copy=copy)
        end = self.axis.date_position(end_date)
        start = end if start_date is None else self.axis.date_position(start_date)
        if start > end:
            raise ValueError("起始日期不能晚于结束日期")
        columns = spec.shape[-1]
        count = (end - start + 1) * columns
        with spec.path.open("rb") as stream:
            stream.seek(start * columns * spec.dtype.itemsize)
            # 以大块read读入，避免逐页访问远程文件映射。
            raw = stream.read(count * spec.dtype.itemsize)
        if len(raw) != count * spec.dtype.itemsize:
            raise ValueError(f"字段读取不完整: {field}")
        values = np.frombuffer(raw, dtype=spec.dtype).reshape(end - start + 1, columns)
        values = values[:, :self.axis.tick_count] if ticks is None else values[:, self.axis.tick_positions(ticks)]
        return values[0] if start_date is None else values
