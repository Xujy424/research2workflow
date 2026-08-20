# GetData

GetData 是 UpdateData 本地矩阵数据库的只读接口。

## 资产目录

默认读取 stock：

    data = DataPool(root)

其他资产只需要指定 asset：

    fund = DataPool(root, asset="fund")
    future = DataPool(root, asset="future")

目录与代码轴约定：

    root/stock   + axis/stock_ticks.npy
    root/fund    + axis/fund_ticks.npy
    root/future  + axis/future_ticks.npy

所有资产共用 axis/dates.npy。

## 读取字段

字段使用资产目录下的斜杠相对路径：

    with DataPool(root) as data:
        close = data.read(
            "d_essentials/close",
            end_date="2024-06-14",
        )

        profit = data.read(
            "fundamental/income_ttm/NetProfit",
            start_date="2024-01-01",
            end_date="2024-06-14",
            ticks=["600000", "000001"],
        )

        forecast = data.read(
            "zyyx/con_forecast/eps",
            end_date="2024-06-14",
        )

路径会转换成：

    root / asset / relative/path.bin

普通字段为 date × code，一致预期为 date × 4 × code，分钟字段为 date × 241 × code。

完整只读 memmap：

    raw = data.load("fundamental/income_ttm/NetProfit")

二维字段 DataFrame：

    frame = data.get_field("d_essentials/close")

UpdateData 更新或扩容前应 close()，之后重新创建 DataPool 或调用 refresh()。