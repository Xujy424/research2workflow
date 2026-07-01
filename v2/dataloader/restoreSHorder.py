import polars as pl
import os



def RestoreOrder(order_dff,deal_dff):

    # 转换数据类型
    deal_dff = deal_dff.with_columns(pl.col("BidApplSeqNum", "OfferApplSeqNum").cast(pl.Int64))
    deal_dff = deal_dff.with_columns(pl.col("TransactTime").str.strptime(pl.Time, "%H:%M:%S%.3f"))
    order_df = order_dff.with_columns(pl.col("TransactTime").str.strptime(pl.Time, "%H:%M:%S%.3f"))

    # 剔除集合竞价
    deal_df = deal_dff.filter(~((pl.col("TransactTime") < pl.time(9, 30, 0)) | (pl.col("TransactTime") >= pl.time(14, 57, 0))))

    # 拆分撤单
    deal = deal_df.filter(~(pl.col("ExecType") == 52)).sort(["ChannelNo", "ApplSeqNum", "SecCode", "TransactTime"])
    cancel = deal_df.filter((pl.col("ExecType") == 52))

    # 判断成交类型
    deal = deal.with_columns(
        pl.when(pl.col('BidApplSeqNum') > pl.col('OfferApplSeqNum'))
        .then(pl.lit('主买'))
        .otherwise(pl.lit('主卖'))
        .alias('DealType')
    )

    # 1. 从成交表提取买卖双方订单并汇总
    deal_buy = deal.filter(pl.col('DealType') == '主买').select([
        "ChannelNo", "BidApplSeqNum", "SecCode", "LastQty", "LastPx", "TransactTime", "FDate"
    ]).rename({"BidApplSeqNum": "ApplSeqNum"}).with_columns(pl.lit("B").alias("Side"))

    deal_sell = deal.filter(pl.col('DealType') == '主卖').select([
        "ChannelNo", "OfferApplSeqNum", "SecCode", "LastQty", "LastPx", "TransactTime", "FDate"
    ]).rename({"OfferApplSeqNum": "ApplSeqNum"}).with_columns(pl.lit("S").alias("Side"))

    deal_all = pl.concat([deal_buy, deal_sell])
    deal_summary = deal_all.group_by(["ChannelNo", "ApplSeqNum", "SecCode", "Side"]).agg([
        pl.sum("LastQty").alias("DealQty"),
        pl.last("LastPx").alias("Price"),  # 一笔主动单可能同时吃掉多笔挂单
        pl.last("TransactTime").alias("TransactTime"),
        pl.last("FDate").alias("FDate")
    ])

    # 2. 使用反连接和半连接分离三种情况
    # 部分成交：同时存在于委托和成交（inner join）
    partial = order_df.join(
        deal_summary.select(["ChannelNo", "ApplSeqNum", "SecCode", "Side", "DealQty"]),
        on=["ChannelNo", "ApplSeqNum", "SecCode", "Side"],
        how="inner"
    ).with_columns([
        (pl.col("OrderQty") + pl.col("DealQty")).alias("OrderQty"),
        pl.lit("主动部分成交").alias("OrderStatus")
    ]).drop("DealQty")

    # 未成交：存在于委托但不存在于成交（anti join）
    untouched = order_df.join(
        deal_summary.select(["ChannelNo", "ApplSeqNum", "SecCode", "Side"]),
        on=["ChannelNo", "ApplSeqNum", "SecCode", "Side"],
        how="anti"
    ).with_columns(pl.lit("挂单被动成交").alias("OrderStatus"))
    untouched = untouched.select(partial.columns)

    # 完全成交：存在于成交但不存在于委托（anti join）
    new = deal_summary.join(
        order_df.select(["ChannelNo", "ApplSeqNum", "SecCode", "Side"]),
        on=["ChannelNo", "ApplSeqNum", "SecCode", "Side"],
        how="anti"
    ).with_columns([
        pl.lit("主动完全成交").alias("OrderStatus"),
        pl.lit(50, dtype=pl.Int64).alias("OrdType"),
        pl.lit(0, dtype=pl.Int64).alias("SeqNo"),
        pl.lit(0, dtype=pl.Int64).alias("__index_level_0__")
    ]).rename({"DealQty": "OrderQty"})
    new = new.select(partial.columns)

    # 3. 合并所有订单
    init_order = pl.concat([partial, untouched, new])

    return init_order




if __name__ == '__main__':

    # home_path = os.path.expanduser('~')
    # wt_data_path = os.path.join(home_path,'data/量化中间数据/l2/逐笔委托数据')
    # cj_data_path = os.path.join(home_path,'data/量化中间数据/l2/逐笔成交数据')
    wt_data_path = f'/data/xujiayi/wt'
    cj_data_path = f'/data/xujiayi/cj'

    for file in os.listdir(wt_data_path):
        dt = file[:8]
        if 'sh' not in file: continue
        print(file)
        order = pl.read_parquet(os.path.join(wt_data_path,file))
        deal = pl.read_parquet(os.path.join(cj_data_path,file.replace('wt','cj')))
        init_order = RestoreOrder(order, deal)
        init_order.write_parquet(os.path.join(wt_data_path,file))
        print(f'{file} as restored!')