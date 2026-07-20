import polars as pl
import datetime as dt
import os
from pathlib import Path



def normalize_date(date: dt.date | dt.datetime | str) -> str:
    if isinstance(date, (dt.datetime, dt.date)):
        return date.strftime("%Y%m%d")
    return str(date).replace("-", "").replace(".", "").strip("/")

def generate_path(root, date):
    date = normalize_date(date)
    rawpath = Path(root)/"raw"/date
    outpath = Path(root)/"proc"/date
    outpath.mkdir(parents=True, exist_ok=True)
    return rawpath, outpath


def process_SZ_level2(rawpath, savefile, outpath, securities=None):

    szcj_scan = pl.scan_parquet(rawpath/'szcj.pq')
    if securities is not None:
        szcj_scan = szcj_scan.filter(pl.col('SecurityID').is_in(list(securities)))
    szcj_merged = szcj_scan.collect()
    szcj_merged = szcj_merged.filter(
        pl.col('MDStreamID')==11
    ).drop(
        ['MDStreamID','SecurityIDSource']
    ).rename({
        'LastPx':'Price',
        'LastQty':'OrderQty'
    }).with_columns([
        pl.col('TransactTime').str.to_time(format="%H:%M:%S%.3f"),
        pl.when(pl.col('BidApplSeqNum')>pl.col('OfferApplSeqNum')).then(pl.lit(1)).otherwise(pl.lit(-1)).alias('Side')
    ])
    szcj = szcj_merged.filter(pl.col('ExecType')==70).drop('ExecType')
    szcancel = szcj_merged.filter(pl.col('ExecType')==52).drop('ExecType')

    szwt_scan = pl.scan_parquet(rawpath/'szwt.pq')
    if securities is not None:
        szwt_scan = szwt_scan.filter(pl.col('SecurityID').is_in(list(securities)))
    szwt = szwt_scan.collect()
    szwt = szwt.filter(
        pl.col('MDStreamID')==11
    ).drop(
        ['MDStreamID','SecurityIDSource']
    ).with_columns([
        pl.col('Side').replace(49,1).replace(50,-1).cast(pl.Int8),
        pl.col('TransactTime').str.to_time(format="%H:%M:%S%.3f")
    ])
    
    if savefile:
        szcj.write_parquet(outpath/'szcj.pq',compression="gzip")
        szcancel.write_parquet(outpath/'szcancel.pq',compression="gzip")
        szwt.write_parquet(outpath/'szwt.pq',compression="gzip")
    
    return szwt, szcj, szcancel


def restoreSHorder(wt, cj):
        
    cj = cj.sort(["ChannelNo", "ApplSeqNum", "SecurityID", "TransactTime"])

    # 剔除集合竞价
    cj_df = cj.filter(~((pl.col("TransactTime") < pl.time(9, 30)) | (pl.col("TransactTime") >= pl.time(14, 57))))

    # 1. 从成交表提取买卖双方订单并汇总
    cj_buy = (
        cj_df
        .select([
            "ChannelNo",
            pl.col("BidApplSeqNum").alias("ApplSeqNum"),
            "SecurityID",
            pl.when(pl.col("BidApplSeqNum") > pl.col("OfferApplSeqNum"))
            .then(pl.col("OrderQty"))
            .otherwise(0)
            .alias("OrderQty"),
            "Price",
            "TransactTime",
        ])
        .with_columns(pl.lit(1).alias("Side"))
    )
    cj_sell = (
        cj_df
        .select([
            "ChannelNo",
            pl.col("OfferApplSeqNum").alias("ApplSeqNum"),
            "SecurityID",
            pl.when(pl.col("OfferApplSeqNum") > pl.col("BidApplSeqNum"))
            .then(pl.col("OrderQty"))
            .otherwise(0)
            .alias("OrderQty"),
            "Price",
            "TransactTime",
        ])
        .with_columns(pl.lit(-1).alias("Side"))
    )
    cj_all = pl.concat([cj_buy, cj_sell])
    cj_summary = cj_all.group_by(["ChannelNo", "ApplSeqNum", "SecurityID", "Side"]).agg([
        pl.sum("OrderQty"),
        pl.last("Price"),  # 一笔主动单可能同时吃掉多笔挂单
        pl.last("TransactTime")
    ])

    # 2. 使用反连接和半连接分离三种情况
    # 部分成交：同时存在于委托和成交（inner join）
    partial = wt.join(
        cj_summary.select(["ChannelNo", "ApplSeqNum", "SecurityID", "Side", "OrderQty"]),
        on=["ChannelNo", "ApplSeqNum", "SecurityID", "Side"],
        how="inner"
    ).with_columns([
        (pl.col("OrderQty") + pl.col("OrderQty_right")).alias("OrderQty"),
        pl.lit("主动部分成交").alias("OrderStatus")
    ]).drop("OrderQty_right")

    # 未成交：存在于委托但不存在于成交（anti join）
    untouched = wt.join(
        cj_summary.select(["ChannelNo", "ApplSeqNum", "SecurityID", "Side"]),
        on=["ChannelNo", "ApplSeqNum", "SecurityID", "Side"],
        how="anti"
    ).with_columns(
        pl.lit("挂单被动成交").alias("OrderStatus")
    )
    untouched = untouched.select(partial.columns)

    # 完全成交：存在于成交但不存在于委托（anti join）
    new = cj_summary.join(
        wt.select(["ChannelNo", "ApplSeqNum", "SecurityID", "Side"]),
        on=["ChannelNo", "ApplSeqNum", "SecurityID", "Side"],
        how="anti"
    ).with_columns([
        pl.lit("主动完全成交").alias("OrderStatus"),
    ])
    new = new.select(partial.columns)

    # 3. 合并所有订单
    init_order = pl.concat([partial, untouched, new], how="vertical_relaxed")
    return init_order

def restoreSHorder_v2(wt, cj):
    cj = cj.sort(["ChannelNo", "ApplSeqNum", "SecurityID", "TransactTime"])

    # 剔除集合竞价。集合竞价阶段 A 行 Qty 按说明已经是原始委托量，不需要用成交加回。
    cj_df = cj.filter(
        ~( (pl.col("TransactTime") < pl.time(9, 30)) | (pl.col("TransactTime") >= pl.time(14, 57)) )
    )

    # 1. 只提取主动方成交用于还原原始委托量
    # BidApplSeqNum > OfferApplSeqNum: 买方主动
    # OfferApplSeqNum > BidApplSeqNum: 卖方主动
    active_buy = (
        cj_df
        .filter(pl.col("BidApplSeqNum") > pl.col("OfferApplSeqNum"))
        .select([
            "ChannelNo",
            pl.col("BidApplSeqNum").alias("ApplSeqNum"),
            "SecurityID",
            "OrderQty",
            "Price",
            "TransactTime",
            pl.col("ApplSeqNum").alias("TradeApplSeqNum"),
        ])
        .with_columns(pl.lit(1).alias("Side"))
    )

    active_sell = (
        cj_df
        .filter(pl.col("OfferApplSeqNum") > pl.col("BidApplSeqNum"))
        .select([
            "ChannelNo",
            pl.col("OfferApplSeqNum").alias("ApplSeqNum"),
            "SecurityID",
            "OrderQty",
            "Price",
            "TransactTime",
            pl.col("ApplSeqNum").alias("TradeApplSeqNum"),
        ])
        .with_columns(pl.lit(-1).alias("Side"))
    )

    active_cj = pl.concat([active_buy, active_sell])

    # 2. 对存在 A 行的订单，只加回 A 行之前的主动成交。
    # 连续竞价中若先成交再剩余挂单，交易所顺序是：
    # T 成交行 -> A 剩余委托行
    # 所以只有 TradeApplSeqNum < A行 ApplSeqNum 的成交，才是需要加回的首次撮合部分。
    # 原始委托量 = A行剩余委托量 + A行之前主动成交量
    pre_add_summary = (
        active_cj
        .join(
            wt.select([
                "ChannelNo",
                "ApplSeqNum",
                "SecurityID",
                "Side",
            ]),
            on=["ChannelNo", "ApplSeqNum", "SecurityID", "Side"],
            how="inner",
        )
        .filter(pl.col("TradeApplSeqNum") < pl.col("ApplSeqNum"))
        .group_by(["ChannelNo", "ApplSeqNum", "SecurityID", "Side"])
        .agg([
            pl.sum("OrderQty").alias("PreDealQty"),
            pl.last("Price"),
            pl.min("TransactTime").alias("FirstTransactTime"),
        ])
    )

    partial = (
        wt
        .join(
            pre_add_summary.select([
                "ChannelNo",
                "ApplSeqNum",
                "SecurityID",
                "Side",
                "PreDealQty",
                "FirstTransactTime",
            ]),
            on=["ChannelNo", "ApplSeqNum", "SecurityID", "Side"],
            how="inner",
        )
        .with_columns([
            (pl.col("OrderQty") + pl.col("PreDealQty")).alias("OrderQty"),
            pl.col("FirstTransactTime").alias("TransactTime"),
            pl.lit("主动部分成交").alias("OrderStatus"),
        ])
        .drop(["PreDealQty", "FirstTransactTime"])
    )

    # 3. 普通新增挂单
    # 集合竞价原始委托
    # 只发生过后续被动成交的挂单
    # 没有成交的委托
    untouched = (
        wt
        .join(
            pre_add_summary.select([
                "ChannelNo",
                "ApplSeqNum",
                "SecurityID",
                "Side",
            ]),
            on=["ChannelNo", "ApplSeqNum", "SecurityID", "Side"],
            how="anti",
        )
        .with_columns(pl.lit("普通委托").alias("OrderStatus"))
    )
    untouched = untouched.select(partial.columns)

    # 4. 没有 A 行的主动完全成交订单：用主动成交汇总构造。
    # 后面计算盘口时再用完整成交扣掉，最终 RemainQty 应为 0。
    # 主动完全成交，没有剩余挂单，所以交易所不会发 A 行
    active_summary = (
        active_cj
        .group_by(["ChannelNo", "ApplSeqNum", "SecurityID", "Side"])
        .agg([
            pl.sum("OrderQty"),
            pl.last("Price"),
            pl.min("TransactTime"),
        ])
    )
    new = (
        active_summary
        .join(
            wt.select(["ChannelNo", "ApplSeqNum", "SecurityID", "Side"]),
            on=["ChannelNo", "ApplSeqNum", "SecurityID", "Side"],
            how="anti",
        )
        .with_columns(pl.lit("主动完全成交").alias("OrderStatus"))
    )
    new = new.select(partial.columns)

    init_order = pl.concat([partial, untouched, new], how="vertical_relaxed")
    return init_order


def process_SH_level2(rawpath, savefile, outpath, securities=None):
    sh_scan = pl.scan_parquet(rawpath/'sh.pq')
    if securities is not None:
        sh_scan = sh_scan.filter(pl.col('SecurityID').is_in(list(securities)))
    sh = sh_scan.collect()
    sh = sh.drop(
        ['TradeMoney']
    ).rename({
        'BizIndex':'ApplSeqNum',
        'Channel':'ChannelNo',
        'TickTime':'TransactTime',
        'Qty':'OrderQty'
    }).with_columns([
        pl.col('TransactTime').str.to_time(format="%H:%M:%S%.3f"),
    ])

    # Type=S carries per-security trading-phase changes (START/TRADE/SUSP/...).
    # Keep it separately so snapshot replay can clear/mask a halted book.
    shstatus = sh.filter(pl.col('Type')=='S').select([
        'ChannelNo', 'SecurityID', 'ApplSeqNum', 'TransactTime',
        pl.col('TickBSFlag').alias('TradingPhaseCode'),
        'LocalTime', 'SeqNo',
    ])
    sh = sh.filter(pl.col('Type')!='S')

    shwt_added = sh.filter(pl.col('Type')=='A').drop('Type').with_columns([
        pl.when(pl.col('BuyOrderNO')!=0).then(pl.col('BuyOrderNO')).otherwise(pl.col('SellOrderNO')).alias('OrderNo'),
        pl.when(pl.col('BuyOrderNO')!=0).then(pl.lit(1)).otherwise(pl.lit(-1)).alias('Side'),
    ]).drop([
        'BuyOrderNO', 'SellOrderNO', 'TickBSFlag', 'LocalTime', 'SeqNo'
    ])

    shcj = sh.filter(
        (pl.col('Type')=='T') | (pl.col('Type')=='D')
    ).join(
        shwt_added.select(['ChannelNo','OrderNo','ApplSeqNum','SecurityID']),  # 获取买单的频道内索引
        left_on=['ChannelNo','BuyOrderNO','SecurityID'],
        right_on=['ChannelNo','OrderNo','SecurityID'],
        how='left',
        suffix='_buy'
    ).join(
        shwt_added.select(['ChannelNo','OrderNo','ApplSeqNum','SecurityID']),  # 获取卖单的频道内索引
        left_on=['ChannelNo','SellOrderNO','SecurityID'],
        right_on=['ChannelNo','OrderNo','SecurityID'],
        how='left',
        suffix='_sell'
    ).rename({
        'ApplSeqNum_buy':'BidApplSeqNum',
        'ApplSeqNum_sell':'OfferApplSeqNum'
    }).with_columns([
        pl.when(pl.col('Type')=='D').then(pl.col('BidApplSeqNum').fill_null(0)).otherwise(pl.col('BidApplSeqNum').fill_null(pl.col('ApplSeqNum'))),
        pl.when(pl.col('Type')=='D').then(pl.col('OfferApplSeqNum').fill_null(0)).otherwise(pl.col('OfferApplSeqNum').fill_null(pl.col('ApplSeqNum'))),
    ]).with_columns([                                               # 判断买卖方向
        pl.when(pl.col('BidApplSeqNum')>pl.col('OfferApplSeqNum')).then(pl.lit(1)).otherwise(pl.lit(-1)).alias('Side')
    ]).drop(['BuyOrderNO','SellOrderNO'])
    
    shcancel = shcj.filter(pl.col('Type')=='D').drop('Type')
    shcj = shcj.filter(pl.col('Type')=='T').drop('Type')
    shwt_added = shwt_added.drop('OrderNo')
    shwt = restoreSHorder_v2(shwt_added, shcj)

    if savefile:
        shcancel.write_parquet(outpath/'shcancel.pq',compression="gzip")
        shcj.write_parquet(outpath/'shcj.pq',compression="gzip")
        shwt.write_parquet(outpath/'shwt.pq',compression="gzip")
        shstatus.write_parquet(outpath/'shstatus.pq',compression="gzip")
    
    return shwt, shcj, shcancel


def get_closebook(exchange, root, date):
    shot = pl.read_parquet(Path(root)/"raw"/f"{date}"/f"{exchange}shot.pq")
    cols = ['SecurityID','UpdateTime'] + ','.join([f'BidPrice{i},BidVolume{i},AskPrice{i},AskVolume{i}' for i in range(1, 11)]).split(',')
    shot = shot.select(cols)
    closing_shot = shot.sort(['SecurityID', 'UpdateTime']).group_by('SecurityID').last()
    close_orderbook = pl.concat([
        closing_shot.select(
            pl.col("SecurityID"),
            pl.lit(i).alias("Level"),

            pl.col(f"BidPrice{i}")
            .cast(pl.Float64, strict=False)
            .alias("BidPrice"),

            pl.col(f"BidVolume{i}")
            .cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
            .alias("BidQty"),

            pl.col(f"AskPrice{i}")
            .cast(pl.Float64, strict=False)
            .alias("AskPrice"),

            pl.col(f"AskVolume{i}")
            .cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
            .alias("AskQty"),
        )
        for i in range(1, 11)
    ]).sort(["SecurityID", "Level"])
    return close_orderbook


def generate_closebook(
    shwt: pl.DataFrame,
    shcj: pl.DataFrame,
    shcd: pl.DataFrame,
    topn: int = 10,
):
    # 1. 委托订单表
    orders = (
        shwt
        .select([
            "ChannelNo",
            "SecurityID",
            "ApplSeqNum",
            "Price",
            "OrderQty",
            "Side",
        ])
    )

    cj = pl.concat([shcj, shcd])

    # 2. 买方订单成交扣减
    buy_filled = (
        cj
        .select([
            "ChannelNo",
            "SecurityID",
            pl.col("BidApplSeqNum").alias("ApplSeqNum"),
            pl.col("OrderQty").alias("DealQty"),
        ])
        .with_columns(pl.lit(1).alias('Side'))
    )

    # 3. 卖方订单成交扣减
    sell_filled = (
        cj
        .select([
            "ChannelNo",
            "SecurityID",
            pl.col("OfferApplSeqNum").alias("ApplSeqNum"),
            pl.col("OrderQty").alias("DealQty"),
        ])
        .with_columns(pl.lit(-1).alias('Side'))
    )

    # 4. 每张订单的累计成交数量
    filled = (
        pl.concat([buy_filled, sell_filled])
        .group_by(["ChannelNo", "SecurityID", "Side", "ApplSeqNum"])
        .agg(
            pl.col("DealQty").sum().alias("FilledQty")
        )
    )

    # 8. 计算每张订单剩余数量
    alive_orders = (
        orders
        .join(
            filled,
            on=["ChannelNo", "SecurityID", "Side", "ApplSeqNum"],
            how="left",
        )
        .with_columns([
            (
                pl.col("OrderQty")
                - pl.col("FilledQty").fill_null(0)
            ).alias("RemainQty")
        ])
        .filter(pl.col("RemainQty") > 0)
    )

    # 9. 聚合成价格档位
    price_level = (
        alive_orders
        .group_by(["SecurityID", "Side", "Price"])
        .agg(
            pl.col("RemainQty").sum().alias("Qty")
        )
    )

    # 10. 买盘：价格从高到低
    bid_book = (
        price_level
        .filter(pl.col("Side") == 1)
        .with_columns(
            pl.col("Price")
            .rank(method="dense", descending=True)
            .over("SecurityID")
            .cast(pl.Int32)
            .alias("Level")
        )
        .filter(pl.col("Level") <= topn)
        .sort(["SecurityID", "Level"])
        .rename({
            "Price": "BidPrice",
            "Qty": "BidQty",
        })
        .select(["SecurityID", "Level", "BidPrice", "BidQty"])
    )

    # 11. 卖盘：价格从低到高
    ask_book = (
        price_level
        .filter(pl.col("Side") == -1)
        .with_columns(
            pl.col("Price")
            .rank(method="dense", descending=False)
            .over("SecurityID")
            .cast(pl.Int32)
            .alias("Level")
        )
        .filter(pl.col("Level") <= topn)
        .sort(["SecurityID", "Level"])
        .rename({
            "Price": "AskPrice",
            "Qty": "AskQty",
        })
        .select(["SecurityID", "Level", "AskPrice", "AskQty"])
    )

    # 12. 合并买卖盘
    close_book = (
        bid_book
        .join(
            ask_book,
            on=["SecurityID", "Level"],
            how="full",
            coalesce=True,
        )
        .sort(["SecurityID", "Level"])
    )

    return close_book, alive_orders





if __name__ == '__main__':

    L2DATA_PATH = "D:/data/l2/"
    rawpath, outpath = generate_path(L2DATA_PATH, '20260625')

    shwt, shcj, shcd = process_SH_level2(rawpath, True, outpath)

    sh_closebook, sh_aliveorders = generate_closebook(shwt, shcj, shcd, topn=10)
    print(sh_closebook.filter(pl.col('SecurityID')==600000))

    sh_closebook_correct = get_closebook('sh', L2DATA_PATH, '20260625')
    print(sh_closebook_correct.filter(pl.col('SecurityID')==600000))



