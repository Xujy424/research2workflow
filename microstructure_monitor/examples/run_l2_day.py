"""Read v2 normalized L2 proc tables without modifying the source files."""
import argparse
from pathlib import Path
import pandas as pd
import polars as pl
from ..schema import MarketData
from ..pipeline import run_daily_monitor


def load_proc_day(root, date, symbols):
    directory = Path(root) / "proc" / date.replace("-", "")
    order_parts, trade_parts, quote_parts = [], [], []
    for exchange, suffix in (("sh", "SH"), ("sz", "SZ")):
        ids = [int(symbol.split(".")[0]) for symbol in symbols if symbol.endswith(suffix)]
        if not ids:
            continue
        for kind, target in (("wt", order_parts), ("cj", trade_parts), ("cancel", order_parts)):
            if not (directory / f"{exchange}{kind}.pq").exists():
                continue
            frame = pl.scan_parquet(directory / f"{exchange}{kind}.pq").filter(pl.col("SecurityID").is_in(ids))
            columns = ["TransactTime", "SecurityID", "Price", "OrderQty", "Side", "ChannelNo"]
            columns += ["ApplSeqNum"] if kind == "wt" else ["BidApplSeqNum", "OfferApplSeqNum"]
            frame = frame.select(columns).with_columns(pl.col("TransactTime").cast(pl.String)).collect().to_pandas()
            frame = frame.rename(columns={"TransactTime": "time", "SecurityID": "symbol", "Price": "price", "OrderQty": "volume", "Side": "direction" if kind == "cj" else "side"})
            frame["symbol"] = frame.symbol.astype(str).str.zfill(6) + "." + suffix
            channel = frame.ChannelNo.astype(str) + ":"
            if kind == "wt":
                frame["order_id"] = channel + frame.ApplSeqNum.astype(str)
                frame["status"] = "submit"
            elif kind == "cancel":
                seq = frame.BidApplSeqNum.where(frame.side == 1, frame.OfferApplSeqNum)
                frame["order_id"] = channel + seq.astype(str)
                frame["status"] = "cancel"
            else:
                frame["buy_order_id"] = channel + frame.BidApplSeqNum.astype(str)
                frame["sell_order_id"] = channel + frame.OfferApplSeqNum.astype(str)
            target.append(frame)
            print(f"Loaded {exchange}{kind}: {len(frame):,} rows", flush=True)
        shot = directory / f"{exchange}_shot_1m.pq"
        if shot.exists():
            scan = pl.scan_parquet(shot).filter(pl.col("SecurityID").is_in(ids))
            if "Level" in scan.collect_schema():
                scan = scan.filter(pl.col("Level") == 1).rename({"BidPrice": "BidPrice1", "AskPrice": "AskPrice1", "BidQty": "BidQty1", "AskQty": "AskQty1"})
            quote = scan.select(["BarTime", "SecurityID", "BidPrice1", "AskPrice1", "BidQty1", "AskQty1"]).with_columns(pl.col("BarTime").cast(pl.String)).collect().to_pandas()
            quote = quote.rename(columns={"BarTime": "time", "SecurityID": "symbol", "BidPrice1": "bid_price1", "AskPrice1": "ask_price1", "BidQty1": "bid_volume1", "AskQty1": "ask_volume1"})
            quote["symbol"] = quote.symbol.astype(str).str.zfill(6) + "." + suffix
            quote_parts.append(quote)
    if not order_parts or not trade_parts:
        raise ValueError("No matching exchange tables or securities")
    # Missing quotes remain absent; no synthetic spread or depth is introduced.
    quotes = pd.concat(quote_parts, ignore_index=True) if quote_parts else pd.DataFrame(columns=["time", "symbol", "bid_price1", "ask_price1"])
    return MarketData(pd.concat(order_parts, ignore_index=True), pd.concat(trade_parts, ignore_index=True), quotes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Z:/stock/l2")
    parser.add_argument("--date", default="20260724")
    parser.add_argument("--symbols", nargs="+", default=["000001.SZ", "600000.SH"])
    args = parser.parse_args()
    data = load_proc_day(args.root, args.date, args.symbols)
    result = run_daily_monitor(args.date, data)
    columns = ["trade.count", "trade.amount", "trade.median", "trade.p99", "trade.large_ratio", "flow.imbalance", "order.count", "intraday.amount_open_share", "intraday.amount_close_share"]
    print(result.features[columns].to_string())
    print(result.regime)
    print(f"Report: {result.report_path}")
    return result


if __name__ == "__main__":
    main()
