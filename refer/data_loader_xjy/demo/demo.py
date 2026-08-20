import sys
from pathlib import Path

sys.path.append("../data_loader")

from data_loader.data_pool import DataPool

if __name__ == "__main__":
    data_path = Path("/data/share/data/mmap")
    # data_name = {"stock": {"d_essentials": ["close", "close_adj"]}}
    data_pool = DataPool(data_path)

    # get close value of di = 1000
    # return a 1d array of (len(stock_tick), )
    print(data_pool.read_data("stock", "d_essentials", "close", 1000))

    # new way to get data, easier to understand
    print(data_pool["stock"]["d_essentials"]["close"][1000])

    # get close value of di = 900 to 1000
    # return a 2d array of (100, len(stock_tick), )
    print(data_pool.read_data("stock", "d_essentials", "close", 1000, 900))
    # new way to get data, easier to understand
    print(data_pool["stock"]["d_essentials"]["close"][900:1000])

    # get the whole data with a form of dataframe
    print(data_pool.get_field("stock", "d_essentials", "close"))

    ## Fundamental Data
    # suggest to use this way to get fundamental data
    print(data_pool["stock"]["f_balancesheet_merge"]["total_assets"][1000])

    print(data_pool["stock"]["d_industry"]["level_1"][1000])
