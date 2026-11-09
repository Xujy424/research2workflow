"""v3 指标使用的指数、行业与 Barra 本地字段映射。"""
from __future__ import annotations

INDEX_MAP = {
    "hs300": "hs300", "zz500": "zz500", "zz1000": "zz1000",
    "zz2000": "zz2000", "zzfull": "zzfull",
    "沪深300": "hs300", "中证500": "zz500", "中证1000": "zz1000",
    "中证2000": "zz2000", "全A": "zzfull",
}

INDUSTRY_MAP = {
    "商贸零售": 0, "轻工制造": 1, "汽车": 2, "美容护理": 3, "房地产": 4,
    "国防军工": 5, "通信": 6, "煤炭": 7, "交通运输": 8, "公用事业": 9,
    "机械设备": 10, "电力设备": 11, "环保": 12, "食品饮料": 13, "计算机": 14,
    "纺织服饰": 15, "家用电器": 16, "医药生物": 17, "钢铁": 18, "社会服务": 19,
    "有色金属": 20, "非银金融": 21, "综合": 22, "建筑装饰": 23, "农林牧渔": 24,
    "银行": 25, "传媒": 26, "基础化工": 27, "建筑材料": 28, "石油石化": 29, "电子": 30,
}

STYLE_MASKS = {
    **{name: f"barra/{name}_mask" for name in (
        "beta", "btop", "size", "nonlinear_size", "momentom", "residual_vol",
        "liquidity", "leverage", "growth1", "earnings_yield",
    )},
    "growth_value": "barra/growth1_mask",
}
