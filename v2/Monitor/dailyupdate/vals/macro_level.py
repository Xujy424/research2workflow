"""宏观水平指标：序列名由注册表指定。"""
def calculate(raw, series):
    # 返回已截断并对齐的原始利率序列，变化与分位由统一汇总器计算。
    return raw[series]
