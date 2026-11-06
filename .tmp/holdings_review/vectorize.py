from pathlib import Path
p=Path('v2/Monitor/v3/moneyflow.py')
s=p.read_text(encoding='utf-8')
a=s.index('    # 每个交易日有哪些基金披露了净值',s.index('def fund_holdings_similarity'))
b=s.index('    # ==========================================================',a)
s=s[:a]+'''    # 直接构造“交易日×基金”矩阵，列内ffill替代逐日更新净值字典。
    nav_daily = (nav.pivot(index='day', columns='fund', values='UnitNV')
        .reindex(index=range(T), columns=range(F)).ffill().to_numpy())
    nav_daily[np.arange(T)[:, None] > last_pos[None, :]] = np.nan

'''+s[b:]
a=s.index('    holding_groups =',s.index('def fund_holdings_similarity'))
b=s.index('\n    return result',a)
s=s[:a]+'''    # 关联报告参数后一次性计算所有持仓系数，避免逐份报告切表。
    reports = reports.sort_values(['fund', 'day', 'ReportDate', 'InfoPublDate']).copy()
    reports['event'] = np.arange(len(reports))
    positions = holdings.merge(reports, on=report_keys, how='inner')
    report_price = price[positions['report_pos'].to_numpy(), positions['stock'].to_numpy()]
    positions['coefficient'] = positions['weight'].to_numpy() * positions['base_nav'].to_numpy() / report_price
    positions['valid'] = (np.isfinite(report_price) & (report_price > 0) &
                          np.isfinite(positions['coefficient']))
    valid_events = positions.groupby('event')['valid'].all()
    reports = reports.loc[reports['event'].isin(valid_events.index[valid_events])]

    # 每日每只基金采用最新报告；迟到的旧报告不能覆盖新报告。
    reports = reports.drop_duplicates(['fund', 'day'], keep='last')
    reports = reports.loc[reports['ReportDate'].eq(reports.groupby('fund')['ReportDate'].cummax())].copy()
    # [day, stop)等价于在下一报告或存续截止日之前fill整份报告。
    reports['stop'] = np.minimum(
        reports.groupby('fund')['day'].shift(-1).fillna(T).to_numpy(dtype=int),
        last_pos[reports['fund'].to_numpy()] + 1)
    intervals = pl.from_pandas(reports.loc[reports['day'] < reports['stop'], ['event', 'day', 'stop']])
    table = (pl.from_pandas(positions[['event', 'fund', 'stock', 'coefficient']])
        .join(intervals, on='event')
        .group_by(['fund', 'stock', 'day', 'stop']).agg(pl.col('coefficient').sum()))

    # 7. 每块展开20个交易日，批量漂移持仓并按日计算横截面相似度。
    # 保留分块循环控制内存；无逐日、逐基金、逐报告的Python循环。
    for start in range(0, T, 20):
        stop = min(start + 20, T)
        block = (table.filter((pl.col('day') < stop) & (pl.col('stop') > start))
            .with_columns(pl.int_ranges(pl.max_horizontal('day', pl.lit(start)),
                                        pl.min_horizontal('stop', pl.lit(stop))).alias('day'))
            .explode('day'))
        if block.is_empty():
            continue
        day, fund, stock = (block[col].to_numpy().astype(int) for col in ('day', 'fund', 'stock'))
        weight = block['coefficient'].to_numpy() * price[day, stock] / nav_daily[day, fund]
        rows = (block.select('day', 'fund', 'stock').with_columns(pl.Series('weight', weight))
            .filter((pl.col('weight').is_finite() & (pl.col('weight') >= 0)).all().over(['day', 'fund'])))
        # 整只基金的保留持仓都有效且总权重大于0，才参与当日比较。
        totals = (rows.group_by(['day', 'fund']).agg(pl.col('weight').sum().alias('total'))
            .filter(pl.col('total') > 0))
        rows = rows.join(totals, on=['day', 'fund'])
        counts = totals.group_by('day').agg(pl.len().alias('count'))
        # 只关联同日共同持仓，fund<fund_j排除自身和反向重复基金对。
        scores = (rows.lazy().join(rows.lazy(), on=['day', 'stock'], suffix='_j')
            .filter(pl.col('fund') < pl.col('fund_j'))
            .group_by(['day', 'fund', 'fund_j'])
            .agg(pl.min_horizontal('weight', 'weight_j').sum().alias('overlap'),
                 pl.col('total').first(), pl.col('total_j').first())
            .with_columns((pl.col('overlap') /
                (pl.col('total') + pl.col('total_j') - pl.col('overlap'))).alias('similarity'))
            .group_by('day').agg(pl.col('similarity').sum()).collect(engine='streaming'))
        # 无共同股票的基金对贡献0；不足两只基金的日期保留NaN。
        daily = counts.join(scores, on='day', how='left').filter(pl.col('count') >= 2)
        n = daily['count'].to_numpy().astype(float)
        result.iloc[daily['day'].to_numpy()] = daily['similarity'].fill_null(0).to_numpy() / (n * (n - 1) / 2)
'''+s[b:]
s=s.replace('from collections import defaultdict\n','')
p.write_text(s,encoding='utf-8')
