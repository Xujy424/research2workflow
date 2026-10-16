"""Evidence-bound Chinese interpretation for the local research report."""
from html import escape
import pandas as pd


def render_conclusions(history, features, date, regime=None):
    prior = history.loc[history.index < pd.Timestamp(date)]
    html = '<section id="conclusions"><h2>分析结论与验证范围</h2>'
    html += f'<p>监控日期：{escape(str(date))}；本次覆盖 {len(features)} 只股票；此前历史记录 {len(prior)} 个交易日。</p>'
    if len(prior) < 20:
        html += '<p class="notice">历史基准不足：以下为当日样本描述，不能据此确认碎片化、尾盘迁移或新市场状态。Z20 至少需要20个此前有效观测，Z60需要60个，一年分位数需要252个。</p>'
    else:
        html += '<p>可结合原值、此前均线、Z值及连续异常观察趋势；状态标签属于研究规则，不能替代样本与口径检查。</p>'
    if not prior.empty:
        latest = prior.iloc[-1]
        current = history.loc[pd.Timestamp(date)]
        html += '<h3>相对上一观测日</h3><ul>'
        comparisons = [('成交额', 'market.trade_amount'), ('成交金额尾部比 P99/P50', 'pooled_trade.tail_ratio'), ('尾盘成交占比', 'intraday.amount_close_share'), ('中证1000与沪深300价差差值', 'gap.csi1000-hs300.liquidity.spread')]
        for label, metric in comparisons:
            before, after = latest.get(metric), current.get(metric)
            if pd.notna(before) and pd.notna(after):
                change = f'{(after/before-1):+.2%}' if before != 0 else '前值为零，不计算相对变化'
                html += f'<li>{label}：{before:.6g} → {after:.6g}；相对变化 {change}。</li>'
        html += '</ul><p>以上是观测日之间的样本变化，不是充分历史基准下的异常或迁移结论。</p>'
    html += '<ul>'
    for symbol, row in features.iterrows():
        amount, ratio, flow = row.get('trade.amount'), row.get('trade.large_ratio'), row.get('flow.imbalance')
        if pd.notna(amount) and pd.notna(ratio) and pd.notna(flow):
            direction = '主动买入占优' if flow > .05 else '主动卖出占优' if flow < -.05 else '主动买卖接近平衡'
            html += f'<li>{escape(str(symbol))}：成交额 {amount/1e8:.2f} 亿元，大单及超大单金额占比 {ratio:.2%}，主动买卖失衡 {flow:.2%}，{direction}。'
            opening, closing = row.get('intraday.amount_open_share'), row.get('intraday.amount_close_share')
            if pd.notna(opening) and pd.notna(closing):
                html += f'早盘成交占比 {opening:.2%}，尾盘成交占比 {closing:.2%}。'
            recovered = row.get('liquidity.bid_depth_recovered_count')
            censored = row.get('liquidity.bid_depth_censored_count')
            duration = row.get('liquidity.bid_depth_recovery_seconds')
            if pd.notna(recovered) and pd.notna(censored):
                html += f'买盘深度恢复事件 {int(recovered)} 次，截尾未确认恢复 {int(censored)} 次。'
                if pd.notna(duration):
                    html += f'已恢复事件平均用时 {duration:.1f} 秒。'
            html += '</li>'
    html += '</ul>'
    quote_column = features.get('quality.quotes_count', pd.Series(0, index=features.index))
    if quote_column.fillna(0).sum() == 0:
        html += '<p class="notice">当前样本没有盘口快照，价差、深度、有效价差及盘口冲击不可评估；这不代表流动性正常。</p>'
    html += '<p>大单仅代表成交规模，不能识别机构身份。股票均值、股票池金额汇总与 pooled_trade 全样本分位数采用不同口径，不能互相替代。早盘30分钟与尾盘连续竞价27分钟的占比也不能直接视为等时长强弱比较。</p>'
    if regime:
        html += '<h3>当前状态</h3><div class="table"><table><thead><tr><th>维度</th><th>状态</th></tr></thead><tbody>'
        names = {'liquidity_regime': '流动性', 'trade_size_regime': '成交规模', 'orderflow_regime': '订单流方向', 'impact_regime': '价格冲击', 'intraday_regime': '日内重心', 'institution_regime': '大单活动代理', 'overall': '综合迁移状态'}
        labels = {'unavailable': '数据不可用', 'insufficient_history': '历史不足', 'INSUFFICIENT_HISTORY': '历史或有效维度不足', 'sell': '主动卖出占优', 'buy': '主动买入占优', 'neutral': '中性', 'opening_dominant': '早盘占优', 'closing_dominant': '尾盘占优', 'balanced': '均衡', 'NORMAL': '未触发迁移规则', 'TRANSITION': '迁移预警', 'NEW_REGIME': '持续异常触发新状态规则'}
        for key, name in names.items():
            value = str(regime.get(key, 'unavailable'))
            html += f'<tr><td>{name}</td><td>{escape(labels.get(value, value))}</td></tr>'
        html += '</tbody></table></div>'
    html += '<details><summary>功能完整性与尚未验证的边界</summary><p>输入清洗、事件关联、多周期聚合、六类指标、分布与日内形态、历史基准、股票池差异、迁移规则及报告主链路已实现。盘口恢复速度仍是观测快照代理；规则尚未经过真实全市场长历史校准；数值分钟特征已统一生成日内形态，逐股历史基准和所有数值指标可在交互页面查询；只存在日频定义的指标不会伪造分钟曲线。工程主链路可运行不等于所有市场结论已验证。</p><p>本页输入覆盖范围以顶部日期、股票数和历史记录为准。软件口径测试不代表市场结论已验证。浏览器交互未实测。指数分组为空时保持缺失值，不能进行该组差异解读。机构活动仅有大单代理，尚无真实投资者身份数据。</p></details></section>'
    return html
