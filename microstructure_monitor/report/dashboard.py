"""Static, self-contained trend report; figures use matplotlib's Agg backend."""
from pathlib import Path
from html import escape
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import pandas as pd
from io import BytesIO
from .conclusions import render_conclusions


def generate_dashboard(history, migration, directory, minute=None, features=None, structural=None, regime=None):
    path = Path(directory) / "trends.html"
    groups = {"Trade size": ["trade.median", "trade.p99", "trade.tail_ratio"],
              "Concentration": ["trade.large_ratio", "trade.xl_ratio"],
              "Order flow": ["flow.imbalance", "order.imbalance"],
              "Intraday shares": ["intraday.amount_open_share", "intraday.amount_close_share"],
              "Liquidity": ["liquidity.spread", "liquidity.depth", "liquidity.impact"],
              "Pooled distribution": ["pooled_trade.p50", "pooled_trade.p99", "pooled_trade.tail_ratio"],
              "Peak times": ["intraday.amount_argmax_minute", "intraday.large_amount_argmax_minute", "intraday.spread_argmax_minute"],
              "Distribution changes": [c for c in history if c.startswith("structural.ks.") and c.endswith(("trade.median", "trade.large_ratio", "liquidity.spread"))],
              "Curve changes": [c for c in history if c.startswith("structural.curve.")],
              "Universe differences": []}
    for prefix in ("universe.", "gap."):
        grouped = {}
        for column in history:
            for suffix in ("liquidity.spread", "trade.large_ratio", "flow.imbalance"):
                if column.startswith(prefix) and column.endswith("." + suffix):
                    group = column[len(prefix):-len(suffix)-1]
                    grouped.setdefault(group, []).append(column)
        for group, columns in grouped.items():
            groups[prefix.rstrip(".") + ": " + group] = columns
    images = []
    titles = []
    for title, columns in groups.items():
        columns = [column for column in columns if column in history and history[column].notna().any()]
        if not columns:
            continue
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 2.8 * len(columns)), squeeze=False)
        for axis, column in zip(axes[:, 0], columns):
            series = history[column]
            axis.plot(history.index, series, label="Daily", color="#196b67", marker=".")
            for window, color in ((5, "#b73858"), (20, "#6b7280"), (60, "#3d7534")):
                average = series.shift(1).rolling(window).mean()
                if average.notna().any():
                    axis.plot(history.index, average, label=f"Prior MA{window}", color=color)
            if len(history) == 1:
                axis.set_xlim(history.index[0]-pd.Timedelta("12h"), history.index[0]+pd.Timedelta("12h"))
                axis.set_xticks(history.index, history.index.strftime("%Y-%m-%d"))
            axis.set_title(column, fontsize=10)
            axis.legend(fontsize=8, ncol=4)
            axis.grid(alpha=.2)
            historical = series.shift(1).rolling(20)
            z20 = (series - historical.mean()) / historical.std().replace(0, float("nan"))
            if z20.notna().any():
                z_axis = axis.twinx()
                z_axis.plot(history.index, z20, label="Z20", color="#a0781c", alpha=.55, linestyle=":")
                z_axis.set_ylabel("Z20", fontsize=8)
                z_axis.axhline(2, color="#a0781c", alpha=.25)
                z_axis.axhline(-2, color="#a0781c", alpha=.25)
        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        titles.append(title)
        images.append(f'<section class="plot" data-title="{escape(title)}"><h2>{escape(title)}</h2><img alt="{escape(title)}" src="data:image/png;base64,{encoded}"></section>')
    if minute is not None:
        prior = sorted(p for p in Path(directory).iterdir() if p.is_dir() and (p/"bars_1min.csv").exists())
        fig, axes = plt.subplots(3, 1, figsize=(10, 9))
        for axis, metric in zip(axes, ("amount", "large_amount", "spread")):
            frames = []
            for day in prior[-21:]:
                frame = pd.read_csv(day/"bars_1min.csv", parse_dates=["time"])
                if not minute.empty and day.name < minute.time.iloc[0].strftime("%Y%m%d"):
                    frames.append(frame)
            curve = minute.groupby(minute.time.dt.strftime("%H:%M"))[metric].mean()
            historical = []
            for frame in frames:
                historical.append(frame.groupby(frame.time.dt.strftime("%H:%M"))[metric].mean())
            if historical:
                baseline = pd.concat(historical, axis=1).mean(axis=1).reindex(curve.index)
                axis.plot(range(len(curve)), baseline, label="Prior observed-day mean", color="#b73858")
            axis.plot(range(len(curve)), curve, label="Today", color="#196b67")
            ticks = list(range(0, len(curve), max(1, len(curve)//8)))
            axis.set_xticks(ticks, [curve.index[i] for i in ticks])
            axis.set_title(metric)
            axis.legend(fontsize=8)
        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        title = "Intraday profiles"
        titles.append(title)
        images.append(f'<section class="plot" data-title="{title}"><h2>{title}</h2><img alt="{title}" src="data:image/png;base64,{encoded}"></section>')
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>微观结构监控 · 研究报告</title><style>body{font:15px Arial,"Microsoft YaHei",sans-serif;margin:24px auto;padding:0 20px;color:#222;max-width:1100px;line-height:1.7}h1{font-size:28px}h2{font-size:21px;margin-top:32px}h3{font-size:17px}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd;white-space:nowrap}.table{overflow:auto}.notice{border-left:3px solid #bc8245;padding:8px 12px;background:#fff8ee}select{max-width:100%;padding:8px;border:1px solid #aaa}summary{cursor:pointer;color:#196b67}nav a{margin-right:18px;color:#196b67}li{margin-bottom:10px}</style>'
    html += '<h1>微观结构监控研究报告</h1><nav><a href="#conclusions">结论</a><a href="#charts">图表</a><a href="#quality">质量</a></nav>'
    if features is not None:
        date = minute.time.max().strftime('%Y-%m-%d') if minute is not None and not minute.empty else history.index.max().strftime('%Y-%m-%d')
        html += render_conclusions(history, features, date, regime)
    html += '<h2 id="charts">图表与历史对照</h2><label for="view">指标视图 </label><select id="view"><option value="all">全部图表</option>' + ''.join(f'<option>{escape(title)}</option>' for title in titles) + '</select>'
    html += "".join(images) + '<h2>Migration history</h2><div class="table">' + migration.tail(60).to_html() + "</div>"
    if features is not None:
        quality_columns = [c for c in features if c.startswith("quality.") or c in ("discovery.quote_match_ratio", "lifecycle.overaccounted_orders")]
        html += '<h2 id="quality">数据质量与覆盖</h2><div class="table">' + features[quality_columns].round(4).to_html() + '</div>'
    if structural is not None:
        html += '<h2>Structural differences</h2><div class="table">' + structural.round(4).to_html() + '</div>'
    html += '<script>document.getElementById("view").addEventListener("change",function(){document.querySelectorAll(".plot").forEach(p=>p.hidden=this.value!=="all"&&p.dataset.title!==this.value)});</script>'
    html += '</html>'
    path.write_text(html, encoding="utf-8")
    return path
