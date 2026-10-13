"""Static, self-contained trend report; figures use matplotlib's Agg backend."""
from pathlib import Path
from html import escape
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def generate_dashboard(history, migration, directory):
    path = Path(directory) / "trends.html"
    groups = {"Trade size": ["trade.median", "trade.p99", "trade.tail_ratio"],
              "Concentration": ["trade.large_ratio", "trade.xl_ratio"],
              "Order flow": ["flow.imbalance", "order.imbalance"],
              "Intraday shares": ["intraday.amount_open_share", "intraday.amount_close_share"],
              "Liquidity": ["liquidity.spread", "liquidity.depth", "liquidity.impact"]}
    images = []
    for title, columns in groups.items():
        columns = [column for column in columns if column in history and history[column].notna().any()]
        if not columns:
            continue
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 2.8 * len(columns)), squeeze=False)
        for axis, column in zip(axes[:, 0], columns):
            series = history[column]
            axis.plot(history.index, series, label="Daily", color="#196b67")
            for window, color in ((5, "#b73858"), (20, "#6b7280"), (60, "#3d7534")):
                axis.plot(history.index, series.rolling(window).mean(), label=f"MA{window}", color=color)
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
        images.append(f'<h2>{escape(title)}</h2><img alt="{escape(title)}" src="data:image/png;base64,{encoded}">')
    html = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Microstructure trends</title><style>body{font:14px Arial;margin:24px;color:#222;max-width:1200px}img{width:100%;height:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}.table{overflow:auto}</style>'
    html += f"<h1>Microstructure trends</h1><p>{len(history)} observed trading days; equal-weight security metrics. Large trades are an activity proxy.</p>"
    html += "".join(images) + '<h2>Migration history</h2><div class="table">' + migration.tail(60).to_html() + "</div>"
    path.write_text(html, encoding="utf-8")
    return path
