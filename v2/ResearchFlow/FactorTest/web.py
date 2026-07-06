"""中文说明：本脚本提供当前模块的量化研究或生产能力。"""


from __future__ import annotations

import html
import sys
from datetime import date, datetime
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from pandas.io.formats.style import Styler


PAGE_TITLE = "因子分析报告"
FIGURE_DPI = 120
TABLE_ROW_HEIGHT = 31
TABLE_HEADER_HEIGHT = 38
TABLE_MAX_HEIGHT = 430
HALF_FIGSIZE = (7.2, 4.8)
FULL_FIGSIZE = (11.5, 4.2)
TALL_FIGSIZE = (11.5, 5.4)
SUPPORTED_FACTOR_SUFFIXES = {".csv", ".parquet", ".pkl", ".pickle", ".feather"}
UNIVERSE_OPTIONS = ("universe", "hs300", "zz500", "zz1000", "zz2000", "a500")


# 中文说明：定义 `OutputSpec`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class OutputSpec:
    title: str
    kind: str
    func_name: str
    width: str = "full"


# 中文说明：定义 `ModuleSpec`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class ModuleSpec:
    key: str
    title: str
    outputs: tuple[OutputSpec, ...]
    accent: str = "#D9EAF7"


MODULES: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        key="signal_test",
        title="PRF 与多期胜率检验",
        outputs=(
            OutputSpec("PRF 检验", "table", "table_PRF_stats", "half"),
            OutputSpec("多期胜率检验", "table", "table_winrate_scan", "half"),
        ),
        accent="#F8C8CC",
    ),
    ModuleSpec(
        key="basic_performance",
        title="收益与基本表现",
        outputs=(
            OutputSpec("月度收益", "table", "table_monthly_ret", "half"),
            OutputSpec("年度收益指标表现", "table", "table_annual_stats", "half"),
            OutputSpec("基本表现", "figure", "plot_basic_performance"),
        ),
        accent="#DDEED6",
    ),
    ModuleSpec(
        key="alpha_distribution",
        title="因子值分布",
        outputs=(
            OutputSpec("因子值分布表", "table", "table_alpha_annual_stats", "half"),
            OutputSpec("因子值分布图", "figure", "plot_alpha_distribution", "half"),
        ),
        accent="#C9C6F4",
    ),
    ModuleSpec(
        key="ic_analysis",
        title="IC 分析",
        outputs=(
            OutputSpec("IC 年度统计指标", "table", "table_ic_annual_stats"),
            OutputSpec("各回报期 IC 分布图", "figure", "plot_ic_distribution", "half"),
            OutputSpec("IC 指标累计时序图", "figure", "plot_ic_contribution", "half"),
        ),
        accent="#F4BDF0",
    ),
    ModuleSpec(
        key="group_return",
        title="分组收益",
        outputs=(
            OutputSpec("分组收益表现", "table", "table_group_stats", "half"),
            OutputSpec("分组收益累计图", "figure", "plot_group_cumret", "half"),
        ),
        accent="#D8D8D8",
    ),
    ModuleSpec(
        key="industry",
        title="行业分析",
        outputs=(
            OutputSpec("行业分域指标", "table", "table_industry_annual_stats", "half"),
            OutputSpec("行业分域表现", "figure", "plot_industry_performance", "half"),
            OutputSpec("行业暴露表现", "table", "table_industry_exposure_stats", "half"),
            OutputSpec("行业暴露收益", "figure", "plot_industry_exposure_ret", "half"),
            OutputSpec("行业持仓结构", "figure", "plot_industry_component"),
        ),
        accent="#FFF1C9",
    ),
    ModuleSpec(
        key="sector",
        title="板块分析",
        outputs=(
            OutputSpec("板块分域指标", "table", "table_sector_annual_stats", "half"),
            OutputSpec("板块分域表现", "figure", "plot_sector_performance", "half"),
            OutputSpec("板块暴露表现", "table", "table_sector_exposure_stats", "half"),
            OutputSpec("板块暴露收益", "figure", "plot_sector_exposure_ret", "half"),
            OutputSpec("板块持仓结构", "figure", "plot_sector_component"),
        ),
        accent="#FFD09A",
    ),
    ModuleSpec(
        key="barra",
        title="Barra 暴露",
        outputs=(
            OutputSpec("Barra 因子暴露表现", "table", "table_barra_exposure_stats", "half"),
            OutputSpec("Barra 因子暴露", "figure", "plot_barra_exposure", "half"),
            OutputSpec("Barra 因子暴露收益时序图", "figure", "plot_barra_exposure_ret"),
        ),
        accent="#F2DED3",
    ),
    # Spearman redundancy module disabled per current FactorTest workflow.
#     ModuleSpec(
#         key="spearman",
#         title="Spearman 秩相关冗余检验",
#         outputs=(OutputSpec("Spearman 秩相关冗余检验", "figure", "plot_corr_redundancy"),),
#         accent="#B9CAE8",
#     ),
    ModuleSpec(
        key="regime_online",
        title="上下线模型检验",
        outputs=(
            OutputSpec("市场状态统计", "table", "table_regime_stats", "half"),
            OutputSpec("市场状态累计收益", "figure", "plot_regime_cumret", "half"),
        ),
        accent="#CFE8E2",
    ),
    ModuleSpec(
        key="shadow_capacity",
        title="影子盘容量检验",
        outputs=(
            OutputSpec("影子盘容量测试", "table", "table_shadow_capacity_test", "half"),
            OutputSpec("容量净值曲线", "figure", "plot_shadow_capacity_curve", "half"),
        ),
        accent="#E7D7F4",
    ),
)

MODULE_BY_KEY = {module.key: module for module in MODULES}


# 中文说明：`main`：执行该名称对应的业务计算，并返回调用方所需结果。
def main() -> None:
    configure_page()
    init_state()
    inject_css()

    render_header()
    render_module_buttons()
    info, factor_df, factor_signature, load_clicked, reset_clicked = render_sidebar()
    handle_cache_reset(reset_clicked)
    handle_data_load(info, factor_df, factor_signature, load_clicked)
    render_report_body()


# 中文说明：`configure_page`：执行该名称对应的业务计算，并返回调用方所需结果。
def configure_page() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="expanded")
    configure_matplotlib()


# 中文说明：`configure_matplotlib`：执行该名称对应的业务计算，并返回调用方所需结果。
def configure_matplotlib() -> None:
    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    selected = next((font for font in font_candidates if font in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D7DEE8",
            "axes.labelcolor": "#253041",
            "xtick.color": "#526071",
            "ytick.color": "#526071",
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "grid.color": "#E8EDF4",
            "grid.linewidth": 0.8,
        }
    )


# 中文说明：`init_state`：执行该名称对应的业务计算，并返回调用方所需结果。
def init_state() -> None:
    defaults = {
        "analyzer": None,
        "base_signature": None,
        "axis_signature": None,
        "active_modules": [],
        "module_cache": {},
        "score_cache": {},
        "uploaded_factor_path": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.active_modules = [
        key for key in st.session_state.active_modules if key in MODULE_BY_KEY
    ]


# 中文说明：`inject_css`：执行该名称对应的业务计算，并返回调用方所需结果。
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1440px;
        }
        [data-testid="stSidebar"] {
            background: #F7F9FC;
            border-right: 1px solid #E0E6EF;
        }
        div.stButton > button {
            border-radius: 8px;
            border: 1px solid #CAD4E1;
            background: #FFFFFF;
            color: #213047;
            min-height: 2.45rem;
            font-weight: 600;
        }
        div.stButton > button:hover {
            border-color: #4F7CAC;
            color: #17446D;
            background: #F3F7FB;
        }
        .report-title {
            font-size: 1.65rem;
            font-weight: 750;
            color: #172033;
            margin: 0 0 0.25rem 0;
        }
        .report-subtitle {
            color: #667386;
            margin-bottom: 1rem;
        }
        .module-shell {
            border: 1px solid #D8E0EB;
            border-radius: 8px;
            background: #FFFFFF;
            margin: 1rem 0;
            overflow: hidden;
        }
        .module-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.55rem 0.8rem;
            border-bottom: 1px solid #D8E0EB;
            font-weight: 700;
            color: #182235;
        }
        .module-body {
            padding: 0.85rem 0.9rem 1rem 0.9rem;
        }
        .output-title {
            font-weight: 700;
            color: #26364D;
            margin: 0.2rem 0 0.45rem 0;
        }
        .codex-info-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
        }
        .codex-info-table td {
            border: 1px solid #DDE4EE;
            padding: 0.48rem 0.66rem;
            vertical-align: top;
        }
        .codex-info-table .key {
            width: 170px;
            background: #F4F7FA;
            color: #415066;
            font-weight: 700;
            white-space: nowrap;
        }
        .codex-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .codex-table th,
        .codex-table td {
            border: 1px solid #DDE4EE;
            padding: 4px 7px;
            text-align: center;
            white-space: nowrap;
        }
        .codex-table th {
            background: #F3F6FA;
            color: #23324A;
            font-weight: 700;
        }
        .score-note {
            margin: 0.55rem 0 0.85rem 0;
            padding: 0.65rem 0.75rem;
            border: 1px solid #DDE4EE;
            border-left: 4px solid #4F7CAC;
            border-radius: 8px;
            background: #F8FAFD;
            color: #2C3B52;
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .score-note.fail {
            border-left-color: #B42318;
            background: #FFF7F5;
        }
        .score-note.watch {
            border-left-color: #D68A00;
            background: #FFF9EC;
        }
        .score-note.good {
            border-left-color: #287C71;
            background: #F2FAF8;
        }
        .score-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.35rem;
        }
        .score-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            background: #EAF0F7;
            padding: 0.12rem 0.48rem;
            font-weight: 700;
            color: #23324A;
        }
        .score-suggestions {
            margin-top: 0.35rem;
            color: #405066;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# 中文说明：`render_header`：渲染研究报告或界面内容。
def render_header() -> None:
    st.markdown("<div class='report-title'>因子分析交互平台</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='report-subtitle'>输入或上传因子文件，加载数据后按模块生成表格、图表，并可导出当前展示内容。</div>",
        unsafe_allow_html=True,
    )


# 中文说明：`render_module_buttons`：渲染研究报告或界面内容。
def render_module_buttons() -> None:
    st.markdown("**分析模块**")
    columns = st.columns(5)
    for index, module in enumerate(MODULES):
        active = module.key in st.session_state.active_modules
        label = f"{'✓ ' if active else ''}{module.title}"
        with columns[index % len(columns)]:
            if st.button(label, key=f"toggle_{module.key}", use_container_width=True):
                toggle_module(module.key)
                st.rerun()


# 中文说明：`render_sidebar`：渲染研究报告或界面内容。
def render_sidebar() -> tuple[dict[str, Any], pd.DataFrame | None, str | None, bool, bool]:
    with st.sidebar:
        st.header("数据与参数")
        factor_path = st.text_input("因子文件地址", value="")
        uploaded_file = st.file_uploader("上传因子文件", type=[s.lstrip(".") for s in SUPPORTED_FACTOR_SUFFIXES])

        st.divider()
        st.subheader("因子基础信息")
        factor_name = st.text_input("因子名称", value="GRU")
        factor_type = st.selectbox("因子方向", ["longshort", "long"], index=0)
        alpha_type = st.selectbox("因子类型", ["深度学习", "量价", "基本面", "资金面", "另类", "其他"], index=0)
        usage = st.text_input("数据用途", value="日频选股")
        universe = st.selectbox("Universe", UNIVERSE_OPTIONS, index=0)
        start_date = st.date_input("起始日期", value=pd.Timestamp("2021-01-01"))
        end_date = st.date_input("结束日期", value=pd.Timestamp("2025-12-16"))
        summary = st.text_area("因子描述", value="", height=90)

        load_clicked = st.button("加载数据", type="primary", use_container_width=True)
        reset_clicked = st.button("安全重置缓存", use_container_width=True)

    info = build_info(factor_name, factor_type, alpha_type, usage, universe, start_date, end_date, summary)
    if not load_clicked:
        return info, None, None, False, reset_clicked

    factor_df, factor_signature = dataframe_from_input(factor_path, uploaded_file)
    return info, factor_df, factor_signature, True, reset_clicked


# 中文说明：`build_info`：构建下游所需对象。
def build_info(
    name: str,
    factor_type: str,
    alpha_type: str,
    usage: str,
    universe: str,
    start_date: Any,
    end_date: Any,
    summary: str,
) -> dict[str, Any]:
    return {
        "name": name.strip() or "未命名因子",
        "factor_type": factor_type,
        "alpha_type": alpha_type,
        "usage": usage.strip(),
        "universe": universe,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "summary": summary.strip(),
    }


# 中文说明：`dataframe_from_input`：执行该名称对应的业务计算，并返回调用方所需结果。
def dataframe_from_input(path: str, uploaded_file: Any | None) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower() or ".csv"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            st.session_state.uploaded_factor_path = tmp.name
        df = read_factor_file(Path(st.session_state.uploaded_factor_path))
        return df, f"upload:{uploaded_file.name}:{uploaded_file.size}"

    clean_path = path.strip().strip('"')
    if not clean_path:
        raise ValueError("请填写因子文件地址，或上传因子文件。")
    file_path = Path(clean_path).expanduser()
    df = read_factor_file(file_path)
    return df, f"path:{file_path.resolve()}:{file_path.stat().st_mtime_ns}"


# 中文说明：`read_factor_file`：读取持久化数据。
def read_factor_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif suffix == ".feather":
        df = pd.read_feather(path)
        df = df.set_index(df.columns[0])
    else:
        raise ValueError(f"不支持的文件类型: {suffix}。支持: {', '.join(sorted(SUPPORTED_FACTOR_SUFFIXES))}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = df.columns.astype(str)
    return df


# 中文说明：`handle_cache_reset`：执行该名称对应的业务计算，并返回调用方所需结果。
def handle_cache_reset(reset_clicked: bool) -> None:
    if not reset_clicked:
        return
    analyzer = st.session_state.analyzer
    if analyzer is None:
        st.warning("当前没有可重置的分析器缓存。")
        return
    analyzer.reset_cache()
    clear_module_cache()
    st.success("已执行 reset_cache，并清空网页模块结果缓存。")


# 中文说明：`handle_data_load`：执行该名称对应的业务计算，并返回调用方所需结果。
def handle_data_load(
    info: dict[str, Any],
    factor_df: pd.DataFrame | None,
    factor_signature: str | None,
    load_clicked: bool,
) -> None:
    if not load_clicked:
        return
    if factor_df is None or factor_signature is None:
        st.error("因子数据为空。")
        return
    if pd.Timestamp(info["start_date"]) > pd.Timestamp(info["end_date"]):
        st.error("起始日期不能晚于结束日期。")
        return

    with st.spinner("正在加载数据并准备基础缓存..."):
        try:
            message = load_or_reset_analyzer(info, factor_df, factor_signature)
        except Exception as exc:
            st.error(f"加载数据失败: {exc}")
            return
    st.success(message)


# 中文说明：`load_or_reset_analyzer`：读取并规范化外部数据。
def load_or_reset_analyzer(info: dict[str, Any], factor_df: pd.DataFrame, factor_signature: str) -> str:
    FactorAnalyzer = load_factor_analyzer_class()
    base_signature = make_base_signature(info, factor_signature)
    axis_signature = make_axis_signature(info)
    analyzer = st.session_state.analyzer

    if analyzer is None or st.session_state.base_signature != base_signature:
        base_info = info.copy()
        base_info["start_date"] = str(factor_df.index.min().date())
        base_info["end_date"] = str(factor_df.index.max().date())
        analyzer = FactorAnalyzer(base_info, factor_df.copy())
        st.session_state.analyzer = analyzer
        st.session_state.base_signature = base_signature
        analyzer.reset_axis(info["start_date"], info["end_date"], info["universe"])
        st.session_state.axis_signature = axis_signature
        clear_module_cache()
        return "已初始化分析器，完成 init、prepare_data，并按当前起止日期和 universe 执行 reset_axis。"

    if st.session_state.axis_signature != axis_signature:
        analyzer.reset_axis(info["start_date"], info["end_date"], info["universe"])
        st.session_state.axis_signature = axis_signature
        clear_module_cache()
        return "已复用当前分析器，并因起止日期或 universe 变化执行 reset_axis。"

    return "数据和轴参数未变化，已保留当前缓存。"


# 中文说明：`load_factor_analyzer_class`：读取并规范化外部数据。
def load_factor_analyzer_class() -> type:
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent
    for path in (str(package_dir), str(project_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        from papertest import FactorAnalyzer  # type: ignore
    except ImportError:
        from .analyzer import FactorAnalyzer
    return FactorAnalyzer


# 中文说明：`make_base_signature`：执行该名称对应的业务计算，并返回调用方所需结果。
def make_base_signature(info: dict[str, Any], factor_signature: str) -> tuple[Any, ...]:
    return (
        factor_signature,
        info["name"],
        info["factor_type"],
        info["alpha_type"],
        info["usage"],
        info["summary"],
    )


# 中文说明：`make_axis_signature`：执行该名称对应的业务计算，并返回调用方所需结果。
def make_axis_signature(info: dict[str, Any]) -> tuple[str, str, str]:
    return (info["start_date"], info["end_date"], info["universe"])


# 中文说明：`clear_module_cache`：执行该名称对应的业务计算，并返回调用方所需结果。
def clear_module_cache() -> None:
    st.session_state.module_cache = {}
    st.session_state.score_cache = {}


# 中文说明：`render_report_body`：渲染研究报告或界面内容。
def render_report_body() -> None:
    analyzer = st.session_state.analyzer
    if analyzer is None:
        st.info("请先在左侧输入或上传因子文件，填写基础信息后点击加载数据。模块可以提前选择，加载后会自动展示。")
        return

    render_status(analyzer)
    render_info_table(analyzer.info_table)
    render_active_score_summary(analyzer)
    for module_key in st.session_state.active_modules:
        render_module(analyzer, MODULE_BY_KEY[module_key])
    render_pdf_export(analyzer)


# 中文说明：`render_status`：渲染研究报告或界面内容。
def render_status(analyzer: Any) -> None:
    start = pd.to_datetime(analyzer.cache["start_date"]).date()
    end = pd.to_datetime(analyzer.cache["end_date"]).date()
    universe = analyzer.info_table.get("universe", "")
    st.caption(f"当前数据区间: {start} 至 {end} | Universe: {universe}")


# 中文说明：`render_active_score_summary`：渲染研究报告或界面内容。
def render_active_score_summary(analyzer: Any) -> None:
    score_rows = []
    for module_key in st.session_state.active_modules:
        module = MODULE_BY_KEY[module_key]
        for spec in module.outputs:
            score_result = compute_score_result(analyzer, spec)
            if score_result is None:
                continue
            score_rows.append(
                {
                    "模块": fix_mojibake(module.title),
                    "分析项": fix_mojibake(spec.title),
                    "档位": score_result.level,
                    "得分": round(float(score_result.score), 2),
                    "硬Gate": "是" if score_result.hard_fail else "否",
                }
            )
    if not score_rows:
        return
    score_df = pd.DataFrame(score_rows)
    hard_fail = bool((score_df["硬Gate"] == "是").any())
    avg_score = float(score_df["得分"].mean())
    conclusion = "硬Gate不通过" if hard_fail else ("通过" if avg_score >= 75 else "观察/待优化")
    st.metric("当前展示模块平均分", f"{avg_score:.1f}", conclusion)
    with st.expander("查看当前展示模块评分概览", expanded=False):
        st.dataframe(score_df, use_container_width=True, height=min(360, 42 + 30 * len(score_df)))


# 中文说明：`render_module`：渲染研究报告或界面内容。
def render_module(analyzer: Any, module: ModuleSpec) -> None:
    st.markdown(
        f"""
        <div class='module-shell'>
            <div class='module-head' style='background:{module.accent};'>
                <span>{html.escape(module.title)}</span>
            </div>
            <div class='module-body'>
        """,
        unsafe_allow_html=True,
    )
    close_col, _ = st.columns([1, 12])
    with close_col:
        if st.button("×", key=f"close_{module.key}", help="关闭该模块"):
            toggle_module(module.key, force_off=True)
            st.rerun()

    render_outputs(analyzer, module)
    st.markdown("</div></div>", unsafe_allow_html=True)


# 中文说明：`render_outputs`：渲染研究报告或界面内容。
def render_outputs(analyzer: Any, module: ModuleSpec) -> None:
    if module.key in {"industry", "sector"}:
        render_domain_outputs(analyzer, module)
        return

    outputs = list(module.outputs)
    index = 0
    while index < len(outputs):
        current = outputs[index]
        nxt = outputs[index + 1] if index + 1 < len(outputs) else None
        if current.width == "half" and nxt is not None and nxt.width == "half":
            cols = st.columns(2)
            for col, spec in zip(cols, (current, nxt)):
                with col:
                    render_titled_output(analyzer, module.key, spec)
            index += 2
        else:
            render_titled_output(analyzer, module.key, current)
            index += 1


# 中文说明：`render_domain_outputs`：渲染研究报告或界面内容。
def render_domain_outputs(analyzer: Any, module: ModuleSpec) -> None:
    outputs = list(module.outputs)
    if len(outputs) < 5:
        return render_outputs_default(analyzer, module)

    first_row = st.columns(2)
    with first_row[0]:
        render_titled_output(analyzer, module.key, outputs[0])
    with first_row[1]:
        render_titled_output(analyzer, module.key, outputs[1])

    second_row = st.columns(2)
    with second_row[0]:
        render_titled_output(analyzer, module.key, outputs[2])
    with second_row[1]:
        render_titled_output(analyzer, module.key, replace_width(outputs[4], "half"))

    render_titled_output(analyzer, module.key, replace_width(outputs[3], "full"))


# 中文说明：`render_outputs_default`：渲染研究报告或界面内容。
def render_outputs_default(analyzer: Any, module: ModuleSpec) -> None:
    outputs = list(module.outputs)
    index = 0
    while index < len(outputs):
        current = outputs[index]
        nxt = outputs[index + 1] if index + 1 < len(outputs) else None
        if current.width == "half" and nxt is not None and nxt.width == "half":
            cols = st.columns(2)
            for col, spec in zip(cols, (current, nxt)):
                with col:
                    render_titled_output(analyzer, module.key, spec)
            index += 2
        else:
            render_titled_output(analyzer, module.key, current)
            index += 1


# 中文说明：`replace_width`：执行该名称对应的业务计算，并返回调用方所需结果。
def replace_width(spec: OutputSpec, width: str) -> OutputSpec:
    return OutputSpec(spec.title, spec.kind, spec.func_name, width)


# 中文说明：`render_titled_output`：渲染研究报告或界面内容。
def render_titled_output(analyzer: Any, module_key: str, spec: OutputSpec) -> None:
    st.markdown(f"<div class='output-title'>{html.escape(spec.title)}</div>", unsafe_allow_html=True)
    try:
        value = compute_output(analyzer, module_key, spec)
        render_output(value, spec)
        render_score_note(analyzer, spec)
    except Exception as exc:
        st.error(f"{spec.title} 计算失败: {exc}")


# 中文说明：`compute_output`：执行该名称对应的业务计算，并返回调用方所需结果。
def compute_output(analyzer: Any, module_key: str, spec: OutputSpec) -> Any:
    cache_key = (module_key, spec.func_name)
    if cache_key in st.session_state.module_cache:
        return st.session_state.module_cache[cache_key]

    if spec.kind == "info":
        value = getattr(analyzer, spec.func_name)
    else:
        value = get_callable(analyzer, spec.func_name)()
    value = normalize_result(value)
    if spec.kind == "figure":
        value = polish_figure(value, spec.width)
    st.session_state.module_cache[cache_key] = value
    return value


# 中文说明：`get_callable`：执行该名称对应的业务计算，并返回调用方所需结果。
def get_callable(analyzer: Any, func_name: str) -> Callable[[], Any]:
    aliases = {
        # Spearman redundancy check disabled; keep alias here for future restoration.
        # "plot_corr_redundancy": ("plot_corr_redundancy", "plot_spearman_redundancy"),
    }
    for candidate in aliases.get(func_name, (func_name,)):
        if hasattr(analyzer, candidate):
            return getattr(analyzer, candidate)
    raise AttributeError(f"分析器不存在函数: {func_name}")


# 中文说明：`normalize_result`：规范化输入或权重。
def normalize_result(value: Any) -> Any:
    if isinstance(value, (pd.DataFrame, pd.Series, Styler)):
        return fix_table_mojibake(value)
    return value


# 中文说明：`render_output`：渲染研究报告或界面内容。
def render_output(value: Any, spec: OutputSpec) -> None:
    if spec.kind == "info":
        render_info_table(value)
    elif spec.kind == "table":
        render_table(value, spec)
    elif spec.kind == "figure":
        st.pyplot(value, use_container_width=True)
    else:
        st.write(value)


SCORE_FUNC_BY_OUTPUT = {
    "table_PRF_stats": "score_prf",
    "table_winrate_scan": "score_winrate",
    "table_monthly_ret": "score_monthly_return",
    "table_annual_stats": "score_annual_stats",
    "plot_basic_performance": "score_basic_performance",
    "table_alpha_annual_stats": "score_alpha_distribution_table",
    "plot_alpha_distribution": "score_alpha_distribution_shape",
    "table_ic_annual_stats": "score_ic_annual_stats",
    "plot_ic_distribution": "score_ic_distribution",
    "plot_ic_contribution": "score_ic_curve",
    "table_group_stats": "score_group_stats",
    "plot_group_cumret": "score_group_curve",
    "table_industry_annual_stats": "score_industry_annual",
    "plot_industry_performance": "score_industry_curve",
    "table_industry_exposure_stats": "score_industry_exposure",
    "plot_industry_exposure_ret": "score_industry_exposure_return",
    "plot_industry_component": "score_industry_component",
    "table_sector_annual_stats": "score_sector_annual",
    "plot_sector_performance": "score_sector_curve",
    "table_sector_exposure_stats": "score_sector_exposure",
    "plot_sector_exposure_ret": "score_sector_exposure_return",
    "plot_sector_component": "score_sector_component",
    "table_barra_exposure_stats": "score_barra_exposure_stats",
    "plot_barra_exposure": "score_barra_exposure_bar",
    "plot_barra_exposure_ret": "score_barra_exposure_return",
    # "plot_corr_redundancy": "score_redundancy",  # Spearman redundancy check disabled.
    "table_regime_stats": "score_regime_stats",
    "plot_regime_cumret": "score_regime_curve",
    "table_shadow_capacity_test": "score_shadow_capacity",
    "plot_shadow_capacity_curve": "score_shadow_capacity_curve",
}


# 中文说明：`render_score_note`：渲染研究报告或界面内容。
def render_score_note(analyzer: Any, spec: OutputSpec) -> None:
    score_result = compute_score_result(analyzer, spec)
    if score_result is None:
        return

    advice = get_upgrade_advice(spec.func_name, score_result)
    note_class = score_note_class(score_result)
    advice_html = ""
    if advice is not None:
        issues = list(getattr(advice, "issues", ()))
        causes = list(getattr(advice, "causes", ()))
        sources = list(getattr(advice, "sources", ()))
        methods = list(getattr(advice, "methods", ()))
        parts = []
        if issues:
            text = "；".join(html.escape(str(item)) for item in issues[:3])
            parts.append(f"<div class='score-suggestions'><b>问题：</b>{text}</div>")
        if causes:
            text = "；".join(html.escape(str(item)) for item in causes[:3])
            parts.append(f"<div class='score-suggestions'><b>指标判断：</b>{text}</div>")
        if sources:
            text = "；".join(html.escape(str(item)) for item in sources[:3])
            parts.append(f"<div class='score-suggestions'><b>收益来源：</b>{text}</div>")
        if methods:
            text = "；".join(html.escape(str(item)) for item in methods[:3])
            parts.append(f"<div class='score-suggestions'><b>改进方向：</b>{text}</div>")
        advice_html = "".join(parts)

    st.markdown(
        "<div class='score-note {note_class}'>"
        "<div class='score-meta'>"
        "<span class='score-pill'>档位：{level}</span>"
        "<span class='score-pill'>得分：{score:.1f}</span>"
        "{gate}"
        "</div>"
        "<div><b>评分依据：</b>{actual}</div>"
        "<div><b>评审意见：</b>{opinion}</div>"
        "{advice_html}"
        "</div>".format(
            note_class=note_class,
            level=html.escape(str(score_result.level)),
            score=float(score_result.score),
            gate="<span class='score-pill'>硬Gate</span>" if score_result.hard_fail else "",
            actual=html.escape(str(score_result.actual)),
            opinion=html.escape(str(score_result.opinion)),
            advice_html=advice_html,
        ),
        unsafe_allow_html=True,
    )


# 中文说明：`compute_score_result`：执行该名称对应的业务计算，并返回调用方所需结果。
def compute_score_result(analyzer: Any, spec: OutputSpec) -> Any | None:
    score_func_name = SCORE_FUNC_BY_OUTPUT.get(spec.func_name)
    if score_func_name is None:
        return None
    cache_key = ("score", spec.func_name)
    if cache_key in st.session_state.score_cache:
        return st.session_state.score_cache[cache_key]
    try:
        from . import score as score_module

        score_func = getattr(score_module, score_func_name)
        score_result = score_func(analyzer)
    except Exception as exc:
        try:
            from .score import ScoreResult

            score_result = ScoreResult(
                actual=f"评分计算失败: {exc}",
                level="待复核",
                score=40.0,
                opinion="请检查 analyzer 缓存、评分规则或人工复核该项。",
            )
        except Exception:
            return None
    st.session_state.score_cache[cache_key] = score_result
    return score_result


# 中文说明：`get_upgrade_advice`：执行该名称对应的业务计算，并返回调用方所需结果。
def get_upgrade_advice(func_name: str, score_result: Any) -> Any | None:
    try:
        from .upgrade import diagnose_upgrades

        return diagnose_upgrades(func_name, score_result)
    except Exception:
        return None


# 中文说明：`score_note_class`：计算评分或监控指标。
def score_note_class(score_result: Any) -> str:
    level = str(getattr(score_result, "level", "")).lower()
    if getattr(score_result, "hard_fail", False) or "fail" in level:
        return "fail"
    if "watch" in level or "待" in level:
        return "watch"
    if "good" in level or "excellent" in level or "pass" in level:
        return "good"
    return ""


# 中文说明：`render_info_table`：渲染研究报告或界面内容。
def render_info_table(info_table: Any) -> None:
    rows = iter_info_rows(info_table)
    html_rows = []
    for key, value in rows:
        html_rows.append(
            "<tr>"
            f"<td class='key'>{html.escape(pretty_text(key))}</td>"
            f"<td>{html.escape(pretty_value(value))}</td>"
            "</tr>"
        )
    st.markdown("<table class='codex-info-table'>" + "".join(html_rows) + "</table>", unsafe_allow_html=True)


# 中文说明：`iter_info_rows`：执行该名称对应的业务计算，并返回调用方所需结果。
def iter_info_rows(info_table: Any) -> Iterable[tuple[Any, Any]]:
    if isinstance(info_table, pd.Series):
        return info_table.items()
    df = normalize_table(info_table)
    if len(df) == 1:
        return df.iloc[0].items()
    return df.stack().items()


# 中文说明：`render_table`：渲染研究报告或界面内容。
def render_table(value: Any, spec: OutputSpec) -> None:
    df = format_display_table(normalize_table(value), spec)
    height = table_height(df, spec)
    st.dataframe(df, use_container_width=True, height=height)


# 中文说明：`format_display_table`：执行该名称对应的业务计算，并返回调用方所需结果。
def format_display_table(df: pd.DataFrame, spec: OutputSpec) -> pd.DataFrame:
    if spec.func_name == "table_monthly_ret":
        return df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    if spec.func_name != "table_shadow_capacity_test":
        return df
    result = df.copy()
    numeric_index = pd.to_numeric(pd.Index(result.index), errors="coerce")
    if numeric_index.notna().any():
        result.index = [
            f"{float(value) / 1e8:.2f}亿" if pd.notna(value) else str(index)
            for index, value in zip(result.index, numeric_index)
        ]
        result.index.name = "Capital"
    return result


# 中文说明：`table_height`：生成诊断表格。
def table_height(df: pd.DataFrame, spec: OutputSpec) -> int:
    full_height = TABLE_HEADER_HEIGHT + TABLE_ROW_HEIGHT * (max(len(df), 1) + 1)
    if is_scroll_table(spec):
        return min(TABLE_MAX_HEIGHT, full_height)
    return full_height


# 中文说明：`is_scroll_table`：执行该名称对应的业务计算，并返回调用方所需结果。
def is_scroll_table(spec: OutputSpec) -> bool:
    return spec.func_name in {"table_industry_annual_stats", "table_industry_exposure_stats"}


# 中文说明：`normalize_table`：规范化输入或权重。
def normalize_table(value: Any) -> pd.DataFrame:
    if isinstance(value, Styler):
        return value.data
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame(name="value")
    return pd.DataFrame(value)


# 中文说明：`fix_table_mojibake`：执行该名称对应的业务计算，并返回调用方所需结果。
def fix_table_mojibake(value: Any) -> Any:
    if isinstance(value, Styler):
        value.data = fix_table_mojibake(value.data)
        return value
    if isinstance(value, pd.Series):
        result = value.copy()
        result.index = [fix_mojibake(x) for x in result.index]
        result.name = fix_mojibake(result.name)
        result = result.map(fix_mojibake)
        return result
    if isinstance(value, pd.DataFrame):
        result = value.copy()
        result.index = [fix_mojibake(x) for x in result.index]
        result.columns = [fix_mojibake(x) for x in result.columns]
        return result.apply(lambda column: column.map(fix_mojibake))
    return value


# 中文说明：`fix_mojibake`：执行该名称对应的业务计算，并返回调用方所需结果。
def fix_mojibake(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        repaired = value.encode("gbk").decode("utf-8")
    except UnicodeError:
        repaired = value
    return COLUMN_ALIASES.get(repaired, COLUMN_ALIASES.get(value, repaired))


COLUMN_ALIASES = {
    "name": "因子名称",
    "factor_type": "因子方向",
    "alpha_type": "因子类型",
    "usage": "数据用途",
    "universe": "Universe",
    "start_date": "起始日期",
    "end_date": "结束日期",
    "summary": "因子描述",
    "year": "年份",
    "horizon": "回报期",
    "precision": "准确率",
    "baseline": "基准胜率",
    "lift": "提升",
    "recall": "召回率",
    "f1": "F1",
    "n_signal": "信号数",
    "mean_ret": "平均收益",
    "win_rate": "胜率",
    "t_stat": "t 值",
    "p_value": "p 值",
    "AnnRet": "年化收益",
    "AnnVol": "年化波动",
    "Sharpe": "夏普比率",
    "MaxDrawdown": "最大回撤",
    "Calmar": "Calmar",
    "Avg Exposure": "平均暴露",
    "Ind Return": "行业收益",
    "Sec Return": "板块收益",
    "Exposure Return": "暴露收益",
    "Group": "分组",
    "Industry": "行业",
    "Sector": "板块",
}


# 中文说明：`polish_figure`：执行该名称对应的业务计算，并返回调用方所需结果。
def polish_figure(fig: Any, width: str) -> Any:
    if not hasattr(fig, "set_size_inches"):
        return fig
    fig.set_dpi(FIGURE_DPI)
    fig.set_size_inches(TALL_FIGSIZE if width == "full" else HALF_FIGSIZE, forward=True)
    for ax in fig.axes:
        ax.grid(True, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D7DEE8")
        ax.spines["bottom"].set_color("#D7DEE8")
        ax.tick_params(axis="both", labelsize=9)
        if axis_uses_datetime(ax):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for text in legend.get_texts():
                text.set_fontsize(8.5)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


# 中文说明：`axis_uses_datetime`：执行该名称对应的业务计算，并返回调用方所需结果。
def axis_uses_datetime(ax: Any) -> bool:
    for line in ax.lines:
        xdata = line.get_xdata()
        if len(xdata) == 0:
            continue
        values = np.asarray(xdata)
        if np.issubdtype(values.dtype, np.datetime64):
            return True
        first = next((item for item in values if item is not None), None)
        if isinstance(first, (pd.Timestamp, datetime, date)):
            return True
    return False


# 中文说明：`pretty_text`：执行该名称对应的业务计算，并返回调用方所需结果。
def pretty_text(value: Any) -> str:
    return str(fix_mojibake(value))


# 中文说明：`pretty_value`：执行该名称对应的业务计算，并返回调用方所需结果。
def pretty_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return str(fix_mojibake(value))


# 中文说明：`toggle_module`：执行该名称对应的业务计算，并返回调用方所需结果。
def toggle_module(module_key: str, force_off: bool = False) -> None:
    active = list(st.session_state.active_modules)
    if module_key in active:
        active.remove(module_key)
    elif not force_off:
        active.append(module_key)
    st.session_state.active_modules = active


# 中文说明：`render_pdf_export`：渲染研究报告或界面内容。
def render_pdf_export(analyzer: Any) -> None:
    st.markdown("**保存 PDF**")
    default_name = f"factor_report_{analyzer.info_table.get('name', 'factor')}.pdf"
    output_path = st.text_input("PDF 保存路径", value=str(Path.cwd() / default_name))
    if st.button("保存 PDF", use_container_width=True):
        if not st.session_state.active_modules:
            st.warning("请至少展示一个模块后再保存 PDF。")
            return
        with st.spinner("正在生成 PDF..."):
            try:
                saved_path = save_pdf_report(analyzer, output_path)
            except Exception as exc:
                st.error(f"保存 PDF 失败: {exc}")
            else:
                st.success(f"PDF 已保存到: {saved_path}")


# 中文说明：`save_pdf_report`：持久化当前状态。
def save_pdf_report(analyzer: Any, output_path: str) -> Path:
    target = Path(output_path).expanduser()
    if target.suffix.lower() != ".pdf":
        target = target.with_suffix(".pdf")
    target.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(target) as pdf:
        add_table_to_pdf(pdf, analyzer.info_table, "因子基础信息")
        for module_key in st.session_state.active_modules:
            module = MODULE_BY_KEY[module_key]
            for spec in module.outputs:
                value = compute_output(analyzer, module.key, spec)
                if spec.kind in {"table", "info"}:
                    add_table_to_pdf(pdf, value, f"{module.title} - {spec.title}")
                elif spec.kind == "figure":
                    pdf.savefig(value, bbox_inches="tight")
    return target


# 中文说明：`add_table_to_pdf`：执行该名称对应的业务计算，并返回调用方所需结果。
def add_table_to_pdf(pdf: PdfPages, value: Any, title: str) -> None:
    df = normalize_table(fix_table_mojibake(value)).copy()
    df = df.round(6) if all(dtype.kind in "biufc" for dtype in df.dtypes) else df
    fig_height = max(3.2, min(16.0, 1.1 + 0.32 * max(len(df), 1)))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=[str(col) for col in df.columns],
        rowLabels=[str(idx) for idx in df.index],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.24)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
