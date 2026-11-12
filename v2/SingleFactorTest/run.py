"""作者：xujiayi | 创建日期：2026-11-11
单因子报告统一入口：factor_pool读取→预处理→依Excel顺序计算→ECharts/表格协议。
支持命令行完整报告及web.py复用同一Runner的局部参数更新。
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import warnings
import numpy as np
import pandas as pd

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "v2.SingleFactorTest"

from .factorbase import FactorBase
from . import reporting

SECTION_ORDER = ("overall", "distribution", "groups", "ic", "domains", "portfolio", "capacity")
SECTION_NAMES = dict(zip(SECTION_ORDER, ("基本表现", "因子值分布", "分组收益", "IC检验", "分域表现", "组合跟踪", "容量测试")))
STYLE_NAMES = ("beta", "btop", "size", "nonlinear_size", "momentom", "residual_vol", "liquidity", "leverage", "growth1", "earnings_yield")
INDEX_NAMES = ("hs300", "zz500", "a500", "zz1000", "zz2000")


def list_factors(root=None):
    """列出本地因子；Params: 含stock/axis的数据根；Return: 按名称排序的元数据列表。"""
    folder = Path(root or os.environ.get("RESEARCH_DATA_ROOT", "Z:/")) / "stock" / "factor_pool"
    if not folder.is_dir():
        raise FileNotFoundError(f"因子目录不存在: {folder}")
    result = []
    for p in sorted(folder.rglob("*.bin")):
        # 名称来自目录枚举，而非用户拼接路径。避免对每个网络盘文件重复resolve。
        stat = p.stat()
        result.append({"name": p.relative_to(folder).with_suffix("").as_posix(),
                       "bytes": stat.st_size, "modified": stat.st_mtime})
    return result


class ReportRunner:
    """单工作线程使用的报告上下文，最多缓存一个因子的矩阵。

    Params: root: 数据根；Return: Runner。网页通过单工作队列串行访问，避免共享实例竞态。
    """
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("RESEARCH_DATA_ROOT", "Z:/"))
        self.base = None
        self.key = None
        self.cache = {}
        self.notes = []
        self._market = None
        self._source = None

    def prepare(self, factor_name, construction, normalization, transform, direction,
                neutralize_industry, neutralize_size, start, end, progress):
        """加载或复用因子；Params: 显式因子信息/进度回调；Return: FactorBase。"""
        name = str(factor_name).replace("\\", "/").removeprefix("factor_pool/").removesuffix(".bin")
        available = {x["name"]: x for x in list_factors(self.root)}
        if name not in available:
            raise ValueError("请选择factor_pool目录中存在的因子名称")
        key = (name, available[name]["modified"], construction, normalization, transform,
               direction, neutralize_industry, neutralize_size, start, end)
        if self.key == key and self.base is not None:
            return self.base
        self.base, self.key, self.cache, self.notes = None, None, {}, []
        progress("loading", "读取本地因子与市场数据", None)
        from .dataio import ReportDataPool
        if self._source is None:
            self._source = ReportDataPool(self.root, asset="stock")
        b = FactorBase(f"factor_pool/{name}", construction, normalization, transform,
                       neutralize_industry, neutralize_size, direction=direction,
                       start=start, end=end, root=self.root, data=self._source)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if self._market is not None and self._market[0] == (start, end):
                _, pct, fields, notes = self._market
                progress("loading", "复用市场数据，读取所选因子", None)
                alpha = pd.DataFrame(self._source.read(b.factor_name, pct.index[-1], pct.index[0]), pct.index, pct.columns)
                b.load(alpha, pct, fields, strict_history=False)
                self.notes.extend(notes)
            else:
                b.load(strict_history=False, progress=lambda msg: progress("loading", msg, None))
        self.notes = [str(x.message) for x in caught]
        self._market = ((start, end), b.pct, b.fields, self.notes)
        # 本地轴可能包含未维护的未来行情行，报告末日使用有收益数据的最后一行。
        valid = b.raw_alpha.notna().any(axis=1) & b.pct.notna().any(axis=1)
        if not valid.any():
            raise ValueError("因子与日收益没有共同有效样本")
        progress("preprocess", "处理因子值并计算当日中性化", None)
        b.process_alpha()
        effective = b.alpha.notna().any(axis=1)
        if not effective.any():
            raise ValueError("因子在当前股票池内没有可分组的有效截面")
        first, last = b.dates[effective][0], b.dates[effective][-1]
        if (last - first).days < 365 * 5:
            self.notes.append(f"有效因子样本不足5年：{first:%Y-%m-%d} 至 {last:%Y-%m-%d}，缺失年份保留空值")
        self.base, self.key = b, key
        return b

    def ensure_domain(self, domain):
        """按需加载注册指数；Params: 域名；Return: None。"""
        if domain.startswith("index:"):
            name = domain.split(":", 1)[1]
            if name not in INDEX_NAMES:
                raise ValueError("未知指数")
            if name not in self.base.domain_weights:
                self.base.load_domains((name,), ())

    def render(self, section, options):
        """单模块重算/缓存；Params: 模块及参数；Return: 面板结果字典。"""
        key = json.dumps([section, options], sort_keys=True, ensure_ascii=False)
        if key in self.cache:
            return self.cache[key]
        self.ensure_domain(options.get("domain", "universe"))
        if section == "domains":
            dtype = options.get("domain_type", "industry")
            if dtype == "index":
                for name in INDEX_NAMES:
                    if (self.root / "stock/index/weight" / f"{name}_weight.bin").is_file():
                        self.ensure_domain(f"index:{name}")
            if dtype == "barra":
                missing = tuple(s for s in STYLE_NAMES if s not in self.base.barra)
                if missing:
                    self.base.load_domains((), missing)
        if section in ("portfolio", "capacity") and options.get("benchmark_method") == "index":
            self.ensure_domain(f"index:{options['benchmark_index']}")
        func = {"overall": reporting.overall, "distribution": reporting.value_distribution,
                "groups": reporting.groups, "ic": reporting.ic, "domains": reporting.domain_report,
                "portfolio": reporting.portfolio, "capacity": reporting.capacity_report}[section]
        result = func(self.base, options)
        self.cache[key] = result
        # 图表协议体积小；最多保留24组参数结果，矩阵始终只有当前因子。
        if len(self.cache) > 24:
            self.cache.pop(next(iter(self.cache)))
        return result


def _options(section, construction, normalization, long_group, short_group, overrides):
    """模块参数校验与显式默认值；Params: 模块/构建配置/覆盖项；Return: 参数字典。"""
    common = {"construction": construction, "long_group": long_group, "short_group": short_group, "domain": "universe"}
    defaults = {
        "overall": common, "distribution": {"processed": True, "year": "Overall"},
        "groups": {"domain": "universe"},
        "ic": {"horizon": 1, "kind": "IC", "years": 5},
        "domains": {**common, "domain_type": "industry", "years": 5, "metric": "exposure"},
        "portfolio": {**common, "normalization": normalization, "rebalance_days": 5,
                      "turnover_limit": 1., "stock_deviation": None, "stock_cap": .05,
                      "industry_deviation": None, "commission": .0003, "stamp_tax": .0005,
                      "execution_price": "vwap", "capital": 1e8, "max_participation": .1,
                      "impact_coefficient": .001, "borrow_rate": 0., "lot_size": 100,
                      "benchmark_enhanced": True, "active_gross": .2,
                      "benchmark_method": "index", "benchmark_index": "hs300"},
    }
    defaults["capacity"] = {**defaults["portfolio"], "capitals": (1e7, 5e7, 1e8, 5e8)}
    unknown = set(overrides) - set(defaults[section])
    if unknown:
        raise ValueError(f"未知{section}参数: {sorted(unknown)}")
    options = {**defaults[section], **overrides}
    if "construction" in options:
        side = options["construction"]
        if side not in ("long", "short", "longshort"):
            raise ValueError("构建方式无效")
        lg, sg = int(options.pop("long_group")), int(options.pop("short_group"))
        if not 1 <= lg <= 10 or not 1 <= sg <= 10 or (side == "longshort" and lg == sg):
            raise ValueError("组号须在1—10，多空组不可相同")
        options["group_numbers"] = ((lg,), (sg,)) if side == "longshort" else (lg,) if side == "long" else (sg,)
    if "years" in options and options["years"] not in (1, 3, 5):
        raise ValueError("年限须为1/3/5")
    if "horizon" in options and options["horizon"] not in (1, 3, 5, 10, 20):
        raise ValueError("收益期限须为1/3/5/10/20日")
    if "rebalance_days" in options and options["rebalance_days"] not in (1, 3, 5, 10, 20):
        raise ValueError("调仓频率须为1/3/5/10/20个交易日")
    if "capitals" in options and (not 1 <= len(options["capitals"]) <= 8 or any(not 0 < float(c) <= 1e12 for c in options["capitals"])):
        raise ValueError("容量档位须为1—8个正数，单档不超过1万亿元")
    return options


def run(factor_name, construction="longshort", normalization="equal", transform="asinh",
        direction=1, neutralize_industry=True, neutralize_size=True, long_group=10,
        short_group=1, start=None, end=None, description="", root=None, output=None,
        sections=None, section_options=None, runner=None, progress=None):
    """按排版顺序执行研究报告，网页与命令行共用此函数。

    Params: factor_name: stock/factor_pool内名称；其余显式参数为因子信息；
        sections: 选算模块（默认全部）；section_options: 内部图表参数映射；
        runner: 可复用上下文；progress(stage,message,payload): 增量回调；output: 可选JSON输出路径。
    Return: {metadata, sections, errors}，缺失值为JSON null，不生成假数据填充图表。
    """
    started = perf_counter()
    progress = progress or (lambda stage, message, payload: None)
    runner = runner or ReportRunner(root)
    selected = tuple(sections) if sections is not None else SECTION_ORDER
    if not selected or not set(selected).issubset(SECTION_ORDER):
        raise ValueError("请求模块无效")
    b = runner.prepare(factor_name, construction, normalization, transform, direction,
                       neutralize_industry, neutralize_size, start, end, progress)
    effective = b.alpha.notna().any(axis=1)
    metadata = {"factor": runner.key[0], "construction": construction, "normalization": normalization,
                "transform": transform, "direction": direction, "description": description,
                "neutralize_industry": neutralize_industry, "neutralize_size": neutralize_size,
                "start": b.dates[0].strftime("%Y-%m-%d"), "end": b.dates[-1].strftime("%Y-%m-%d"),
                "effective_start": b.dates[effective][0].strftime("%Y-%m-%d"),
                "effective_end": b.dates[effective][-1].strftime("%Y-%m-%d"),
                "stocks": len(b.ticks), "days": len(b.dates), "notes": runner.notes,
                "years": list(range(b.dates[-1].year - 5, b.dates[-1].year + 1))}
    report = {"metadata": metadata, "sections": {}, "errors": {}}
    progress("metadata", "因子数据准备完成", metadata)
    for section in SECTION_ORDER:
        if section not in selected:
            continue
        progress(section, f"正在计算{SECTION_NAMES[section]}", None)
        try:
            overrides = (section_options or {}).get(section, {})
            if section == "capacity":
                overrides = {**(section_options or {}).get("portfolio", {}), **overrides}
            options = _options(section, construction, normalization, long_group, short_group, overrides)
            payload = runner.render(section, options)
            report["sections"][section] = payload
            progress(section, f"{SECTION_NAMES[section]}已完成", payload)
        except (ValueError, KeyError, FileNotFoundError, PermissionError) as error:
            report["errors"][section] = str(error)
            progress(section, str(error), {"error": str(error)})
    report["elapsed_seconds"] = round(perf_counter() - started, 2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从stock/factor_pool生成单因子报告")
    parser.add_argument("factor_name")
    parser.add_argument("--root", default=None)
    parser.add_argument("--construction", choices=("long", "short", "longshort"), default="longshort")
    parser.add_argument("--normalization", choices=("equal", "market_cap", "factor"), default="equal")
    parser.add_argument("--transform", choices=("asinh", "softsign", "piecewise", "rank_gaussian", "yeo_johnson", "none"), default="asinh")
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output", default="artifacts/single_factor/report.json")
    args = vars(parser.parse_args())
    result = run(**args, progress=lambda stage, message, payload: print(message, flush=True))
    print(f"报告已保存；模块错误数: {len(result['errors'])}")
