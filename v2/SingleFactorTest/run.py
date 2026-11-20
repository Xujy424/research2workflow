"""浣滆€咃細xujiayi | 鍒涘缓鏃ユ湡锛?026-11-11
鍗曞洜瀛愭姤鍛婄粺涓€鍏ュ彛锛歠actor_pool璇诲彇鈫掗澶勭悊鈫掍緷Excel椤哄簭璁＄畻鈫扙Charts/琛ㄦ牸鍗忚銆?鏀寔鍛戒护琛屽畬鏁存姤鍛婂強web.py澶嶇敤鍚屼竴Runner鐨勫眬閮ㄥ弬鏁版洿鏂般€?"""
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
SECTION_NAMES = dict(zip(SECTION_ORDER, ("鍩烘湰琛ㄧ幇", "鍥犲瓙鍊煎垎甯?, "鍒嗙粍鏀剁泭", "IC妫€楠?, "鍒嗗煙琛ㄧ幇", "缁勫悎璺熻釜", "瀹归噺娴嬭瘯")))
STYLE_NAMES = ("beta", "btop", "size", "nonlinear_size", "momentom", "residual_vol", "liquidity", "leverage", "growth1", "earnings_yield")
INDEX_NAMES = ("hs300", "zz500", "a500", "zz1000", "zz2000")


def list_factors(root=None):
    """鍒楀嚭鏈湴鍥犲瓙锛汸arams: 鍚玸tock/axis鐨勬暟鎹牴锛汻eturn: 鎸夊悕绉版帓搴忕殑鍏冩暟鎹垪琛ㄣ€?""
    folder = Path(root or os.environ.get("RESEARCH_DATA_ROOT", "Z:/")) / "stock" / "factor_pool"
    if not folder.is_dir():
        raise FileNotFoundError(f"鍥犲瓙鐩綍涓嶅瓨鍦? {folder}")
    result = []
    for p in sorted(folder.rglob("*.bin")):
        # 鍚嶇О鏉ヨ嚜鐩綍鏋氫妇锛岃€岄潪鐢ㄦ埛鎷兼帴璺緞銆傞伩鍏嶅姣忎釜缃戠粶鐩樻枃浠堕噸澶峳esolve銆?        stat = p.stat()
        result.append({"name": p.relative_to(folder).with_suffix("").as_posix(),
                       "bytes": stat.st_size, "modified": stat.st_mtime})
    return result


class ReportRunner:
    """鍗曞伐浣滅嚎绋嬩娇鐢ㄧ殑鎶ュ憡涓婁笅鏂囷紝鏈€澶氱紦瀛樹竴涓洜瀛愮殑鐭╅樀銆?
    Params: root: 鏁版嵁鏍癸紱Return: Runner銆傜綉椤甸€氳繃鍗曞伐浣滈槦鍒椾覆琛岃闂紝閬垮厤鍏变韩瀹炰緥绔炴€併€?    """
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
        """鍔犺浇鎴栧鐢ㄥ洜瀛愶紱Params: 鏄惧紡鍥犲瓙淇℃伅/杩涘害鍥炶皟锛汻eturn: FactorBase銆?""
        name = str(factor_name).replace("\\", "/").removeprefix("factor_pool/").removesuffix(".bin")
        available = {x["name"]: x for x in list_factors(self.root)}
        if name not in available:
            raise ValueError("璇烽€夋嫨factor_pool鐩綍涓瓨鍦ㄧ殑鍥犲瓙鍚嶇О")
        key = (name, available[name]["modified"], construction, normalization, transform,
               direction, neutralize_industry, neutralize_size, start, end)
        if self.key == key and self.base is not None:
            return self.base
        self.base, self.key, self.cache, self.notes = None, None, {}, []
        progress("loading", "璇诲彇鏈湴鍥犲瓙涓庡競鍦烘暟鎹?, None)
        if self._source is None:
            from v2.GetData import DataPool
            self._source = DataPool(self.root, asset="stock")
        b = FactorBase(f"factor_pool/{name}", construction, normalization, transform,
                       neutralize_industry, neutralize_size, direction=direction,
                       start=start, end=end, root=self.root, data=self._source)
        b.load()
        # 鏈湴杞村彲鑳藉寘鍚湭缁存姢鐨勬湭鏉ヨ鎯呰锛屾姤鍛婃湯鏃ヤ娇鐢ㄦ湁鏀剁泭鏁版嵁鐨勬渶鍚庝竴琛屻€?        valid = b.raw_alpha.notna().any(axis=1) & b.pct.notna().any(axis=1)
        if not valid.any():
            raise ValueError("鍥犲瓙涓庢棩鏀剁泭娌℃湁鍏卞悓鏈夋晥鏍锋湰")
        progress("preprocess", "澶勭悊鍥犲瓙鍊煎苟璁＄畻褰撴棩涓€у寲", None)
        b.process_alpha()
        effective = b.alpha.notna().any(axis=1)
        if not effective.any():
            raise ValueError("鍥犲瓙鍦ㄥ綋鍓嶈偂绁ㄦ睜鍐呮病鏈夊彲鍒嗙粍鐨勬湁鏁堟埅闈?)
        first, last = b.dates[effective][0], b.dates[effective][-1]
        if (last - first).days < 365 * 5:
            self.notes.append(f"鏈夋晥鍥犲瓙鏍锋湰涓嶈冻5骞达細{first:%Y-%m-%d} 鑷?{last:%Y-%m-%d}锛岀己澶卞勾浠戒繚鐣欑┖鍊?)
        self.base, self.key = b, key
        return b

    def ensure_domain(self, domain):
        """鎸夐渶鍔犺浇娉ㄥ唽鎸囨暟锛汸arams: 鍩熷悕锛汻eturn: None銆?""
        if domain.startswith("index:"):
            name = domain.split(":", 1)[1]
            if name not in INDEX_NAMES:
                raise ValueError("鏈煡鎸囨暟")
            if name not in self.base.domain_weights:
                self.base.load(indices=(name,))

    def render(self, section, options):
        """鍗曟ā鍧楅噸绠?缂撳瓨锛汸arams: 妯″潡鍙婂弬鏁帮紱Return: 闈㈡澘缁撴灉瀛楀吀銆?""
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
                    self.base.load(styles=missing)
        if section in ("portfolio", "capacity") and options.get("benchmark_method") == "index":
            self.ensure_domain(f"index:{options['benchmark_index']}")
        func = {"overall": reporting.overall, "distribution": reporting.value_distribution,
                "groups": reporting.groups, "ic": reporting.ic, "domains": reporting.domain_report,
                "portfolio": reporting.portfolio, "capacity": reporting.capacity_report}[section]
        result = func(self.base, options)
        self.cache[key] = result
        # 鍥捐〃鍗忚浣撶Н灏忥紱鏈€澶氫繚鐣?4缁勫弬鏁扮粨鏋滐紝鐭╅樀濮嬬粓鍙湁褰撳墠鍥犲瓙銆?        if len(self.cache) > 24:
            self.cache.pop(next(iter(self.cache)))
        return result


def _options(section, construction, normalization, long_group, short_group, overrides):
    """妯″潡鍙傛暟鏍￠獙涓庢樉寮忛粯璁ゅ€硷紱Params: 妯″潡/鏋勫缓閰嶇疆/瑕嗙洊椤癸紱Return: 鍙傛暟瀛楀吀銆?""
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
        raise ValueError(f"鏈煡{section}鍙傛暟: {sorted(unknown)}")
    options = {**defaults[section], **overrides}
    if "construction" in options:
        side = options["construction"]
        if side not in ("long", "short", "longshort"):
            raise ValueError("鏋勫缓鏂瑰紡鏃犳晥")
        lg, sg = int(options.pop("long_group")), int(options.pop("short_group"))
        if not 1 <= lg <= 10 or not 1 <= sg <= 10 or (side == "longshort" and lg == sg):
            raise ValueError("缁勫彿椤诲湪1鈥?0锛屽绌虹粍涓嶅彲鐩稿悓")
        options["group_numbers"] = ((lg,), (sg,)) if side == "longshort" else (lg,) if side == "long" else (sg,)
    if "years" in options and options["years"] not in (1, 3, 5):
        raise ValueError("骞撮檺椤讳负1/3/5")
    if "horizon" in options and options["horizon"] not in (1, 3, 5, 10, 20):
        raise ValueError("鏀剁泭鏈熼檺椤讳负1/3/5/10/20鏃?)
    if "rebalance_days" in options and options["rebalance_days"] not in (1, 3, 5, 10, 20):
        raise ValueError("璋冧粨棰戠巼椤讳负1/3/5/10/20涓氦鏄撴棩")
    if "capitals" in options and (not 1 <= len(options["capitals"]) <= 8 or any(not 0 < float(c) <= 1e12 for c in options["capitals"])):
        raise ValueError("瀹归噺妗ｄ綅椤讳负1鈥?涓鏁帮紝鍗曟。涓嶈秴杩?涓囦嚎鍏?)
    return options


def run(factor_name, construction="longshort", normalization="equal", transform="asinh",
        direction=1, neutralize_industry=True, neutralize_size=True, long_group=10,
        short_group=1, start=None, end=None, description="", root=None, output=None,
        sections=None, section_options=None, runner=None, progress=None):
    """鎸夋帓鐗堥『搴忔墽琛岀爺绌舵姤鍛婏紝缃戦〉涓庡懡浠よ鍏辩敤姝ゅ嚱鏁般€?
    Params: factor_name: stock/factor_pool鍐呭悕绉帮紱鍏朵綑鏄惧紡鍙傛暟涓哄洜瀛愪俊鎭紱
        sections: 閫夌畻妯″潡锛堥粯璁ゅ叏閮級锛泂ection_options: 鍐呴儴鍥捐〃鍙傛暟鏄犲皠锛?        runner: 鍙鐢ㄤ笂涓嬫枃锛沺rogress(stage,message,payload): 澧為噺鍥炶皟锛沷utput: 鍙€塉SON杈撳嚭璺緞銆?    Return: {metadata, sections, errors}锛岀己澶卞€间负JSON null锛屼笉鐢熸垚鍋囨暟鎹～鍏呭浘琛ㄣ€?    """
    started = perf_counter()
    progress = progress or (lambda stage, message, payload: None)
    runner = runner or ReportRunner(root)
    selected = tuple(sections) if sections is not None else SECTION_ORDER
    if not selected or not set(selected).issubset(SECTION_ORDER):
        raise ValueError("璇锋眰妯″潡鏃犳晥")
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
    progress("metadata", "鍥犲瓙鏁版嵁鍑嗗瀹屾垚", metadata)
    for section in SECTION_ORDER:
        if section not in selected:
            continue
        progress(section, f"姝ｅ湪璁＄畻{SECTION_NAMES[section]}", None)
        try:
            overrides = (section_options or {}).get(section, {})
            if section == "capacity":
                overrides = {**(section_options or {}).get("portfolio", {}), **overrides}
            options = _options(section, construction, normalization, long_group, short_group, overrides)
            payload = runner.render(section, options)
            report["sections"][section] = payload
            progress(section, f"{SECTION_NAMES[section]}宸插畬鎴?, payload)
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
    parser = argparse.ArgumentParser(description="浠巗tock/factor_pool鐢熸垚鍗曞洜瀛愭姤鍛?)
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
    print(f"鎶ュ憡宸蹭繚瀛橈紱妯″潡閿欒鏁? {len(result['errors'])}")


