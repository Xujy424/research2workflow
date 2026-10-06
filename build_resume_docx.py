import html
import shutil
import zipfile
from pathlib import Path


SRC = Path(r"c:\Users\xujiayi\Desktop\个人材料\简历综合\三类简历\许佳宜个人简历-量化研究支持岗.docx")
OUT = Path.cwd() / "许佳宜个人简历-量化研究员-更新版.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def esc(text):
    return html.escape(text, quote=False)


def r(text, bold=False, size=21, font="SimSun"):
    b = "<w:b/>" if bold else ""
    return (
        "<w:r><w:rPr>"
        f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="{font}"/>'
        f"{b}<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>"
        "</w:rPr>"
        f"<w:t xml:space=\"preserve\">{esc(text)}</w:t></w:r>"
    )


def p(text="", bold=False, size=21, align=None, before=0, after=0, line=240, indent=None, hanging=None):
    jc = f'<w:jc w:val="{align}"/>' if align else ""
    ind = ""
    if indent is not None:
        attrs = f'w:left="{indent}"'
        if hanging is not None:
            attrs += f' w:hanging="{hanging}"'
        ind = f"<w:ind {attrs}/>"
    return (
        "<w:p><w:pPr>"
        f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>'
        f"{jc}{ind}</w:pPr>{r(text, bold=bold, size=size)}</w:p>"
    )


def bullet(text, level=0):
    left = 360 + level * 240
    return p("•  " + text, size=20, after=22, line=232, indent=left, hanging=240)


def section(title):
    return (
        "<w:p><w:pPr>"
        '<w:spacing w:before="230" w:after="34"/>'
        '<w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" w:color="000000"/></w:pBdr>'
        "</w:pPr>"
        f"{r(title, bold=True, size=23)}"
        "</w:p>"
    )


def twips(width_inches):
    return int(width_inches * 1440)


def row(left_lines, right_lines=None):
    right_lines = right_lines or []
    def cell(lines, width, align=None):
        ps = []
        for i, (text, bold, size) in enumerate(lines):
            ps.append(p(text, bold=bold, size=size, align=align, after=0, line=220))
        return (
            "<w:tc><w:tcPr>"
            f'<w:tcW w:w="{width}" w:type="dxa"/>'
            '<w:tcBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/></w:tcBorders>'
            "</w:tcPr>" + "".join(ps) + "</w:tc>"
        )
    return (
        "<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders><w:top w:val=\"nil\"/><w:left w:val=\"nil\"/><w:bottom w:val=\"nil\"/><w:right w:val=\"nil\"/><w:insideH w:val=\"nil\"/><w:insideV w:val=\"nil\"/></w:tblBorders>"
        "</w:tblPr><w:tblGrid><w:gridCol w:w=\"7200\"/><w:gridCol w:w=\"2160\"/></w:tblGrid><w:tr>"
        + cell(left_lines, 7200)
        + cell(right_lines, 2160, "right")
        + "</w:tr></w:tbl>"
    )


def document_xml():
    body = []
    body.append(p("许佳宜", bold=True, size=30, align="center", after=20, line=260))
    body.append(p("电话：（+86）136-1297-9211 | 邮箱：quant_jy@mail.ustc.edu.cn", size=20, align="center", after=80))

    body.append(section("教育背景"))
    body.append(row(
        [("中国科学技术大学（保研）", True, 22), ("管理学院 MF 中心 | 金融硕士（量化金融）", False, 20)],
        [("", True, 21), ("2022.9 - 2025.6", False, 20)],
    ))
    body.append(bullet("相关课程：深度学习、金融衍生工具、随机分析、数据结构与数据库、数值方法、数据挖掘、金融数据分析、应用统计方法、金融经济学、金融风险管理、证券投资分析"))
    body.append(row(
        [("暨南大学", True, 22), ("国际商学院 | 金融工程", False, 20)],
        [("", True, 21), ("2018.9 - 2022.6", False, 20)],
    ))
    body.append(bullet("相关课程：高等数学、线性代数、概率论与数理统计、金融工程、风险管理、微观经济学、宏观经济学、计量经济学、公司金融、财务管理"))

    body.append(section("工作经历"))
    body.append(p("浙商基金管理有限公司", bold=True, size=23, after=40, line=220))
    body.append(p("多因子选股与组合研究", bold=True, size=21, line=220, indent=120))
    body.append(bullet("因子研究：围绕量价结构、资金流向与分析师预期构建选股因子，统一去极值、横截面标准化及行业市值中性化流程；结合 IC、RankIC、分组收益与回撤检验因子预测能力及收益稳定性。"))
    body.append(bullet("策略评估：搭建单因子与事件驱动回测框架，支持多空组合、基准对冲及指数增强场景；纳入信号滞后、可交易性、调仓频率与交易成本，并通过成交参与率和冲击成本模拟评估策略容量。"))
    body.append(bullet("多因子建模：按经济含义对因子分类，运用相关性筛选、聚类及正交化控制信息冗余，结合 PCA、PLS、ICIR 加权与岭回归等方法合成 Alpha 信号，衔接选股研究与组合构建。"))
    body.append(bullet("组合与研究工程：基于因子风险模型和凸优化生成目标权重，将行业暴露、个股权重、换手与流动性约束纳入优化；建设矩阵化数据存储、增量更新、因子版本管理及监控流程，支持研究结果复现与持续跟踪。"))

    body.append(p("端到端深度学习选股模型研究", bold=True, size=21, before=70, line=220, indent=120))
    body.append(bullet("模型与训练：研究 GRU、Transformer、TimesNet 及图注意力网络等架构，搭建数据处理、模型训练与预测评估流程，探索监督学习、自监督预训练及多阶段训练在股票收益预测中的应用。"))
    body.append(bullet("模型拓展：融合 Barra 风险因子与量价特征构建 Risk-Attention，测试集 IC 较 GRU 提升 6%-7%；研究自适应 GCN，10 日 IC 达 0.14；结合 DWT 多尺度特征与 SAC 探索交易策略，并尝试逆波兰表达式与 PPO 因子挖掘。"))
    body.append(bullet("损失函数设计：比较 MSE、Pearson IC 与可微 RankIC 训练目标，尝试按指数成分和行业划分样本域、加权聚合域内 RankIC，以研究不同股票群体的预测差异；探索引入时序约束的排序损失。"))
    body.append(bullet("标签与实验评估：构建 1、5、10、20 日前瞻收益、横截面标准化收益及行业市值中性化收益标签，探索基于历史收益相关性的邻近股票组内标准化标签；结合滚动训练、交叉验证与 Bagging，以样本外 IC、RankIC 及其信息比率比较预测能力与稳定性。"))

    body.append(section("实习经历"))
    jobs = [
        (
            "上海洛书投资管理有限公司", "期货低频部 | 量化研究实习生", "上海", "2023.9 - 2024.1",
            [
                "使用计量模型检验棕榈油主要产地降雨量、产量与期货价格之间的关系，并将预报产量或降雨量生成的交易信号用于回测，验证关系稳定性。",
                "整合不同数据商的厂库数据预测螺纹钢、线材表观需求，分析需求预测信号对价格的回测表现。",
                "运用格兰杰因果检验研究化工产业链中冰醋酸对上游甲醇和下游 PTA 的领先-滞后效应。"
            ],
        ),
        (
            "长城证券股份有限公司", "资产管理部 | 量化研究员助理", "深圳", "2021.11 - 2022.2",
            [
                "协助检验 WorldQuant101 量价因子在选股中的表现，梳理因子构建中的量价含义，并依据分组收益单调性筛选因子以服务指数增强策略。",
                "参考券商研报，构建行业景气度指标，为行业轮动策略提供研究支持。"
            ],
        ),
    ]
    for company, role, loc, dates, bullets in jobs:
        body.append(row([(company, True, 22), (role, False, 20)], [("", True, 21), (dates, False, 20)]))
        for item in bullets:
            body.append(bullet(item))

    body.append(section("研究经历"))
    body.append(row(
        [("工作论文《SGLNet: A self-supervised framework for stock price prediction》", True, 21), ("自监督股票价格预测框架", False, 20)],
        [("", True, 21), ("2024.8", False, 20)],
    ))
    body.append(bullet("通过构造全局与局部对比学习损失，提出自监督学习框架，从原始股票价格序列直接提取表征信息，提升模型自主学习数据表示的能力，并为下游预测器或分类器提供更丰富的信息。"))

    body.append(section("专业技能"))
    body.append(bullet("语言：CET-4 612/710，CET-6 608/710"))
    body.append(bullet("编程与工具：Python（sklearn、PyTorch 等）、SQL、C++；熟悉 VN.PY 自动化交易开源框架"))
    body.append(bullet("证书与知识体系：通过 CFA 二级、FRM 待持证、CQF 持证"))

    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="360" w:footer="360" w:gutter="0"/>'
        '<w:cols w:space="720"/><w:docGrid w:linePitch="312"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W}"><w:body>' + "".join(body) + sect + "</w:body></w:document>"
    )


def main():
    shutil.copyfile(SRC, OUT)
    xml = document_xml().encode("utf-8")
    tmp = OUT.with_suffix(".tmp.docx")
    with zipfile.ZipFile(OUT, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "word/document.xml":
                zout.writestr(item, xml)
            else:
                zout.writestr(item, zin.read(item.filename))
    tmp.replace(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
