from pathlib import Path
p=Path('.ppt_build/build_ai.mjs')
s=p.read_text(encoding='utf-8')
replacements={
'Q1整理宏观与风格指标，结合各类因子，为现有时序模型补充市场状态输入。':'Q1统一L2、预期及成长因子的生效时点，构建大小盘、成长价值与风险状态样本。',
'Q2比较有无风格感知的模型，检查不同市场阶段的因子权重与样本外表现。':'Q2借鉴PRIME加入状态输入与门控，对比无状态GRU和分位规则，检验未来因子收益。',
'设置1、3、5日收益预测头，先以等权损失训练，与单一5日标签模型对比。':'设置1、3、5日预测头，各自拟合真实收益，以等权损失训练，对比单一5日标签。',
'5日RankIC与成本后收益改善后，再研究自适应标签权重，另留独立样本检验。':'先检验5日RankIC与成本后收益，优于单任务后再试自适应权重，独立样本留作终检。',
'模型验证：固定基线，保留实验依据':'滚动验证：识别模型的真实增量',
'统一股票池、回测口径和数据切分。训练与验证按标签期限隔离，保留多数据源对照实验。':'Q3分别检验状态门控与多任务的贡献，比较分域RankIC、分组单调性及成本后超额。',
'Q3至Q4交付AI模型对照报告与实验配置，说明增益来源、适用场景及失效条件。':'Q4跟踪通过检验的模拟盘，归档数据口径、模型配置与失效条件，训练验证按标签期隔离。',
'微观结构\\n与预期因子':'L2增强指增\\n特征构造',
'Q1至Q2积累Level2、成长及一致预期因子，统一清洗与检验。\\n以机器学习比较新增因子的预测贡献，剔除冗余信号。':'Q1按时段、订单规模及主动方向拆分委托与成交样本。\\n提取订单流不平衡、交易聚集和冲击后流动性恢复，补充成长与预期因子。',
'风格感知\\n动态配权':'L2增强指增\\nAI筛选组合',
'Q2利用AI模型识别风格状态，研究不同因子的条件权重。\\n对比等权与ICIR配置，检验轮动收益是否来自风格暴露。':'Q2用机器学习检验因子增量，控制行业市值及动量反转的影响。\\n按ICIR和相关性筛选，加入风险模型、换手与容量约束。',
'成本与\\n交易约束':'状态感知配置\\n指标与面板',
'固定行业、个股及换手约束，检查AI信号的成本后收益。\\nQ3将验证通过的组合纳入模拟盘，记录预测与执行偏差。':'Q1建立大小盘、行业轮动、ETF申赎及流动性压力指标库。\\n输出多窗口历史分位，检验对未来因子收益的预测力，区分输入与监控。',
'模型跟踪\\n人工复核':'状态感知配置\\nAI动态权重',
'将风格和流动性指标接入监控面板，跟踪模型衰减。\\nQ3至Q4按月复盘超额与预警，人工决定保留或调整策略。':'Q2比较等权、固定ICIR、分位规则与AI状态门控配置。\\nQ3做成本压力测试，Q4模拟盘复盘超额、回撤和换手，保留人工调整。'
}
for a,b in replacements.items():s=s.replace(a,b)
Path('.ppt_build/build_details.mjs').write_text(s,encoding='utf-8')
s=Path('.ppt_build/finish_ai.ps1').read_text(encoding='utf-8-sig')
s=s.replace('2027年目标：验证AI风格感知与多任务学习的模型增益。','2027年目标：以AI风格感知与多期限监督升级2项模型组件。').replace('2027年目标：形成AI驱动的因子轮动与模拟跟踪方案。','2027年目标：储备L2增强指增与AI状态感知配置两项策略。')
s=s.replace("@('因子输入','AI轮动','组合回测','监控复盘')","@('L2特征','AI选股','市场状态','AI配权')")
s=s.replace(' Set-Text ($s.Shapes.Item($headingNames[$i])) $heading 14 -1',' Set-Text ($s.Shapes.Item($headingNames[$i])) $heading 14 -1\n $s.Shapes.Item($headingNames[$i]).TextFrame.TextRange.ParagraphFormat.Alignment=2')
s=s.replace('2027年度计划_许佳宜_AI重点与差异排版.pptx','2027年度计划_许佳宜_AI研究与策略_详细版.pptx')
extra='参考用户提供的2027年度经营计划_量化研究_结合月报周报_更新版.pptx。策略一对行业市值、动量与反转做残差化或中性控制。策略二输出1、3、6、12、36个月历史分位，候选因子含动量、反转、价值、成长及低波。统一风险和交易约束后比较各类配置，报告超额、回撤、换手及容量。季度节点为拟定计划。'
s=s.replace('模拟盘仅纳入通过样本外与成本检验的原型，不将待研究功能写作已有成果。','模拟盘仅纳入通过样本外与成本检验的原型，不将待研究功能写作已有成果。'+extra)
Path('.ppt_build/finish_details.ps1').write_text(s,encoding='utf-8-sig')
v=Path('.ppt_build/validate_ai.mjs').read_text(encoding='utf-8').replace('2027年度计划_许佳宜_AI重点与差异排版.pptx','2027年度计划_许佳宜_AI研究与策略_详细版.pptx').replace('ai-candidate.pptx','details-candidate.pptx').replace('ai-validation.json','details-validation.json')
Path('.ppt_build/validate_details.mjs').write_text(v,encoding='utf-8')
