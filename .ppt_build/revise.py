from pathlib import Path
p=Path('.ppt_build/finish.ps1')
s=p.read_text(encoding='utf-8-sig')
s=s.replace('业务目标：完善公告事件模型与文本因子的研究验证。','2027年目标：验证风格感知与多期限预测的模型增益。').replace('业务目标：完成多因子选股与风险控制的策略验证。','2027年目标：形成选股因子、风格轮动与监控的策略原型。')
s=s.replace("@('事件样本库','模型对照检验','因子持续跟踪')","@('风格感知','多期限预测','统一验证')").replace("@('多因子选股','风险约束组合','模拟盘复盘')","@('选股因子','风格轮动','监控与复盘')")
s=s.replace('2027年度计划_许佳宜_新技术与新策略.pptx','2027年度计划_许佳宜_结合月报周报修订.pptx')
start=s.index(' $notes=');end=s.index('\n $s.NotesPage',start)
s=s[:start]+""" $notes=if($j -eq 1){'依据用户补充的本周周报及月报截图。PRIME风格感知与标签期限为学习和探索内容，未视为验证通过。2027年季度节点为拟定安排。多任务等权损失为首轮基线，自适应权重仅在5日RankIC和成本后收益改善后研究。论文题名由用户提供，本页未引用其未经核验的实验结论。'}else{'依据用户提供的月报截图：微观结构与成长因子持续积累，多数据源固定参数和分域损失训练测试已开展，Level2及一致预期因子、市场监控面板、风格研判优化仍在推进。2027年计划不沿用月报进度百分比作为年度完成率，模拟盘仅纳入验证通过的原型。'}"""+s[end:]
Path('.ppt_build/finish_revision.ps1').write_text(s,encoding='utf-8-sig')
v=Path('.ppt_build/validate.mjs').read_text(encoding='utf-8').replace('2027年度计划_许佳宜_新技术与新策略.pptx','2027年度计划_许佳宜_结合月报周报修订.pptx').replace('validated-candidate.pptx','revision-candidate.pptx').replace('validation.json','revision-validation.json')
Path('.ppt_build/validate_revision.mjs').write_text(v,encoding='utf-8')
