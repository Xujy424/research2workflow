import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const base='C:/Users/xujiayi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const req=createRequire(base+'/runtime.cjs');
const {Presentation,PresentationFile}=await import(pathToFileURL(req.resolve('@oai/artifact-tool')).href);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const pages=[[
['风格感知模型：检验PRIME思路的适用性',true],
['Q1建立宏观、风格与交易状态指标，结合不同类别因子，形成模型输入与基线。',false],
['Q2对比有无风格感知的模型，按市场阶段检查因子权重变化及样本外增益。',false],
['多期限预测：先做共享GRU与三个预测头',true],
['分别预测未来1、3、5日收益，以真实收益监督，先将三个任务损失等权求和。',false],
['Q2对比单一5日标签，独立测试集的5日RankIC和成本后收益改善后，再试自适应权重。',false],
['验证流程：统一回测口径与实验记录',true],
['承接多数据源、分域损失训练测试，固定股票池与切分，训练及验证集之间留出标签隔离期。',false],
['Q3—Q4交付模型对照报告与可复现实验配置，明确有效场景及失效条件。',false]
],[
['选股因子：积累Level2与一致预期信号',true],
['Q1—Q2持续研究订单流、买卖盘深度及一致预期变化，结合已有成长因子检验增量。',false],
['扩展单因子回测，统一检查分组收益、衰减与换手，将成本后仍有效的信号纳入候选池。',false],
['因子轮动：比较固定配置与风格条件配置',true],
['Q2以等权、ICIR为对照，利用风格研判指标与模型输出调整因子权重。',false],
['加入行业、个股和换手约束，按市场阶段比较超额与回撤，Q3筛选模拟盘原型。',false],
['市场监控：支持策略跟踪与人工研判',true],
['Q1完成监控基线，将风格、波动及流动性指标接入面板，区分选股信号与风险提示。',false],
['Q3—Q4跟踪通过检验的原型，按月复盘收益来源与预警效果，交付策略说明及调整建议。',false]
]];
for(const rows of pages){const s=p.slides.add();let top=183;for(const [text,bold] of rows){let h=bold?29:49;const box=s.shapes.add({geometry:'textbox',position:{left:bold?273:298,top,width:bold?540:510,height:h},fill:'none',line:{fill:'none',width:0}});box.text=text;box.text.style={fontFamily:'微软雅黑',fontSize:18.667,bold,color:'#000000',verticalAlignment:'top',autoFit:'none'};box.text.insets=0;top+=h+(bold?1:5);}}
await (await PresentationFile.exportPptx(p)).save('D:/3. projects/xjy/research2workflow/.ppt_build/body.pptx');
