import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const req=createRequire('C:/Users/xujiayi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/runtime.cjs');
const {Presentation,PresentationFile}=await import(pathToFileURL(req.resolve('@oai/artifact-tool')).href);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const rows=[
['AI风格感知：条件输入与状态门控',true],
['Q1统一L2、预期及成长因子的生效时点，构建大小盘、成长价值与风险状态样本。',false],
['Q2借鉴PRIME加入状态输入与门控，对比无状态GRU和分位规则，检验未来因子收益。',false],
['多任务学习：共享GRU，预测不同期限',true],
['设置1、3、5日预测头，各自拟合真实收益，以等权损失训练，对比单一5日标签。',false],
['先检验5日RankIC与成本后收益，优于单任务后再试自适应权重，独立样本留作终检。',false],
['滚动验证：识别模型的真实增量',true],
['Q3分别检验状态门控与多任务的贡献，比较分域RankIC、分组单调性及成本后超额。',false],
['Q4跟踪通过检验的模拟盘，归档数据口径、模型配置与失效条件，训练验证按标签期隔离。',false]
];
const slide=p.slides.add();let top=183;
for(const [text,bold] of rows){const h=bold?29:49;const s=slide.shapes.add({geometry:'textbox',position:{left:bold?273:298,top,width:bold?540:510,height:h},fill:'none',line:{fill:'none',width:0}});s.text=text;s.text.style={fontFamily:'微软雅黑',fontSize:18.667,bold,color:'#000000',autoFit:'none'};s.text.insets=0;top+=h+(bold?1:5);}
const second=p.slides.add();
const stages=[
['L2增强指增\n特征构造','Q1按时段、订单规模及主动方向拆分委托与成交样本。\n提取订单流不平衡、交易聚集和冲击后流动性恢复，补充成长与预期因子。'],
['L2增强指增\nAI筛选组合','Q2用机器学习检验因子增量，控制行业市值及动量反转的影响。\n按ICIR和相关性筛选，加入风险模型、换手与容量约束。'],
['状态感知配置\n指标与面板','Q1建立大小盘、行业轮动、ETF申赎及流动性压力指标库。\n输出多窗口历史分位，检验对未来因子收益的预测力，区分输入与监控。'],
['状态感知配置\nAI动态权重','Q2比较等权、固定ICIR、分位规则与AI状态门控配置。\nQ3做成本压力测试，Q4模拟盘复盘超额、回撤和换手，保留人工调整。']
];
for(let i=0;i<4;i++)for(let j=0;j<2;j++){const s=second.shapes.add({geometry:'textbox',position:{left:270+i*245,top:j?350:285,width:210,height:j?295:65},fill:'none',line:{fill:'none',width:0}});s.text=stages[i][j];s.text.style={fontFamily:'微软雅黑',fontSize:18.667,bold:j===0,color:'#000000'};}
await (await PresentationFile.exportPptx(p)).save('D:/3. projects/xjy/research2workflow/.ppt_build/body_ai.pptx');
