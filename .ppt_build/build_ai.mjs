import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const req=createRequire('C:/Users/xujiayi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/runtime.cjs');
const {Presentation,PresentationFile}=await import(pathToFileURL(req.resolve('@oai/artifact-tool')).href);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const rows=[
['AI风格感知：条件输入与状态门控',true],
['Q1整理宏观与风格指标，结合各类因子，为现有时序模型补充市场状态输入。',false],
['Q2比较有无风格感知的模型，检查不同市场阶段的因子权重与样本外表现。',false],
['多任务学习：共享GRU，预测不同期限',true],
['设置1、3、5日收益预测头，先以等权损失训练，与单一5日标签模型对比。',false],
['5日RankIC与成本后收益改善后，再研究自适应标签权重，另留独立样本检验。',false],
['模型验证：固定基线，保留实验依据',true],
['统一股票池、回测口径和数据切分。训练与验证按标签期限隔离，保留多数据源对照实验。',false],
['Q3至Q4交付AI模型对照报告与实验配置，说明增益来源、适用场景及失效条件。',false]
];
const slide=p.slides.add();let top=183;
for(const [text,bold] of rows){const h=bold?29:49;const s=slide.shapes.add({geometry:'textbox',position:{left:bold?273:298,top,width:bold?540:510,height:h},fill:'none',line:{fill:'none',width:0}});s.text=text;s.text.style={fontFamily:'微软雅黑',fontSize:18.667,bold,color:'#000000',autoFit:'none'};s.text.insets=0;top+=h+(bold?1:5);}
const second=p.slides.add();
const stages=[
['微观结构\n与预期因子','Q1至Q2积累Level2、成长及一致预期因子，统一清洗与检验。\n以机器学习比较新增因子的预测贡献，剔除冗余信号。'],
['风格感知\n动态配权','Q2利用AI模型识别风格状态，研究不同因子的条件权重。\n对比等权与ICIR配置，检验轮动收益是否来自风格暴露。'],
['成本与\n交易约束','固定行业、个股及换手约束，检查AI信号的成本后收益。\nQ3将验证通过的组合纳入模拟盘，记录预测与执行偏差。'],
['模型跟踪\n人工复核','将风格和流动性指标接入监控面板，跟踪模型衰减。\nQ3至Q4按月复盘超额与预警，人工决定保留或调整策略。']
];
for(let i=0;i<4;i++)for(let j=0;j<2;j++){const s=second.shapes.add({geometry:'textbox',position:{left:270+i*245,top:j?350:285,width:210,height:j?295:65},fill:'none',line:{fill:'none',width:0}});s.text=stages[i][j];s.text.style={fontFamily:'微软雅黑',fontSize:18.667,bold:j===0,color:'#000000'};}
await (await PresentationFile.exportPptx(p)).save('D:/3. projects/xjy/research2workflow/.ppt_build/body_ai.pptx');
