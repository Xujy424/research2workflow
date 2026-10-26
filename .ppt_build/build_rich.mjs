import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const req=createRequire('C:/Users/xujiayi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/runtime.cjs');
const {Presentation,PresentationFile}=await import(pathToFileURL(req.resolve('@oai/artifact-tool')).href);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const navy='#392B91',ink='#16243B';
function txt(s,name,text,x,y,w,h,size=18.667,bold=false,color=ink){const a=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=text;a.text.style={fontFamily:'微软雅黑',fontSize:size,bold,color,autoFit:'none'};a.text.insets=0;return a;}
function table(s,name,values,x,y,w,h,widths){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});t.borders.assign({fill:'#D7DDEA',width:0.7,style:'solid'});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?navy:'#F1F4F9';cell.text.style={fontFamily:'微软雅黑',fontSize:18.667,bold:r===0,color:r===0?'#FFFFFF':ink};}return t;}
let s=p.slides.add();
txt(s,'intro','以风格感知和多期限监督改造现有时序模型，形成2项可复用的AI研究组件。',272,137,950,38,21.333,true);
table(s,'technical-route',[
['数据与因子','市场状态','AI模型结构','增量验证'],
['L2量价、成长与预期\n统一时点、缺失及异常值','大小盘、成长价值\n趋势反转、行业轮动\n历史风险分位','共享GRU主干\n状态门控与1/3/5日预测头','无状态与单期限基线\n滚动检验及成本后评价']
],272,189,958,115,[239,239,240,240]);
txt(s,'heading-a','改造一  PRIME式风格感知',272,327,456,29,21.333,true,navy);
txt(s,'heading-b','改造二  多期限标签联合学习',764,327,466,29,21.333,true,navy);
const a=[
'建立状态指标与各类因子收益的历史样本，先用分位规则检验风格对因子强弱的解释力。',
'将市场状态作为GRU的条件输入或门控变量，与无状态模型比较，观察不同阶段的权重变化。',
'滚动训练并逐项移除状态特征，检验前瞻预测增量，区分事后解释与可交易信号。'];
const b=[
'共享GRU主干，三个预测头分别拟合未来1、3、5日真实收益，初始任务损失等权求和。',
'先与单一5日标签比较RankIC、收益单调性和成本后收益，确认增益后再试自适应任务权重。',
'训练与验证按标签期隔离，在验证集选择参数，独立测试集终检，避免反复调参透支样本。'];
for(let i=0;i<3;i++){txt(s,'bullet-a'+i,a[i],281,365+i*75,447,70);txt(s,'bullet-b'+i,b[i],773,365+i*75,457,70);}
table(s,'milestones',[
['Q1  数据与基线','Q2  模型原型','Q3  增量验收','Q4  组件交付'],
['统一数据口径\n状态库与单任务基线','状态门控与多任务\n固定实验对照组','消融、分域与成本检验\n记录有效及失效场景','通过检验后模拟跟踪\n模型配置与报告入库']
],272,607,958,70,[239,239,240,240]);
s=p.slides.add();
txt(s,'intro','储备L2微观结构增强指增与AI状态感知配置两项策略，以成本后表现和模拟盘验收。',272,137,958,40,21.333,true);
txt(s,'strategy1','策略一\nL2微观结构\n增强指增',272,205,172,82,21.333,true,navy);
txt(s,'strategy1-purpose','订单行为形成\n日频信号，补充\n成长与预期因子。',272,305,178,75);
txt(s,'s1-head','信号构造与AI筛选',478,199,435,28,21.333,true,navy);
txt(s,'bullet-s1a','按时段、订单规模及主动方向拆分样本，构造订单流不平衡、交易聚集和冲击后流动性恢复因子。',487,239,426,71);
txt(s,'bullet-s1b','控制行业市值和动量反转，以机器学习检验新增信息，结合ICIR、相关性及分组收益筛选。',487,318,426,71);
txt(s,'s1-eval','组合约束与验收',956,199,274,28,21.333,true,navy);
txt(s,'bullet-s1c','接入风险模型，限制行业偏离、个股权重及换手，按成交额检验容量。',965,239,265,71);
txt(s,'bullet-s1d','比较基础组合与新增因子组合，报告成本后超额、回撤和换手。',965,318,265,71);
txt(s,'strategy2','策略二\nAI状态感知\n动态因子配置',272,421,180,83,21.333,true,navy);
txt(s,'strategy2-purpose','提前识别因子强弱，\n检验动态配权对\n固定配置的增益。',272,521,180,75);
txt(s,'s2-head','状态指标与动态配权',478,415,435,28,21.333,true,navy);
txt(s,'bullet-s2a','建设大小盘、行业轮动、ETF申赎与流动性压力指标，输出多窗口历史分位，区分模型输入与监控。',487,455,426,71);
txt(s,'bullet-s2b','以动量、反转、价值、成长和低波为候选，比较等权、固定ICIR、分位规则与AI状态门控配置。',487,534,426,71);
txt(s,'s2-eval','前瞻检验与模拟跟踪',956,415,274,28,21.333,true,navy);
txt(s,'bullet-s2c','固定风险和交易约束，检验未来因子收益，拆解风格暴露与配权贡献。',965,455,265,71);
txt(s,'bullet-s2d','监控模型衰减和预警，按月复盘，人工决定保留、降权或调整。',965,534,265,71);
table(s,'acceptance',[
['Q1  基础准备','Q2  原型研究','Q3  严格检验','Q4  策略储备'],
['指标库、基准与股票池','因子筛选、配置对照','滚动样本外与成本压力','模拟盘、说明书与复盘']
],272,632,958,45,[239,239,240,240]);
await (await PresentationFile.exportPptx(p)).save('D:/3. projects/xjy/research2workflow/.ppt_build/body_rich.pptx');
