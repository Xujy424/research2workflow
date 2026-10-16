'use strict';
const $=s=>document.querySelector(s);
const colors=['#087f78','#327aa5','#c14e67','#a87a20'];
const periods=[['全部历史',0,129,''],['2016—2017',0,23,'供给侧改革 / 漂亮50'],['2018',24,35,'去杠杆 / 贸易摩擦'],['2019—2021初',36,61,'核心资产 / 机构抱团'],['2021—2023',62,95,'赛道切换 / 小微盘 / 红利'],['2024上半年',96,101,'微盘流动性冲击'],['2024四季度—2025',105,119,'政策修复 / 科技成长'],['2026至10月',120,129,'动量扩张后再平衡']];
const events=[[0,'熔断'],[24,'信用收缩'],[61,'抱团瓦解'],[96,'微盘冲击'],[104,'政策修复']];
const modules=[
 {id:'style',name:'风格与定价方向',question:'谁在获得定价溢价？风格切换是否持续？',series:[['小盘 − 大盘累计收益','%',0,24],['成长 − 价值累计收益','%',2,20],['动量 − 反转累计收益','%',4,18]]},
 {id:'breadth',name:'趋势与市场宽度',question:'指数上涨是否伴随多数股票与行业共同改善？',series:[['基准净值','倍',1,0.4],['高于 20 日均线比例','%',52,23],['收益横截面离散度','%',2.4,1.3]]},
 {id:'earnings',name:'盈利与估值支撑',question:'估值扩张有没有盈利上修支撑？',series:[['估值历史分位','%',55,28],['FY1 盈利预期修正','%',0,7],['盈利预测上调覆盖率','%',48,20]]},
 {id:'macro',name:'利率与信用环境',question:'折现率与融资约束是否发生方向变化？',series:[['10 年国债收益率','%',2.8,0.7],['AA − AAA 信用利差','bp',95,40],['社融同比增速','%',10,3]]},
 {id:'flows',name:'资金结构与风格迁移',question:'增量资金流向哪里？成交定价权向谁集中？',series:[['小微盘成交额占比','%',33,13],['风格 ETF 估算净流量','亿元',0,130],['融资余额变化','%',0,6]]},
 {id:'risk',name:'拥挤、流动性与尾部压力',question:'集中交易是否伴随退出困难与下行压力？',series:[['头部 5% 股票成交占比','%',38,15],['Amihud 历史分位','%',48,29],['跌停股票比例','%',1.2,1],['波动率历史分位','%',50,31]]}
];
const catalog=[
 ['风格与定价方向','大小盘相对强度；微盘相对收益；成长价值、红利、质量、高低 Beta、高低波、动量反转相对收益','价差累计收益 / 滚动收益时序；分组收益条形图','日频行情与风格定义'],
 ['风格与定价方向','PE、PB、EP、BP、PEG、股息率、ERP；大小盘估值分位差；估值扩张 ΔlogPE','估值与盈利分图；历史分位时序','估值 / 财务 / 预期数据；亏损样本单列'],
 ['趋势与市场宽度','上涨比例；均线上方比例；趋势一致性；行业扩散与行业收益分化','指数、宽度分图共享时间轴；行业热力图','宽度指标为展示补充，定义待固定'],
 ['趋势与市场宽度','连续下跌比例；收益横截面离散度；行业内离散度；龙头 / 尾部相对强弱；3—12 月及残差动量','离散度时序；龙头尾部价差；市值分层对比','日频行情、行业、市值与中性化结果'],
 ['盈利与产业景气','ROE、ROIC、利润同比、盈利稳定性、现金流与利润匹配度','分组聚合时序；行业截面点图','按财报披露日可用，非每日新观测'],
 ['盈利与产业景气','FY1 / FY2 一致预期 EPS 修正；目标价调整；上调 / 下调比例；盈利超预期比例','预期修正、覆盖率、估值分图；行业发散条形图','需预期修正历史快照；FY1/FY2 非等同未来12个月'],
 ['盈利与产业景气','行业高频景气数据；行业集中度；龙头市场份额','行业独立时序；盈利与估值贡献拆解','各行业口径不同；盈利估值拆解不等同 Brinson'],
 ['利率与信用环境','利率、10 年国债收益率、社融；AA / AAA 与民企信用利差','分图共享时间轴；事件竖线','宏观与信用数据按发布日期'],
 ['利率与信用环境','财务杠杆；质押比例；潜在平仓压力；政策、贸易摩擦、去杠杆等背景','风险组占比时序；事件标注','平仓线依赖假设；事件为标签'],
 ['资金结构与迁移','ETF 份额与估算净流量；公募发行 / AUM / 仓位变化；机构持股比例；重仓基金数量','资金变化时序；风格占比堆叠图','份额与价格变化分离；季报披露滞后'],
 ['资金结构与迁移','北向历史持仓与成交；两融余额与流向；市值 / 风格 / 行业成交额占比','各资金来源分图；成交占比时序','北向历史口径与可得区间需核对；成交不等于净流入'],
 ['拥挤与集中交易','Σ AUM × 持仓权重；基金持仓占流通市值比例；重仓数量、持仓重叠与集中度','披露期时序；拥挤与估值散点','规模加权持仓需除流通市值以便跨股票比较'],
 ['拥挤与集中交易','成交额 HHI；头部 5% 成交占比；估值分位；因子相关性与持仓重叠','集中度与风险分图；相关性热力图','HHI 与头部占比是不同指标；相关对象分别标明'],
 ['日频流动性与尾部','成交额、日均成交额分布、换手、成交萎缩、Amihud、波动率、跌停家数 / 比例','风险时序；市值组分位热力图；流动性箱线图','Amihud 使用绝对收益 / 成交额，统一单位'],
 ['日频流动性与尾部','偏度、峰度、最差 5 日收益、VaR、CVaR、最大回撤','滚动风险时序；收益分布；压力区间标记','说明收益对象、窗口和置信水平'],
 ['因子健康与交易约束','IC、RankIC、20 / 60 / 120 日 ICIR、胜率、分层单调性、纯因子收益、成本后收益','市场页仅摘要；细节在因子检测界面','FactorTest/analyzer 组织；标签与年化口径明确'],
 ['因子健康与交易约束','IC 衰减、半衰期、信号自相关、行业 / Barra 暴露、生产因子相关性','衰减曲线、暴露与相关性图','半衰期不可强制套用非单调衰减；需持仓或因子矩阵'],
 ['因子健康与交易约束','换手、滑点、可成交性、容量、资金规模收益衰减、压力退出天数','成本净收益比较、容量曲线、压力测试','依赖持仓、成本与成交参与率假设；因子检测负责'],
 ['微观结构 · 暂缓','大单净流入、主动买入、订单不平衡、撤单、订单流自相关、价差、深度与恢复、Kyle λ、冲击与保留率','后续微观结构时序与事件路径','本期不接入市场面板'],
 ['微观结构 · 暂缓','历史跳跃频率、ACD 预测与标准化跳跃强度、跳跃聚集、1 / 5 / 15 / 30 分钟冲击持续性','跳跃强度与保留率联动；事件路径','仅用已完成事件；ACD 倒数强度定义依赖分布'],
 ['微观结构 · 暂缓','LiquidityCoverage；极端盘口容量','买盘容量 / 待卖数量；压力情景','需订单 / 持仓 / 参与率；不是通用市场状态指标']
];
const states=['流动性宽松＋风险偏好上行','机构抱团＋趋势集中','信用收缩＋避险','经济弱复苏＋低利率','产业景气持续上修','震荡＋无清晰主线','拥挤交易集中平仓','政策冲击＋超跌修复'];
const factors=['价值','质量','成长','预期修正','红利','低波','动量','反转','量价','资金流','小市值'];
const priors=[[0,0,1,1,-1,-1,1,0,1,1,1],[-1,1,1,1,0,0,1,-1,0,1,-1],[0,1,-1,-1,1,1,0,-1,-1,0,-1],[1,1,0,0,1,1,0,0,0,0,0],[0,1,1,1,0,0,1,0,0,1,0],[0,0,0,0,0,0,-1,1,1,0,0],[0,0,-1,0,1,1,-1,-1,-1,-1,-1],[0,0,1,0,-1,-1,0,1,1,1,1]];
const mechanisms=['融资与风险预算改善，成长、高弹性及趋势信号值得优先检验。','盈利质量与持续配置可能支持趋势；估值扩张和持仓重叠增加尾部风险。','信用与流动性约束提高防御、现金流质量的重要性；超跌不必然反转。','弱增长与低利率可能支持红利、价值防御；成长取决于盈利与利率的共同变化。','盈利预期上修可能支持成长、产业动量及预期修正，需剥离行业收益。','趋势缺乏连续性时可研究反转与量价，但净收益需覆盖成本。','强制减仓可能使动量与反转同时受损；优先检查退出约束。','政策修复可支持超跌与高弹性信号，短期反转及后续趋势要分阶段验证。'];
const dates=Array.from({length:130},(_,i)=>`${2016+Math.floor(i/12)}-${String(i%12+1).padStart(2,'0')}`);
let start=0,end=129,hoverIndex=null,selected=[1,6];
function data(m,j){const s=m.series[j],pool=['all','large','mid','small'].indexOf($('#pool').value);return dates.map((_,i)=>{const pulse=Math.exp(-Math.pow((i-96)/2,2));const wave=Math.sin(i/8+j*1.3)+0.4*Math.cos(i/3+j);let value=s[2]+s[3]*(wave*.65+Math.sin(i/25+j)*.35)+(pool-1)*s[3]*.09;if(m.id==='risk')value+=s[3]*pulse;if(s[1]==='%'&&s[2]>20)value=Math.max(1,Math.min(99,value));if(s[0].includes('跌停'))value=Math.max(0,value+5*pulse);return value;});}
function percentile(values){return values.map(v=>100*values.filter(x=>x<=v).length/values.length);}
function setup(){
 $('#period').innerHTML=periods.map((p,i)=>`<option value="${i}">${p[0]}</option>`).join('');
 $('#timeline').innerHTML=periods.slice(1).map((p,i)=>`<button data-period="${i+1}"><strong>${p[0]}</strong>${p[3]}</button>`).join('');
 $('#modules').innerHTML=modules.map(m=>`<section class="module" id="${m.id}"><div class="module-head"><div><h2>${m.name}</h2><p>${m.question}</p></div><span class="status">模拟月度序列</span></div><div class="module-controls">${m.series.map((s,j)=>`<label class="series-toggle"><input type="checkbox" data-module="${m.id}" data-series="${j}" checked><i style="--c:${colors[j]}"></i>${s[0]}</label>`).join('')}</div><div class="chart-layout"><div class="chart-shell"><canvas aria-label="${m.name}多维时序图" role="img"></canvas><div class="tooltip" hidden></div></div><div class="snapshot"></div></div><details class="cross"><summary>截面对比 · 观察窗口末端</summary><canvas aria-label="${m.name}分组对比" role="img"></canvas><p>分组值为独立模拟结果，用于检查比较布局。</p></details></section>`).join('');
}
// Keep chart and evidence updates in one shared observation window.
function fit(canvas,height){const w=Math.max(220,canvas.parentElement.clientWidth),dpr=window.devicePixelRatio||1;canvas.style.height=height+'px';canvas.width=w*dpr;canvas.height=height*dpr;const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,height);c.font='11px Segoe UI, Microsoft YaHei';return [c,w,height];}
function visible(m){return m.series.map((_,i)=>i).filter(j=>document.querySelector(`[data-module="${m.id}"][data-series="${j}"]`).checked);}
function draw(m){const el=$('#'+m.id),canvas=el.querySelector('.chart-shell canvas'),js=visible(m),rank=$('#scale').value==='percentile',height=rank?240:Math.max(145,js.length*112+30),[c,w,h]=fit(canvas,height),left=w<420?50:62,right=18,top=18,bottom=28,plotW=w-left-right;const vals=js.map(j=>rank?percentile(data(m,j)):data(m,j));const laneH=(h-top-bottom)/(rank?1:Math.max(1,js.length));const x=i=>left+(i-start)/Math.max(1,end-start)*plotW;
 if(!js.length){c.fillStyle='#617079';c.fillText('请选择至少一个指标',left,50);}
 js.forEach((j,k)=>{const arr=vals[k],segment=arr.slice(start,end+1);let lo=rank?0:Math.min(...segment),hi=rank?100:Math.max(...segment);let pad=(hi-lo)*.12||1;if(!rank){lo-=pad;hi+=pad;}const yTop=top+(rank?0:k*laneH),yBottom=yTop+laneH-24,y=v=>yBottom-(v-lo)/(hi-lo)*(laneH-40);
 if(!rank||k===0){for(let t=0;t<3;t++){const v=lo+(hi-lo)*t/2,yy=y(v);c.strokeStyle='#e4e9ec';c.beginPath();c.moveTo(left,yy);c.lineTo(w-right,yy);c.stroke();c.fillStyle='#617079';c.textAlign='right';c.fillText(rank?Math.round(v)+'%':v.toFixed(Math.abs(hi)>50?0:1),left-7,yy+3);}c.textAlign='left';c.fillStyle='#617079';c.fillText(rank?'历史分位 (%)':m.series[j][0]+' ('+m.series[j][1]+')',left,yTop+1);}
 c.save();c.beginPath();c.rect(left,yTop,plotW,laneH);c.clip();if($('#events').checked&&(!rank||k===0)){events.filter(e=>e[0]>=start&&e[0]<=end).forEach(e=>{c.strokeStyle='#c9d4d9';c.setLineDash([3,3]);c.beginPath();c.moveTo(x(e[0]),yTop+10);c.lineTo(x(e[0]),yBottom);c.stroke();c.setLineDash([]);});}c.strokeStyle=colors[j];c.lineWidth=1.8;c.beginPath();for(let i=start;i<=end;i++){if(i===start)c.moveTo(x(i),y(arr[i]));else c.lineTo(x(i),y(arr[i]));}c.stroke();if(hoverIndex!==null&&hoverIndex>=start&&hoverIndex<=end){c.strokeStyle='#82959e';c.lineWidth=1;c.beginPath();c.moveTo(x(hoverIndex),yTop+8);c.lineTo(x(hoverIndex),yBottom);c.stroke();c.fillStyle=colors[j];c.beginPath();c.arc(x(hoverIndex),y(arr[hoverIndex]),3,0,Math.PI*2);c.fill();}c.restore();});
 c.fillStyle='#617079';c.textAlign='center';const ticks=w<420?3:5;for(let t=0;t<ticks;t++){const i=Math.round(start+(end-start)*t/(ticks-1));c.textAlign=t===0?'left':t===ticks-1?'right':'center';c.fillText(dates[i],x(i),h-8);}c.textAlign='left';if($('#events').checked){const es=events.filter(e=>e[0]>=start&&e[0]<=end);el.querySelector('.status').textContent='模拟数据'+(es.length?' · '+es.map(e=>dates[e[0]]+' '+e[1]).join(' / '):'');}else el.querySelector('.status').textContent='模拟月度序列';
 el.querySelector('.snapshot').innerHTML='<h3>窗口末端 · '+dates[end]+'</h3>'+js.map(j=>{const a=data(m,j),v=a[end],change=v-a[Math.max(start,end-3)];return `<div class="snapshot-row"><span>${m.series[j][0]}</span><b>${v.toFixed(2)} ${m.series[j][1]}</b><small>近3月 ${change>=0?'+':''}${change.toFixed(2)}</small></div>`;}).join('');
 if(el.querySelector('details').open)drawCross(m);
 canvas.onpointermove=e=>{const rect=canvas.getBoundingClientRect();hoverIndex=Math.max(start,Math.min(end,Math.round(start+(e.clientX-rect.left-left)/plotW*(end-start))));modules.forEach(mm=>draw(mm));const tip=el.querySelector('.tooltip');tip.hidden=false;tip.innerHTML='<strong>'+dates[hoverIndex]+'</strong>'+js.map(j=>'<div>'+m.series[j][0]+'：'+data(m,j)[hoverIndex].toFixed(2)+' '+m.series[j][1]+'</div>').join('');tip.style.left=Math.min(Math.max(0,e.clientX-rect.left+10),Math.max(0,w-260))+'px';tip.style.top='20px';};
 canvas.onpointerleave=()=>{hoverIndex=null;document.querySelectorAll('.tooltip').forEach(t=>t.hidden=true);modules.forEach(mm=>draw(mm));};}
function drawCross(m){const canvas=$('#'+m.id+' .cross canvas'),[c,w]=fit(canvas,170),names=['大盘','中盘','小微盘'],j=visible(m)[0]??0,base=data(m,j)[end];c.fillStyle='#617079';c.fillText(m.series[j][0]+' ('+m.series[j][1]+')',10,16);names.forEach((n,i)=>{const v=base+(i-1)*m.series[j][3]*.15,y=48+i*37;c.fillStyle='#617079';c.fillText(n,8,y+12);c.fillStyle=colors[0];c.fillRect(64,y,Math.max(5,(w-155)*(0.45+i*.12)),15);c.fillStyle='#202b30';c.fillText(v.toFixed(2),Math.max(75,w-70),y+12);});}
function matrix(){const mode=$('#matrix-mode').value;$('#matrix').innerHTML='<thead><tr><th>市场状态 / 因子类型</th>'+factors.map(f=>'<th>'+f+'</th>').join('')+'</tr></thead><tbody>'+states.map((s,r)=>'<tr><th>'+s+'</th>'+factors.map((f,k)=>{const v=priors[r][k],label=mode==='prior'?(v>0?'＋':v<0?'−':'·'):mode==='ic'?((v*.025+Math.sin(r+k)*.006).toFixed(3)):String(45+r*29+k*7);return `<td><button class="${v>0?'pos':v<0?'neg':''} ${selected[0]===r&&selected[1]===k?'selected':''}" data-r="${r}" data-k="${k}" aria-label="${s}，${f}，${label}">${label}</button></td>`;}).join('')+'</tr>').join('')+'</tbody>';document.querySelectorAll('#matrix button').forEach(b=>b.onclick=()=>{selected=[Number(b.dataset.r),Number(b.dataset.k)];matrix();});const [r,k]=selected,v=priors[r][k];$('#evidence').innerHTML=`<div><h3>${states[r]} → ${factors[k]}</h3><p>${v>0?'优先检验':v<0?'谨慎使用':'方向依赖具体机制与阶段'}。${mechanisms[r]}</p><p class="note">这些方向是文档机制的归纳，非已验证的收益结论。</p></div><div><h3>证据与反证</h3><p>查看上方风格、资金、盈利与风险是否相互支持；集中度上升而盈利修正下降时，应复核趋势的收益来源。</p><p class="note">验证入口：条件 IC、成本后收益、样本数量、跨年度与样本外一致性。当前样本未计算真实匹配度。</p></div>`;}
function renderCatalog(){const filter=$('#catalog-filter').value,groups=[...new Set(catalog.map(c=>c[0]))];$('#catalog-content').innerHTML=groups.filter(g=>filter==='all'||g===filter).map(g=>`<details class="catalog-group" ${filter!=='all'?'open':''}><summary>${g}</summary><div class="table-wrap"><table><thead><tr><th>指标 / 方法</th><th>展示形式</th><th>数据与口径约束</th></tr></thead><tbody>${catalog.filter(c=>c[0]===g).map(c=>'<tr>'+c.slice(1).map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')}</tbody></table></div></details>`).join('');}
function update(){start=Number($('#start').value);end=Number($('#end').value);if(start>=end){end=Math.min(129,start+1);start=Math.min(start,end-1);$('#start').value=start;$('#end').value=end;}$('#range-label').textContent=dates[start]+' → '+dates[end];const p=periods[Number($('#period').value)];$('#context-title').textContent=(start===p[1]&&end===p[2]?p[0]:'自定义窗口')+' · '+$('#pool').selectedOptions[0].text;document.querySelectorAll('#timeline button').forEach(b=>b.classList.toggle('active',periods[Number(b.dataset.period)][1]===start&&periods[Number(b.dataset.period)][2]===end));modules.forEach(draw);}
setup();
$('#catalog-filter').innerHTML+=[...new Set(catalog.map(c=>c[0]))].map(g=>`<option>${g}</option>`).join('');renderCatalog();matrix();
function choose(i){$('#period').value=i;$('#start').value=periods[i][1];$('#end').value=periods[i][2];update();}
$('#period').onchange=e=>choose(Number(e.target.value));document.querySelectorAll('[data-period]').forEach(b=>b.onclick=()=>choose(Number(b.dataset.period)));$('#reset').onclick=()=>choose(0);['start','end','pool','scale','events'].forEach(id=>$('#'+id).addEventListener('input',update));document.querySelectorAll('[data-series]').forEach(e=>e.onchange=update);document.querySelectorAll('.cross').forEach((e,i)=>e.ontoggle=()=>{if(e.open)drawCross(modules[i]);});$('#matrix-mode').onchange=matrix;$('#catalog-filter').onchange=renderCatalog;let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(update,100);});update();
