$ErrorActionPreference='Stop'
$root='D:\3. projects\xjy\research2workflow'
$app=New-Object -ComObject PowerPoint.Application
$p=$app.Presentations.Open("$root\.ppt_build\template.pptx",0,0,0)
$body=$app.Presentations.Open("$root\.ppt_build\body.pptx",-1,0,0)
for($i=$p.Slides.Count;$i -ge 1;$i--){if($i -ne 8){$p.Slides.Item($i).Delete()}}
$null=$p.Slides.Item(1).Duplicate()
for($j=1;$j -le 2;$j++){
 $s=$p.Slides.Item($j)
 $s.Shapes.Item('文本框 6').Delete()
 $s.Shapes.Item('矩形: 圆角 8').Delete()
 $title=$s.Shapes.Item('文本框 2').TextFrame.TextRange
 $title.Text='三、业务计划-研究部-'+@('新技术','新策略')[$j-1]
 $title.Font.Name='微软雅黑'
 $title.Font.Size=28
 $title.Font.Bold=-1
 $s.Shapes.Item('文本框 5').TextFrame.TextRange.Text=@('2027年目标：验证风格感知与多期限预测的模型增益。','2027年目标：形成选股因子、风格轮动与监控的策略原型。')[$j-1]
 $s.Shapes.Item('灯片编号占位符 2').TextFrame.TextRange.Text=[string]$j
 $labels=if($j -eq 1){@('风格感知','多期限预测','统一验证')}else{@('选股因子','风格轮动','监控与复盘')}
 $g=$s.Shapes.Item('组合 7');$labelIndex=0
 for($n=1;$n -le $g.GroupItems.Count;$n++){
  $sh=$g.GroupItems.Item($n)
  if($sh.HasTextFrame -and $sh.TextFrame.HasText){$sh.TextFrame.TextRange.Text=$labels[$labelIndex];$sh.TextFrame.TextRange.Font.Name='微软雅黑';$sh.TextFrame.TextRange.Font.Size=16;$labelIndex++}
 }
 for($k=1;$k -le $body.Slides.Item($j).Shapes.Count;$k++){
  $body.Slides.Item($j).Shapes.Item($k).Copy()
  $r=$s.Shapes.Paste();$sh=$r.Item(1)
  $sh.TextFrame.TextRange.Font.Name='微软雅黑'
  $sh.TextFrame.TextRange.Font.Size=14
  $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Visible=-1
  $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Character=if(($k-1)%3 -eq 0){9642}else{10146}
  $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Font.Name='Segoe UI Symbol'
  $sh.TextFrame.TextRange.ParagraphFormat.LineRuleWithin=-1
  $sh.TextFrame.TextRange.ParagraphFormat.SpaceWithin=1.3
  $sh.TextFrame.MarginLeft=10
  $sh.TextFrame.MarginRight=0
  $sh.TextFrame.MarginTop=0
  $sh.TextFrame.MarginBottom=0
 }
 $notes=if($j -eq 1){'依据用户补充的本周周报及月报截图。PRIME风格感知与标签期限为学习和探索内容，未视为验证通过。2027年季度节点为拟定安排。多任务等权损失为首轮基线，自适应权重仅在5日RankIC和成本后收益改善后研究。论文题名由用户提供，本页未引用其未经核验的实验结论。'}else{'依据用户提供的月报截图：微观结构与成长因子持续积累，多数据源固定参数和分域损失训练测试已开展，Level2及一致预期因子、市场监控面板、风格研判优化仍在推进。2027年计划不沿用月报进度百分比作为年度完成率，模拟盘仅纳入验证通过的原型。'}
 $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=$notes
}
$p.SaveAs("$root\output\2027年度计划_许佳宜_结合月报周报修订.pptx",24)
for($j=1;$j -le 2;$j++){$p.Slides.Item($j).Export("$root\.ppt_build\final$j.png",'PNG',1600,900)}
$body.Close();$p.Close();$app.Quit()
