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
 $s.Shapes.Item('文本框 5').TextFrame.TextRange.Text=@('业务目标：完善公告事件模型与文本因子的研究验证。','业务目标：完成多因子选股与风险控制的策略验证。')[$j-1]
 $s.Shapes.Item('灯片编号占位符 2').TextFrame.TextRange.Text=[string]$j
 $labels=if($j -eq 1){@('事件样本库','模型对照检验','因子持续跟踪')}else{@('多因子选股','风险约束组合','模拟盘复盘')}
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
 $notes=if($j -eq 1){'工作基础：上季度参考中的事件驱动、LLM因子方向；本项目v2/ResearchFlow单因子检验、因子登记及机器学习模块。季度节点与交付物为2027年拟定计划，未将研究模块表述为实盘成果。'}else{'工作基础：v2/ResearchFlow/README_v2.md；v2/Monitor/risk.py、style.py、micro.py；上季度参考中的组合优化与动态权重方向。拟定计划以样本外和模拟盘检验为前提，不作收益承诺。'}
 $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=$notes
}
$p.SaveAs("$root\output\2027年度计划_许佳宜_新技术与新策略.pptx",24)
for($j=1;$j -le 2;$j++){$p.Slides.Item($j).Export("$root\.ppt_build\final$j.png",'PNG',1600,900)}
$body.Close();$p.Close();$app.Quit()
