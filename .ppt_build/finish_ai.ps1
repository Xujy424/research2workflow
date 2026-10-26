$ErrorActionPreference='Stop'
$root='D:\3. projects\xjy\research2workflow'
$app=New-Object -ComObject PowerPoint.Application
$p=$app.Presentations.Open("$root\.ppt_build\template.pptx",0,0,0)
$body=$app.Presentations.Open("$root\.ppt_build\body_ai.pptx",-1,0,0)
for($i=$p.Slides.Count;$i -ge 1;$i--){if($i -ne 6 -and $i -ne 8){$p.Slides.Item($i).Delete()}}
$p.Slides.Item(2).MoveTo(1)
function Set-Text($shape,$text,$size,$bold){
 $shape.TextFrame.TextRange.Text=$text
 $shape.TextFrame.TextRange.Font.Name='微软雅黑'
 $shape.TextFrame.TextRange.Font.Size=$size
 $shape.TextFrame.TextRange.Font.Bold=[string]$bold
}
$s=$p.Slides.Item(1)
$s.Shapes.Item('文本框 6').Delete();$s.Shapes.Item('矩形: 圆角 8').Delete()
Set-Text ($s.Shapes.Item('文本框 2')) '三、业务计划-研究部-新技术' 28 -1
Set-Text ($s.Shapes.Item('文本框 5')) '2027年目标：验证AI风格感知与多任务学习的模型增益。' 16 0
Set-Text ($s.Shapes.Item('灯片编号占位符 2')) '1' 11 0
$labels=@('AI风格感知','多任务学习','样本外验证');$ix=0;$g=$s.Shapes.Item('组合 7')
for($n=1;$n -le $g.GroupItems.Count;$n++){$sh=$g.GroupItems.Item($n);if($sh.HasTextFrame -and $sh.TextFrame.HasText){Set-Text $sh $labels[$ix] 16 -1;$ix++}}
for($k=1;$k -le $body.Slides.Item(1).Shapes.Count;$k++){
 $body.Slides.Item(1).Shapes.Item($k).Copy();$range=$s.Shapes.Paste();$sh=$range.Item(1)
 $sh.TextFrame.TextRange.Font.Name='微软雅黑';$sh.TextFrame.TextRange.Font.Size=14
 $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Visible=-1
 $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Character=if(($k-1)%3 -eq 0){9642}else{10146}
 $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Font.Name='Segoe UI Symbol'
 $sh.TextFrame.TextRange.ParagraphFormat.LineRuleWithin=-1;$sh.TextFrame.TextRange.ParagraphFormat.SpaceWithin=1.3
 $sh.TextFrame.MarginLeft=10;$sh.TextFrame.MarginRight=0;$sh.TextFrame.MarginTop=0;$sh.TextFrame.MarginBottom=0
}
$s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text='依据用户周报、月报及项目代码拟定2027年计划。PRIME风格感知和标签自适应均为待检验方向。自适应权重在验证集选择，独立样本仅用于最终检验。GRU与机器学习即本页AI技术的具体应用，不等同于大语言模型。'
$s=$p.Slides.Item(2)
$s.Shapes.Item('矩形: 圆角 6').Delete()
Set-Text ($s.Shapes.Item('文本框 2')) '三、业务计划-研究部-新策略' 28 -1
Set-Text ($s.Shapes.Item('文本框 3')) '2027年目标：形成AI驱动的因子轮动与模拟跟踪方案。' 16 0
Set-Text ($s.Shapes.Item('灯片编号占位符 2')) '2' 11 0
$stageNames=@('矩形: 圆角 8','矩形: 圆角 12','矩形: 圆角 13','矩形: 圆角 14')
$stageLabels=@('因子输入','AI轮动','组合回测','监控复盘')
$headingNames=@('文本框 28','文本框 34','文本框 40','文本框 46')
$bodyNames=@('文本框 29','文本框 35','文本框 41','文本框 47')
for($i=0;$i -lt 4;$i++){
 Set-Text ($s.Shapes.Item($stageNames[$i])) $stageLabels[$i] 14 -1
 $heading=$body.Slides.Item(2).Shapes.Item(2*$i+1).TextFrame.TextRange.Text
 Set-Text ($s.Shapes.Item($headingNames[$i])) $heading 14 -1
 $sh=$s.Shapes.Item($bodyNames[$i]);$txt=$body.Slides.Item(2).Shapes.Item(2*$i+2).TextFrame.TextRange.Text
 Set-Text $sh $txt 14 0
 $sh.TextFrame.MarginLeft=8;$sh.TextFrame.MarginRight=4
 $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Visible=-1
 $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Character=9642
 $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Font.Name='Arial'
 $sh.TextFrame.TextRange.ParagraphFormat.LineRuleWithin=-1
 $sh.TextFrame.TextRange.ParagraphFormat.SpaceWithin=1.3
 $sh.TextFrame.TextRange.ParagraphFormat.LineRuleAfter=0
 $sh.TextFrame.TextRange.ParagraphFormat.SpaceAfter=16
}
$s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text='依据用户周报与月报拟定2027年计划。AI用于因子增量检验、风格感知和条件配权，监控与复盘保留人工判断。模拟盘仅纳入通过样本外与成本检验的原型，不将待研究功能写作已有成果。'
$p.SaveAs("$root\output\2027年度计划_许佳宜_AI重点与差异排版.pptx",24)
for($j=1;$j -le 2;$j++){$p.Slides.Item($j).Export("$root\.ppt_build\ai$j.png",'PNG',1600,900)}
$body.Close();$p.Close();$app.Quit()
