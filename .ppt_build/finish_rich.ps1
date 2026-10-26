$ErrorActionPreference='Stop'
$root='D:\3. projects\xjy\research2workflow'
$app=New-Object -ComObject PowerPoint.Application
$p=$app.Presentations.Open("$root\.ppt_build\template.pptx",0,0,0)
$body=$app.Presentations.Open("$root\.ppt_build\body_rich.pptx",-1,0,0)
for($i=$p.Slides.Count;$i -ge 1;$i--){if($i -ne 8){$p.Slides.Item($i).Delete()}}
$null=$p.Slides.Item(1).Duplicate()
for($j=1;$j -le 2;$j++){
 $s=$p.Slides.Item($j)
 for($k=$s.Shapes.Count;$k -ge 1;$k--){$sid=$s.Shapes.Item($k).Id;if($sid -notin @(2,3,4,5)){$s.Shapes.Item($k).Delete()}}
 $title=$s.Shapes.Item('文本框 2').TextFrame.TextRange
 $title.Text='三、业务计划-研究部-'+@('新技术','新策略')[$j-1]
 $title.Font.Name='微软雅黑';$title.Font.Size=28;$title.Font.Bold=-1
 $s.Shapes.Item('灯片编号占位符 2').TextFrame.TextRange.Text=[string]$j
 for($k=1;$k -le $body.Slides.Item($j).Shapes.Count;$k++){
  $body.Slides.Item($j).Shapes.Item($k).Copy();$r=$s.Shapes.Paste();$sh=$r.Item(1)
  if($sh.HasTextFrame){
   $sh.TextFrame.MarginTop=0;$sh.TextFrame.MarginBottom=0;$sh.TextFrame.MarginLeft=0;$sh.TextFrame.MarginRight=0
   $sh.TextFrame.TextRange.ParagraphFormat.LineRuleWithin=-1
   $sh.TextFrame.TextRange.ParagraphFormat.SpaceWithin=1.2
   if($sh.Name -like 'bullet*'){
    $sh.TextFrame.MarginLeft=10
    $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Visible=-1
    $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Character=10146
    $sh.TextFrame.TextRange.ParagraphFormat.Bullet.Font.Name='Segoe UI Symbol'
   }
  }
  if($sh.HasTable){
   $t=$sh.Table
   for($a=1;$a -le $t.Rows.Count;$a++){for($b=1;$b -le $t.Columns.Count;$b++){
    $c=$t.Cell($a,$b).Shape
    $c.TextFrame.TextRange.Font.Name='微软雅黑';$c.TextFrame.TextRange.Font.Size=14
    $c.TextFrame.MarginLeft=8;$c.TextFrame.MarginRight=5;$c.TextFrame.MarginTop=3;$c.TextFrame.MarginBottom=3
    $c.TextFrame.TextRange.ParagraphFormat.LineRuleWithin=-1;$c.TextFrame.TextRange.ParagraphFormat.SpaceWithin=1.12
   }}
   if($j -eq 1 -and $sh.Top -lt 200){$t.Rows.Item(1).Height=20;$t.Rows.Item(2).Height=66}
  }
 }
 $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text='依据用户周报、月报、项目代码与用户提供的2027年度经营计划_量化研究_结合月报周报_更新版.pptx拟定。状态门控、多任务学习与AI配权均为待验证研究，季度节点及交付物是2027年计划。指标可输出1、3、6、12、36个月历史分位，须满足历史长度。模型调参使用验证集，独立测试仅用于最终检验，边界按标签期限隔离。仅将通过样本外与成本检验的原型纳入模拟盘，不承诺收益。'
}
$p.SaveAs("$root\output\2027年度计划_许佳宜_AI量化研究_充实重排版.pptx",24)
for($j=1;$j -le 2;$j++){$p.Slides.Item($j).Export("$root\.ppt_build\rich$j.png",'PNG',1600,900)}
$body.Close();$p.Close();$app.Quit()
