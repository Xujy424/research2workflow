import zipfile, copy, re
from pathlib import Path
from lxml import etree as E
B=Path(__file__).parent
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
for k,v in ns.items(): E.register_namespace(k,v)
def xml(b): return E.fromstring(b)
def out(x): return E.tostring(x,xml_declaration=True,encoding='UTF-8',standalone=True)
def idof(s):
 n=s.find('.//p:cNvPr',ns)
 return n.get('id') if n is not None else ''
def settext(s,text,size=1400,bold=False):
 t=s.find('p:txBody',ns)
 if t is None:return
 for c in list(t):
  if c.tag.endswith('}p'):t.remove(c)
 for line in text.split('\n'):
  p=E.SubElement(t,'{'+ns['a']+'}p'); pp=E.SubElement(p,'{'+ns['a']+'}pPr',algn='ctr' if idof(s) not in ['3','6'] else 'l')
  r=E.SubElement(p,'{'+ns['a']+'}r'); rp=E.SubElement(r,'{'+ns['a']+'}rPr',sz=str(size),b='1' if bold else '0')
  E.SubElement(rp,'{'+ns['a']+'}latin',typeface='微软雅黑'); E.SubElement(rp,'{'+ns['a']+'}ea',typeface='微软雅黑')
  E.SubElement(r,'{'+ns['a']+'}t').text=line
with zipfile.ZipFile(B/'template.pptx') as z,zipfile.ZipFile(B/'body.pptx') as body:
 data={n:z.read(n) for n in z.namelist()}
 src=xml(data['ppt/slides/slide8.xml'])
 for j,num in enumerate([5,8]):
  root=copy.deepcopy(src);tree=root.find('p:cSld/p:spTree',ns)
  for s in list(tree):
   sid=idof(s)
   if sid in ['7','9']:tree.remove(s)
   elif sid=='3':settext(s,'三、业务计划-研究部-'+['新技术','新策略'][j],2800,True)
   elif sid=='5':settext(s,str(j+1),1100)
   elif sid=='6':settext(s,['业务目标：完善公告事件模型与文本因子的研究验证。','业务目标：完成多因子选股与风险控制的策略验证。'][j],1600)
   elif sid=='8':
    labels=[['事件样本库','模型对照检验','因子持续跟踪'],['多因子选股','风险约束组合','模拟盘复盘']][j]
    txts=[x for x in s.findall('.//p:sp',ns) if x.find('p:txBody',ns)is not None]
    for x,label in zip(txts,labels):settext(x,label,1600,True)
  b=xml(body.read(f'ppt/slides/slide{j+1}.xml')).find('p:cSld/p:spTree',ns)
  for k,shape in enumerate(list(b)[2:]):
   shape=copy.deepcopy(shape)
   shape.find('.//p:cNvPr',ns).set('id',str(100+k))
   for pr in shape.findall('.//a:rPr',ns):
    pr.set('sz','1400');E.SubElement(pr,'{'+ns['a']+'}ea',typeface='微软雅黑')
   for p in shape.findall('.//a:p',ns):
    pp=p.find('a:pPr',ns)
    if pp is None:pp=E.Element('{'+ns['a']+'}pPr');p.insert(0,pp)
    pp.set('marL','152400' if k%3 else '190500');pp.set('indent','-152400' if k%3 else '-190500')
    for c in list(pp):
     if c.tag.split('}')[-1] in ['buNone','buChar','buFont','lnSpc']:pp.remove(c)
    sp=E.SubElement(pp,'{'+ns['a']+'}lnSpc');E.SubElement(sp,'{'+ns['a']+'}spcPct',val='130000')
    E.SubElement(pp,'{'+ns['a']+'}buFont',typeface='Wingdings' if k%3 else 'Arial')
    E.SubElement(pp,'{'+ns['a']+'}buChar',char='➢' if k%3 else '▪')
   tree.append(shape)
  data[f'ppt/slides/slide{num}.xml']=out(root)
  rel=xml(data['ppt/slides/_rels/slide8.xml.rels'])
  for x in list(rel):
   if 'notesSlide' in x.get('Type',''):rel.remove(x)
  data[f'ppt/slides/_rels/slide{num}.xml.rels']=out(rel)
 pres=xml(data['ppt/presentation.xml']);lst=pres.find('p:sldIdLst',ns)
 keep=[]
 for idx,x in enumerate(list(lst),1):
  if idx in [5,8]:keep.append(copy.deepcopy(x))
 for x in list(lst):lst.remove(x)
 for x in keep:lst.append(x)
 data['ppt/presentation.xml']=out(pres)
 with zipfile.ZipFile(B/'draft.pptx','w',zipfile.ZIP_DEFLATED) as dst:
  for n,v in data.items():dst.writestr(n,v)
