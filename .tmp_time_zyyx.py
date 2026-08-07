import importlib.util, numpy as np, pymssql, time, sys
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
p=r'v2/dataloader/stock/update_zyyx_xjy.py'; s=importlib.util.spec_from_file_location('zyyx_xjy',p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m)
ticks=np.load(r'D:\data\axis\ticks.npy',allow_pickle=True); date='2024-06-28'; cfg=m.AnalystFactorConfig()
url=URL.create('mssql+pymssql',username='zyyxReader',password='zyyx!5893@Fund',host='10.110.0.106',database='zyyx',query={'charset':'utf8'}); engine=create_engine(url,connect_args={'tds_version':'7.0','charset':'utf8'}); conn=engine.connect(); jy=pymssql.connect(server='10.10.0.102',user='jydbReader',password='jy@9043!Reader',database='jydb',charset='cp936')
def run(name, fn):
 t=time.time(); x=fn(); print(name, getattr(x,'shape',None), round(time.time()-t,2),flush=True); return x
run('annual23',lambda:m.load_annual_actuals(2022,'2023-06-28',ticks,jy))
run('forecast23',lambda:m.load_forecasts(2022,conn,'2023-06-28',cfg,False))
run('annual24',lambda:m.load_annual_actuals(2023,'2024-06-28',ticks,jy))
run('forecast24',lambda:m.load_forecasts(2023,conn,'2024-06-28',cfg,False))
run('annual25',lambda:m.load_annual_actuals(2024,'2025-06-28',ticks,jy))
run('forecast25',lambda:m.load_forecasts(2024,conn,'2025-06-28',cfg,False))
run('forecast22',lambda:m.load_forecasts(2021,conn,'2022-06-28',cfg,False))
run('annual23',lambda:m.load_annual_actuals(2022,'2023-06-28',ticks,jy))
run('forecast23',lambda:m.load_forecasts(2022,conn,'2023-06-28',cfg,False))
run('annual24',lambda:m.load_annual_actuals(2023,'2024-06-28',ticks,jy))
run('forecast24',lambda:m.load_forecasts(2023,conn,'2024-06-28',cfg,False))



