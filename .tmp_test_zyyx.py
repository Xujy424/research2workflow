import importlib.util, numpy as np, pymssql, time, sys
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
p=r'v2/dataloader/stock/update_zyyx_xjy.py'
s=importlib.util.spec_from_file_location('zyyx_xjy',p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m)
ticks=np.load(r'D:\data\axis\ticks.npy',allow_pickle=True)
url=URL.create('mssql+pymssql',username='zyyxReader',password='zyyx!5893@Fund',host='10.110.0.106',database='zyyx',query={'charset':'utf8'})
engine=create_engine(url,connect_args={'tds_version':'7.0','charset':'utf8'}); conn=engine.connect()
jy=pymssql.connect(server='10.10.0.102',user='jydbReader',password='jy@9043!Reader',database='jydb',charset='cp936')
t=time.time(); x=m.calc_con_forecast('2024-06-28',ticks,conn,jy,m.AnalystFactorConfig()); print(x.shape, x.head(), 'seconds',time.time()-t)

