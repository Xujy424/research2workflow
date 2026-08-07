import pymssql,time
from sqlalchemy import create_engine,text
from sqlalchemy.engine import URL
u=URL.create('mssql+pymssql',username='zyyxReader',password='zyyx!5893@Fund',host='10.110.0.106',database='zyyx',query={'charset':'utf8'})
t=time.time(); e=create_engine(u,connect_args={'tds_version':'7.0','charset':'utf8'}); c=e.connect(); print('zyyx connect',time.time()-t, c.execute(text('select 1')).scalar(),flush=True)
t=time.time(); j=pymssql.connect(server='10.10.0.102',user='jydbReader',password='jy@9043!Reader',database='jydb',charset='cp936'); q=j.cursor(); q.execute('select 1'); print('jy connect',time.time()-t,q.fetchone(),flush=True)
