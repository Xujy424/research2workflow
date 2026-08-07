from sqlalchemy import create_engine,text
from sqlalchemy.engine import URL
u=URL.create('mssql+pymssql',username='zyyxReader',password='zyyx!5893@Fund',host='10.110.0.106',database='zyyx',query={'charset':'utf8'})
e=create_engine(u,connect_args={'tds_version':'7.0','charset':'utf8'})
q='''SELECT TOP 50 y1,m1,y2,m2,COUNT(*) n FROM rpt_author_information
WHERE (y1 IS NOT NULL AND (y1<1753 OR y1>9999 OR COALESCE(m1,1) NOT BETWEEN 1 AND 12))
   OR (y2 IS NOT NULL AND (y2<1753 OR y2>9999 OR COALESCE(m2,1) NOT BETWEEN 1 AND 12))
GROUP BY y1,m1,y2,m2 ORDER BY n DESC'''
with e.connect() as c:
 for row in c.execute(text(q)): print(row)
