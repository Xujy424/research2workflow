from sqlalchemy import create_engine,text
from sqlalchemy.engine import URL
u=URL.create('mssql+pymssql',username='zyyxReader',password='zyyx!5893@Fund',host='10.110.0.106',database='zyyx',query={'charset':'utf8'})
e=create_engine(u,connect_args={'tds_version':'7.0','charset':'utf8'})
with e.connect() as c:
 for row in c.execute(text("SELECT COLUMN_NAME,DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='rpt_author_information' ORDER BY ORDINAL_POSITION")):
  print(row)
