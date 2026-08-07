import time
from sqlalchemy import create_engine,text
from sqlalchemy.engine import URL
u=URL.create('mssql+pymssql',username='zyyxReader',password='zyyx!5893@Fund',host='10.110.0.106',database='zyyx',query={'charset':'utf8'}); e=create_engine(u,connect_args={'tds_version':'7.0','charset':'utf8'}); c=e.connect()
queries={
'period':"SELECT COUNT(*) FROM rpt_forecast_stk f JOIN rpt_report_author ra ON ra.report_id=f.report_id WHERE f.create_date>='2022-05-31' AND f.entrytime<='2026-05-29'",
'industry':"SELECT COUNT(*) FROM qt_indus_constituents WHERE standard_code='908' AND industry_level=2",
'pairs':"WITH p AS (SELECT DISTINCT ra.author_id,f.stock_code FROM rpt_forecast_stk f JOIN rpt_report_author ra ON ra.report_id=f.report_id WHERE f.create_date>='2025-05-29' AND f.entrytime<='2026-05-29') SELECT COUNT(*) FROM p"
}
for n,q in queries.items():
 t=time.time(); print(n,c.execute(text(q)).scalar(),round(time.time()-t,2),flush=True)
