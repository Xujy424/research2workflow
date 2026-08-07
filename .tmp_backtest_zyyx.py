import importlib.util,sys,numpy as np,pymssql,time
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
print('before import',flush=True); p=r'v2/dataloader/stock/update_zyyx_xjy.py'; s=importlib.util.spec_from_file_location('zyyx_xjy',p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m)
print('after import',flush=True); ticks=np.load(r'D:\data\axis\ticks.npy',allow_pickle=True); cfg=m.AnalystFactorConfig()
print('after ticks',flush=True); u=URL.create('mssql+pymssql',username='zyyxReader',password='zyyx!5893@Fund',host='10.110.0.106',database='zyyx',query={'charset':'utf8'}); e=create_engine(u,connect_args={'tds_version':'7.0','charset':'utf8'}); c=e.connect(); print('after zyyx',flush=True); j=pymssql.connect(server='10.10.0.102',user='jydbReader',password='jy@9043!Reader',database='jydb',charset='cp936')
for name in ['load_annual_actuals','load_forecasts','get_fit_data','load_analyst_history','build_pmafe_samples','calc_accwt','load_month_end_dates']:
    original=getattr(m,name)
    def timed(*args,_name=name,_fn=original,**kwargs):
        t0=time.time(); print('start',_name,flush=True)
        result=_fn(*args,**kwargs)
        print('done',_name,getattr(result,'shape',None),round(time.time()-t0,2),flush=True)
        return result
    setattr(m,name,timed)

t=time.time(); out=m.backtest_consensus('2026-05-01','2026-06-30',ticks,c,j,cfg,5); print('seconds',time.time()-t); print(out['correlations']); print(out['summary']); out['correlations'].write_csv('.tmp_consensus_corr.csv'); out['summary'].write_csv('.tmp_consensus_groups.csv'); out['cumulative_returns'].write_csv('.tmp_consensus_cumulative.csv')



