# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:47:11 2021

@author: huangyue
"""


'''
文档说明：

'''
import pymssql
import argparse
import ZSFundRequestClient as zs
import sys
import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 连接聚源的参数
server_jydb = "10.10.0.102"
user_jydb = "jydb"
password_jydb = "jydb"

# 连接infodb的参数
server_zs = "infodb"

d1 = datetime.timedelta(days=1)

'''本项目的数据在FI_SeriesIndicators里面以CB_fit_PremiumRate_with_ConvertValue_开头'''
variable_prefix = 'FICC|LiquidityInidcatorNarrowSense'    # 狭义流动性指标

# %% 类
class CB_industry_info:  
    
    # @staticmethod
    def get_tradingdaySeries(cursor,begdate,enddate):
        df=0
        return df
    
    @staticmethod
    def get_table(**kwargs):
        begdate = kwargs.get('begdate', None)
        enddate = kwargs.get('enddate', None)
        
        pass

    @staticmethod
    def get_picdata(**kwargs):
        date_df = CB_industry_info.get_tradingdaySeries(cursor_zs,updatedate,updatedate)
        pass
    
# %% 提取数据
# 交易日日期序列
def get_tradingdate(cursor,begdate,enddate):
    str_sql = '''
            select TradingDate from QT_TradingDayNew
            where
            SecuMarket = 83 
            and IFTradingDay=1
            and TradingDate between '%s' and '%s' 
            order by TradingDate'''% (begdate,enddate)

    # 数据提取
    cursor.execute(str_sql)
    date_df = pd.DataFrame(cursor.fetchall(),columns=['date'])    
    
    return date_df

'''提取一个表中多列数据：示例'''
def get_many_data(cursor,begdate,enddate=None):
    str_columns = 'TradingDay,MaturityType,\
        WeightAveragePrice'.replace(' ','')  # 删除所有空格
    list_columns = str_columns.split(',')
    if enddate is None:
        str_sql = '''
                SELECT %s
                FROM Bond_RepuExIBQuote 
                WHERE
                MaturityType in ('R001','DR001','R007','DR007')    --早上七点一刻更新
                AND RepurchaseNature in (21,22)
                AND TradingDay = '%s'
                ORDER BY TradingDay,MaturityType
                '''% (str_columns,begdate)
    else:
        str_sql = '''
                SELECT %s
                FROM Bond_RepuExIBQuote 
                WHERE
                MaturityType in ('R001','DR001','R007','DR007')    --早上七点一刻更新
                AND RepurchaseNature in (21,22)
                AND TradingDay between '%s' and '%s'
                ORDER BY TradingDay,MaturityType
                '''% (str_columns,begdate,enddate)
    cursor.execute(str_sql)
    
    tmp_table = pd.DataFrame(cursor.fetchall(), columns=list_columns)            

    return tmp_table

'''提取一个表中单列数据：示例'''
def get_data(cursor,begdate,enddate=None):
    if enddate is None:
        str_sql = '''
            SELECT TradingDay,TenderRate FROM Bond_CBRepo
            where RepurchaseVariety = 1 and RepurchaseType = 2 --只保留了7天逆回购
            and TradingDay > '%s'
            ORDER BY TradingDay
                '''% (begdate)
    else:
        str_sql = '''
            SELECT TradingDay,TenderRate FROM Bond_CBRepo
            where RepurchaseVariety = 1 and RepurchaseType = 2 --只保留了7天逆回购
            and TradingDay between '%s' and '%s'
            ORDER BY TradingDay
                '''% (begdate,enddate)

    cursor.execute(str_sql)
    tmp_table = pd.DataFrame(cursor.fetchall(), columns=['date','rate']).astype({'rate':float})          

    return tmp_table   

'''提取EDB数据'''
def get_EDB_data(ID,IDname ='rate'):
    Zs = zs.RequestZSFundDateServer()
    EDB_data = Zs.request_client('http://dataservice/api/EDBData/New?code='+ID,None,'get')
    EDB_data = pd.DataFrame(EDB_data[0]['CodeData']).rename(columns={'FDate':'date','Value':IDname})\
        .astype({'date':'<M8[ns]'})
    return EDB_data

'''提取WSD数据'''
def get_WSD_data(secCodes = '127044.SZ',indicator  = 'ytm_cb',
                 startDate = '2023-01-01',endDate = '2024-02-01'):
    Zs = zs.RequestZSFundDateServer()
    tmpurl = 'http://dataservice/api/WSDQuant/New?secCodes='+secCodes\
                +'&indicator='+indicator+'&startDate='+ startDate \
                +'&endDate='+endDate+'&isFromCache=false&param=Period%3DD'
    
    WSD_data = Zs.request_client(tmpurl, None,'get')
    
    WSD_data = pd.DataFrame(WSD_data[0]['CodeData']).rename(columns={'FDate':'date','Value':indicator})\
        .astype({'date':'<M8[ns]'})
    return WSD_data

test = get_WSD_data(secCodes='000015.OF',indicator='fund_inital')

'''提取历史保存的数据'''
def get_existedinfo(cursor, variable_prefix, begdate=None, enddate=None):
    # 提取历史上的模型拟合数据
    # enddate当天的数据不提取
    if (begdate==enddate) | (begdate==None):
        str_existedinfo = '''
                        select * from FI_SeriesIndicators
                        where Indicator like '%s_%%'
                        and KeyDate < '%s'
                        ''' %(variable_prefix,enddate)
    else:
        str_existedinfo = '''
                        select * from FI_SeriesIndicators
                        where Indicator like '%s_%%'
                        and KeyDate>='%s' and KeyDate < '%s'
                        ''' %(variable_prefix,begdate,enddate)
    cursor.execute(str_existedinfo)
    existed_info = pd.DataFrame(cursor.fetchall(),
                                columns=['Indicator','SecuCode','KeyDate','Value'])
    
    # 修正DataFrame的数据保存方式
    existed_info = pd.pivot(existed_info, values='Value', index='KeyDate', columns='Indicator')
    existed_info.columns = list(a[len(variable_prefix)+1:] for a in existed_info.columns)
    existed_info = existed_info.reset_index().rename(columns={'KeyDate':'date'}).astype({'date':'<M8[ns]'})
 
    return existed_info

'''获取更新的日期'''
def getParams():  # 获取参数
    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('--InsertDate', type=str, default=None, help='插入日期')
    args = parser.parse_args()
    return args.InsertDate

'''检查数据是否更新完毕'''
def check_data_update(check_data,updatedate,check_all=False):
    if (check_data[check_data['date']==updatedate].shape[0]==0):
        # 没有更新update日期的数据
        exit_main()
    if check_all & (check_data[check_data['date']==updatedate].isna().any(axis=1).values[0]):
        # 没有更新完毕update日期的数据
        exit_main()

# %% 数据清洗
# 宽表转长表
def reverse_pivot(df):
    '''
    传入一个宽表：
    
    index  col1     col2     col3  
    1      value11  value12  value13
    2      value21  value22  value23
    3      value31  value32  value33
    
    返回一张长表:
    multiindex  value    
    1  col1    value11  
    1  col2    value12  
    1  col3    value13  
    2  col1    value21  
    2  col2    value22  
    2  col3    value23  
    3  col1    value31  
    3  col2    value32  
    3  col3    value33  
    '''
    df_new = pd.DataFrame(columns = ['value'],
                     data = np.concatenate(df.values),
                     index = pd.MultiIndex.from_product([df.index,df.columns]))
    return df_new

def get_clean_data_for_model():

    pass

# %% 数据输出
'''输出指标'''
def save_indicator(save_data,variable_prefix,indiname):
    save_data = pd.DataFrame({'Value':save_data[indiname].values,
                              'KeyDate':save_data['date'].apply(lambda x:x.strftime('%Y-%m-%d'))}).dropna()
    save_data['SecuCode'] = variable_prefix+'_'+indiname
    save_data['Indicator'] = variable_prefix+'_'+indiname
    save_list = list(tuple(a) for a in save_data[['Indicator','SecuCode','KeyDate','Value']].values)
    cursor_zs.executemany("INSERT INTO FI_SeriesIndicators(Indicator,SecuCode,KeyDate,Value) VALUES (%s, %s, %s, %s)",save_list)
    conn_zs.commit()

'''退出主函数'''
def exit_main():
    cursor_jydb.close()
    conn_jydb.close()
    cursor_zs.close()
    conn_zs.close()     
    sys.exit(0)
    
# %% main
if __name__ == '__main__':
    conn_zs = pymssql.connect(host=server_zs, database="DataCenter", charset='cp936')
    cursor_zs = conn_zs.cursor()
    
    conn_jydb = pymssql.connect(server_jydb, user_jydb, password_jydb, 'JYDB', charset='utf8')
    cursor_jydb = conn_jydb.cursor()

    '''
    ##########################################################################################
    # step1：确定本次更新的时间范围
    ##########################################################################################
    '''   
    # updatedate = '2021-07-07'
    updatedate = getParams()
    begdate = datetime.datetime(2014,1,1).strftime('%Y-%m-%d') 
    
    # 将更新日期设置为最近的一个交易日
    date_df = get_tradingdate(cursor_jydb, begdate, updatedate)
    updatedate = date_df['date'].iloc[-1].strftime('%Y-%m-%d')
    enddate = updatedate
    
    '''
    ##########################################################################################
    # step2：提取数据
    ##########################################################################################
    '''
    # (1) 提取数据


    # 关闭服务器
    cursor_jydb.close()
    conn_jydb.close()
    
    # (2) 如果提取的数据范围不符合预期，就退出程序
    if (repo_data_raw[repo_data_raw['date']==updatedate].shape[0]==0):
        # 没有更新update日期的数据
        exit_main()
    if repo_data_raw[repo_data_raw['date']==updatedate][['R001','R007','DR001','DR007']].isna().any(axis=1).values[0]:
        # 没有更新完毕update日期的数据
        exit_main()
        

    
    # (3)提取历史数据
    existed_info = get_existedinfo(cursor_zs, variable_prefix, begdate=begdate, enddate=enddate)

    '''
    ##########################################################################################
    # step3：模型拟合 
    ##########################################################################################
    '''
 
    
 
    '''
    ##########################################################################################
    # step4：数据存储
    ##########################################################################################
    '''
    conn_zs = pymssql.connect(host=server_zs, database="DataCenter", charset='cp936')
    cursor_zs = conn_zs.cursor()
    # (1)清空本项目当天的所有数据
    cursor_zs.execute('''
                      DELETE FROM FI_SeriesIndicators 
                      where Indicator like '%s_%%'
                      and KeyDate='%s' 
                      '''%(variable_prefix,updatedate))
    conn_zs.commit()
    
    # (2)提取需要保存的数据
    save_data = corridor_data[corridor_data['date']==updatedate]
    
    # (3)保存数据
    save_indicator(save_data[['date','dispersion']],variable_prefix,'dispersion')

    # 关闭服务器
    cursor_zs.close()
    conn_zs.close()

# %% 绘图查看
