import os
import time
import operator
import multiprocessing
import numpy as np
import pandas as pd
import tushare as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sts
from datetime import datetime
from collections import namedtuple

TABLE_STOCKS_BASIC = 'stock_basic_list'
DownloadDir = './stockdata/'

def adfuller_check_smols(code1, code2, start_date = '2011-10-10', end_date = '2014-09-30'):
    m = str(code1)
    n = str(code2)
    file1 = DownloadDir + "h_kline_" + code1 + ".csv"
    file2 = DownloadDir + "h_kline_" + code2 + ".csv"
    if not os.path.exists(file1) or not os.path.exists(file1):
        return

    kline1 = pd.read_csv(file1, parse_dates=['date'], index_col='date', date_parser=tudateparser)
    kline2 = pd.read_csv(file2, parse_dates=['date'], index_col='date', date_parser=tudateparser)
    #print kline1.head()
    price_of_1 = kline1[end_date:start_date]
    price_of_2 = kline2[end_date:start_date]

    combination = price_of_1.join(price_of_2, how='inner', lsuffix='l', rsuffix='r')
    combination.dropna()

    closeprice_of_1 = combination['closel'].reset_index(drop=True)
    closeprice_of_2 = combination['closer'].reset_index(drop=True)
    
    if len(closeprice_of_1) != 0 and len(closeprice_of_2) != 0:
        X = sm.add_constant(closeprice_of_1)
        model = sm.OLS(endog=closeprice_of_2, exog=X)
        result = model.fit()
#        print result.summary()
        spread = result.resid
        stat = sts.adfuller(x=spread)
        adf = stat[0]
        pvalue = stat[1]
        critical_values = stat[4]
        pair = m + '+' + n

        return adf < critical_values['10%']
#       for(k, v) in critical_values.items():
#           print k, v
#        spread2 = closeprice_of_2 - closeprice_of_1*result.params.closel
#        sta2 = sts.adfuller(spread, 1)
#        print sta2

def adfuller_check_online(code1, code2):
#for i in range(len(potentialPair)):
    m = str(code1)
    n = str(code2)
    price_of_1 = ts.get_hist_data(m, start='2011-10-10', end='2014-09-30')
    price_of_2 = ts.get_hist_data(n, start='2011-10-10', end='2014-09-30')
    price_of_1.to_csv(code1+"20111010-2016-03-05.csv")
    price_of_2.to_csv(code1+"20111010-2016-03-05.csv")
    closeprice_of_1 = price_of_1['close']
    closeprice_of_2 = price_of_2['close']

    if len(closeprice_of_1) != 0 and len(closeprice_of_2) != 0:
        model = pd.ols(y=closeprice_of_2, x=closeprice_of_1, intercept=True)   # perform ols on these two stocks
        spread = closeprice_of_2 - closeprice_of_1*model.beta['x']
        spread = spread.dropna()
        sta = sts.adfuller(spread, 1)
        pair = m + '+' + n
        print pair + ": adfuller result " 
        print sta

#date example 2011/10/13
tudateparser = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
def adfuller_check(code1, code2, start_date = '2011-10-10', end_date = '2014-09-30'):
#for i in range(len(potentialPair)):
    m = str(code1)
    n = str(code2)
    file1 = DownloadDir + "h_kline_" + code1 + ".csv"
    file2 = DownloadDir + "h_kline_" + code2 + ".csv"
    if not os.path.exists(file1) or not os.path.exists(file1):
        return

    kline1 = pd.read_csv(file1, parse_dates=['date'], index_col='date', date_parser=tudateparser)
    kline2 = pd.read_csv(file2, parse_dates=['date'], index_col='date', date_parser=tudateparser)
    #print kline1.head()
    price_of_1 = kline1[end_date:start_date]
    price_of_2 = kline2[end_date:start_date]

    combination = price_of_1.join(price_of_2, how='inner', lsuffix='l', rsuffix='r')
    combination.dropna()
    
    closeprice_of_1 = combination['closel']
    closeprice_of_2 = combination['closer']

    if len(closeprice_of_1) != 0 and len(closeprice_of_2) != 0:
        model = pd.ols(y=closeprice_of_2, x=closeprice_of_1, intercept=True)   # perform ols on these two stocks
        spread = closeprice_of_2 - closeprice_of_1*model.beta['x']
        spread = spread.dropna()
        sta = sts.adfuller(spread, 1)
        pair = m + '+' + n
        return sta
'''
        print pair + ": adfuller result "
        print sta
'''
def adfuller_check2(df):
    adfuller_check_smols(df[0], df[1])

def adfuller_check3(df):
    print df
    adfuller_check(df.code1, df.code2)

def check_all_dir():
    print 'starting adf checking'
    stock_list = pd.read_csv(TABLE_STOCKS_BASIC + '.csv', dtype=str)
    code = stock_list['code']
    reindexed_code = code.reset_index(drop=True)
    reindexed_code = reindexed_code[100:200]
    reindexed_code = reindexed_code.reset_index(drop=True)
    stockPool = pd.DataFrame(columns=['code1','code2'])
    print len(reindexed_code)

    for i in range(len(reindexed_code)):
        for j in range(i+1, len(reindexed_code)):
            stockPool = stockPool.append({'code1':str(reindexed_code[i]), 'code2':str(reindexed_code[j])}, ignore_index=True)

    stockPool.apply(adfuller_check2, axis=1)
'''not working
    try:
        pool = multiprocessing.Pool(processes=2)
        pool.map(adfuller_check3, stockPool)
        pool.close()
        pool.join()

    except Exception as e:
        print str(e)
    print 'all stock checked'
'''

## Main functionality
def main():
    time1 = time.time()
    #adfuller_check2("601002", "600815")
    #adfuller_check_smols("601002", "600815")
    # chedk all stock pairing in list book
    check_all_dir()

    time2 = time.time()
    print "running time(s): ", time2-time1
if __name__ == "__main__":
    # Execute Main functionality
    main()