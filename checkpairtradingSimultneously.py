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
#date example 2011/10/13
tudateparser = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

def adfuller_check_smols(closeprice_of_1, closeprice_of_2):

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

        return adf < critical_values['10%']
#       for(k, v) in critical_values.items():
#           print k, v
#        spread2 = closeprice_of_2 - closeprice_of_1*result.params.closel
#        sta2 = sts.adfuller(spread, 1)
#        print sta2

def simulate_check_5days(close1, close2):
    
    # check every 5days
    # period = 250 working days in a year
    jump = 1
    period = 250

    if close1.count() < period:
        return
    
    index_start = 0
    index_end = index_start + period

    while index_end < close1.count():
        part_close1 = close1[index_start:index_end]
        part_close2 = close2[index_start:index_end]
        bRet = adfuller_check_smols(part_close1, part_close2)
        index_start += jump
        index_end += jump
    print index_start/5

def adfuller_check_price(code1, code2, start_date = '2011-10-10', end_date = '2014-09-30'):
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

    # regroup quotation according to date index
    combination = price_of_1.join(price_of_2, how='inner', lsuffix='l', rsuffix='r')
    combination.dropna()

    closeprice_of_1 = combination['closel'].reset_index(drop=True)
    closeprice_of_2 = combination['closer'].reset_index(drop=True)
    return simulate_check_5days(closeprice_of_1, closeprice_of_2)

def adfuller_check2(df):
    adfuller_check_price(df[0], df[1])

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

## Main functionality
def main():
    time1 = time.time()

    adfuller_check_price("601002", "600815")
    # chedk all stock pairing in list book
    #check_all_dir()

    time2 = time.time()
    print "running time(s): ", time2-time1
if __name__ == "__main__":
    # Execute Main functionality
    main()