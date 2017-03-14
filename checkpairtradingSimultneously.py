#coding:utf-8
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

'''
    linear regression with Stochastic Gradient Decent mothod
'''
def linregSGD(x, y, a, b):
# -------------------------------------------随机梯度下降算法----------------------------------------------------------

    # 两种终止条件
    loop_max = 10000   # 最大迭代次数(防止死循环)
    epsilon = 1e-6    

    alpha = 0.001       # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
    diff = 0.           
    errorA = a
    errorB = b
    count = 0           # 循环次数
    finish = 0          # 终止标志
    m = len(x) # 训练数据点数目

    while count < loop_max:
        count += 1

        # 遍历训练数据集，不断更新权值
        for i in range(m):  
            diff = a + b * x[i] - y[i]  # 训练集代入,计算误差值

            # 采用随机梯度下降算法,更新一次权值只使用一组训练数据
            a = a - alpha * diff
            b = b - alpha * diff * x[i]

            # ------------------------------终止条件判断-----------------------------------------
            # 若没终止，则继续读取样本进行处理，如果所有样本都读取完毕了,则循环重新从头开始读取样本进行处理。

        # ----------------------------------终止条件判断-----------------------------------------
        # 注意：有多种迭代终止条件，和判断语句的位置。终止判断可以放在权值向量更新一次后,也可以放在更新m次后。
        if ((a-errorA)*(a-errorA) + (b-errorB)*(b-errorB)) < epsilon:     # 终止条件：前后两次计算出的权向量的绝对误差充分小  
            finish = 1
            break
        else:
            errorA = a
            errorB = b
    print 'loop count = %d' % count,  '\tweight:[%f, %f]' % (a, b)
    return a, b

def adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b):

    if len(closeprice_of_1) != 0 and len(closeprice_of_2) != 0:
        alpha, beta = linregSGD(x=closeprice_of_1, y=closeprice_of_2, a=a, b=b)

        spread = closeprice_of_2 - closeprice_of_1*beta - alpha
        adfstat, pvalue, usedlag, nobs, critvalues, icbest = sts.adfuller(x=spread)

        return adfstat < critvalues['5%']
'''        
        print adfstat
        for(k, v) in critvalues.items():
            print k, v
'''

def adfuller_check_smols(closeprice_of_1, closeprice_of_2):

    if len(closeprice_of_1) != 0 and len(closeprice_of_2) != 0:
        X = sm.add_constant(closeprice_of_1)
        model = sm.OLS(endog=closeprice_of_2, exog=X)
        result = model.fit()
        #print result.summary()
        print result.params
        spread = result.resid
        adfstat, pvalue, usedlag, nobs, critvalues, icbest = sts.adfuller(x=spread)

        return adfstat < critvalues['5%']
'''
        print adfstat
        for(k, v) in critvalues.items():
            print k, v
'''
'''
        spread2 = closeprice_of_2 - closeprice_of_1*result.params.closel
        sta2 = sts.adfuller(spread, 1)
        print sta2
'''
def compare_algorithm(code1, code2, start_date = '2013-10-10', end_date = '2014-09-30'):
    
    closeprice_of_1, closeprice_of_2 = load_process(code1, code2, start_date, end_date)

    time1 = time.time()
    result = adfuller_check_smols(closeprice_of_1, closeprice_of_2)
    time2 = time.time()
    print "smols running time(s): ", time2-time1
    
    time3 = time.time()
    a = -1
    b = -1
    result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
    time4 = time.time()
    print "sgd running time(s): ", time4-time3

    time5 = time.time()
    a = 0
    b = 0
    result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
    time6 = time.time()
    print "sgdmiddle running time(s): ", time6-time5

def adfuller_check_price_sgd(code1, code2, start_date = '2011-10-10', end_date = '2014-09-30', linrreg="SMOLS"):

    closeprice_of_1, closeprice_of_2 = load_process(code1, code2, start_date, end_date)

    if linrreg == "SMOLS" :
        result = adfuller_check_smols(closeprice_of_1, closeprice_of_2)
    elif linrreg == "SGD" :
        a = 0
        b = 0
        result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
    elif linrreg == "SGDMiddle" :
        a = 0
        b = 0
        result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
    else :
        result = ""
    return result

def load_process(code1, code2, start_date, end_date):
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
    return closeprice_of_1, closeprice_of_2

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

    #adfuller_check_price("601002", "600815")
    # chedk all stock pairing in list book
    #check_all_dir()

    #adfuller_check_price_sgd("601002", "600815",start_date = '2013-10-10', 
    #            end_date = '2014-09-30', linrreg="SMOLS")           #"SGD")
    compare_algorithm("601002", "600815",start_date = '2013-10-10', end_date = '2014-09-30')

    time2 = time.time()
    print "running time(s): ", time2-time1
if __name__ == "__main__":
    # Execute Main functionality
    main()