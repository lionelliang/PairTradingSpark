## Spark Application - execute with spark-submit

## Imports
import csv
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
from StringIO import StringIO
from collections import namedtuple
from operator import add, itemgetter
from pyspark import SparkConf, SparkContext
from pyspark import SQLContext

## Module Constants
APP_NAME = "ADF Spark Application"
TABLE_STOCKS_BASIC = 'stock_basic_list'
TABLE_STOCKS_PAIRS = 'stock_pairing_list'
DownloadDir = './stockdata/'
## Closure Functions

def save_stk_pairings():
	stock_list = pd.read_csv(TABLE_STOCKS_BASIC + '.csv', dtype=str)
	code = stock_list['code']
	reindexed_code = code.reset_index(drop=True)
	reindexed_code = reindexed_code[100:200]
	reindexed_code = reindexed_code.reset_index(drop=True)
	stockPool = pd.DataFrame(columns=['code1','code2'], dtype=str)
	print len(reindexed_code)

	for i in range(len(reindexed_code)):
	    for j in range(i+1, len(reindexed_code)):
	        stockPool = stockPool.append({'code1':str(reindexed_code[i]), 'code2':str(reindexed_code[j])}, ignore_index=True)

	stockPool.to_csv(TABLE_STOCKS_PAIRS + '.csv', header=False, index=False)

# input: int or string
# output: string
def getSixDigitalStockCode(code):
    strZero = ''
    for i in range(len(str(code)), 6):
        strZero += '0'
    return strZero + str(code)

def split(line):
	"""
	Operator function for splitting a line with csv module
	"""
	reader = csv.reader(StringIO(line))
	return reader.next()

#date example 2011/10/13
tudateparser = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
def adfuller_check(code1, code2, start_date = '2011-10-10', end_date = '2014-09-30'):
    m = getSixDigitalStockCode(code1)
    n = getSixDigitalStockCode(code2)
    file1 = DownloadDir + "h_kline_" + m + ".csv"
    file2 = DownloadDir + "h_kline_" + n + ".csv"
    if not os.path.exists(file1) or not os.path.exists(file1):
        return

    kline1 = pd.read_csv(file1, parse_dates=['date'], index_col='date', date_parser=tudateparser)
    kline2 = pd.read_csv(file2, parse_dates=['date'], index_col='date', date_parser=tudateparser)
    #print kline1.head()
    price_of_1 = kline1[end_date:start_date]
    price_of_2 = kline2[end_date:start_date]

    closeprice_of_1 = price_of_1['close'].dropna()
    closeprice_of_2 = price_of_2['close'].dropna()

    print combination.head()

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

def adfuller_check2(row):
	return adfuller_check(row[0], row[1])

def check_all_dir(sc):
    print 'starting adf checking'
    stockPool = sc.textFile(TABLE_STOCKS_PAIRS + '.csv').map(split)
    stats = stockPool.map(adfuller_check2)
    print stats.first()
    print stats.count()

## Main functionality
def main(sc):
    time1 = time.time()
    #adfuller_check2("601002", "600815")
    # check all stock pairing in list book
    #save_stk_pairings()
    check_all_dir(sc)

    time2 = time.time()
    print "running time(s): ", time2-time1

if __name__ == "__main__":
	# Configure Spark
	conf = SparkConf().setAppName(APP_NAME)
	conf = conf.setMaster("local[*]")
	sc = SparkContext(conf=conf)

	# Execute Main functionality
	main(sc)
