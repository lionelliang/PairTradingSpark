#coding:utf-8
## Spark Application - execute with spark-submit

## Imports
import csv
import os
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as sts

from StringIO import StringIO
from pyspark import SparkConf, SparkContext

import pymongo_spark
from pymongo import MongoClient

pymongo_spark.activate()

## Module Constants
APP_NAME = "ADF Spark Application"
TABLE_STOCKS_BASIC = 'stock_basic_list'
TABLE_STOCKS_PAIRS = 'stock_pairing_list45'
TABLE_WEIGHT = 'stock_linrreg.csv'
DownloadDir = './stockdata/'
weightdict = {}     #previous weight dict broadcast

#mongo db config
MONGO_HOST = '127.0.0.1'
MONGO_TABLE_WEIGHT = 'stock.linrreg'
MONGO_TABLE_WEIGHT_SAVED = 'stock.linrregsaved'
MONGO_TABLE_STOCKS_PAIRS = 'stock.pairs'
MONGO_TABLE_STOCKS_PAIRS_ALL = 'stock.pairsall'
MONGO_DB_QUOTATION = 'quotation'
MONGO_TABLE_PREFIX = 'kline_'

## Closure Functions
#date example 2011/10/13
tudateparser = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

def save_stk_pairings():
    stock_list = pd.read_csv(TABLE_STOCKS_BASIC + '.csv', dtype=str)

    list_code = stock_list['code'].values.tolist()
    #list_code = list_code[100:200]

    print len(list_code)
    list_pool = list(itertools.combinations(list_code, 2))
    stockPool = pd.DataFrame(list_pool, columns=['code1','code2'])
    print stockPool.head()
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

# 功能：从csv文件中读取一个字典
# 输入：文件名称，keyIndex,valueIndex
def readDictCSV(fileName="", dataDict = {}):
    if not os.path.exists(fileName) :
        return {}
    with open(fileName, "r") as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            dataDict[str(row[0])] = [float(row[1]), float(row[2])]
        csvFile.close()
    return dataDict

# 功能：将一字典写入到csv文件中
# 输入：文件名称，数据字典
def writeDictCSV(fileName="", dataDict={}):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for k,v in dataDict.iteritems():
            csvWriter.writerow([str(k), v[0], v[1]])
        csvFile.close()

def writeRddCSV(fileName, rdd, sqlContext):
    df = sqlContext.createDataFrame(rdd)
    #print df.first()
    #df.write.format("com.databricks.spark.csv").save(fileName)
    df.toPandas().to_csv(fileName, header=False, index=False)
    '''
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        rdd.foreach(lambda elem: writeElem(csvWriter, elem)) 
        csvFile.close()
    '''
def writeElem(csvWriter, elem):
    csvWriter.writerow(elem[0], elem[1][1], elem[1][2])

def toCSVLine(data):
  return ','.join(str(d) for d in data)

'''
mongo db operation
'''
def _connect_mongo(host, port):
    """ A util for making a connection to mongo """
    conn = MongoClient(host, port)
    return conn

def get_connection_mongo(host, port):
    connMongo = MongoClient(host, port)
    return connMongo

def read_mongo(db, collection, query={}, column={}, host=MONGO_HOST, port=27017, username=None, password=None):
    """ Read from Mongo and Store into DataFrame """
    # Connect to MongoDB
    connMongo = _connect_mongo(host, port)
    dbMongo = connMongo[db]
    # Make a query to the specific DB and Collection
    cursor = dbMongo[collection].find(query, column)
    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))
    connMongo.close()
    return df

def readCollectionMongo(collection):
    return sc.mongoRDD('mongodb://'+MONGO_HOST+':27017/'+collection)

def writeCollectionMongo(rdd, collection):
    rdd.saveToMongoDB('mongodb://'+MONGO_HOST+':27017/'+collection)

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
    finish = False          # 终止标志
    m = len(x)          # 训练数据点数目

    while count < loop_max:
        #count += 1

        # 遍历训练数据集，不断更新权值
        for i in range(m):
            count += 1
            diff = a + b * x[i] - y[i]  # 训练集代入,计算误差值

            # 采用随机梯度下降算法,更新一次权值只使用一组训练数据
            a = a - alpha * diff
            b = b - alpha * diff * x[i]

            if ((a-errorA)*(a-errorA) + (b-errorB)*(b-errorB)) < epsilon:     
                # 终止条件：前后两次计算出的权向量的绝对误差充分小  
                finish = 1
                break
            else:
                errorA = a
                errorB = b
        if finish == True:     # 跳出循环
            break

    #print 'loop count = %d' % count,  '\tweight:[%f, %f]' % (a, b)
    return finish, a, b

def adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b):

    if len(closeprice_of_1) >= 10 and len(closeprice_of_2) >= 10:
        # adfuller won't work if data is not enough
        finish, alpha, beta = linregSGD(x=closeprice_of_1, y=closeprice_of_2, a=a, b=b)

        if not finish:
            return False, a, b
        spread = closeprice_of_2 - closeprice_of_1*beta - alpha
        spread.dropna()
        adfstat, pvalue, usedlag, nobs, critvalues, icbest = sts.adfuller(x=spread, maxlag=1)

        return adfstat < critvalues['5%'], alpha, beta
    else:
        return False, 0, 0
'''        
        print adfstat
        for(k, v) in critvalues.items():
            print k, v
'''

def load_process(code1, code2, start_date, end_date):
    m = getSixDigitalStockCode(code1)
    n = getSixDigitalStockCode(code2)
    file1 = DownloadDir + "h_kline_" + m + ".csv"
    file2 = DownloadDir + "h_kline_" + n + ".csv"

    if (not os.path.exists(file1)) or (not os.path.exists(file1)):
        return {},{}

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

def load_process_data_mongo(code1, code2, start_date, end_date):
    m = getSixDigitalStockCode(code1)
    n = getSixDigitalStockCode(code2)
    collection1 = MONGO_TABLE_PREFIX + m
    collection2 = MONGO_TABLE_PREFIX + n

    query = {"date": {"$gte": start_date, "$lt": end_date}}
    column = {"date":1, "close":1, "_id":0}

    kline1 = read_mongo(MONGO_DB_QUOTATION, collection1, query, column)
    kline2 = read_mongo(MONGO_DB_QUOTATION, collection2, query, column)

    if kline1.empty or  kline2.empty:
        return {},{}

    kline1['date'] =  pd.to_datetime(kline1['date'], format='%Y-%m-%d')
    kline2['date'] =  pd.to_datetime(kline2['date'], format='%Y-%m-%d')
    kline1.index = kline1['date'].tolist()
    kline2.index = kline2['date'].tolist()

    #price_of_1 = kline1[end_date:start_date]
    #price_of_2 = kline2[end_date:start_date]
    price_of_1 = kline1
    price_of_2 = kline2

    # regroup quotation according to date index
    combination = price_of_1.join(price_of_2, how='inner', lsuffix='l', rsuffix='r')
    combination = combination.dropna()

    closeprice_of_1 = combination['closel'].reset_index(drop=True)
    closeprice_of_2 = combination['closer'].reset_index(drop=True)
    return closeprice_of_1, closeprice_of_2

def adfuller_check_price_sgd(code1, code2, start_date = '2013-10-10', end_date = '2015-09-30'):
    closeprice_of_1, closeprice_of_2 = load_process(code1, code2, start_date, end_date)
    if len(closeprice_of_1)<=1 or len(closeprice_of_1)<=1:
        return
#    time5 = time.time()
    if weightdict.has_key(code1+code2):     # get previous weight
        a = weightdict[code1+code2][0]
        b = weightdict[code1+code2][1]
        #print weightdict[code1+code2]
    else:
        #print "not find w"
        np.random.seed(2)
        a, b = np.random.randn(2)

    result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
#    time6 = time.time()
#    print "sgdmiddle running time(s): ", time6-time5
    weightdict[code1+code2] = [result[1], result[2]]    # update weight data
    return result[0]

def adfuller_check_sgd_withweight(code1, code2, a, b, start_date = '2013-10-10', end_date = '2015-09-30'):
    closeprice_of_1, closeprice_of_2 = load_process_data_mongo(code1, code2, start_date, end_date)
    #closeprice_of_1, closeprice_of_2 = load_process(code1, code2, start_date, end_date)
    if len(closeprice_of_1)<=1 or len(closeprice_of_1)<=1:
        #print "without data, you shall not pass"
        return {"stk1":code1, "stk2":code2, "flag":0, "a":0, "b":0}

    if not a or not b or (a==0 and b==0):     # get previous weight
        #print "not find w"
        #np.random.seed(2)
        #a, b = np.random.randn(2)
        a = 0
        b = 0

    result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)

    return {"stk1":code1, "stk2":code2, "flag":np.float64(result[0]).item(),  \
    "a":np.float64(result[1]).item(), "b":np.float64(result[2]).item()}

def adfuller_check(code1, code2, start_date = '2013-10-10', end_date = '2015-09-30'):
    m = getSixDigitalStockCode(code1)
    n = getSixDigitalStockCode(code2)
    file1 = DownloadDir + "h_kline_" + m + ".csv"
    file2 = DownloadDir + "h_kline_" + n + ".csv"
    if not os.path.exists(file1) or not os.path.exists(file1):
        return False

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
        spread = result.resid
        stat = sts.adfuller(x=spread)
        adf = stat[0]
        pvalue = stat[1]
        critical_values = stat[4]
        pair = m + '+' + n

        return adf < critical_values['5%']

def adfuller_check2(row):
    #return adfuller_check(row[0], row[1])
    #return adfuller_check_price_sgd(row[0], row[1], start_date = '2013-10-10', end_date = '2014-09-30')
	return adfuller_check_price_sgd(row[0], row[1], start_date = '2013-10-10', end_date = '2015-09-30')

def adfuller_check4(code1, code2, a, b):
    return adfuller_check_sgd_withweight(code1, code2, a, b, start_date = '2013-10-10', end_date = '2015-09-30')

def check_all_dir(sc):
    
    stockPool = readCollectionMongo(MONGO_TABLE_STOCKS_PAIRS)      # load weight file
    print stockPool.take(2)

    print "starting adf checking"

    #adfResult = stockPool.map(adfuller_check2)
    #adfResult = stockPool.filter(adfuller_check2)
    # row seems to be a dict
    adfResult  = stockPool.map(lambda f: (adfuller_check4(f["stk1"], f["stk2"], f["a"], f["b"])))

    #adfResult.collect()

    print "%d <<<pairings" % adfResult.count()
    print adfResult.first()

    print "write to mongo db"
    try:
        writeCollectionMongo(adfResult, MONGO_TABLE_WEIGHT_SAVED)
    except Exception, e:
        writeCollectionMongo(adfResult, MONGO_TABLE_WEIGHT_SAVED)
    

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
