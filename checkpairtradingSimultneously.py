#coding:utf-8
import os
import time
import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as sts

TABLE_STOCKS_BASIC = 'stock_basic_list'
DownloadDir = './stockdata/'
DATA_TIME = 'running-time'
#date example 2011/10/13
tudateparser = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
weightdict = {}     #previous weight dict

# 功能：从csv文件中读取一个字典
# 输入：文件名称，keyIndex,valueIndex
def readDictCSV(fileName="", dataDict = {}):
    if not os.path.exists(fileName) :
        return
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

    if len(closeprice_of_1) != 0 and len(closeprice_of_2) != 0:
        finish, alpha, beta = linregSGD(x=closeprice_of_1, y=closeprice_of_2, a=a, b=b)

        if not finish:
            return False, a, b
        spread = closeprice_of_2 - closeprice_of_1*beta - alpha
        spread.dropna()
        adfstat, pvalue, usedlag, nobs, critvalues, icbest = sts.adfuller(x=spread)

        return adfstat < critvalues['5%'], alpha, beta
    else:
        return False, 0, 0
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
    print "trading days:", len(closeprice_of_1)
    if len(closeprice_of_1)<=1 or len(closeprice_of_1)<=1:
        return

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

    time7 = time.time()
    #a = 0
    #b = 0
    np.random.seed(2)
    a, b = np.random.randn(2)
    result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
    time8 = time.time()
    print "sgd00 running time(s): ", time8-time7

    time5 = time.time()
    #a = 0.189965
    #b = 0.4243
    if weightdict.has_key(code1+code2):     # get previous weight
        a = weightdict[code1+code2][0]
        b = weightdict[code1+code2][1]
        #print weightdict[code1+code2]
    else:
        #print "not find w"
        np.random.seed(2)
        a, b = np.random.randn(2)
    result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
    weightdict[code1+code2] = [result[1], result[2]]
    time6 = time.time()
    print "sgdmiddle running time(s): ", time6-time5

    time9 = time.time()
    if weightdict.has_key(code1+code2):
        a = weightdict[code1+code2][0]
        b = weightdict[code1+code2][1]
    else:
        print "not find w"
        np.random.seed(2)
        a, b = np.random.randn(2)
    result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
    weightdict[code1+code2] = [result[1], result[2]]
    time10 = time.time()

    print "sgdsavedvalue running time(s): ", time10-time9

def adfuller_check_price_sgd(code1, code2, linrreg="SMOLS", start_date = '2013-10-10', end_date = '2014-09-30'):

    closeprice_of_1, closeprice_of_2 = load_process(code1, code2, start_date, end_date)

    if linrreg == "SMOLS" :
        result = adfuller_check_smols(closeprice_of_1, closeprice_of_2)
    elif linrreg == "SGD" :
        a = 0
        b = 0
        result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
    elif linrreg == "SGDMiddle" :
        if weightdict.has_key(code1+code2):
            a = weightdict[code1+code2][0]
            b = weightdict[code1+code2][1]
        else:
            print "not find w"
            np.random.seed(2)
            a, b = np.random.randn(2)   
        result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
        weightdict[code1+code2] = [result[1], result[2]]
    else :
        result = ""
    return result


def load_process(code1, code2, start_date, end_date):
    m = str(code1)
    n = str(code2)
    file1 = DownloadDir + "h_kline_" + m + ".csv"
    file2 = DownloadDir + "h_kline_" + n + ".csv"
    if not os.path.exists(file1) or not os.path.exists(file1):
        return {}, {}

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


def adfuller_check_general(closeprice_of_1, closeprice_of_2, code1, code2, linrreg="SMOLS"):

    if linrreg == "SMOLS" :
        result = adfuller_check_smols(closeprice_of_1, closeprice_of_2)
    elif linrreg == "SGD" :
        a = 0
        b = 0
        result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
    elif linrreg == "SGDMiddle" :
        if weightdict.has_key(code1+code2):
            a = weightdict[code1+code2][0]
            b = weightdict[code1+code2][1]
        else:
            print "not find w"
            np.random.seed(2)
            a, b = np.random.randn(2)
        result = adfuller_check_sgd(closeprice_of_1, closeprice_of_2, a, b)
        weightdict[code1+code2] = [result[1], result[2]]
    elif linrreg == "SGDRegress" :
        a = 0
        b = 0
        result = linregSGD(x=closeprice_of_1, y=closeprice_of_2, a=a, b=b)
    elif linrreg == "SGDRegressMiddle" :
        if weightdict.has_key(code1+code2):
            a = weightdict[code1+code2][0]
            b = weightdict[code1+code2][1]
        else:
            print "not find w"
            np.random.seed(2)
            a, b = np.random.randn(2)
        result = linregSGD(x=closeprice_of_1, y=closeprice_of_2, a=a, b=b)
        weightdict[code1+code2] = [result[1], result[2]]
    else :
        result = ""
    return result


def simulate_check_5days(close1, close2, code1, code2):
    # check every 5days
    # period = 250 working days in a year
    jump = 5
    period = 250

    if close1.count() < period:
        return
    time1 = time.time()
    timerowlist = []

    index_start = 0
    index_end = index_start + period

    while index_end < close1.count():

        #index_start = index_end - jump -10  # 非优化版本注释掉

        part_close1 = close1[index_start:index_end].reset_index(drop=True)
        part_close2 = close2[index_start:index_end].reset_index(drop=True)
        bRet = adfuller_check_general(part_close1, part_close2, code1, code2, "SGDRegress") #SGDMiddle, SGD, SGDRegressMiddle, SGDRegress
        timerowlist.append(time.time()-time1)

        index_end += jump
        
    print index_end/jump
    timeDF = pd.DataFrame(timerowlist)
    timeDF.to_csv(DATA_TIME + '.csv', header=False, index=False)

def adfuller_check_price(code1, code2, start_date = '2013-10-10', end_date = '2014-09-30'):
    m = str(code1)
    n = str(code2)
    file1 = DownloadDir + "h_kline_" + m + ".csv"
    file2 = DownloadDir + "h_kline_" + n + ".csv"
    if not os.path.exists(file1) or not os.path.exists(file1):
        return

    kline1 = pd.read_csv(file1, parse_dates=['date'], index_col='date', date_parser=tudateparser)
    kline2 = pd.read_csv(file2, parse_dates=['date'], index_col='date', date_parser=tudateparser)
    #print kline1.head()
    price_of_1 = kline1[end_date:]
    price_of_2 = kline2[end_date:]

    # regroup quotation according to date index
    combination = price_of_1.join(price_of_2, how='inner', lsuffix='l', rsuffix='r')
    combination.dropna()

    closeprice_of_1 = combination['closel'].reset_index(drop=True)[0:1500]
    closeprice_of_2 = combination['closer'].reset_index(drop=True)[0:1500]
    return simulate_check_5days(closeprice_of_1, closeprice_of_2, code1, code2)

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
            stockPool = stockPool.append({'code1':str(reindexed_code[i]),  \
            'code2':str(reindexed_code[j])}, ignore_index=True)

    stockPool.apply(adfuller_check2, axis=1)

## Main functionality
def main():
    time1 = time.time()

    #adfuller_check_price("601002", "600815")
    # chedk all stock pairing in list book
    #check_all_dir()

    #adfuller_check_price_sgd("601002", "600815", linrreg="SMOLS", start_date = '2013-10-10', end_date = '2014-09-30')           #"SGD")
    readDictCSV("Linrgre.csv", weightdict)
    #compare_algorithm("601002", "600815",start_date = '2013-10-10', end_date = '2014-09-30') #2014-07-30 trading days: 192; 2014-09-30 trading days: 233
    #compare_algorithm("601002", "600815", start_date = '2013-10-10', end_date = '2014-09-30') #2014-07-30 trading days: 192; 2014-09-30 trading days: 233
    adfuller_check_price("601002", "600815", start_date = '2013-10-10', end_date = '2016-05-30')
    writeDictCSV("Linrgre.csv", weightdict)
    time2 = time.time()
    print "running time(s): ", time2-time1

if __name__ == "__main__":
    # Execute Main functionality
    main()