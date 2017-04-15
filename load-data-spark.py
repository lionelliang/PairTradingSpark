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
from pyspark import SparkConf, SparkContext, SQLContext

## Module Constants
APP_NAME = "ADF Spark Application"
TABLE_STOCKS_BASIC = 'stock_basic_list'
TABLE_STOCKS_PAIRS = 'stock_pairing_list45'
TABLE_WEIGHT = 'stock_linrreg.csv'
DownloadDir = './stockdata/'
#weightdict = {}     #previous weight dict broadcast

## Closure Functions
#date example 2011/10/13
tudateparser = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

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

def load_data(sc):
    stockPool = sc.textFile(TABLE_STOCKS_BASIC + '.csv').map(split)
    #print stockPool.first()
    
    adfResult  = stockPool.map(lambda f: (adfuller_check3(f[0], f[1], 
                                                weight_lookup.value.get(str(f[0])+str(f[1]), 0))))

## Main functionality
def main(sc):
    time1 = time.time()
    load_data(sc)

    time2 = time.time()
    print "running time(s): ", time2-time1

if __name__ == "__main__":
	# Configure Spark
	conf = SparkConf().setAppName(APP_NAME)
	conf = conf.setMaster("local[*]")
	sc = SparkContext(conf=conf)

	# Execute Main functionality
	main(sc)
