#coding:utf-8
## Spark Application - execute with spark-submit

## Imports
import os
import time
import pandas as pd
import commands

## Module Constants
APP_NAME = "ADF Spark Application"
TABLE_STOCKS_BASIC = 'stock_basic_list'
TABLE_STOCKS_PAIRS = 'stock_pairing_list45'
TABLE_WEIGHT = 'stock_linrreg.csv'
DownloadDir = './stockdata/'

#mongo db config
MONGO_TABLE_WEIGHT = 'stock.linrreg'
MONGO_TABLE_WEIGHT_SAVED = 'stock.linrregsaved'
MONGO_TABLE_STOCKS_PAIRS = 'stock.pairs'
MONGO_DB_QUOTATION = 'quotation'

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
	        stockPool = stockPool.append({'code1':str(reindexed_code[i]), 
            'code2':str(reindexed_code[j])}, ignore_index=True)

	stockPool.to_csv(TABLE_STOCKS_PAIRS + '.csv', header=False, index=False)

# input: int or string
# output: string
def getSixDigitalStockCode(code):
    strZero = ''
    for i in range(len(str(code)), 6):
        strZero += '0'
    return strZero + str(code)

def mongo_import(stk):
    code = getSixDigitalStockCode(stk)
    config = 'mongoimport ' + DownloadDir + 'h_kline_' + code + \
    '.csv --type csv --headerline -d ' + MONGO_DB_QUOTATION + \
    ' -c ' + 'kline_' + code
    print config
    #(status, output) = commands.getstatusoutput(config) # ERRO'{' 不是内部或外部命令
    os.system(config)

def load_data():
    stock_list = pd.read_csv(TABLE_STOCKS_BASIC + '.csv', dtype=str)
    code = stock_list['code']
    reindexed_code = code.reset_index(drop=True)
    reindexed_code = reindexed_code[100:200]
    reindexed_code = reindexed_code.reset_index(drop=True)
    print len(reindexed_code)

    reindexed_code.apply(mongo_import)
    #for i in range(len(reindexed_code)):
    #    mongo_import(reindexed_code[i])


## Main functionality
def main():
    time1 = time.time()
    load_data()

    time2 = time.time()
    print "running time(s): ", time2-time1

if __name__ == "__main__":
	# Execute Main functionality
	main()
