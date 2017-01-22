import operator
import numpy as np
import statsmodels.tsa.stattools as sts
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tushare as ts
import pandas as pd
from datetime import datetime
from scipy.stats.stats import pearsonr

DownloadDir = './stockdata/'
'''
selector = pd.read_csv('selector.csv', index_col=0)
selector_code = selector['code'][100:110]
reselectorcode = selector_code.reset_index(drop=True)
stockPool = []
rank = {}
Rank = {}
for i in range(10):
    stockPool.append(str(reselectorcode[i]))




for i in range(10):
    for j in range(i+1,10):
        if i != j:
            # get the price of stock from TuShare
            price_of_i = ts.get_hist_data(stockPool[i], start='2012-01-01', end='2013-01-01')
            price_of_j = ts.get_hist_data(stockPool[j], start='2012-01-01', end='2013-01-01')
            # combine the close price of the two stocks and drop the NaN
            closePrice_of_ij = pd.concat([price_of_i['close'], price_of_j['close']], axis=1)
            closePrice_of_ij = closePrice_of_ij.dropna()
            # change the column name in the dataFrame
            closePrice_of_ij.columns = ['close_i', 'close_j']
            # calculate the daily return and drop the return of first day cause it is NaN.
            ret_of_i = ((closePrice_of_ij['close_i'] - closePrice_of_ij['close_i'].shift())/closePrice_of_ij['close_i'].shift()).dropna()
            ret_of_j = ((closePrice_of_ij['close_j'] - closePrice_of_ij['close_j'].shift())/closePrice_of_ij['close_j'].shift()).dropna()
            # calculate the correlation and store them in rank1
            if len(ret_of_i) == len(ret_of_j):
                correlation = np.corrcoef(ret_of_i.tolist(), ret_of_j.tolist())
                m = stockPool[i] + '+' + stockPool[j]
                rank[m] = correlation[0, 1]
    rank1 = sorted(rank.items(), key=operator.itemgetter(1))
    potentialPair = [list(map(int, item[0].split('+'))) for item in rank1]
    potentialPair = potentialPair[-5:]
'''

def adfuller_check2(code1, code2):
#for i in range(len(potentialPair)):
    m = str(code1)
    n = str(code2)
    kline1 = pd.read_csv(DownloadDir + "h_kline_" + code1 + ".csv", parse_dates='date', index_col='date')
    kline2 = pd.read_csv(DownloadDir + "h_kline_" + code2 + ".csv")

    price_of_1 = kline1['2011-10-10':'2016-03-05']
    price_of_2 = kline2['2011-10-10':'2016-03-05']

    closeprice_of_1 = price_of_1['close']
    closeprice_of_2 = price_of_2['close']

    if len(closeprice_of_1) != 0 and len(closeprice_of_2) != 0:
        model = pd.ols(y=closeprice_of_2, x=closeprice_of_1, intercept=True)   # perform ols on these two stocks
        spread = closeprice_of_2 - closeprice_of_1*model.beta['x']
        spread = spread.dropna()
        sta = sts.adfuller(spread, 1)
        pair = m + '+' + n
        print pair + ": adfuller result " + sta

def adfuller_check(code1, code2):
#for i in range(len(potentialPair)):
    m = str(code1)
    n = str(code2)
    price_of_1 = ts.get_hist_data(m, start='2011-10-10', end='2016-03-05')
    price_of_2 = ts.get_hist_data(n, start='2011-10-10', end='2016-03-05')
    price_of_1.to_csv(code1+"20111010-2016-03-05.csv")
    price_of_2.to_csv(code1+"20111010-2016-03-05.csv")
    closeprice_of_1 = price_of_1['close']
    closeprice_of_2 = price_of_2['close']

    if len(closeprice_of_1) != 0 and len(closeprice_of_2) != 0:
        model = sm.OLS(closeprice_of_2, closeprice_of_1)
        result = model.fit()
        spread = closeprice_of_2 - closeprice_of_1*result.params[1]
        spread = spread.dropna()
        sta = sts.adfuller(spread, 1)
        pair = m + '+' + n
        print pair + ": adfuller result " 
        print sta

## Main functionality
def main():
    # 获取所有股票的历史K线
    adfuller_check("601002", "600815")

if __name__ == "__main__":
    # Execute Main functionality
    main()