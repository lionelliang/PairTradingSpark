import operator
import numpy as np
import statsmodels.tsa.stattools as sts
import matplotlib.pyplot as plt
import tushare as ts
import pandas as pd
from datetime import datetime
from scipy.stats.stats import pearsonr

sector = pd.read_csv('sector.csv', index_col=0)
sector_code = sector['code'][100:110]
resectorcode = sector_code.reset_index(drop=True)
stockPool = []
rank = {}
Rank = {}
for i in range(10):
    stockPool.append(str(resectorcode[i]))




for i in range(10):
    for j in range(i+1,10):
        if i != j:
                # get the price of stock from TuShare
                price_of_i = ts.get_hist_data(stockPool[i], start='2012-01-01', end='2013-01-01')
                price_of_j = ts.get_hist_data(stockPool[j], start='2012-01-01', end='2013-01-01')
                # combine the close price of the two stocks and drop the NaN
                closePrice_of_ij = pd.concat([price_of_i['close'], price_of_j['close']], axis = 1)
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
                    rank[m] = correlation[0,1]
    rank1 = sorted(rank.items(), key=operator.itemgetter(1))
    potentialPair = [list(map(int, item[0].split('+'))) for item in rank1]
    potentialPair = potentialPair[-5:]




    for i in range(len(potentialPair)):
    m = str(potentialPair[i][0])
    n = str(potentialPair[i][1])
    price_of_1 = ts.get_hist_data(m, start='2012-01-01', end='2013-01-01')
    price_of_2 = ts.get_hist_data(n, start='2012-01-01', end='2013-01-01')

    closeprice_of_1 = price_of_1['close']
    closeprice_of_2 = price_of_2['close']

    if len(closeprice_of_1) != 0 and len(closeprice_of_2) != 0:
        model = pd.ols(y=closeprice_of_2, x=closeprice_of_1, intercept=True)   # perform ols on these two stocks
        spread = closeprice_of_2 - closeprice_of_1*model.beta['x']
        spread = spread.dropna()
        sta = sts.adfuller(spread, 1)
        pair = m + '+' + n
        Rank[pair] = sta[0]
        rank2 = sorted(Rank.items(), key=operator.itemgetter(1))


    