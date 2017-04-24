#coding:utf-8
import pandas as pd
import itertools

TABLE_STOCKS_BASIC = 'stock_basic_list'
TABLE_STOCKS_PAIRS = 'stock_pairing_list_all'
TABLE_WEIGHT = 'stock_linrreg.csv'
DownloadDir = './stockdata/'


def save_stk_pairings():
    stock_list = pd.read_csv(TABLE_STOCKS_BASIC + '.csv', dtype=str)

    list_code = stock_list['code'].values.tolist()
    #list_code = list_code[100:200]

    print len(list_code)
    list_pool = list(itertools.combinations(list_code, 2))
    stockPool = pd.DataFrame(list_pool, columns=['code1','code2'])
    print stockPool.head()
    stockPool.to_csv(TABLE_STOCKS_PAIRS + '.csv', header=False, index=False)


def main():
    save_stk_pairings()

if __name__ == '__main__':
    main()