import tushare as ts
import pandas as pd
import os
import datetime
import multiprocessing

TABLE_STOCKS_BASIC = 'stock_basic_list'
DownloadDir = './stockdata/'
#DownloadDir = os.path.pardir + '/stockdata/'
# os.path.pardir: 上级目录

# 补全股票代码(6位股票代码)
# input: int or string
# output: string
def getSixDigitalStockCode(code):
    strZero = ''
    for i in range(len(str(code)), 6):
        strZero += '0'
    return strZero + str(code)

def download_stock_basic_info():

    try:
        df = ts.get_stock_basics()
        #直接保存到csv
        print 'stock to csv'
        df.to_csv('stock_basic_list.csv');
        print 'download csv finish'
    except Exception as e:
        print str(e)
# 默认为上市日期到今天的K线数据
# 可指定开始、结束日期：格式为"2015-06-28"
def download_stock_kline(code, date_start='', date_end=datetime.date.today()):
    
    code = getSixDigitalStockCode(code) # 将股票代码格式化为6位数字

    try:
        fileName = 'h_kline_' + str(code) + '.csv'

        writeMode = 'w'
        if os.path.exists(DownloadDir+fileName):
            #print (">>exist:" + code)
            df = pd.DataFrame.from_csv(path=DownloadDir+fileName)

            se = df.head(1).index #取已有文件的最近日期
            dateNew = se[0] + datetime.timedelta(1)
            date_start = dateNew.strftime("%Y-%m-%d")
            #print date_start
            writeMode = 'a'

        if date_start == '':
            date_start = "19900101"
            date = datetime.datetime.strptime(str(date_start), "%Y%m%d")
            date_start = date.strftime('%Y-%m-%d')
        date_end = date_end.strftime('%Y-%m-%d')  

        # 已经是最新的数据
        if date_start >= date_end:
            df = pd.read_csv(DownloadDir+fileName)
            return df

        print 'download ' + str(code) + ' k-line >>>begin (', date_start+u' 到 '+date_end+')'
        df_qfq = ts.get_h_data(str(code), start=date_start, end=date_end) # 前复权

        '''
        df_nfq = ts.get_h_data(str(code), start=date_start, end=date_end) # 不复权
        df_hfq = ts.get_h_data(str(code), start=date_start, end=date_end) # 后复权

        if df_qfq is None or df_nfq is None or df_hfq is None:
            return None

        df_qfq['open_no_fq'] = df_nfq['open']
        df_qfq['high_no_fq'] = df_nfq['high']
        df_qfq['close_no_fq'] = df_nfq['close']
        df_qfq['low_no_fq'] = df_nfq['low']
        df_qfq['open_hfq']=df_hfq['open']
        df_qfq['high_hfq']=df_hfq['high']
        df_qfq['close_hfq']=df_hfq['close']
        df_qfq['low_hfq']=df_hfq['low']
        '''

        if writeMode == 'w':
            df_qfq.to_csv(DownloadDir+fileName)
        else:

            df_old = pd.DataFrame.from_csv(DownloadDir + fileName)

            # 按日期由远及近
            df_old = df_old.reindex(df_old.index[::-1])
            df_qfq = df_qfq.reindex(df_qfq.index[::-1])

            df_new = df_old.append(df_qfq)
            #print df_new

            # 按日期由近及远
            df_new = df_new.reindex(df_new.index[::-1])
            df_new.to_csv(DownloadDir+fileName)
            #df_qfq = df_new

        print '\ndownload ' + str(code) +  ' k-line finish'
        return pd.read_csv(DownloadDir+fileName)

    except Exception as e:
        print str(e)        


    return None

# 获取所有股票的历史K线
def download_all_stock_history_k_line():
    print 'download all stock k-line'
    try:
        df = pd.DataFrame.from_csv(TABLE_STOCKS_BASIC + '.csv')
        pool = multiprocessing.Pool(processes=10)
        pool.map(download_stock_kline, df.index)
        pool.close()
        pool.join()  

    except Exception as e:
        print str(e)
    print 'download all stock k-line'

## Main functionality

def main():
	# 获取所有股票的历史K线
    #download_stock_kline("600446")

	download_stock_basic_info()
	download_all_stock_history_k_line()

if __name__ == "__main__":
	# Execute Main functionality
	main()