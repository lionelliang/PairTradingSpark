import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.analysis.function.Log;
    
public class PairTrading implements IHStrategy{
        int count = 0;
        double zScore;
        double beta;
        double shareStock1;
        double shareStock2;
        double spread;
        double betShare;
        double buyShare;
        double portfolioValue;
        double dailyReturn;
        double initialCash;
        @Override
        public void init(IHInformer informer, IHInitializers initializers) {
            
    
            String stockId1 = "600815.XSHG";
            String stockId2 = "601002.XSHG";
            double closePrice[][] = new double[200][2];
            // 这些参数值是在研究部分获取的
            double beta = 0.418142479833;
            double mean=7.27385228021;
            double std  = 0.41596412236;
    
    
    
            int numRows = closePrice.length;
            int numCols = closePrice[0].length;
            int period = 199;
        
    
    
            initializers.instruments((universe) -> universe.add(stockId1, stockId2));
            initializers.shortsell().allow();
            initializers.events().statistics((stats, info, trans) -> {
                //获取两只股票的日线数据
                double[] closePxInStockId1 = stats.get(stockId1).history(period + 1, HPeriod.Day).getClosingPrice();
                double[] closePxInStockId2 = stats.get(stockId2).history(period + 1, HPeriod.Day).getClosingPrice();
                //每次对冲的多头头寸控制为当前持有现金的0.6
                betShare = info.portfolio().getAvailableCash()*0.6/closePxInStockId2[199];   
                portfolioValue = info.portfolio().getPortfolioValue();
                dailyReturn = info.portfolio().getDailyReturn();
                initialCash = info.portfolio().getInitialCash();
                buyShare = beta*betShare;
                //此处为引入的止损。当每天的收益连续四天以上为负的时候则止损
                if (dailyReturn < 0){
                    count = count + 1;
                }
                if (count > 4){
                   if (shareStock1 > 0){
                        trans.sell(stockId1).shares(shareStock1).commit();
                        }
                   if (shareStock1 < 0){
                       trans.buy(stockId1).shares(-shareStock1).commit();
                    }    
                    if (shareStock2 > 0){
                        trans.sell(stockId2).shares(shareStock2).commit();
                    }
                    if (shareStock2 < 0){
                        trans.buy(stockId2).shares(-shareStock2).commit();
                    }
                    count = 0;
                }
                if (buyShare < 100){
                    buyShare =100;
                }
                shareStock1 = info.position(stockId1).getNonClosedTradeQuantity();
                shareStock2 = info.position(stockId2).getNonClosedTradeQuantity();
                //计算两只股票之间的价差
                spread = closePxInStockId2[199] - beta*closePxInStockId1[199];
                //计算zScore
                zScore = (spread - mean)/std;
                informer.plot("zScore", zScore);
    //当入场信号来的时候，进入市场  
                if ((zScore > 1.1  ) && (shareStock1 == 0) && (shareStock2 == 0)){               
                    trans.sell(stockId2).shares(betShare).commit();
                    trans.buy(stockId1).shares(buyShare).commit();
                }
                if ((zScore < -1.5) && (shareStock2 == 0) && (shareStock1 == 0)){ 
                    trans.sell(stockId1).shares(buyShare).commit();
                    trans.buy(stockId2).shares(betShare).commit();
                }
    //当出场信号来的时候，离开市场
                if ((zScore < 0.8) && (zScore > -1.0) && (shareStock1 != 0) && (shareStock2 != 0) ){
                    if (shareStock1 > 0){
                        trans.sell(stockId1).shares(shareStock1).commit();
                    }
                    if (shareStock1 < 0){
                        trans.buy(stockId1).shares(-shareStock1).commit();
                    }    
                    if (shareStock2 > 0){
                        trans.sell(stockId2).shares(shareStock2).commit();
                    }
                    if (shareStock2 < 0){
                        trans.buy(stockId2).shares(-shareStock2).commit();
                        
                    }
                }            
            });
        }
    }
