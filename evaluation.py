import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 拼接预测数据
def flattn_data_set(true_data: np.array, predict_data_set: np.array):
    pre_len = predict_data_set.shape[1]
    
    flattn_data = []
    for i in range(pre_len):
        seq = np.array(predict_data_set[i: : pre_len]).reshape(-1)
        flattn_data.append(seq[pre_len - i - 1:])
    min_len = min([len(seq) for seq in flattn_data])
    
    for i in range(pre_len): 
        flattn_data[i] = flattn_data[i][: min_len]

    true_data = true_data[pre_len - 1: min_len + pre_len - 1]
    return np.array(true_data), np.array(flattn_data)


# 计算趋势指标
def accuracy_metric(true_data: np.array, predict_data_set: np.array):
    from math import sqrt
    assert(len(predict_data_set.shape) == 2)

    (_, unit_len) = predict_data_set.shape
    
    true_up, true_down, false_up, false_down = 0, 0, 0, 0
    for i in range(unit_len - 1):
        if true_data[i+1] >= true_data[i]:
            up, down = True, False
        else:
            up, down = False, True

        for seq in predict_data_set:
            if seq[i+1] >= seq[i]:
                pre_up, pre_down = True, False
            else:
                pre_up, pre_down = False, True

            true_up    += (1 if up   & pre_up   else 0)
            true_down  += (1 if down & pre_down else 0)
            false_up   += (1 if down & pre_up   else 0)
            false_down += (1 if up   & pre_down else 0)
    
    acc = (true_up + true_down) / (true_up + true_down + false_up + false_down)
    if sqrt((true_up + false_up)*(true_up + false_down)*(true_down + false_up)*(true_down + false_down)) != 0:
        mcc = (true_up * true_down - false_up * false_down) / sqrt((true_up + false_up)*(true_up + false_down)*(true_down + false_up)*(true_down + false_down))
    else:
        mcc = np.inf

    return acc, mcc


# 获取美国当天无风险利率
def get_rf(rf_date):
    # 无风险收益率
    rf_date = str(rf_date)
    rf_date = time.strftime('%m/%d/%Y', time.strptime(rf_date,'%Y%m%d'))
    rf_table = pd.read_csv("daily-treasury-rates.csv")
    rf = rf_table.loc[rf_table['Date'] == rf_date]['10 Yr'].values[-1] * 0.01
    
    return rf


# 验证策略有效性
def trade_verify(true_close_data, true_label_data, trade_class, detail_len=50, result_save_root=None):
    trade=trade_class(2e9,np.array(true_close_data),np.array([true_label_data]))
    plt.figure(figsize=(12,6))
    plt.subplots_adjust(top=0.995, bottom=0.05, left=0.04, right=0.995, hspace=0.1, wspace=0.08)
    plt.subplot(2,2,1)
    plt.plot(true_close_data[:detail_len],label=str(detail_len)+" days' trend")
    plt.legend(loc="best")
    plt.subplot(2,2,2)
    plt.plot(trade.nop_assets[:detail_len],label=str(detail_len)+" days' asset without operate")
    plt.plot(trade.assets[0][:detail_len],label=str(detail_len)+" days' asset with operate")
    plt.legend(loc="best")

    plt.subplot(2,2,3)
    plt.plot(true_close_data,label="trend of test set")
    plt.legend(loc="best")
    plt.subplot(2,2,4)
    plt.plot(trade.nop_assets,label="asset without operate of test set")
    plt.plot(trade.assets[0],label="asset with operate of test set")
    plt.legend(loc="best")
    plt.show()
    if result_save_root is not None:
        plt.savefig(result_save_root + "strategy.jpg")
        print("Save image to " + result_save_root + "strategy.jpg")


# 计算统计指标
def calc_stat_standrad(model_data):
    from math import sqrt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    up_same,down_same = [0 for _ in range(len(model_data)-1)],[0 for _ in range(len(model_data)-1)]
    up_total,down_total = 0,0
    for i in range(len(model_data[0])-1):
        true_trend = model_data[0][i+1]>model_data[0][i]
        up_total += 1 if true_trend else 0
        down_total += 0 if true_trend else 1
        for j in range(len(model_data)-1):
            pred_trend = model_data[j+1][i+1]>model_data[j+1][i]
            up_same[j] += 1 if (not true_trend^pred_trend) & true_trend else 0 
            down_same[j] += 1 if (not true_trend^pred_trend) & (not true_trend) else 0 
    up_percent = np.mean(np.array(up_same)/up_total)
    down_percent = np.mean(np.array(down_same)/down_total)
    total_percent = np.mean((np.array(up_same)+np.array(down_same))/(up_total+down_total))

    stand = []
    for i in range(1,6):
        stand.append([mean_absolute_error(model_data[0],model_data[i]),mean_squared_error(model_data[0],model_data[i]),
              sqrt(mean_squared_error(model_data[0],model_data[i])),r2_score(model_data[0],model_data[i])])
    stand = np.mean(np.array(stand),axis=0)

    return pd.DataFrame(np.hstack(([up_percent,down_percent,total_percent],stand)).reshape(1,-1),
             columns=["up_trend","down_trend","total_trend","MAE","MSE","RMSE","R2"])


class trade_strategy_split:
    def __init__(self, cash: float, true_close: np.array, predict_data: np.array, fee: float=0.0009):
        self.groups = len(predict_data)
        self.initial = cash
        self.fee = fee # 手续费
        # 现金额度，持有股票总量，总资产变化，策略操作点
        self.cash, self.assets = [cash/self.groups for _ in range(self.groups)], [[cash/self.groups] for _ in range(self.groups)]
        self.buy_point, self.sell_point = [[] for _ in range(self.groups)], [[] for _ in range(self.groups)]
        
        # 多组预测结果进行交易结果
        for i in range(self.groups):
            self.operation(i, true_close, predict_data[i])

        # 不操作资产变动曲线
        # 已经考虑天数与预测结果一致，买入卖出扣除手续费
        self.nop_assets = [cash]
        nop_share = cash // (true_close[0] * 100 * (1 + fee)) * 100
        nop_cash = cash - nop_share * true_close[0] * (1 + fee)
        for p in true_close[: predict_data.shape[1] - 1]:
            self.nop_assets.append(nop_cash + nop_share * p)
        self.nop_assets.append(nop_cash + nop_share * true_close[predict_data.shape[1] - 1] * (1 - fee))

    def operation(self, index: int, true_close, predict_close):
        share = 0
        for i in range(0, len(predict_close)-1):
            up = True if predict_close[i+1] > predict_close[i] else False
            down = True if predict_close[i+1] < predict_close[i] else False

            # tomorrow > today buy
            if up and share == 0:
                unit_price = true_close[i] * (1 + self.fee)
                # 购入股票数额
                share = self.cash[index] // (unit_price * 100) * 100
                # 剩余现金数额
                self.cash[index] -= share * unit_price
                # 记录购入点
                self.buy_point[index].append([i, true_close[i]])

            # tomorrow < today sell
            if down and share > 0:
                # 卖出股票，现金额度变化
                self.cash[index] += share * true_close[i] * (1 - self.fee)
                # 售出股票数额
                share = 0
                # 记录售出点
                self.sell_point[index].append([i, true_close[i]])
            # 记录当天总资产
            self.assets[index].append(self.cash[index] + share * true_close[i])
        # 记录最后一天总资产
        self.assets[index].append(self.cash[index] + share * true_close[-1] * (1 - self.fee))

    def operate_result(self):
        total = sum([l[-1] for l in self.assets])
        return self.initial, total, self.nop_assets[-1]

    def calculate_standard(self, rf: float):
        asset = np.array(self.assets).sum(axis=0)

        # 股票自身波动年化收益
        income_rate = pd.DataFrame(self.nop_assets).pct_change().dropna().values.reshape(-1)
        daily_rate_mean = ((1+income_rate).prod())**(1/income_rate.shape[0])-1
        no_op_annual_rate = (1+daily_rate_mean)**252 - 1

        # 策略收益
        income_rate = pd.DataFrame(asset).pct_change().dropna().values.reshape(-1)

        # 最大回撤
        max_list = np.maximum.accumulate(asset)
        mdd = max((max_list - asset)/max_list)

        # 日均收益率
        daily_rate_mean = ((1+income_rate).prod())**(1/income_rate.shape[0])-1
        # 年化收益率 & 波动率
        annual_rate = (1+daily_rate_mean)**252 - 1
        annual_vol = income_rate.std()*(252**(1/2))
        # 夏普比率
        sharpe_ratio = (annual_rate-rf)/annual_vol

        # 风险价值模型（Value at Risk, VaR）
        var1 = np.percentile(income_rate, 1)
        var5 = np.percentile(income_rate, 5)
        var32 = np.percentile(income_rate, 32)

        col = ["ARR","MDD","Sharpe","VaR 1%","VaR 5%","VaR 32%","Benchmark ARR"]
        std = [round(annual_rate, 4), round(mdd, 4), round(sharpe_ratio, 4), round(var1,4), round(var5,4), round(var32,4), round(no_op_annual_rate,4)]
        return pd.DataFrame(np.array(std).reshape(1,-1), columns=col)

        '''
        # 市场收益
        market_rate["close_rate"] = market_rate.close.pct_change()
        market_rate = market_rate.loc[market_rate['trade_date'].isin(date)].close_rate.reset_index(drop=True)
        # 无风险投资日均收益
        daily_rf = (1 + rf)**(1/252) - 1
        # 策略超额收益
        stategy_excess = income_rate - daily_rf
        # 市场超额收益
        market_excess = pd.DataFrame(market_rate - daily_rf)
        market_excess["alpha"] = 1
        # Alpha
        import statsmodels.api as stm
        lr = stm.OLS(stategy_excess, market_excess).fit()
        daily_alpha = lr.params["alpha"]
        annual_alpha = (1+daily_alpha)**252-1

        return [date.iloc[0], date.iloc[-1], round(annual_rate, 4), round(mdd, 4), round(sharpe_ratio, 4), round(annual_alpha, 4), round(var1,4),round(var5,4),round(var10,4)]
        '''

    def show_trade_result(self):
        tickssize = 6
        labelsize = 7
        titlesize = 8
        legensize = 8
        plt.figure(figsize=(3.3, 4), dpi=200)
        plt.subplots_adjust(top=0.955, bottom=0.09, left=0.14, right=1, hspace=0.35)
        plt.subplot(211)
        plt.plot(range(len(self.close)), self.close, "lightblue", label="Stock Close of Each Trade Day")
        plt.scatter(self.buy_point[:, 0], self.buy_point[:, 1], c='r', label="Buy Point", marker=".")
        plt.scatter(self.sell_point[:, 0], self.sell_point[:, 1], c='g', label="Sell Point", marker=".")
        plt.xlim((-5, len(self.close) + 5))
        plt.xticks(fontsize=tickssize)
        plt.yticks(fontsize=tickssize)
        plt.xlabel("Number of Days", fontsize=labelsize)
        plt.ylabel("Price", fontsize=labelsize)
        plt.legend(loc="best", fontsize=legensize)
        plt.title("(a). Forecast Curve & Real Curve", fontsize=titlesize)
        plt.grid(linestyle='-.')

        plt.subplot(212)
        plt.plot(range(len(self.each_income)), self.each_income, "b", label="Percentage of Amount")
        plt.plot(range(len(self.true_remove)), self.true_remove, label="Percentage of 000016SH income")
        plt.xlim((-5, len(self.close) + 5))
        plt.xticks(fontsize=tickssize)
        plt.yticks(fontsize=tickssize)
        plt.xlabel("Number of Days", fontsize=labelsize)
        plt.ylabel("Percentage", fontsize=labelsize)
        plt.legend(loc="best", fontsize=legensize)
        plt.title("(b). Change in Percentage of Amount", fontsize=titlesize)
        plt.grid(linestyle='-.')


class trade_strategy_split_ma:
    def __init__(self, cash: float, true_close: np.array, predict_data: np.array, ma_day: int=5, fee: float=0.0009):
        self.groups = len(predict_data)
        self.ma_day = ma_day
        self.initial = cash
        self.fee = fee  # 手续费
        # 现金额度，持有股票总量，总资产变化，策略操作点
        self.cash,self.assets = [cash/self.groups for _ in range(self.groups)],[[cash/self.groups] for _ in range(self.groups)]
        self.buy_point,self.sell_point = [[] for _ in range(self.groups)],[[] for _ in range(self.groups)]
        
        # 多组预测结果进行交易结果
        for i in range(self.groups):
            self.operation(i, true_close, predict_data[i])

        # 不操作资产变动曲线
        # 已经考虑天数与预测结果一致，买入卖出扣除手续费
        self.nop_assets = [cash]
        nop_share = int(cash/true_close[0]/100)*100
        nop_cash = cash - nop_share*true_close[0]*(1+fee)
        for p in true_close[:predict_data.shape[1]-1]:
            self.nop_assets.append(nop_cash + nop_share * p)
        self.nop_assets.append(nop_cash + nop_share*true_close[predict_data.shape[1]-1]*(1-fee))

    def operation(self, index: int, true_close, predict_data):
        share = [0 for _ in range(self.ma_day)]
        cash = [self.cash[index]/self.ma_day for _ in range(self.ma_day)]
        for i in range(len(predict_data)-self.ma_day):
            pr = i % self.ma_day # prior
            # tomorrow > today buy
            if predict_data[i+self.ma_day] > predict_data[i+self.ma_day-1] and cash[pr]/(true_close[i]*100)>=1:
                share[pr] = int(cash[pr]/(true_close[i]*100))*100
                cash[pr] -= share[pr]*true_close[i]*(1+self.fee)
                self.buy_point[index].append([i, true_close[i]])
            # tomorrow < today sell
            if predict_data[i+self.ma_day] < predict_data[i+self.ma_day-1] and share[pr]>0:
                cash[pr] += share[pr]*true_close[i]*(1-self.fee)
                share[pr] = 0
                self.sell_point[index].append([i, true_close[i]])
            # record each day's assets
            self.assets[index].append(sum(cash)+sum(share)*true_close[i])

        for i in range(len(predict_data)-self.ma_day,len(predict_data)):
            pr = i % self.ma_day # prior
            cash[pr] += share[pr]*true_close[i]*(1-self.fee)
            share[pr] = 0
            self.sell_point[index].append([i, true_close[i]])
            # record each day's assets
            self.assets[index].append(sum(cash)+sum(share)*true_close[i]*(1-self.fee))
        # record cash of each data
        self.cash[index] = sum(cash)

    def operate_result(self):
        total = sum([l[-1] for l in self.assets])
        return self.initial, total, self.nop_assets[-1]

    def calculate_standard(self, rf: float):
        asset = np.array(self.assets).sum(axis=0)

        # 股票自身波动年化收益
        income_rate = pd.DataFrame(self.nop_assets).pct_change().dropna().values.reshape(-1)
        daily_rate_mean = ((1+income_rate).prod())**(1/income_rate.shape[0])-1
        no_op_annual_rate = (1+daily_rate_mean)**252 - 1

        # 策略收益
        income_rate = pd.DataFrame(asset).pct_change().dropna().values.reshape(-1)

        # 最大回撤
        max_list = np.maximum.accumulate(asset)
        mdd = max((max_list - asset)/max_list)

        # 日均收益率
        daily_rate_mean = ((1+income_rate).prod())**(1/income_rate.shape[0])-1
        # 年化收益率 & 波动率
        annual_rate = (1+daily_rate_mean)**252 - 1
        annual_vol = income_rate.std()*(252**(1/2))
        # 夏普比率
        sharpe_ratio = (annual_rate-rf)/annual_vol

        # 风险价值模型（Value at Risk, VaR）
        var1 = np.percentile(income_rate,1)
        var5 = np.percentile(income_rate,5)
        var32 = np.percentile(income_rate,32)

        col = ["ARR","MDD","Sharpe","VaR 1%","VaR 5%","VaR 32%","Benchmark ARR"]
        std = [round(annual_rate, 4), round(mdd, 4), round(sharpe_ratio, 4), round(var1,4), round(var5,4), round(var32,4), round(no_op_annual_rate,4)]
        return pd.DataFrame(np.array(std).reshape(1,-1), columns=col)

    def show_trade_result(self):
        tickssize = 6
        labelsize = 7
        titlesize = 8
        legensize = 8
        plt.figure(figsize=(3.3, 4), dpi=200)
        plt.subplots_adjust(top=0.955, bottom=0.09, left=0.14, right=1, hspace=0.35)
        plt.subplot(211)
        plt.plot(range(len(self.close)), self.close, "lightblue", label="Stock Close of Each Trade Day")
        plt.scatter(self.buy_point[:, 0], self.buy_point[:, 1], c='r', label="Buy Point", marker=".")
        plt.scatter(self.sell_point[:, 0], self.sell_point[:, 1], c='g', label="Sell Point", marker=".")
        plt.xlim((-5, len(self.close) + 5))
        plt.xticks(fontsize=tickssize)
        plt.yticks(fontsize=tickssize)
        plt.xlabel("Number of Days", fontsize=labelsize)
        plt.ylabel("Price", fontsize=labelsize)
        plt.legend(loc="best", fontsize=legensize)
        plt.title("(a). 买卖操作图", fontsize=titlesize)
        plt.grid(linestyle='-.')

        asset = np.array(self.assets).sum(axis=0)
        plt.subplot(212)
        plt.plot(range(len(asset)), asset, "b", label="策略交易资产走势")
        plt.plot(range(len(self.nop_assets)), self.nop_assets, label="长期持有资产走势")
        plt.xlim((-5, len(self.close) + 5))
        plt.xticks(fontsize=tickssize)
        plt.yticks(fontsize=tickssize)
        plt.xlabel("Number of Days", fontsize=labelsize)
        plt.ylabel("Assets", fontsize=labelsize)
        plt.legend(loc="best", fontsize=legensize)
        plt.title("(b). 资产走势", fontsize=titlesize)
        plt.grid(linestyle='-.')


# 评测指标类
class evaluate_indicator:
    def __init__(self, cash: float, true_close_data: np.array, predict_close_data: np.array, risk_free: float, name: str = None, fee: float=0.0009):
        # 对齐预测数据以及真实数据
        true_close, predict_close = flattn_data_set(true_close_data, predict_close_data)
        # 预测数据的组数
        groups = len(predict_close)

        self.initial = cash
        self.fee = fee # 手续费
        # 现金额度，持有股票总量，总资产变化，策略操作点
        self.cash, self.assets_set = [cash / groups for _ in range(groups)], [[cash / groups] for _ in range(groups)]
        self.buy_point, self.sell_point = [[] for _ in range(groups)], [[] for _ in range(groups)]
        
        # 多组预测结果进行交易结果
        for i in range(groups):
            self.__operation__(i, true_close, predict_close[i])
        self.assets = np.array(self.assets_set).sum(axis=0)

        # 不操作资产变动曲线
        # 已经考虑天数与预测结果一致，买入卖出扣除手续费
        self.nop_assets = [cash]
        nop_share = cash // true_close[0] * (1 + fee)
        nop_cash = cash % true_close[0] * (1 + fee)
        for p in true_close[: -1]:
            self.nop_assets.append(nop_cash + nop_share * p)
        self.nop_assets.append(nop_cash + nop_share * true_close[-1] * (1 - fee))
        self.nop_assets = np.array(self.nop_assets)

        self.name = name
        self.__calculate_standard__(risk_free)
        self.acc, self.mcc = accuracy_metric(true_close, predict_close)

    def __operation__(self, index: int, true_close, predict_close):
        share = 0
        for i in range(0, len(predict_close)-1):
            up = True if predict_close[i+1] > predict_close[i] else False
            down = True if predict_close[i+1] < predict_close[i] else False

            # tomorrow > today -> buy
            if up and share == 0:
                # number of shareholdings
                share = self.cash[index] // (true_close[i] * (1 + self.fee))
                # remain cash
                self.cash[index] = self.cash[index] % (true_close[i] * (1 + self.fee))
                # point of purchase
                self.buy_point[index].append([i, true_close[i]])

            # tomorrow < today -> sell
            if down and share > 0:
                # remain cash
                self.cash[index] += share * true_close[i] * (1 - self.fee)
                # number of shareholdings
                share = 0
                # point of sell
                self.sell_point[index].append([i, true_close[i]])

            # total assets each day
            self.assets_set[index].append(self.cash[index] + share * true_close[i])
        # total assets
        self.assets_set[index].append(self.cash[index] + share * true_close[-1] * (1 - self.fee))

    def __calculate_standard__(self, rf: float):
        # no operator asset
        self.nop_irr = self.nop_assets[-1] / self.initial
        nop_rate = pd.DataFrame(self.nop_assets).pct_change().dropna()

        nop_annual_rate = ((1 + nop_rate).prod()**(1 / len(nop_rate)))**252 - 1
        nop_annual_vol = nop_rate.std(ddof=0)*(252**(1/2))
        self.nop_sharpe_ratio = ((nop_annual_rate - rf) / nop_annual_vol).values[-1]
        
        # operator asset
        self.irr = self.assets[-1] / self.initial
        rate = pd.DataFrame(self.assets).pct_change().dropna()

        annual_rate = ((1 + rate).prod()**(1 / len(rate)))**252 - 1
        annual_vol = rate.std(ddof=0)*(252**(1/2))
        self.sharpe_ratio = ((annual_rate - rf) / annual_vol).values[-1]
    
    def standard(self):
        col = ["irr", "benchmark irr", "sharpe", "benchmark sharpe", "acc", "mcc"]
        std = [round(std, 4) for std in [self.irr, self.nop_irr, self.sharpe_ratio, self.nop_sharpe_ratio, self.acc, self.mcc]]
        return pd.DataFrame(np.array(std).reshape(1,-1), columns=col, index=[self.name])


# 评测指标类——预测结果求平均
class evaluate_indicator_mean:
    def __init__(self, cash: float, true_close_data: np.array, predict_close_data: np.array, risk_free: float, name: str = None, fee: float=0.0009):
        # 对齐预测数据以及真实数据
        true_close, predict_close = flattn_data_set(true_close_data, predict_close_data)
        predict_close = predict_close.mean(axis=0)
        self.true_close, self.predict_close = true_close, predict_close

        self.initial = cash
        self.fee = fee # 手续费
        # 现金额度，持有股票总量，总资产变化，策略操作点
        self.cash, self.assets = cash, [cash]
        self.buy_point, self.sell_point = [], []
        
        # 多组预测结果进行交易结果
        self.__operation__(true_close, predict_close)

        # 不操作资产变动曲线
        # 已经考虑天数与预测结果一致，买入卖出扣除手续费
        self.nop_assets = [cash]
        nop_share = cash // true_close[0] * (1 + fee)
        nop_cash = cash % true_close[0] * (1 + fee)
        for p in true_close[: -1]:
            self.nop_assets.append(nop_cash + nop_share * p)
        self.nop_assets.append(nop_cash + nop_share * true_close[-1] * (1 - fee))
        self.nop_assets = np.array(self.nop_assets)

        self.name = name
        self.__calculate_standard__(risk_free)
        self.acc, self.mcc = accuracy_metric(true_close, np.array([predict_close]))

    def __operation__(self, true_close, predict_close):
        share = 0
        for i in range(0, len(predict_close)-1):
            up = True if predict_close[i+1] > predict_close[i] else False
            down = True if predict_close[i+1] < predict_close[i] else False

            # tomorrow > today -> buy
            if up and share == 0:
                # number of shareholdings
                share = self.cash // (true_close[i] * (1 + self.fee))
                # remain cash
                self.cash = self.cash % (true_close[i] * (1 + self.fee))
                # point of purchase
                self.buy_point.append([i, true_close[i]])

            # tomorrow < today -> sell
            if down and share > 0:
                # remain cash
                self.cash += share * true_close[i] * (1 - self.fee)
                # number of shareholdings
                share = 0
                # point of sell
                self.sell_point.append([i, true_close[i]])

            # total assets each day
            self.assets.append(self.cash + share * true_close[i])
        # total assets
        self.assets.append(self.cash + share * true_close[-1] * (1 - self.fee))
        self.assets = np.array(self.assets)

    def __calculate_standard__(self, rf: float):
        # no operator asset
        self.nop_irr = self.nop_assets[-1] / self.initial
        nop_rate = pd.DataFrame(self.nop_assets).pct_change().dropna()

        nop_annual_rate = ((1 + nop_rate).prod()**(1 / len(nop_rate)))**252 - 1
        nop_annual_vol = nop_rate.std(ddof=0)*(252**(1/2))
        self.nop_sharpe_ratio = ((nop_annual_rate - rf) / nop_annual_vol).values[-1]
        
        # operator asset
        self.irr = self.assets[-1] / self.initial
        rate = pd.DataFrame(self.assets).pct_change().dropna()

        annual_rate = ((1 + rate).prod()**(1 / len(rate)))**252 - 1
        annual_vol = rate.std(ddof=0)*(252**(1/2))
        self.sharpe_ratio = ((annual_rate - rf) / annual_vol).values[-1]
    
    def standard(self):
        col = ["irr", "benchmark irr", "sharpe", "benchmark sharpe", "acc", "mcc"]
        std = [round(std, 4) for std in [self.irr, self.nop_irr, self.sharpe_ratio, self.nop_sharpe_ratio, self.acc, self.mcc]]
        return pd.DataFrame(np.array(std).reshape(1,-1), columns=col, index=[self.name])
