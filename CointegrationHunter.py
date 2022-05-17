import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vecm
from itertools import combinations
import sys
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc, random_walk
import matplotlib.pyplot as plt
import math
from FileManager import FileManager


'''
# Example on how to use this class

commodities = ["CORN", "SOYB", "CPER", "GLD", "WEAT", "SLV"]

stocks = commodities

start = '2019-1-1'
end = '2022-1-1'

hunter = CointegrationHunter(total_stocks_to_maintain=3, r=0, pass_criteria=0.95, use_returns=False)
hunter.findBestCointegrations(stocks, start, end)
hunter.best_cointegration_results

# Here we can use differnt dates to confirm out of sample cointegration exists
hunter.testBestCombinations("2019-1-1", "2022-1-20")
hunter.best_cointegration_results

# Here we test the best results for stationarity and then sort them by p_value to determine ones that
# reject the null hypothesis that there is no stationarity

results = hunter.test_results_for_stationarity()
results.sort_values('p_value')



'''



class CointegrationHunter:
    
    def __init__(self, total_stocks_to_maintain, r, pass_criteria, use_returns, k_ar_diff=12) -> None:

        self.cash = 0
        self.data = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.stocks = []
        self.stock_combinations = []
        self.best_cointegration_results = pd.DataFrame()
        self.unique_stock_list_from_best_combinations = None
        self.total_stocks_to_maintain = total_stocks_to_maintain
        self.r = r
        self.pass_criteria = pass_criteria
        self.use_returns = use_returns
        self.test_data = None
        self.use_adj_close = False
        self.k_ar_diff = k_ar_diff
        self.results = pd.DataFrame()



    def get_cointegrated_combinations(self):

        if len(self.stocks) == 0:
            sys.exit("Stocks array is empty")

        self.stock_combinations = list(combinations(self.stocks, self.total_stocks_to_maintain))


    def getStockData(self, stocks, start, end, use_returns):

        fm = FileManager()


        data = pd.DataFrame()
        returns = pd.DataFrame()
        self.stocks = stocks

        for stock in stocks:

            # print(f"*** Getting data for {stock} ***")

            fm.get_data(stock, start, end)

            prices = fm.data

            # prices = yf.download(stock, start, end)

            if self.use_adj_close:
                data[stock] = prices['Adj Close']

            else:
                data[stock] = prices['Close']

            returns[stock] = np.append(
                data[stock][1:].reset_index(drop=True) / data[stock][:-1].reset_index(drop=True) - 1, 0)
                # data[stock][1:] / data[stock][:-1] - 1, 0)

        if use_returns:
            self.data = returns.cumsum()
        else:
            self.data = data




    def test_stock_combos(self):

        results = pd.DataFrame()

        for stock_combo in self.stock_combinations:

            test_dataset = self.data[list(stock_combo)]

            jt = vecm.coint_johansen(test_dataset, det_order=0, k_ar_diff=self.k_ar_diff)


            passing_threshold_count = 0

            for r in range(0, self.total_stocks_to_maintain):

                # print(f"Checking for {r} relationships")

                pass_threshold_1, eig_confidence_level = self.analyze_critical_values(jt.max_eig_stat, jt.max_eig_stat_crit_vals, r, self.pass_criteria)

                pass_threshold_2, trace_confidence_level = self.analyze_critical_values(jt.trace_stat, jt.trace_stat_crit_vals, r, self.pass_criteria)

                if pass_threshold_1 and pass_threshold_2:
                    # print(f"Passed both tests")
                    passing_threshold_count = passing_threshold_count + 1
                else:
                    pass
                    # print("Did not pass both tests")

            if passing_threshold_count == self.total_stocks_to_maintain - 1:

                results = results.append(
                    {
                        "stock_combo": stock_combo,
                        "jt": jt,
                        "confidence_level_eig": eig_confidence_level,
                        "confidence_level_trace": trace_confidence_level,
                        
                    }, ignore_index=True
                )
        
        return results

    def findBestCointegrations(self, stocks, start, end):

        results = pd.DataFrame()

        # Get all the stock data
        self.getStockData(stocks, start=start, end=end, use_returns=self.use_returns)

        # Get all possible combinations of the stock
        self.get_cointegrated_combinations()

        # Here we test the combinations to see which ones are cointegrated
        results = self.test_stock_combos()
        
        self.best_cointegration_results = results

        if len(self.best_cointegration_results) == 0:
            print("No results passed Cointegration Hunter requirements")

    def analyze_critical_values(self, stat, critical_values, r, pass_criteria):
        
        pass_threshold = False

        confidence_level = 0

        if stat[r] > critical_values[r][0]:

            confidence_level = 0.9

            if stat[r] > critical_values[r][1]:
                
                confidence_level = 0.95

                if stat[r] > critical_values[r][2]:

                    confidence_level = 0.99

            # print(f"Reject Null Hypothesis r <= {r} at {confidence_level*100}% Level")
            
            if pass_criteria <= confidence_level:
                pass_threshold = True
                
            return pass_threshold, confidence_level

        else:
            # print(f"Fail to Reject Null Hypothesis r <= {r} at 90% Level")
            
            return pass_threshold, confidence_level


    def testBestCombinations(self, start, end):
        
        # Here we get a unique list of stocks from the bet combinations
        self.getUniqueListFromBestCombinations()

        # Now we download the data for the new timeframe on the unique list of stocks
        self.getStockData(self.unique_stock_list_from_best_combinations, start, end, use_returns=self.use_returns)

        # Here we set the best combination array from the last iteration to self.stock_combinations
        self.stock_combinations = list(self.best_cointegration_results["stock_combo"])

        # Then we test the stock combinations that did well before against the new data with the different times
        results = self.test_stock_combos()

        # Here we return the results
        self.best_cointegration_results = results

    def getUniqueListFromBestCombinations(self):
        
        x = 0

        # initialize a null list
        unique_list = []

        while x < len(self.best_cointegration_results):
            # print(list(self.best_cointegration_results.iloc[x]['stock_combo']))
            
            combo = list(self.best_cointegration_results.iloc[x]['stock_combo'])
            
            for stock in combo:
                
                if stock not in unique_list:
                    unique_list.append(stock)
            
            x = x + 1
            
        self.unique_stock_list_from_best_combinations = unique_list




    def create_weighted_portfolio_column(self, selected_index):

        selected_result = self.best_cointegration_results.iloc[selected_index]

        results = self.best_cointegration_results.iloc[selected_index].jt

        # print(results.evec[0])

        jt_vectors = results.evec[0]

        stock_combo = list(self.best_cointegration_results.iloc[selected_index].stock_combo)

        self.test_data = self.data[stock_combo].copy()


        x = 0

        weight_columns = []

        # print(f"Columns: {test_data.columns}")

        for column in self.test_data:
            self.test_data[f"{column}_wt"] = self.test_data[column] * jt_vectors[x]

            weight_columns.append(f"{column}_wt")
            x = x + 1

        self.test_data['total'] = 0
        for column in weight_columns:
            # print(column)
            self.test_data['total'] = self.test_data['total'] + self.test_data[column]

        return self.test_data['total']


    def plot_selected_best_cointegration_results(self, best_result_index):

        results = self.best_cointegration_results.iloc[best_result_index].jt
        # results = self.results

        print(results.evec[0])

        jt_vectors = results.evec[0]

        stock_combo = list(self.best_cointegration_results.iloc[best_result_index].stock_combo)

        self.test_data = self.data[stock_combo].copy()

        x = 0

        weight_columns = []

        print(f"Columns: {self.test_data.columns}")

        for column in self.test_data:
            self.test_data[f"{column}_wt"] = self.test_data[column] * jt_vectors[x]
            weight_columns.append(f"{column}_wt")
            x = x + 1

        self.test_data['total'] = 0
        for column in weight_columns:
            # print(column)
            self.test_data['total'] = self.test_data['total'] + self.test_data[column]



        self.test_data['mean'] = self.test_data['total'].mean()

        print(f"Mean Value: {self.test_data['mean'].iloc[-1]}")

        self.test_data['z_score'] = self.z_score(self.test_data['total'])


        self.test_data[['z_score']].plot()

        plt.axhline(0, color='black')
        plt.axhline(1.5, color='red', linestyle='--')
        plt.axhline(-1.5, color='red', linestyle='--')


        self.test_data[['total', 'mean', 'z_score']].plot()

        self.test_data[list(self.best_cointegration_results.iloc[best_result_index]['stock_combo'])].plot()

    # def get_hurst_exponent(self, time_series, max_lag=20):
    #     """Returns the Hurst Exponent of the time series"""
    #
    #     lags = range(2, max_lag)
    #
    #     # variances of the lagged differences
    #     tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    #
    #     # calculate the slope of the log plot -> the Hurst Exponent
    #     reg = np.polyfit(np.log(lags), np.log(tau), 1)
    #
    #     return reg[0]

    def z_score(self, series):
        return (series - series.mean()) / np.std(series)



    def test_results_for_stationarity(self):

        best_results = self.best_cointegration_results

        results = pd.DataFrame()

        x = 0
        while x < len(best_results):

            stock_list = list(self.best_cointegration_results.iloc[x]['stock_combo'])

            jt = self.best_cointegration_results.iloc[x].jt

            test_dataset = self.data.copy()

            test_dataset['total'] = self.create_weighted_portfolio_column(x)

            self.this_test_dataset = test_dataset

            # test_dataset['total'] = test_dataset[stock_list].sum(axis=1)

            adf_result = adfuller(test_dataset['total'], autolag='AIC')

            half_life = self.get_half_life(test_dataset)

            # self.hurst_exponent = self.get_hurst_exponent(self.test_data['total'].values)

            H, c, data = compute_Hc(test_dataset['total'].values)

            # print(f"X: {x}, P Value: {adf_result[1]}")

            if H < 0.5:

                results = results.append({
                    "x": x,
                    "stock_combo": stock_list,
                    "p_value": adf_result[1],
                    "half_life": half_life,
                    "hurst_exponent": H,
                    "jt": jt,
                    "k_ar_diff": self.k_ar_diff


                }, ignore_index=True)

            x = x + 1

        self.results = results

        return results



    def get_half_life(self, series):

        series['total_lag'] = series['total'].shift(1)
        series.dropna(inplace=True)
        series['returns'] = series['total'] - series['total_lag']
        y = series['returns']
        x = series['total_lag']
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()
        beta = results.params.total_lag
        half_life = -math.log(2)/beta
        return half_life







