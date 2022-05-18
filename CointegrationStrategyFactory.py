import datetime
import sys
from CointegrationHunter import CointegrationHunter
from CointegrationBacktester import CointegrationBacktester
from CointegrationOptimizer import CointegrationOptimizer
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from tabulate import tabulate
import statsmodels.api as sm
import math
import pickle


class CointegrationStrategyFactory():

    def __init__(self, data):

        self._in_sample_start_date = data[
            'in_sample_start_date']  # this is in sample because this is what we use to build the strategy
        self._in_sample_end_date = data['in_sample_end_date']
        self._out_sample_end_date = data['out_sample_end_date']
        self._reward_risk_ratio = data['reward_risk_ratio']

        self._number_of_stocks = data['number_of_stocks']
        self._starting_cash = data['starting_cash']
        self._minimum_number_of_trades = data['minimum_number_of_trades']
        self._sector = data['sector']
        self._hurst_exponent_max = data['hurst_exponent_max']
        self._p_value_max = data['p_value_max']
        self._jt_pass_criteria = data['jt_pass_criteria']
        self._drawdown_filter_multiplier = data['drawdown_filter_multiplier']
        self._max_allowable_optimization_combos = data['max_allowable_optimization_combos']
        self._z_score_lookback = data['z_score_lookback']
        self._z_score_open_threshold = data['z_score_open_threshold']


        self._symbols = []
        self._hunter_results = pd.DataFrame()
        self._hunter = None
        self._optimizer = None
        self._best_iterations = pd.DataFrame()
        self._backtester = None
        self.equity_df = pd.DataFrame()
        self._best_iteration_backtests = []

        print("Initializing Strategy Factory")

    def run(self):

        print("Running...")

        # Here we hunt for cointegrated stocks
        print("Getting stocks")
        self._get_stocks()
        print("Hunting for cointegrations")
        self._hunt_for_cointegrations()
        print("First stage filter - here we test all the combinations and filter")
        self._first_stage_filter()
        print("Backtesting best iterations - here we backtest the best iterations from the first stage filter")
        self._backtest_best_iterations()
        print("Creating Summary")
        self._get_results_summary()


    def _get_results_summary(self):

        print("-------------- Summary Results -------------")
        self.equity_df['total'] = self.equity_df[self.equity_df.columns].sum(axis=1)
        self.equity_df[self._in_sample_start_date:]['total'].plot(figsize=(15, 10))

        self.equity_df[self._in_sample_start_date:self._in_sample_end_date]['total'].plot(figsize=(15, 10))
        self.equity_df[self._in_sample_end_date:self._out_sample_end_date]['total'].plot(figsize=(15, 10))
        plt.show()

        self.equity_df[self._in_sample_start_date:self._in_sample_end_date]['total'].plot(figsize=(15, 10))
        plt.show()
        self.equity_df[self._in_sample_end_date:self._out_sample_end_date]['total'].plot(figsize=(15, 10))
        plt.show()

    def _get_equity_regression_results(self, start_date, end_date):

        reg_df = self.equity_df[start_date:end_date]

        reg_df.insert(0, 'day', range(0, len(reg_df)))

        y = reg_df['total']
        x = reg_df['day']
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()


        if results.bse['day'] == 0:
            reward_risk_ratio = 0
        else:
            reward_risk_ratio = results.params['day'] / results.bse['day']

        equity_smoothness = math.sqrt(results.ssr / (results.nobs - 2))

        r_squared_adj = results.rsquared_adj

        return {

            'result': results,
            'reward_risk_ratio': reward_risk_ratio,
            'equity_smoothness': equity_smoothness,
            'r_squared_adj': r_squared_adj

        }

    def _calculate_in_sample_results(self):

        results = self._get_results(self._in_sample_start_date, self._in_sample_end_date)

        self.is_trading_days = results['trading_days']
        self.is_cumulative_return = results['cumulative_return']
        self.is_annualized_return = results['annualized_return']
        self.is_profit = results['profit']
        self.is_reward_risk_ratio = results['reward_risk_ratio']
        self.is_equity_smoothness = results['equity_smoothness']
        self.is_r_squared_adj = results['r_squared_adj']
        self.is_max_drawdown = results['max_drawdown']
        self.is_return_over_max_drawdown = results['return_over_max_drawdown']
        self.is_daily_return_mean = results['daily_return_mean']
        self.is_daily_return_mean = results['daily_return_std']
        self.is_daily_sharpe_ratio = results['daily_sharpe_ratio']
        self.is_annual_sharpe_ratio = results['annual_sharpe_ratio']
        self.is_start_equity = results['start_equity']
        self.is_end_equity = results['end_equity']

    def _calculate_out_sample_results(self):

        results = self._get_results(self._in_sample_end_date, self._out_sample_end_date)

        self.os_trading_days = results['trading_days']
        self.os_cumulative_return = results['cumulative_return']
        self.os_annualized_return = results['annualized_return']
        self.os_profit = results['profit']
        self.os_reward_risk_ratio = results['reward_risk_ratio']
        self.os_equity_smoothness = results['equity_smoothness']
        self.os_r_squared_adj = results['r_squared_adj']
        self.os_max_drawdown = results['max_drawdown']
        self.os_return_over_max_drawdown = results['return_over_max_drawdown']
        self.os_daily_return_mean = results['daily_return_mean']
        self.os_daily_return_mean = results['daily_return_std']
        self.os_daily_sharpe_ratio = results['daily_sharpe_ratio']
        self.os_annual_sharpe_ratio = results['annual_sharpe_ratio']
        self.os_start_equity = results['start_equity']
        self.os_end_equity = results['end_equity']


    def _get_results(self, start_date, end_date):

        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        trading_days = (end - start).days

        cumulative_return = (self.equity_df[start_date:end_date]['total'].iloc[-1] / self.equity_df[start_date:end_date]['total'].iloc[0] - 1)

        annualized_return = ((1 + cumulative_return)**(365/trading_days)) - 1

        start_equity = self.equity_df[start_date:end_date]['total'][0]
        end_equity = self.equity_df[start_date:end_date]['total'][-1]

        profit = end_equity - start_equity

        max_drawdown = self._get_max_drawdown(start_date, end_date)

        reg_results = self._get_equity_regression_results(self._in_sample_start_date, self._in_sample_end_date)

        return_ove_max_dd = cumulative_return / max_drawdown

        daily_returns_mean = self.equity_df[start_date:end_date]['total'].pct_change().mean()
        daily_returns_std = self.equity_df[start_date:end_date]['total'].pct_change().std()

        daily_sharpe_ratio = daily_returns_mean / daily_returns_std
        annual_sharpe_ratio = daily_sharpe_ratio * (252 ** 0.5)

        return {
            'trading_days': trading_days,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'profit': profit,
            'max_drawdown': max_drawdown,
            'reward_risk_ratio': reg_results['reward_risk_ratio'],
            'equity_smoothness': reg_results['equity_smoothness'],
            'r_squared_adj': reg_results['r_squared_adj'],
            'return_over_max_drawdown': return_ove_max_dd,
            'daily_return_mean': daily_returns_mean,
            'daily_return_std': daily_returns_std,
            'daily_sharpe_ratio': daily_sharpe_ratio,
            'annual_sharpe_ratio': annual_sharpe_ratio,
            'start_equity': start_equity,
            'end_equity': end_equity,

        }

    def _get_max_drawdown(self, start_date, end_date):

        df = self.equity_df[start_date:end_date]

        # We are going to use a trailing 252 trading day window
        window = len(df)

        # Calculate the max drawdown in the past window days for each day in the series.
        # Use min_periods=1 if you want to let the first 252 days data have an expanding window
        Roll_Max = df['total'].rolling(window,min_periods=1).max()
        Daily_Drawdown = df['total'] / Roll_Max - 1.0

        # Next we calculate the minimum (negative) daily drawdown in that window.
        # Again, use min_periods=1 if you want to allow the expanding window
        Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()

        return -min(Max_Daily_Drawdown)

    def _calculate_results(self):

        self._calculate_in_sample_results()
        self._calculate_out_sample_results()

    def print_best_iteration_summaries(self, ignore_stopped_out=False):

        for bt in self._best_iteration_backtests:

            if ignore_stopped_out:

                if not bt._stop_loss_triggered:
                    bt._print_backtest2_summary()

            else:

                bt._print_backtest2_summary()


    def print_is_summary(self):

        self._calculate_results()

        print("________________________________________________________________________________")
        print("____________________________ In Sample Summary _________________________________")
        print("________________________________________________________________________________")

        print("")

        data = [["Annualized Return", self.is_annualized_return, "Cumulative Return", self.is_cumulative_return],
                ['Gross Profit', self.is_profit],
                ['Reward Risk Ratio', self.is_reward_risk_ratio, 'Equity Smoothness', self.is_equity_smoothness],
                ['Return Over Max Drawdown', self.is_return_over_max_drawdown],

                ]
        print(tabulate(data, headers=["Item", "value", "Item", "Value"]))

        print("")
        print("____________________________ Hypothesis Test _________________________________")
        print("")

        data = [
            ['Daily Sharpe Ratio', self.is_daily_sharpe_ratio, 'Sharpe Ratio', self.is_annual_sharpe_ratio],
            ['Number of Days', self.is_trading_days]]
        print(tabulate(data, headers=["Item", "value", "Item", "Value"]))

        print("****** Summary *******")
        print(f"Annualized Return: {round(self.is_annualized_return * 100, 2)}%")
        print(f"Max Drawdown: {round(self.is_max_drawdown * 100, 1)}%")
        print(f"Total Trading Days: {self.is_trading_days}")
        print(f"Start Equity: {round(self.is_start_equity, 2)}")
        print(f"End Equity: {round(self.is_end_equity, 2)}")
        print(f"Profits: {round(self.is_profit)}")
        print(f"Cumulative Return: {round(self.is_cumulative_return * 100, 2)}%")
        # print(f"Fees: {round(self.fees, 2)}")
        # print(f"Total Trades: {len(self.trade_ledger)}")
        # print("Open Trades")
        #
        # for trade in self.trades:
        #
        #     print(f"{trade.quantity} shares of {trade.symbol} purchased on {trade.open_date} at {round(trade.open_price, 2)}")
        #
        # print("Open Orders: ")
        # if self.order is None:
        #     print("No open orders")
        # else:
        #
        #     print("************************************")
        #     print("********** ORDER NOTICE ************")
        #
        #     print(f"{self.order['order_string']}")
        #


    def print_os_summary(self):

        self._calculate_results()

        print("________________________________________________________________________________")
        print("____________________________ Out Sample Summary _________________________________")
        print("________________________________________________________________________________")

        print("")

        data = [["Annualized Return", self.os_annualized_return, "Cumulative Return", self.os_cumulative_return],
                ['Gross Profit', self.os_profit],
                ['Reward Risk Ratio', self.os_reward_risk_ratio, 'Equity Smoothness', self.os_equity_smoothness],
                ['Return Over Max Drawdown', self.os_return_over_max_drawdown],

                ]
        print(tabulate(data, headers=["Item", "value", "Item", "Value"]))

        print("")
        print("____________________________ Hypothesis Test _________________________________")
        print("")

        data = [
            ['Daily Sharpe Ratio', self.os_daily_sharpe_ratio, 'Sharpe Ratio', self.os_annual_sharpe_ratio],
            ['Number of Days', self.os_trading_days]]
        print(tabulate(data, headers=["Item", "value", "Item", "Value"]))

        print("****** Summary *******")
        print(f"Annualized Return: {round(self.os_annualized_return * 100, 2)}%")
        print(f"Max Drawdown: {round(self.os_max_drawdown * 100, 1)}%")
        print(f"Total Trading Days: {self.os_trading_days}")
        print(f"Start Equity: {round(self.os_start_equity, 2)}")
        print(f"End Equity: {round(self.os_end_equity, 2)}")
        print(f"Profits: {round(self.os_profit)}")
        print(f"Cumulative Return: {round(self.os_cumulative_return * 100, 2)}%")
        # print(f"Fees: {round(self.fees, 2)}")
        # print(f"Total Trades: {len(self.trade_ledger)}")
        # print("Open Trades")
        #
        # for trade in self.trades:
        #
        #     print(f"{trade.quantity} shares of {trade.symbol} purchased on {trade.open_date} at {round(trade.open_price, 2)}")
        #
        # print("Open Orders: ")
        # if self.order is None:
        #     print("No open orders")
        # else:
        #
        #     print("************************************")
        #     print("********** ORDER NOTICE ************")
        #
        #     print(f"{self.order['order_string']}")
        #

    def _get_stocks(self):

        stock_options = {

            'utilities': ['AES', 'LNT', 'AEE', 'AEP', 'AWK', 'ATO', 'CNP', 'CMS', 'ED', 'D', 'DTE', 'DUK', 'EIX', 'ETR',
                          'EVRG', 'ES', 'EXC', 'FE', 'NEE', 'NI', 'NRG', 'PNW', 'PPL', 'PEG', 'SRE', 'SO', 'WEC',
                          'XEL'],
            'micro_energy': ['AE', 'ALTO', 'BKEP', 'DLNG', 'EGY', 'GPP', 'LEU', 'NC', 'REI', 'USDP'],
            'micro_metals_mining': ['GORO', 'MSB', 'SYNL', 'ZEUS'],
            'industrials': ['MMM', 'AOS', 'ALK','ALLE','AAL','AME','BA','CHRW','CAT','CTAS','CPRT','CSX','CMI','DE',
                            'DAL','DOV','ETN','EMR','EFX','EXPD','FAST','FDX','FTV','FBHS','GD','GE','HON','HWM','HII','IEX',
                            'ITW','JBHT','J','JCI','LHX','LDOS','LMT','MAS','NLSN','NSC','NOC','ODFL','PCAR','PH','PNR',
                            'PWR','RTX','RSG','RHI','ROK','ROL','ROP','SNA','LUV','SWK','TXT','TT','TDG','UNP','UAL','UPS','URI',
                            'VRSK','GWW','WAB','WM','XYL'],
            'energy': ['APA','BKR','CVX','COP','CTRA','DVN','FANG','EOG','XOM','HAL','HES','MRO','MPC','OXY','OKE','PSX',
                       'PXD','SLB','VLO','WMB'],
            'communication_services': ['ATVI','GOOG','T','CHTR','CMCSA','DISCA','DISCK','DISH','EA','FB','FOXA','FOX','IPG',
                                       'LYV','LUMN','MTCH','NFLX','NWSA','NWS','OMC','TMUS','TTWO','DIS','TWTR','VZ'],
            'consumer_discretionary': ['AAP','AMZN','APTV','AZO','BBWI','BBY','BKNG','BWA','CZR','KMX','CCL','CMG','DHI','DRI',
                                       'DG','DLTR','DPZ','EBAY','ETSY','EXPE','F','GPS','GRMN','GM','GPC','HBI','HAS','HLT',
                                       'HD','LVS','LEG','LEN','LKQ','LOW','MAR','MCD','MGM','MHK','NWL','NKE','NCLH','NVR','ORLY',
                                        'PENN','POOL','PHM','PVH','RL','ROST','RCL','SBUX','TPR','TGT','TSLA','TJX','TSCO','ULTA',
                                       'UAA','UA','VFC','WHR','WYNN','YUM',],
            'consumer_staples': ['ADM','MO','CPB','CHD','CLX','KO','CL','CAG','STZ','COST','EL','GIS','HRL','SJM','K','KMB','KHC','KR','LW',
                            'MKC','TAP','MDLZ','MNST','PEP','PM','PG','SYY','HSY','TSN','WBA','WMT',],
            'real_estate': ['ARE','AMT','AVB','BXP','CBRE','CCI','DLR','DRE','EQIX','EQR','ESS','EXR',
                            'FRT','PEAK','HST','IRM','KIM','MAA','PLD','PSA','O','REG','SBAC','SPG','UDR','VTR',
                            'VNO','WELL','WY'],
            'materials': ['APD','ALB','AMCR','AVY','BLL','CF','DD','EMN','FMC','FCX','IFF','IP',
                          'LIN','LYB','MLM','NEM','NUE','PKG','PPG','SEE','SHW','MOS','VMC','WRK'],
            'commodities': ["CORN", "SOYB", "CPER", "GLD", "WEAT", "SLV"],
            'financials': ['AFL','ALL','AXP','AIG','AMP','AON','AJG','AIZ','BAC','BLK','BK','BRO','COF','CBOE','SCHW','CB',
                           'CINF','C','CFG','CME','CMA','DFS','RE','FITB','FRC','BEN','GL','GS','HBAN','ICE','IVZ','JPM',
                           'KEY','LNC','L','MTB','MKTX','MMC','MET','MCO','MS','MSCI','NDAQ','NTRS','PBCT','PNC','PFG','PGR','PRU','RJF',
                           'RF','SPGI','STT','SIVB','SYF','TROW','HIG','TRV','TFC','USB','WRB','WFC','ZION',],
            'barge_companies': ['AEP', 'NBR', 'TDW', 'TRGP', 'CNX', 'KEX', 'HCC', 'GBX', 'MPC', 'MATX'],
            'test': ['AAL','ALK','ALLE','AOS','DAL','DE','DOV','ETN','FAST','FTV','GE','IEX','JCI','LDOS','LUV','NOC','PCAR','SWK','UAL','URI','WAB'],
            'food': ['CF', 'NTR', 'MOS', 'ADM'],
            'global_developed_markets': ['VEA', 'IEFA', 'EFA', 'BNDX', 'VCIT', 'VCSH', 'IXUS',
                                         'SCHF', 'VT', 'IGSB', 'ACWI', 'EFV', 'MINT', 'GDX', 'SPDW',
                                         'IGIB', 'FLOT', 'SRLN', 'BBEU', 'EFG', 'SMH', 'GUNR', 'VSS', 'FNDF'],


        }


        copper_companies = ['BHP', 'FCX', 'TECK', 'SCCO', 'RIO']

        etfs = ['QQQ', 'SPY', 'XLP', 'XLRE', 'XLE', 'XLU', 'XLK', 'XLB', 'XLY', 'XLI', 'XLC', 'XLV']

        airlines = ['ALK',
                    'AZUL',
                    'CEA',
                    'ZNH',
                    'VLRS',
                    'CPA',
                    'DAL',
                    'GOL',
                    'LUV',
                    'SAVE',
                    'ALGT',
                    'AAL',
                    'HA',
                    'JBLU',
                    'MESA',
                    'RYAAY',
                    'SKYW',
                    'UAL']


        self._symbols = stock_options[self._sector]

        # self._symbols = energy

    def _hunt_for_cointegrations(self):
        # stocks = airlines

        print("Hunting for cointegration")

        start = self._in_sample_start_date
        end = self._in_sample_end_date
        stocks_to_maintain = self._number_of_stocks
        use_returns = True

        r = stocks_to_maintain - 1

        k_ar_diff_range = range(1, 13)

        for k_ar_diff in k_ar_diff_range:

            print(f"Testing k_ar_diff {k_ar_diff}")

            self._hunter = CointegrationHunter(total_stocks_to_maintain=stocks_to_maintain, r=r, pass_criteria=self._jt_pass_criteria,
                                         use_returns=use_returns, k_ar_diff=k_ar_diff)

            print("Finding best cointegrations")
            self._hunter.findBestCointegrations(self._symbols, start, end)

            print("Testing for stationarity")
            results = self._hunter.test_results_for_stationarity()

            self.results = results

            if len(results) > 0:

                results = results[
                    (results['hurst_exponent'] < self._hurst_exponent_max) & (results['p_value'] < self._p_value_max)]

                self._hunter_results = self._hunter_results.append(results, ignore_index=True)

            print(f"Found: {len(results)} results using k_ar_diff = {k_ar_diff}")

        print(f"Total hunter results: {len(self._hunter_results)}")

    def _optimize_for_best_combinations(self):

        stock_input = []

        self._df = self._hunter_results.loc[0:self._max_allowable_optimization_combos-1]

        # for index, row in self._hunter_results.iterrows():

        print(f"Total Stock Combinations to Run Through: {len(self._df)}")

        for index, row in self._df.iterrows():

            stock_input.append({
                'stocks': row['stock_combo'],
                'initial_vectors': row['jt'].evec[0]
            })

        parameter_list = {

            'start_date': [self._in_sample_start_date],
            'is_end_date': [self._in_sample_end_date],
            'os_end_date': [self._out_sample_end_date],
            'z_score_lookback': self._z_score_lookback,
            'half_life_multiplier': [1],
            'vector_lookback': [100],
            'z_score_open_threshold': self._z_score_open_threshold,
            'z_score_close_threshold': [0],
            'max_days_in_trade': [100],
            'use_returns': [True],
            'use_kalman_filter': [False],
            'hurst_exponent_limit': [1],
            'k_ar_diff': [2],
            'cash': [self._starting_cash],
            'stock_input': stock_input,
            'drawdown_filter_multiplier': [None],

        }

        total = 1
        for item in parameter_list:
            total = total * len(parameter_list[item])
        print(f'Total Iterations to Optimize: {total}')

        self._optimizer = CointegrationOptimizer(parameter_list, show_iteration_details=False)
        self._optimizer.run_optimization(plot_results=False)

    def _first_stage_filter(self):

        # Here we do a first level filtering where we take the best iterations of each optimization and then
        # select it to be run forward

        self._optimize_for_best_combinations()
        optimizer_results = self._optimizer._optimization_results

        df = optimizer_results[optimizer_results['reward_risk_ratio'] > self._reward_risk_ratio]
        df = df[df['total_trades'] > self._minimum_number_of_trades]

        # Here we pull out the unique combinations that have a reward risk ratio of over 20
        unique_symbols = df['symbol_string'].unique()

        for symbols in unique_symbols:
            # Here we get the best iteration of each stock combination and add it to the best_iteations array
            best_iteration_of_stocks = df[df['symbol_string'] == symbols]
            best_iteration = best_iteration_of_stocks[
                best_iteration_of_stocks['reward_risk_ratio'] == best_iteration_of_stocks['reward_risk_ratio'].max()]

            self._best_iterations = self._best_iterations.append(best_iteration)

    def _backtest_best_iterations(self):

        if len(self._best_iterations) == 0:

            sys.exit("No iterations passed the filter requirements")

        # starting_cash_per_iteration = self._starting_cash / len(self._best_iterations)

        starting_cash_per_iteration = self._starting_cash

        for index, row in self._best_iterations.iterrows():
            parameter_list = {

                'start_date': self._in_sample_start_date,
                'is_end_date': self._in_sample_end_date,
                'os_end_date': self._out_sample_end_date,
                'z_score_lookback': row['z_score_lookback'],
                'half_life_multiplier': row['half_life_multiplier'],
                'vector_lookback': row['vector_lookback'],
                'z_score_open_threshold': row['z_score_open_threshold'],
                'z_score_close_threshold': row['z_score_close_threshold'],
                'max_days_in_trade': row['max_days_in_trade'],
                'use_returns': True,
                'use_kalman_filter': row['use_kalman_filter'],
                'hurst_exponent_limit': row['hurst_exponent_limit'],
                # 'use_rolling_lookback_for_z_score': row['use_rolling_lookback_for_z_score'],
                'k_ar_diff': row['k_ar_diff'],
                'use_fixed_average_price': row['use_fixed_average_price'],
                'cash': starting_cash_per_iteration,
                # 'use_rolling_lookback_for_jt_test': row['use_rolling_lookback_for_jt_test'],
                'stock_input': {'stocks': row['symbols'], 'initial_vectors': row['initial_vectors']},
                'drawdown_filter_multiplier': self._drawdown_filter_multiplier,

            }

            self._backtester = CointegrationBacktester()
            self._backtester.configure(parameter_list)
            self._backtester.run_backtest_2(print_summary=True)

            self.equity_df[index] = self._backtester.model_dataset['equity']
            self._best_iteration_backtests.append(self._backtester)

    def save(self, filename):

        file_pi = open('savedStrategies/'+filename+'.obj', 'wb')
        pickle.dump(self, file_pi)



