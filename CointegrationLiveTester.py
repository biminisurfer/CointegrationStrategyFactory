import pickle
import pandas as pd
import matplotlib.pyplot as plt
from FileManager import FileManager
from tabulate import tabulate

from IPython.display import display
from scipy.stats import ttest_ind

import numpy as np
import copy

from PerformanceMeasurements import PerformanceMeasurements

'''
This class loads the CointegrationStrategyFactory and then uses the data to create a live test

data = {
    
    'live_test_start_date': '2022-5-18',
    'filename': 'LiveTradeIndustrials',
    'max_backtests': 5,
    'max_allowable_allocation': 0.75,
    'initial_cash': 4000,
    'description': 'This is the first real live trade that we are starting on 2022-05-18'
}

lt = CointegrationLiveTester(data)

lt.run(print_backtest_summary=True)
lt.plot_combined_strategies()
lt.perform_t_test(alpha=0.05, verbose=True)
lt.perform_monte_carlo(total_trading_days=60, simulations=100, plot_results=True)

'''

class CointegrationLiveTester():

    def __init__(self, data):

        self.live_test_start_date = data['live_test_start_date']
        self._filename = data['filename']
        self._max_backtests = data['max_backtests']
        self._max_allowable_allocation = data['max_allowable_allocation']
        self._initial_cash = data['initial_cash']


        self.is_end_date = None
        self.backtests = []
        self.os_data = pd.DataFrame()
        self.is_data = pd.DataFrame()
        self.open_trades = pd.DataFrame()
        self.trade_ledger = pd.DataFrame()
        self.backtest_df = pd.DataFrame()
        self.symbols = []
        self.price_df = pd.DataFrame()
        self.orders = pd.DataFrame()
        self.order_summary = pd.DataFrame()
        self.in_sample_trades = pd.DataFrame()
        self.is_mean_daily_return = None
        self.os_mean_daily_return = None

        self._load_sf_from_file()
        self._filter_non_stopped_out_strategies()
        # self._combine_strategies()
        self.backtests_up_to_today = []
        self.backtests_live_trade = []

        self._get_backtest_df()
        self._get_symbols_from_backtests()
        self._create_price_df()




    def _load_sf_from_file(self):

        print("Loading Saved Strategy from File")

        file_pi2 = open(f'savedStrategies/{self._filename}.obj', 'rb')
        self.sf = pickle.load(file_pi2)
        self.is_end_date = self.sf._in_sample_end_date

    def _filter_non_stopped_out_strategies(self):

        print("Filtering out non-stopped out strategies")

        for bt in self.sf._best_iteration_backtests:

            if not bt._stop_loss_triggered:

                self.backtests.append(bt)



    def _combine_strategies(self):

        index = 0

        for bt in self.backtests:
            self.os_data[f'bt_{index}_equity'] = bt.model_dataset['equity']

            index = index + 1

        self.os_data['total'] = self.os_data[self.os_data.columns].sum(axis=1)



    def plot_combined_strategies(self):

        print("************** Combined Strategies **************")
        # self._combine_strategies()

        self.os_data[:self.is_end_date]['total'].plot(figsize=(15,10))
        self.os_data[self.is_end_date:]['total'].plot(figsize=(15,10))
        plt.show()

    def _calculate_summary(self):

        self._combine_strategies()

        self.is_data['total'] = self.is_data[self.is_data.columns].sum(axis=1)

        self.is_mean_daily_return = self.is_data['total'].pct_change().mean()
        self.os_mean_daily_return = self.os_data['total'].pct_change().mean()
        self.is_daily_std = self.is_data['total'].pct_change().std()
        self.os_daily_std = self.os_data['total'].pct_change().std()
        self.is_max_drawdown = self._get_max_drawdown(self.is_data['total'], len(self.is_data['total']))
        self.os_max_drawdown = self._get_max_drawdown(self.os_data['total'], len(self.os_data['total']))

        pm = PerformanceMeasurements()

        self.os_start_equity = self.os_data['total'][0]
        self.os_end_equity = self.os_data['total'][-1]
        self.os_profit = self.os_end_equity - self.os_start_equity
        self.os_total_return = self.os_profit / self.os_start_equity
        self.os_total_days = (self.os_data.iloc[-1].name - self.os_data.iloc[0].name).days
        self.os_daily_return = self.os_total_return / self.os_total_days
        self.os_annualized_return = ((1 + self.os_daily_return) ** 365) - 1

        if self.os_max_drawdown == 0:
            self.os_return_over_max_drawdown = 0
        else:
            self.os_return_over_max_drawdown = self.os_total_return / -self.os_max_drawdown

        if self.os_daily_std == 0:
            self.os_sharpe_ratio = 0
            self.os_daily_sharpe_ratio = 0

        else:
            self.os_daily_sharpe_ratio = self.os_mean_daily_return / self.os_daily_std
            self.os_sharpe_ratio = self.os_daily_sharpe_ratio * (252 ** 0.5)

    def print_summary(self):



        print("OS Performance")

        data = [

            ["Total Profit", self.os_profit],
            ["Total Return", self.os_total_return],
            ["Annualized Return", self.os_annualized_return],
            ["Sharpe Ratio", self.os_sharpe_ratio],
            ["Return Over Max Drawdown", self.os_return_over_max_drawdown],

            ["Max Drawdown", self.os_max_drawdown],



            ["Start Equity", self.os_start_equity],
            ["End Equity", self.os_end_equity],
            ["Total Days Trading", self.os_total_days],
            ["Daily Return", self.os_daily_return],

        ]
        print(tabulate(data, headers=["OS Metric", "Value"]))

        print("")

        print("IS vs. OS Comparison")

        data = [

            ["Mean Daily Return", self.is_mean_daily_return, self.os_mean_daily_return, self.os_mean_daily_return - self.is_mean_daily_return],

            ["Mean Daily Std", self.is_daily_std, self.os_daily_std, self.os_daily_std - self.is_daily_std],
            ["Max Drawdown", self.is_max_drawdown, self.os_max_drawdown, self.os_max_drawdown - self.is_max_drawdown],

            # ["Total Trades", self.in_sample_trades, self.os_max_drawdown, self.os_max_drawdown - self.is_max_drawdown],

        ]
        print(tabulate(data, headers=["Item", "In Sample", "Out Sample", "Variance"]))



    def print_summaries(self):

        for bt in self.backtests:

            bt._print_backtest2_summary()

    def _get_backtest_df(self):

        print("Getting backtest df")

        for bt in self.backtests:

            self.backtest_df = self.backtest_df.append(bt.__dict__, ignore_index=True)


    def _get_symbols_from_backtests(self):

        print("Getting symbols from backtests")

        live_test_symbols = []

        for symbol_array in self.backtest_df.symbols:

            for symbol in symbol_array:

                if symbol not in live_test_symbols:
                    live_test_symbols.append(symbol)

        self.symbols = live_test_symbols

    def _create_price_df(self):

        print("Creating Price Dataframe")

        price_df = pd.DataFrame()

        fm = FileManager()

        for symbol in self.symbols:
            fm.get_data(symbol, self.live_test_start_date, None)

            price_df[f'{symbol}_open'] = fm.data['Open']
            price_df[f'{symbol}_close'] = fm.data['Close']

        self.price_df = price_df

    def _filter_backtests(self):

        print("Filtering backtests")

        # Here we only use the top 5 backtests
        self.backtest_df = self.backtest_df.sort_values('reward_risk_ratio', ascending=False).head(self._max_backtests)

        backtests = []

        for index, bt in self.backtest_df.iterrows():

            backtests.append(self.backtests[index])

        self.backtests = backtests

    def _get_open_trade_value(self):

        pass

    def _get_max_drawdown(self, series, window):


        # Calculate the max drawdown in the past window days for each day in the series.
        # Use min_periods=1 if you want to let the first 252 days data have an expanding window
        roll_max = series.rolling(window, min_periods=1).max()
        daily_drawdown = series / roll_max - 1.0

        # Next we calculate the minimum (negative) daily drawdown in that window.
        # Again, use min_periods=1 if you want to allow the expanding window
        max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

        return max_daily_drawdown.min()

    def get_current_positions(self):

        array = []

        df = self.open_trades

        print("Current Positions")

        symbols = df.symbol.unique()

        for symbol in symbols:
            quantity = df[df['symbol'] == symbol]['quantity'].sum()

            array.append({
                'symbol': symbol,
                'quantity': quantity
            })

        return pd.DataFrame(array)


    def get_target_positions(self):

        array = []

        df = self.orders[self.orders['executed'] == 0]

        for symbol in df['symbol'].unique():
            array.append({
                'symbol': symbol,
                'shares': df[df['symbol'] == symbol]['shares'].sum()
            })

        order_summary = pd.DataFrame(array)

        array = []

        open_trade_summary_df = pd.DataFrame()

        if len(self.open_trades) > 0:


            open_trade_summary_df['symbol'] = self.open_trades.symbol.unique()

            for index, row in open_trade_summary_df.iterrows():
                symbol = row.symbol

                current_quantity_in_portfolio = self.open_trades[self.open_trades['symbol'] == symbol]['quantity'].sum()

                if len(order_summary) == 0:

                    order_amount = 0

                else:

                    order_amount = order_summary[order_summary['symbol'] == symbol]['shares'].sum()

                target_quantity = current_quantity_in_portfolio + order_amount

                array.append({
                    'symbol': row.symbol,
                    'target_quantity': target_quantity

                })

        return pd.DataFrame(array)

    def get_order_summary(self):

        # This method summarizes the orders by summing the total for each symbol

        array = []

        symbols = self.orders.symbol.unique()

        for symbol in symbols:

            shares = self.orders[self.orders['symbol'] == symbol]['shares'].sum()

            array.append({
                'symbol': symbol,
                'quantity': shares
            })

        return pd.DataFrame(array)

    def perform_monte_carlo(self, total_trading_days=None, simulations=100, plot_results=True):

        '''
        The method performs a monte carlo simulation on the in_sample dataset

        :param simulations: this determines how many simulations to run
        :param plot_results: default = True, determines if we plot the results
        :param total_trading_days: default = None, this parameter is to specify how many days to run each simulation
        a default of none means that it will default to the length of the is_data (in sample data) that was used

        :return:
        '''

        self._monte_carlo_df = pd.DataFrame()

        self._monte_carlo_result_df = pd.DataFrame() # This is a summary of results from each simulation

        starting_equity = self._initial_cash * len(self.backtests)

        return_cum_sum_columns = []
        return_columns = []
        equity_columns = []
        max_daily_drawdown_array = []

        summary_df = pd.DataFrame()

        print("")
        print("***************************** Running Monte Carlo *********************************")
        print(f"Total Trading Days: {total_trading_days}")
        print(f"Simulations: {simulations}")

        returns = self.is_data['total'].pct_change()
        # returns = self.os_data['total'].pct_change()

        return_summary = returns.describe()

        for simulation in range(simulations):
            simulated_returns = np.random.normal(return_summary['mean'], return_summary['std'],
                                                 int(return_summary['count']))

            self._monte_carlo_df[f'return_{simulation}'] = simulated_returns[0:total_trading_days]

            self._monte_carlo_df[f'log_ret_{simulation}'] = np.log(1 + self._monte_carlo_df[f'return_{simulation}'])

            self._monte_carlo_df[f'log_ret_cum_sum_{simulation}'] = self._monte_carlo_df[f'log_ret_{simulation}'].cumsum()

            self._monte_carlo_df[f'return_cum_sum_{simulation}'] = np.exp(self._monte_carlo_df[f'log_ret_cum_sum_{simulation}'])

            self._monte_carlo_df[f'equity_{simulation}'] = self._monte_carlo_df[f'return_cum_sum_{simulation}'] * self._initial_cash

            max_daily_drawdown = self._get_max_drawdown(self._monte_carlo_df[f'equity_{simulation}'], total_trading_days)

            final_equity = self._monte_carlo_df.iloc[-1][f'equity_{simulation}']

            total_return = (final_equity - starting_equity) / starting_equity

            self._monte_carlo_result_df = self._monte_carlo_result_df.append({

                'max_drawdown': max_daily_drawdown,
                'final_equity': final_equity,
                'total_return': total_return,

            }, ignore_index=True)

            return_cum_sum_columns.append(f'return_cum_sum_{simulation}')
            return_columns.append(f'return_{simulation}')
            equity_columns.append(f'equity_{simulation}')


        if plot_results:
            self._monte_carlo_df[return_cum_sum_columns].plot(figsize=(15, 10), legend=False, title='Total Returns')
            self._monte_carlo_df[equity_columns].plot(figsize=(15, 10), legend=False, title='Equity')
            pd.DataFrame({'Max Drawdown': max_daily_drawdown_array}).hist(bins=100)



    def perform_t_test(self, alpha=0.05, verbose=True):

        '''
        This function peforms a t-test on the in sample (is_data) and out sample (os_data) to determine how
        close the out of sample data represents the in sample dataset

        :param alpha: what level of confidence to use in determining whether we pass the null hypotheses
        Note here that a higher alpha means that the out of sample data will have to deviate more to be considered
        not of the same type of data as the in sample. Default here is 0.05

        :param verbose: default = True, determines if we want to print the message out

        :return: stat, p, message
        '''

        is_return_data = self.is_data['total'].pct_change().dropna().values
        os_return_data = self.os_data['total'].pct_change().dropna().values

        stat, p = ttest_ind(is_return_data, os_return_data)
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            message = 'Probably the same distribution'
            null_hypothesis = True

        else:
            message = 'Probably different distributions'
            null_hypothesis = False

        if verbose:
            print(message)

        return {
            'stat': stat,
            'p': p,
            'message': message,
            'null_hypothesis': null_hypothesis
        }


    def run(self, print_backtest_summary=False):

        print("Running Live Tester")

        self._filter_backtests()

        index = 0

        print(f"Total Strategy Components: {len(self.backtests)}")

        for bt in self.backtests:

            bt.max_allowable_allocation = self._max_allowable_allocation

            bt.initial_cash = self._initial_cash

            print("Run Backtest Up To Today")
            bt.run_backtest_up_to_today(print_summary=print_backtest_summary)

            # new_bt = copy.deepcopy(bt)
            # self.backtests_up_to_today.append(new_bt)

            # Here we add the trades from the backtest to the trade ledger
            if len(bt.trade_ledger) > 0:
                self.trade_ledger = self.trade_ledger.append(
                    bt.trade_ledger[bt.trade_ledger['close_date'] >= self.live_test_start_date], ignore_index=True)

            self.is_data[f'bt_{index}_equity'] = bt.model_dataset['equity'][:self.is_end_date]

            print("Live Trade")

            bt.live_trade(self.live_test_start_date, print_summary=print_backtest_summary)

            # self.backtests_live_trade.append(bt)

            self.open_trades = self.open_trades.append(bt.open_trades_df, ignore_index=True)

            self.orders = self.orders.append(bt.broker.get_open_orders(), ignore_index=True)

            # if not bt._stop_loss_triggered:
            #     self.backtest_df = self.backtest_df.append(bt.__dict__, ignore_index=True)

            index = index + 1

        self._calculate_summary()

        print("****************************** Completed Running Backtests *******************************")

        # Here we sum it is_data dataset

        print('\n\n\n')
        print('************************************** Live Closed Trades ****************************************************')

        display(self.trade_ledger)

        print('************************************** Live Open Trades ****************************************************')

        display(self.open_trades)


        print('************************************** Live Open Orders ****************************************************')

        self.order_summary = self.get_order_summary()

        display(self.order_summary)

        print('************************************ Current Positions ***************************************************')

        display(self.get_current_positions().sort_values('symbol'))

        print(
            '************************************** Target Positions ****************************************************')

        display(self.get_target_positions().sort_values('symbol'))

        print("************************************ Order Summary ***************************************************")
        display(self.get_order_summary().sort_values('symbol'))


