import matplotlib.pyplot as plt

import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm

import math
import statsmodels.tsa.vector_ar.vecm as vecm
from statsmodels.tsa.stattools import adfuller
from Trade import Trade
import datetime
from tabulate import tabulate
from pykalman import KalmanFilter
from hurst import compute_Hc, random_walk

import scipy.stats
from FileManager import FileManager
from numpy import linalg as LinAlgError
import sys
from IPython.display import display
from Broker import Broker

'''
# Example on how to use this class


backtester = CointegrationBacktester()
stocks = ['AEP', 'SRE', 'XEL']


backtest_data = {
    "start_date": "2017-1-1",
    "end_date": "2022-1-22",
    "symbols": stocks,
    "lookback": 100,
    "cash": 5000,
    'z_score_open_threshold': 0,
    'z_score_close_threshold': .1,
    "bollinger_length": 20, # this is the bollinger of the fast z score
    "bollinger_dev": 1 # This is teh bollinger dev of the fast z score

}
backtester.configure(backtest_data, download_new_data=download_new_data)
backtester.run_backtest()
backtester.trading_df.set_index('date', drop=True, inplace=True)
backtester.print_summary()


'''
class CointegrationBacktester():

    def __init__(self):

        self.start_date = None
        self.os_end_date = None
        self.symbols = None
        self.cash = None
        self.initial_cash = 0
        self.vector_lookback = None
        self.z_score_lookback = None
        self.use_returns = False
        self.use_adj_close = False

        self.initial_vectors = []
        self.z_score = 0
        self.fixed_average_price = 0
        self.use_fixed_average_price = False
        self.cost_per_trade = 4.00

        # These are the thresholds where we open and close positions based on the current z_score
        # A positive value will initiate a short position
        self.z_score_open_threshold = None
        self.z_score_close_threshold = None
        self.daily_sharpe_ratio = 0

        self.open_trade_value = 0
        self.open_profits = 0
        self.equity = 0
        self.data = pd.DataFrame()
        self.price_data = None
        self.open_data = None
        # self.max_allowable_allocation = 0.75
        self.max_allowable_allocation = 0.5

        self.date = None
        self.trading_df = pd.DataFrame()
        self.market_position = "flat"
        self.allow_trades = False
        self.order = None
        self.orders = []
        self.order_ledger = pd.DataFrame()
        self.trades = []
        self.trade_ledger = pd.DataFrame()
        self.fees = 0
        self.verbose = False
        self.daily_returns_std = 0
        self.daily_returns_mean = 0
        self.sharpe_ratio = 0
        self.max_drawdown = 0
        self.cumulative_return = 0
        # Z Score Close Threshold: 0.25
        # Rolling Vector Lookback: 1
        # P Value Threshold: 1
        # Stocks: ['CTRA', 'KMI', 'OXY', 'WMB']
        # Initial Vectors: [0.47268191, 0.39158712, 0.03933233, 0.02413786]
        # K AR DIFF: 2
        # Vector Lookback: 100
        # Z Score Lookback: 100
        #
        self.profits = 0
        self.total_days = 0
        self.annualized_return = 0
        self.return_over_max_drawdown = 0
        self.open_trades_next_day = False
        self.use_entire_trading_df_for_lookback = False
        self.bollinger_length = None
        self.bollinger_dev = None
        self.jt_sample = pd.DataFrame()
        self.lookback_sample = pd.DataFrame()
        self.k_ar_diff = 0

        self.r_squared_adj = 0

        self.total_trades = 0
        self.total_winning_trades = 0
        self.total_losing_trades = 0
        self.gross_profit = 0
        self.gross_loss = 0
        self.net_profit = 0
        self.average_winning_trade = 0
        self.average_losing_trade = 0
        self.percentage_winning_trades = 0
        self.equity_smoothness = 0
        self.reward_risk_ratio = 0
        self.first_trade_date = None
        self.vectors = pd.DataFrame()
        self.vector_df = pd.DataFrame()
        self.day = 0
        self.trading_day = 0
        self.trading_data = pd.DataFrame()
        self.kalman_vector_df = pd.DataFrame()
        self.kalman_half_life = None
        self.hurst_exponent = None
        self.use_kalman_filter = False
        self.hurst_exponent_limit = 0.5
        self.equity_df = pd.DataFrame()
        self.shares = []
        self.exit_all_trades = False
        self.days_in_trade = 0
        self.max_days_in_trade = 0
        self.use_half_life_for_z_score_lookback = False
        self.half_life_multiplier = 1
        self.use_rolling_lookback_for_z_score = False
        self.model_dataset = pd.DataFrame()
        self._stop_loss_triggered = False
        self.force_download = False
        self.max_allowable_drawdown = 0
        self.live_trade_start_date = None
        self.broker = None
        self.live_trading = False


    def configure(self, data, download_new_data=True):

        self.symbols = data['stock_input']['stocks']
        self.cash = data['cash']
        self.initial_cash = self.cash
        self.z_score_lookback = data['z_score_lookback']
        self.vector_lookback = data['vector_lookback']

        # These are the thresholds where we open and close positions based on the current z_score
        # A positive value will initiate a short position
        self.z_score_open_threshold = data['z_score_open_threshold']
        self.z_score_close_threshold = data['z_score_close_threshold']
        self.k_ar_diff = data['k_ar_diff']
        self.use_returns = data['use_returns']
        self.initial_vectors = data['stock_input']['initial_vectors']
        self.use_kalman_filter = data['use_kalman_filter']
        self.hurst_exponent_limit = data['hurst_exponent_limit']
        self.max_days_in_trade = data['max_days_in_trade']
        self._drawdown_filter_multiplier = data['drawdown_filter_multiplier'] # If set to None then we don't use this filter

        self.start_date = data['start_date']
        self.is_end_date = data['is_end_date']  # This is in sample end date
        self.os_end_date = data['os_end_date']  # this is out of sample end date

        self.allow_trades = True
        self.open_trade_value = 0
        self.open_profits = 0
        self.equity = 0
        self.trading_df = pd.DataFrame()
        self.market_position = "flat"
        self.order = None
        # self.price_data = pd.DataFrame()
        # self.data = pd.DataFrame()
        self.orders = []
        self.order_ledger = pd.DataFrame()
        self.trades = []
        self.trade_ledger = pd.DataFrame()
        self.jt_sample = pd.DataFrame()
        self.lookback_sample = pd.DataFrame()
        self.total_trades = 0
        self.total_winning_trades = 0
        self.total_losing_trades = 0
        self.gross_profit = 0
        self.gross_loss = 0
        self.net_profit = 0
        self.average_winning_trade = 0
        self.average_losing_trade = 0
        self.percentage_winning_trades = 0
        self.equity_smoothness = 0
        self.reward_risk_ratio = 0
        self.equity_df = pd.DataFrame()
        self.days_in_trade = 0
        # Here we download the stock data
        self.get_stock_data()
        self.order = None
        self.equity = 0
        self.fees = 0
        self.daily_returns_std = 0
        self.daily_returns_mean = 0
        self.sharpe_ratio = 0
        self.max_drawdown = 0
        self.cumulative_return = 0
        self.profits = 0
        self.total_days = 0
        self.annualized_return = 0
        self.return_over_max_drawdown = 0
        self.open_trades_next_day = False
        self.first_trade_date = None
        self.force_download = False
        self.kalman_vector_df = pd.DataFrame()
        self.kalman_half_life = 0
        self.vector_df = pd.DataFrame()
        self.initialize_equity_df()
        self.initialize_dataset()

        self.broker = Broker({
            'symbols': self.symbols,
            'initial_vectors': self.initial_vectors,
            'start_date': self.start_date,
            'commission': self.cost_per_trade,
            'max_allowable_allocation': self.max_allowable_allocation
        })



    def reset(self):
        self.cash = self.initial_cash
        self.allow_trades = True
        self.open_trade_value = 0
        self.open_profits = 0
        self.equity = 0
        self.trading_df = pd.DataFrame()
        self.market_position = "flat"
        self.order = None
        self.orders = []
        self.order_ledger = pd.DataFrame()
        self.trades = []
        self.trade_ledger = pd.DataFrame()
        self.jt_sample = pd.DataFrame()
        self.lookback_sample = pd.DataFrame()
        self.total_trades = 0
        self.total_winning_trades = 0
        self.total_losing_trades = 0
        self.gross_profit = 0
        self.gross_loss = 0
        self.net_profit = 0
        self.average_winning_trade = 0
        self.average_losing_trade = 0
        self.percentage_winning_trades = 0
        self.equity_smoothness = 0
        self.reward_risk_ratio = 0
        self.equity_df = pd.DataFrame()
        self.days_in_trade = 0
        # Here we download the stock data
        self.get_stock_data()
        self.order = None
        self.equity = 0
        self.fees = 0
        self.daily_returns_std = 0
        self.daily_returns_mean = 0
        self.sharpe_ratio = 0
        self.max_drawdown = 0
        self.cumulative_return = 0
        self.profits = 0
        self.total_days = 0
        self.annualized_return = 0
        self.return_over_max_drawdown = 0
        self.open_trades_next_day = False
        self.first_trade_date = None
        self.kalman_vector_df = pd.DataFrame()
        self.kalman_half_life = 0
        self.vector_df = pd.DataFrame()
        self.initialize_equity_df()
        self.initialize_dataset()

        self.broker = Broker({
            'symbols': self.symbols,
            'initial_vectors': self.initial_vectors,
            'start_date': self.start_date,
            'commission': self.cost_per_trade,
            'max_allowable_allocation': self.max_allowable_allocation
        })

    def initialize_dataset(self):

        x = 0

        synthetic_equity_name_array = []

        self.model_dataset['synthetic_equity'] = 0
        self.model_dataset['position'] = 0
        self.model_dataset['position_signal'] = 0
        self.model_dataset['equity'] = self.initial_cash
        self.model_dataset['cash'] = 0
        self.model_dataset['margin'] = 0
        self.model_dataset['broker_fees'] = 0

        for symbol in self.symbols:

            self.model_dataset[f'{symbol}_daily_return'] = self.model_dataset[f'{symbol}_adj_close'].pct_change()
            self.model_dataset[f'{symbol}_cum_sum_return'] = self.model_dataset[f'{symbol}_daily_return'].cumsum()
            self.model_dataset[f'{symbol}_synthetic_equity'] = self.model_dataset[f'{symbol}_cum_sum_return'] * self.initial_vectors[x]
            self.model_dataset['synthetic_equity'] = self.model_dataset['synthetic_equity'] + self.model_dataset[f'{symbol}_synthetic_equity']
            self.model_dataset[f'{symbol}_position'] = 0
            self.model_dataset[f'{symbol}_quantity'] = 0
            self.model_dataset[f'{symbol}_equity'] = 0

            x = x + 1

        self.model_dataset['synthetic_equity_std'] = self.model_dataset['synthetic_equity'].rolling(
            int(self.z_score_lookback)).std()

        self.model_dataset['synthetic_equity_mean'] = self.model_dataset['synthetic_equity'].rolling(
            int(self.z_score_lookback)).mean()

        self.model_dataset['z_score'] = (self.model_dataset['synthetic_equity'] - self.model_dataset[
            'synthetic_equity_mean']) / self.model_dataset['synthetic_equity_std']

        self.model_dataset['normalized_std'] = self.model_dataset['synthetic_equity_std'] / self.model_dataset['synthetic_equity_mean']

    def get_stock_data(self):

        data = pd.DataFrame()
        returns = pd.DataFrame()# This is the dataframe for the returns
        open_data = pd.DataFrame()# This dataframe has open data on it

        self.open_data = pd.DataFrame()
        self.data = pd.DataFrame()
        self.model_dataset = pd.DataFrame()

        # Here we subtract the lookback days from the start date to ensure we get the data at the beginning of the
        # dataset
        start_datetime = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
        dataset_start_date = start_datetime - datetime.timedelta(self.vector_lookback * 1.5)
        dataset_start_date_string = datetime.datetime.strftime(dataset_start_date, "%Y-%m-%d")

        fm = FileManager()

        for stock in self.symbols:

            fm.get_data(stock, dataset_start_date_string, self.os_end_date, force_download=self.force_download)

            prices = fm.data

            # self.new_prices = prices

            # This is when we downloaded each time
            # self.original_prices = yf.download(stock, dataset_start_date_string, self.os_end_date)

            self.model_dataset[f'{stock}_adj_close'] = prices['Adj Close']
            self.model_dataset[f'{stock}_open'] = prices['Open']
            self.model_dataset[f'{stock}_close'] = prices['Close']
            self.model_dataset[f'{stock}_high'] = prices['High']
            self.model_dataset[f'{stock}_low'] = prices['Low']
            self.model_dataset[f'{stock}_volume'] = prices['Volume']

            if self.use_adj_close:
                data[stock] = prices['Adj Close']

            else:
                data[stock] = prices['Close']

            open_data[stock] = prices['Open']
            returns[stock] = np.append(
                data[stock][1:].reset_index(drop=True) / data[stock][:-1].reset_index(drop=True) - 1, 0)

        self.price_data = data

        if self.use_returns:
            self.data[self.symbols] = self.price_data[self.symbols].pct_change()
            self.data[self.symbols] = self.data.cumsum()

        else:
            self.data = self.price_data

        self.data['day'] = 0
        # days_before_trade = len(self.data[:self.start_date])
        # self.open_data = open_data.dropna()
        self.open_data = open_data

        days_before_trade = (len(self.open_data)) - len(self.data[self.start_date:]['day'])

        self.days_before_trade = days_before_trade

        #I use this since I end up dropping na values and this reduces the number by 1 when we use returns
        # if self.use_returns:
        #     days_before_trade = days_before_trade - 1

        self.data['day'] = range(-days_before_trade, len(self.data[self.start_date:]['day']))
        self.price_data['day'] = self.data['day']

        self.open_data['day'] = range(-days_before_trade, len(self.data[self.start_date:]['day']))

        self.trading_data['day'] = range(0, len(self.trading_data))
        self.data = self.data.astype({"day": int})


    def initialize_equity_df(self):

        columns = ['date', 'cash']

        self.shares = []
        for symbol in self.symbols:
            self._message(symbol)
            columns.append(symbol + '_price')
            columns.append(symbol + '_qty')
            self.shares.append(0)

        self.equity_df = pd.DataFrame(columns=columns)
        self.equity_df.set_index('date')



    def get_half_life(self, trading_df):

        series = trading_df.copy()
        series['total_lag'] = series['synthetic_equity'].shift(1)
        series.dropna(inplace=True)
        series['returns'] = series['synthetic_equity'] - series['total_lag']
        y = series['returns']
        x = series['total_lag']
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()
        beta = results.params.total_lag
        half_life = -math.log(2) / beta
        return half_life

    def get_z_score(self, series):

        if self.use_fixed_average_price:
            mean = self.fixed_average_price
        else:
            mean = series.mean()

        return (series - mean) / np.std(series)
        # return (series - series.mean()) / np.std(series)

    def get_open_profit(self, prices):

        self.open_profits = 0
        x = 0
        for trade in self.trades:
            self.open_profits = self.open_profits + trade.update_trade(prices[x])
            x = x + 1

        return self.open_profits

    def _get_position_signal(self):

        date = self.bar.name

        self._message(f"Position is: {self.position}")

        self._message(
            f"Z score of {self.bar['z_score']}, Open Threshold: {self.z_score_open_threshold}, Close Threshold: {self.z_score_close_threshold}")

        if self.position == 0:

            if self.bar['z_score'] >= self.z_score_open_threshold:

                self._message(f"Z score of {self.bar['z_score']} is above open threshold of {self.z_score_open_threshold}, sending short signal")

                self.model_dataset.at[date, 'position_signal'] = -1


            elif self.bar['z_score'] <= -self.z_score_open_threshold:

                self._message(
                    f"Z score of {self.bar['z_score']} is below open negative open threshold of {-self.z_score_open_threshold}, sending long signal")

                self.model_dataset.at[date, 'position_signal'] = 1

            else:

                self._message(
                    f"Z score of {self.bar['z_score']} is not outside of the open threshold of {self.z_score_open_threshold}, sending flat signal")

                self.model_dataset.at[date, 'position_signal'] = 0


        # If we are in a long position
        elif self.position == 1:

            if self.bar['z_score'] >= -self.z_score_close_threshold:
                self._message(
                    f"Z score of {self.bar['z_score']} is greater than close threshold of {-self.z_score_close_threshold}, sending short signal to close long position")

                self.model_dataset.at[date, 'position_signal'] = -1

        elif self.position == -1:

            if self.bar['z_score'] <= self.z_score_close_threshold:

                self._message(
                    f"Z score of {self.bar['z_score']} is less than close threshold of {self.z_score_close_threshold}, sending long signal to close short position")

                self.model_dataset.at[date, 'position_signal'] = 1


        else:
            sys.exit('Position code is incorrect. Something is wrong here')

    def set_quantities(self, position_signal):

        self._message(f"Cash on hand to open position: {self.cash}")

        self.model_dataset.at[self.bar.name, 'position'] = position_signal

        index = 0
        total_abs_open_value = 0
        for symbol in self.symbols:

            vector = self.initial_vectors[index] * position_signal
            open_price = self.bar[f"{symbol}_open"]
            open_value = vector * open_price
            abs_open_value = abs(open_value)
            total_abs_open_value = total_abs_open_value + abs_open_value
            self._message(f"Open price for {symbol} is {open_price}, Vector is : {vector}, Open Value is: {open_value}")
            index = index + 1

        self._message(f"Total abs open value: {total_abs_open_value}")

        vector_multiplier = (self.cash * self.max_allowable_allocation) / total_abs_open_value

        self._message(f"Vector multiplier: {vector_multiplier}")


        index = 0
        total_purchase_cost = 0

        self.commission = 0

        for symbol in self.symbols:
            shares = math.floor(self.initial_vectors[index] * vector_multiplier) * position_signal
            open_price = self.bar[f"{symbol}_open"]
            purchase_cost = abs(shares * open_price)
            total_purchase_cost = total_purchase_cost + purchase_cost
            self._message(f"Purchase {shares} shares of {symbol} at open price of {open_price} for value of {purchase_cost}")
            self.model_dataset.at[self.bar.name, f'{symbol}_quantity'] = shares
            self.commission = self.commission + self.cost_per_trade
            index = index + 1


        self.model_dataset.at[self.date, 'commission'] = self.commission

        self._message(f"Total commission: {self.commission}")
        self._message(f"Total purchase cost: {total_purchase_cost}")

        if total_purchase_cost > self.cash:

            print(f"Backtester: Total purchase cost of {total_purchase_cost} exceeds cash of {self.cash}, bugging out")



    def add_positions_to_trade_ledger(self):

        self.trades = []

        for symbol in self.symbols:

            quantity = self.model_dataset.loc[self.date][f'{symbol}_quantity']

            price = self.model_dataset.loc[self.date][f'{symbol}_open']

            trade = Trade({

                "date": self.date,
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "order_date": self.previous_bar.name

            })

            self.trades.append(trade)

    def _close_positions_on_trade_ledger(self):

        index = 0

        self._message("*** Closing Trades ***")

        self._message(f"Cash on Hand Before Closing: {self.cash}")

        for trade in self.trades:

            price = self.model_dataset.loc[self.date][f"{trade.symbol}_open"]

            closed_profit = trade.close_trade({
                "date": self.date,
                "price": price
            })

            self._message(
                f"Selling {trade.quantity} shares of {trade.symbol} for {price} with profit of {closed_profit}")

            self.cash = self.cash - trade.close_trade_fee

            self.fees = trade.close_trade_fee + self.fees

            self.trade_ledger = self.trade_ledger.append(trade.__dict__, ignore_index=True)

            index = index + 1

        self.trades = []


    def open_position(self, position_signal):

        if position_signal == 1:
            self._message(f'Position signal is to go long')
        elif position_signal == -1:
            self._message(f'Position signal is to go short')

        self.set_quantities(position_signal)
        self.add_positions_to_trade_ledger()
        self.set_cash_after_opening_positions()
        self.set_margin()
        self._calculate_equity()
        self.model_dataset.at[self.date, 'position'] = position_signal


    def _update_positions(self):

        for symbol in self.symbols:

            self.model_dataset.at[self.date, f'{symbol}_quantity'] = self.previous_bar[f'{symbol}_quantity']

        self.cash = self.model_dataset.at[self.date, 'cash'] = self.previous_bar['cash']
        self.margin = self.model_dataset.at[self.date, 'margin'] = self.previous_bar['margin']

        self._calculate_equity()
        self.position = self.model_dataset.at[self.date, 'position'] = self.previous_bar['position']

        self._message(f"Updated position: {self.position}")

    def set_margin(self):

        self.margin = self.previous_bar['margin']

        for symbol in self.symbols:

            quantity = self.model_dataset.loc[self.date][f'{symbol}_quantity']

            if quantity < 0:

                self.margin = (abs(quantity) * self.bar[f'{symbol}_open']) + self.margin


        self.model_dataset.at[self.date, 'margin'] = self.margin

        self._message(f"Margin account at {self.margin}")


    def _calculate_equity(self):

        bar = self.model_dataset.loc[self.bar.name]

        self.stock_equity = 0

        self.short_position_value = 0

        for symbol in self.symbols:

            symbol_value = bar[f'{symbol}_quantity'] * bar[f'{symbol}_close']

            self._message(f"Stock: {symbol} value is {symbol_value}")

            if symbol_value > 0:

                self.stock_equity = self.stock_equity + symbol_value

            elif symbol_value < 0:

                self.short_position_value = self.short_position_value + abs(symbol_value)

        self.equity = self.model_dataset.at[self.bar.name, 'equity'] = bar['cash'] + self.stock_equity + self.margin + (self.margin - self.short_position_value)

        self._message(f"Stock Equity: {self.stock_equity}")
        self._message(f"Cash: {self.cash}")
        self._message(f"Short Position Value: {self.short_position_value}")
        self._message(f"Margin Value: {self.margin}")
        self._message(f"Total Equity Calculated: {self.equity}")

    def set_cash_after_opening_positions(self):

        self._message(f'Cash is : {self.cash}')

        self.cash = self.previous_bar['cash']
        self.model_dataset.at[self.date, 'cash'] = self.cash

        self._message(f'Setting cash to previous cash of {self.cash}')

        for symbol in self.symbols:
            price = self.model_dataset.loc[self.bar.name][f'{symbol}_open']
            quantity = self.model_dataset.loc[self.bar.name][f'{symbol}_quantity']
            cost = abs(price * quantity)
            self.cash = self.cash - cost

            self._message(f'Buying {symbol}, {quantity} shares at {price}')

        self.cash = self.cash - self.commission

        self._message(f'Cash: {self.cash} after removing {self.commission} commission')

        self.model_dataset.at[self.bar.name, 'cash'] = self.cash

        self._message(f"Cash calculated to be {self.cash}")



    def _set_cash_after_closing_positions(self):

        self._message("Setting cash after closing position")
        self._message(f"Cash before closing positions: {self.cash}")

        self.commission = 0

        # Here we move the margin account to cash
        self._message(f"Moving {self.previous_bar['margin']} from margin account to cash")
        self.cash = self.cash + self.previous_bar['margin']
        self.model_dataset.at[self.date, 'cash'] = self.cash
        self._message(f"Cash after moving margin account: {self.cash}")


        self._message("Moving cash from equities to cash account")

        amount_owed_from_short_sale = 0
        amount_back_from_liquidating_long = 0

        for symbol in self.symbols:

            # Here we calculate the cash value of each equity and add the value to cash. Note that if the qty is
            # negative then we subtract the value from cash as we have already moved the margin account to cash prior

            open_price = self.bar[f'{symbol}_open']
            quantity = self.previous_bar[f'{symbol}_quantity']
            cost = open_price * quantity
            self._message(f"{symbol} open price: {open_price}, Quantity: {quantity}, Cost: {cost}")

            # If the quantity is negative then it is a short sale and therefore we add the cost
            if quantity < 0:

                amount_owed_from_short_sale = amount_owed_from_short_sale - cost
                self._message(f"Amount owed from short sale: {amount_owed_from_short_sale}")

            else:

                amount_back_from_liquidating_long = amount_back_from_liquidating_long + cost
                self._message(f"Amount back from liquidating long: {amount_back_from_liquidating_long}")

            # Here we add the commission
            self.commission = self.commission + self.cost_per_trade
            self.model_dataset.at[self.date, 'broker_fees'] = self.commission

            # Here we set the quantities to zero
            self.model_dataset.at[self.date, f'{symbol}_quantity'] = 0

        self.cash = self.cash + amount_back_from_liquidating_long + (self.margin - amount_owed_from_short_sale)

        # Set the margin account to zero
        self.model_dataset.at[self.date, 'margin'] = 0
        self.margin = 0


        self.cash = self.cash - self.commission
        self.model_dataset.at[self.date, 'cash'] = self.cash

        self._message(f"Cash after closing positions: {self.cash}")


    def _close_positions(self):

        # Note that when we close a position we change the qty to 0 on the day we close the position

        self._close_positions_on_trade_ledger()

        self._set_cash_after_closing_positions()

        self.margin = 0
        self.model_dataset.at[self.date, 'margin'] = 0


        self._calculate_equity()
        self.model_dataset.at[self.date, 'position'] = 0
        self.position = 0

    def _execute_on_signal(self):

        self._message("Executing on signal")

        if self.previous_bar is None:
            return


        # Here we deal with the stop loss
        if self._stop_loss_triggered:

            if self.position == 0:

                self._update_positions()

            else:

                self._close_positions()
                self.broker.execute_orders(self.date)

        else:

            date = self.bar.name

            self.position = self.previous_bar['position']

            self._message(f"Previous position: {self.position}")

            self._message(f"Previous bar position signal is {self.previous_bar['position_signal']}")

            if self.position == 0:

                self._message("Not currently in a position, evaluating signal to open position")

                if self.previous_bar['position_signal'] != 0:

                    self._message("***** Position signal is not flat, opening position *****")

                    self.open_position(self.previous_bar['position_signal'])

                    self.broker.execute_orders(self.date)

                else:

                    self._update_positions()

                    self._message("Position signal is flat, remaining flat")

            elif self.position == 1 or self.position == -1:

                self._message("Currently in a position, evaluating signal to close position")

                if self.previous_bar['position_signal'] == 1 and self.position == -1:

                    self._message(f"We are in short position and the signal is telling us to go flat. Closing short position")

                    self._close_positions()

                    self.broker.execute_orders(self.date)

                elif self.previous_bar['position_signal'] == -1 and self.position == 1:

                    self._message("We are in long position and the signal is telling us to go flat, Closing long position")

                    self._close_positions()

                    self.broker.execute_orders(self.date)

                else:

                    self._update_positions()

                    self._message("Position signal is flat, staying in positions")

            else:

                sys.exit("Incorrect position signal. Check for error")

        self.bar.at['equity'] = self.equity


    def _place_orders(self):

        self._message("Placing Orders from signal")

        self.bar = self.model_dataset.loc[self.date]

        position_signal = self.bar.position_signal

        if self.previous_bar is None:
            return

        # Here we deal with the stop loss
        if self._stop_loss_triggered:

            self._message("Stop loss triggered")

            if self.position == 0:

                self._message("We are in neutral position, no orders to place")

            else:

                self._message("Closing all positions")

                self.broker.place_orders_to_close_all_positions(self.date)

        else:

            self._message(f"Previous position: {self.position}")

            self._message(f"Previous bar position signal is {position_signal}")

            if self.position == 0:

                self._message("Not currently in a position, evaluating signal to open position")

                if position_signal != 0:

                    self._message("***** Position signal is not flat, opening position *****")

                    self.broker.place_orders_for_market_open(
                        position_signal=position_signal,
                        order_date=self.date,
                        total_cash_available=self.cash
                    )

                else:

                    self._message("Position signal is flat, remaining flat")

            elif self.position == 1 or self.position == -1:

                self._message("Currently in a position, evaluating signal to close position")

                if position_signal == 1 and self.position == -1:

                    self._message(
                        f"We are in short position and the signal is telling us to go flat. Closing short position")

                    self.broker.place_orders_to_close_all_positions(self.date)


                elif position_signal == -1 and self.position == 1:

                    self._message(
                        "We are in long position and the signal is telling us to go flat, Closing long position")

                    self.broker.place_orders_to_close_all_positions(self.date)


                else:

                    self._message("Position signal is flat, staying in positions")

            else:

                sys.exit("Incorrect position signal. Check for error")



    def perform_hypothesis_test(self):

        number_of_days = len(self.model_dataset[self.start_date:self.os_end_date])

        self.sharpe_ratio_critical_value = self.daily_sharpe_ratio * (number_of_days ** .5)

        self.sharpe_ratio_p_value = scipy.stats.norm.sf(abs(self.sharpe_ratio_critical_value))


    def _complete_backtest_2_calculations(self):

        first_trade_date = self.model_dataset.index[0]
        self.first_trade_date_string = first_trade_date.strftime("%Y-%m-%d")

        self.model_dataset['daily_return'] = self.model_dataset['equity'].pct_change()

        # self.trading_df = self.model_dataset[self.start_date:self.os_end_date].copy()

        self.model_dataset = self.model_dataset[self.start_date:self.os_end_date]

        self.daily_returns_mean = self.model_dataset['daily_return'].mean()

        self.daily_returns_std = self.model_dataset['daily_return'].std()

        if self.daily_returns_std == 0:
            self.sharpe_ratio = 0
            self.daily_sharpe_ratio = 0

        else:
            self.daily_sharpe_ratio = self.daily_returns_mean / self.daily_returns_std
            self.sharpe_ratio = self.daily_sharpe_ratio * (252 ** 0.5)


        self.perform_hypothesis_test()

        self.max_drawdown = self.get_max_drawdown_backtester_2()

        self.cumulative_return = (self.model_dataset['equity'].iloc[-1] / self.model_dataset['equity'].iloc[0] - 1)

        self.profits = self.model_dataset['equity'].iloc[-1] - self.model_dataset['equity'].iloc[0]
        self.get_annualized_return()

        self.get_equity_regression_results()

        if self.max_drawdown == 0:
            self.return_over_max_drawdown = 0
        else:
            self.return_over_max_drawdown = self.cumulative_return / self.max_drawdown


        self.total_trades = len(self.trade_ledger)

        if len(self.trade_ledger) > 0:
            self.total_winning_trades = len(self.trade_ledger[self.trade_ledger['closed_profit'] > 0])
            self.total_losing_trades = len(self.trade_ledger[self.trade_ledger['closed_profit'] < 0])
            self.gross_profit = self.trade_ledger[self.trade_ledger['closed_profit'] > 0]['closed_profit'].sum()
            self.gross_loss = self.trade_ledger[self.trade_ledger['closed_profit'] < 0]['closed_profit'].sum()
            self.net_profit = self.gross_profit + self.gross_loss

        if self.total_winning_trades == 0:
            self.average_winning_trade = 0
        else:
            self.average_winning_trade = self.gross_profit / self.total_winning_trades

        if self.total_losing_trades == 0:
            self.average_losing_trade = 0
        else:
            self.average_losing_trade = self.gross_loss / self.total_losing_trades

        if self.total_trades == 0:
            self.percentage_winning_trades = 0
        else:
            self.percentage_winning_trades = self.total_winning_trades / self.total_trades

        self.model_dataset['sma'] = self.model_dataset[['synthetic_equity']].rolling(int(self.z_score_lookback)).mean()
        self.model_dataset['std'] = self.model_dataset[['synthetic_equity']].rolling(int(self.z_score_lookback)).std()
        self.model_dataset['upper_open'] = self.model_dataset['sma'] + (self.model_dataset['std'] * self.z_score_open_threshold)
        self.model_dataset['lower_open'] = self.model_dataset['sma'] - (self.model_dataset['std'] * self.z_score_open_threshold)
        self.model_dataset['upper_close'] = self.model_dataset['sma'] + (self.model_dataset['std'] * self.z_score_close_threshold)
        self.model_dataset['lower_close'] = self.model_dataset['sma'] - (self.model_dataset['std'] * self.z_score_close_threshold)

        self._calculate_open_profit()

    def _calculate_open_profit(self):

        df = self.trade_ledger

        if len(df) == 0:

            df['open_profit'] = 0

        else:

            df['open_profit'] = (df['close_price'] - df['open_price']) * df['quantity']


    def run_backtest_2(self, print_summary=True, verbose=False, max_allowable_drawdown=None, live_trading=False):

        self.live_trading = live_trading

        self.bar = None

        self.previous_bar = None

        self.verbose = verbose

        self.position = 0

        self.trading_day = 0



        self._message(f"Max Allowable DD in run_backtest_2: {max_allowable_drawdown}")

        if not self.live_trading:

            if max_allowable_drawdown is None:
                self._set_max_allowable_drawdown(self.start_date, self.is_end_date)
            else:
                self.max_allowable_drawdown = max_allowable_drawdown

                self._message(f"Confirm max allowable drawdown is: {self.max_allowable_drawdown}")

        for index in self.model_dataset[self.start_date:].index:

            self._message(20*" - ")

            self.bar = self.model_dataset.loc[index]
            self.date = self.bar.name

            self._message(f"Date: {self.date}, Index: {index}, Previous Position: {self.position}")

            if self.trading_day == 0:

                self._message(f"Cash is set at original {self.cash}")

                self.model_dataset.at[self.date, 'cash'] = self.cash

                self.model_dataset.at[self.date, 'equity'] = self.cash

            self._execute_on_signal()

            # Here we get the position signal
            # If we are not in a position
            self._message("Getting Position Signal")
            self._get_position_signal()
            self._set_stop_loss_signal()
            self._place_orders()

            # The last thing we do is set the previous bar to this bar
            self.previous_bar = self.model_dataset.loc[index]

            self.position = self.model_dataset.loc[index]['position']
            self.cash = self.model_dataset.loc[index]['cash']

            self.trading_day = self.trading_day + 1

        self._get_open_trades_df()

        self._complete_backtest_2_calculations()

        if print_summary:
            self._print_backtest2_summary()



    def _set_stop_loss_signal(self):

        # If the stop loss has been triggered then we keep it triggered the rest of the iteration

        if self._stop_loss_triggered:
            return

        if self.trading_day == 0:
            return

        if self.date < datetime.datetime.strptime(self.is_end_date, '%Y-%m-%d'):
            return

        # First we determine if we want a drawdown filter

        if self._drawdown_filter_multiplier == None:

            self._stop_loss_triggered = False

        else:
            # Here we get the max daily drawdown of the in sample period

            if not self.live_trading:
                self._set_max_allowable_drawdown(self.start_date, self.is_end_date)

            today_equity = self.bar['equity']
            yesterdays_equity = self.previous_bar['equity']

            drawdown = (today_equity - yesterdays_equity) / yesterdays_equity

            if drawdown < self.max_allowable_drawdown:

                print(f"Stop Loss Triggered: {self.date}, Drawdown: {drawdown}, Max Allowable: {self.max_allowable_drawdown}")

                self._stop_loss_triggered = True

            else:

                self._stop_loss_triggered = False




    def _get_drawdown(self, start_date, end_date):
        window = 2
        min_periods = 1
        df = self.model_dataset
        roll_max = df['equity'][start_date:end_date].rolling(window, min_periods).max()
        daily_drawdown = df['equity'][start_date:end_date] / roll_max - 1
        max_daily_drawdown = daily_drawdown.rolling(window, min_periods).min()
        return max_daily_drawdown.min()  # Maximum in sample drawdown

    def _set_max_allowable_drawdown(self, start_date, end_date):

        self._message("Setting max allowable drawdown")

        self._message(f"drawdown_filter_multiplier is set to: {self._drawdown_filter_multiplier}")

        if self._drawdown_filter_multiplier == None:
            self.max_allowable_drawdown = -100000
        else:
            drawdown = self._get_drawdown(start_date, end_date)
            self.max_allowable_drawdown = self._drawdown_filter_multiplier * drawdown
        self._message(f"Max allowable drawdown: {self.max_allowable_drawdown}")


    def _get_results(self, start_date, end_date):

        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        trading_days = (end - start).days

        cumulative_return = (self.model_dataset[start_date:end_date]['equity'].iloc[-1] / self.model_dataset[start_date:end_date]['equity'].iloc[0] - 1)

        annualized_return = ((1 + cumulative_return)**(365/trading_days)) - 1

        start_equity = self.model_dataset[start_date:end_date]['equity'][0]
        end_equity = self.model_dataset[start_date:end_date]['equity'][-1]

        profit = end_equity - start_equity

        max_drawdown = self._get_max_drawdown(start_date, end_date)

        reg_results = self._get_equity_regression_results(self._in_sample_start_date, self._in_sample_end_date)

        return_ove_max_dd = cumulative_return / max_drawdown

        daily_returns_mean = self.model_dataset[start_date:end_date]['equity'].pct_change().mean()
        daily_returns_std = self.model_dataset[start_date:end_date]['equity'].pct_change().std()

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



    def _print_backtest2_summary(self):

        print("____________________________ Summary _________________________________")

        print(f"Symbols: {self.symbols}")
        print(f"Vectors: {self.initial_vectors}")

        print("")

        data = [["Annualized Return", self.annualized_return, "Cumulative Return", self.cumulative_return],
                ["Net Profit", self.net_profit,],
                ['Gross Profit', self.gross_profit, 'Gross Loss', self.gross_loss],
                ['Reward Risk Ratio', self.reward_risk_ratio, 'Equity Smoothness', self.equity_smoothness],
                ['Return Over Max Drawdown', self.return_over_max_drawdown],

                ]
        print(tabulate(data, headers=["Item", "value", "Item", "Value"]))


        print("")
        print("____________________________ Hypothesis Test _________________________________")
        print("")

        data = [
            ['Daily Sharpe Ratio', self.daily_sharpe_ratio, 'Sharpe Ratio', self.sharpe_ratio],
            ['Number of Days', len(self.model_dataset), 'Sharpe Ratio Critical Value', self.sharpe_ratio_critical_value],
            ['Sharpe Ratio P Value:', self.sharpe_ratio_p_value],
        ]
        print(tabulate(data, headers=["Item", "value", "Item", "Value"]))

        print("****** Summary *******")
        print(f"Annualized Return: {round(self.annualized_return * 100, 2)}%")
        print(f"Max Drawdown: {round(self.max_drawdown * 100, 1)}%")
        print(f"Total Trading Days: {self.total_days}")
        print(f"Start Equity: {round(self.model_dataset['equity'][0], 2)}")
        print(f"End Equity: {round(self.model_dataset['equity'][-1], 2)}")
        print(f"Profits: {round(self.profits)}")
        print(f"Cumulative Return: {round(self.cumulative_return * 100, 2)}%")
        print(f"Fees: {round(self.fees, 2)}")
        print(f"Total Trades: {len(self.trade_ledger)}")
        print("Open Trades")

        for trade in self.trades:

            print(f"{trade.quantity} shares of {trade.symbol} purchased on {trade.open_date} at {round(trade.open_price, 2)}")


        print("Open Orders: ")
        if self.order is None:
            print("No open orders")
        else:

            print("************************************")
            print("********** ORDER NOTICE ************")

            print(f"{self.order['order_string']}")

        self.model_dataset[['equity']].plot(figsize=(15, 10))
        self.model_dataset[['z_score']].plot(figsize=(15, 10))
        self.model_dataset[['synthetic_equity']].plot(figsize=(15, 10))

        plt.show()










    def run_backtest(self):

        day = 0

        self.trading_day = 0
        df = self.data.copy()

        # End trading_df on end date and start on start date
        self.vector_df = pd.DataFrame(columns=self.data.columns)
        self.vector_df.drop(columns=['day'], inplace=True)
        self.kalman_vector_df = pd.DataFrame(columns=self.data.columns)
        self.kalman_vector_df.drop(columns=['day'], inplace=True)

        # Here we enter the daily loop
        last_trading_day = self.data.iloc[-1].day


        while self.trading_day < int(last_trading_day):

            self.bar = self.data[self.data['day'] == self.trading_day]

            self.date = self.bar.index

            # print(day)

            self._message("**************** New Date *****************")
            self._message(f"Date: {self.date}")
            self._message("Execute Orders before day starts")
            self.execute_order(date=self.date)





            # if self.use_kalman_filter:
            #     # Here we check to see if the kalman half life is less than the min z score lookback
            #     if day <= self.z_score_lookback or self.kalman_half_life < self.z_score_lookback:
            #         self.kalman_half_life = int(self.z_score_lookback)
            #     first_sample_day = self.trading_day - self.kalman_half_life
            #
            # else:
            #     first_sample_day = self.trading_day - int(self.z_score_lookback)


            first_sample_day = self.trading_day - int(self.vector_lookback)

            # Here we get the sample lookback dataset and calculate teh Johansen Test
            self.lookback_sample = self.data[
                (self.data['day'] >= first_sample_day) & (self.data['day'] <= self.trading_day)]

            self.lookback_sample_prices = self.price_data[
                (self.price_data['day'] >= first_sample_day) & (self.price_data['day'] <= self.trading_day)]

            self.prices = self.lookback_sample_prices.iloc[-1][self.symbols]

            self.jt_sample = self.lookback_sample[self.symbols]

            self.jt_sample = self.jt_sample.dropna()

            self.jt = vecm.coint_johansen(self.jt_sample, det_order=0, k_ar_diff=self.k_ar_diff)


            # self.vectors = np.array(self.initial_vectors)
            # self.vectors = self.jt.evec[0]

            # Here I use the Kalman Filter to calculate the vectors

            if self.use_kalman_filter:

                kf = KalmanFilter(transition_matrices=[1],
                                  observation_matrices=[1],
                                  initial_state_mean=0,
                                  initial_state_covariance=1,
                                  observation_covariance=1,
                                  transition_covariance=.0001)

                kalman_vectors = []

                self.vector_df.loc[len(self.vector_df)] = self.jt.evec[0]

                for symbol in self.symbols:
                    vectors = self.vector_df[symbol].values
                    mean, cov = kf.filter(vectors)
                    kalman_vectors.append(mean[-1][0])

                self.vectors = np.array(kalman_vectors)
                self.kalman_vector_df.loc[len(self.kalman_vector_df)] = kalman_vectors

            # End using Kalman filter


            else:

                self.vectors = np.array(self.initial_vectors)


            self.jt_sample['synthetic_equity'] = (self.jt_sample[self.symbols] * self.vectors).sum(axis=1)


            self.jt_sample['z_score'] = (self.jt_sample['synthetic_equity'] - self.jt_sample[
                'synthetic_equity'].mean()) / self.jt_sample['synthetic_equity'].std()




            if len(self.jt_sample['synthetic_equity']) < 25:
                H = .5
            else:
                H, c, data = compute_Hc(self.jt_sample['synthetic_equity'].values)


            self.hurst_exponent = H

            # self.hurst_exponent = self.get_hurst_exponent(self.jt_sample['synthetic_equity'].values)

            if self.use_returns:

                self.totals = self.data.loc[self.date][self.symbols].values * self.vectors
                absolute_totals = [abs(x) for x in self.totals]
                self.synthetic_equity = self.totals.sum()

            else:

                self.totals = self.prices * self.vectors
                absolute_totals = [abs(x) for x in self.totals]
                self.synthetic_equity = self.totals.sum()



            # Here we come up with a vector multiplier in order to properly
            # allocate the exposure we want to the securities

            self.vector_multiplier = self.cash / sum(absolute_totals)
            multiplied_vectors = self.vectors * self.vector_multiplier

            self._message("******************  START  *****************************")
            self._message(f"Cash: {self.cash}")
            self._message(f"Raw Vectors: {self.vectors}")
            self._message(f"Prices: {self.prices}")
            self._message(f"Vector Multiplier: {self.vector_multiplier}")
            self._message(f"Multiplied Vectors (shares): {multiplied_vectors}")

            self._message(f"Synthetic Equity: {self.synthetic_equity}")

            self._message(f"Totals: {self.totals}")
            self._message(f"Absolute Totals: {absolute_totals}")

            if self.first_trade_date is None:
                self.first_trade_date = self.date


            # Here we are using a piece of the dataframe that includes today and the number of lookback days following
            # This is to avoid lookahead bias

            self.lookback_sample = self.jt_sample

            adf_result = adfuller(self.lookback_sample['synthetic_equity'], autolag='AIC')
            p_value = adf_result[1]
            self.half_life = self.get_half_life(self.lookback_sample)

            if(day == 0 or not self.use_kalman_filter):

                self.kalman_half_life = self.z_score_lookback

            else:

                half_life = self.trading_df['half_life'].values

                kf = KalmanFilter(transition_matrices=[1],
                                  observation_matrices=[1],
                                  initial_state_mean=0,
                                  initial_state_covariance=1,
                                  observation_covariance=1,
                                  transition_covariance=.0001)

                mean, cov = kf.filter(half_life)

                self.kalman_half_life = mean[-1][0]

            if self.use_half_life_for_z_score_lookback:

                if self.half_life <= 0:
                    final_z_score_lookback = self.z_score_lookback
                else:
                    final_z_score_lookback = int(self.half_life * self.half_life_multiplier)

            else:
                final_z_score_lookback = self.z_score_lookback



            if self.use_rolling_lookback_for_z_score:
                self.z_score = self.get_z_score(self.lookback_sample['synthetic_equity']).iloc[-1]

            else:
                self.z_score = self.get_z_score(self.lookback_sample['synthetic_equity'][-final_z_score_lookback:]).iloc[-1]


            self.manual_upper_synthetic_equity_threshold = 450
            self.manual_lower_synthetic_equity_threshold = 75



            # Determine if we should enter a position by seeing if the z_score is above or below the threshold


            if self.hurst_exponent < self.hurst_exponent_limit:
                self.allow_trades = True
            else:
                self.allow_trades = False

            if self.hurst_exponent > self.hurst_exponent_limit:
                self.exit_all_trades = True
            else:
                self.exit_all_trades = False

            if self.days_in_trade >= self.max_days_in_trade:
                self.exit_all_trades = True


            # if self.synthetic_equity > self.manual_upper_synthetic_equity_threshold or self.synthetic_equity < self.manual_lower_synthetic_equity_threshold:
            #     self.exit_all_trades = True
            #     self.allow_trades = False
            #     print("exiting all trades bc ouside threshpld")



            if self.market_position == "flat":

                self.days_in_trade = 0

                shares = multiplied_vectors
                shares = [int(round(num, 0)) for num in shares]

                # The code below is for using the fast z and bollinger points on the z score as entries


                # Here we would open a short position as the z_score is above the threshold
                if self.z_score > self.z_score_open_threshold and self.allow_trades:
                    shares = [-x for x in shares]

                    self.market_position = 'short'

                    # Here we place orders to be executed the next day
                    self.place_order(self.date, self.vectors, "open")


                # Here I open a long position as the z_score is below the low threshold
                elif self.z_score < -self.z_score_open_threshold and self.allow_trades:

                    self.market_position = 'long'

                    # Here we place orders to be executed the next day
                    self.place_order(self.date, self.vectors, "open")


                # Here we do not see a position to enter as the z_score is between entry bounds
                else:
                    shares = multiplied_vectors * 0


            # If we are long or shortexecute_order
            else:


                shares = self.trading_df.iloc[-1].shares


                # Here we close the long trade if the z_score goes above the negative close threshold
                if (self.z_score > self.z_score_close_threshold and self.market_position == "long") or self.exit_all_trades:
                    self.market_position = "flat"
                    shares = 0

                    self.place_order(self.date, self.vectors, "close")

                # Here we close the short trade if the z score goes below the upper close threshold
                if (self.z_score < -self.z_score_close_threshold and self.market_position == "short") or self.exit_all_trades:
                    # Since we are shorting here we actually have to buy that stock at this price

                    self.market_position = "flat"
                    shares = 0

                    self.place_order(self.date, self.vectors, "close")

                self.days_in_trade = self.days_in_trade + 1


            self.get_open_profit(self.prices)

            # self.equity = self.open_trade_value + self.cash

            self.get_equity()

            # self.equity = self.open_profits + self.cash + self.open_trade_value

            self.trading_df = self.trading_df.append({
                "date": self.date[0],
                "synthetic_equity": self.synthetic_equity,
                "p_value": p_value,
                "half_life": self.half_life,
                "z_score": self.z_score,
                "equity": self.equity,
                "shares": shares,
                "cash": self.cash,
                "open_profits": self.open_profits,
                "market_position": self.market_position,
                "multiplied_vectors": multiplied_vectors,
                "prices": self.prices,
                "open_trade_value": self.open_trade_value,
                "fast_rolling_p_value": self.fast_rolling_p_value,
                "slow_rolling_p_value": self.slow_rolling_p_value,
                "kalman_half_life": self.kalman_half_life,
                "hurst_exponent": self.hurst_exponent,

            }, ignore_index=True)

            self.update_equity_df()

            self.trading_day = self.trading_day + 1


            day = day + 1

        self.trading_df.set_index('date', drop=True, inplace=True)

        first_trade_date = self.first_trade_date.strftime("%Y-%m-%d")
        self.first_trade_date_string = first_trade_date
        self.trading_df['daily_return'] = self.trading_df['equity'].pct_change(1)
        self.daily_returns_mean = self.trading_df['daily_return'].mean()
        self.daily_returns_std = self.trading_df['daily_return'].std()

        if self.daily_returns_std == 0:
            self.sharpe_ratio = 0
        else:
            self.sharpe_ratio = self.daily_returns_mean / self.daily_returns_std
            self.sharpe_ratio = self.sharpe_ratio * (252**0.5)

        self.max_drawdown = self.get_max_drawdown()
        self.cumulative_return = (self.trading_df['equity'].iloc[-1] / self.trading_df['equity'].iloc[0] - 1)

        self.profits = self.trading_df['equity'].iloc[-1] - self.trading_df['equity'].iloc[0]
        self.get_annualized_return()

        self.get_equity_regression_results()

        if self.max_drawdown == 0:
            self.return_over_max_drawdown = 0
        else:
            self.return_over_max_drawdown = self.cumulative_return / self.max_drawdown

        self.total_trades = len(self.trade_ledger)


        if len(self.trade_ledger) > 0:

            self.total_winning_trades = len(self.trade_ledger[self.trade_ledger['closed_profit'] > 0])
            self.total_losing_trades = len(self.trade_ledger[self.trade_ledger['closed_profit'] < 0])
            self.gross_profit = self.trade_ledger[self.trade_ledger['closed_profit'] > 0]['closed_profit'].sum()
            self.gross_loss = self.trade_ledger[self.trade_ledger['closed_profit'] < 0]['closed_profit'].sum()
            self.net_profit = self.gross_profit + self.gross_loss




        if self.total_winning_trades == 0:
            self.average_winning_trade = 0
        else:
            self.average_winning_trade = self.gross_profit / self.total_winning_trades

        if self.total_losing_trades == 0:
            self.average_losing_trade = 0
        else:
            self.average_losing_trade = self.gross_loss / self.total_losing_trades

        if self.total_trades == 0:
            self.percentage_winning_trades = 0
        else:
            self.percentage_winning_trades = self.total_winning_trades / self.total_trades

        self.trading_df['sma'] = self.trading_df[['synthetic_equity']].rolling(self.z_score_lookback).mean()
        self.trading_df['std'] = self.trading_df[['synthetic_equity']].rolling(self.z_score_lookback).std()
        self.trading_df['upper_open'] = self.trading_df['sma'] + (self.trading_df['std'] * self.z_score_open_threshold)
        self.trading_df['lower_open'] = self.trading_df['sma'] - (self.trading_df['std'] * self.z_score_open_threshold)


    def update_equity_df(self):

        equity_dict = {}

        index = 0

        equity_dict['date'] = self.date[0]

        equity_dict['cash'] = self.cash

        for symbol in self.symbols:
            equity_dict[f"{symbol}_price"] = self.prices[f'{symbol}']
            equity_dict[f"{symbol}_qty"] = self.shares[index]
            index = index + 1

        self.equity_df = self.equity_df.append(equity_dict, ignore_index=True)

    def get_equity(self):


        self._message(f"Getting Equity on {self.date}")

        self._message(f"Equity: {self.equity}")

        if len(self.trades) == 0:
            self.open_trade_value = 0
            self.equity = self.cash
            self._message(f"No Open Trades, equity = cash = {self.cash}")

        else:

            index = 0

            trade_values = 0

            self._message("Getting Trade Values")

            for trade in self.trades:

                price = self.prices[index]

                quantity = trade.quantity

                if quantity >= 0:

                    trade_value = price * quantity

                else:

                    trade_value = (trade.open_price * -quantity) + ((trade.open_price - price) * -quantity)

                trade_values = trade_values + trade_value

                open_value = quantity * trade.open_price

                self._message(f"Trade: {trade.symbol}, Qty: {quantity}, Open Price: {trade.open_price}, Open Value: {open_value}, Price: {price}, Value: {trade_value}")


                index = index + 1

            self.equity = self.cash + trade_values

            self._message(f"Open Trades, equity = cash + trade values")

            self._message(f"Cash: {self.cash}")
            self._message(f"Trade Values: {trade_values}")
            self._message(f"Equity: {self.equity}")




    def execute_order(self, date):
        if self.order is None:
            return False
        else:

            vectors = self.order['vectors']
            order_date = self.order['date']

            if self.order['type'] == "open":
                self.open_trades(date, vectors, order_date)
                # print(f"Cash After Trade Openings: {self.cash}")
            elif self.order['type'] == "close":
                self.close_trades(date)

            self.order = None

    def place_order(self, date, vectors, type):

        self.index = 0

        self._message("*** Placing Orders ***")

        trade_value = 0

        cash_available_to_trade = self.cash * self.max_allowable_allocation

        absolute_vectors = [abs(x) for x in vectors]
        prices = self.open_data.loc[date][self.symbols]
        vector_prices = absolute_vectors * prices
        vector_prices_sum = vector_prices.sum()
        vector_multiplier = cash_available_to_trade / vector_prices_sum
        self.shares = vectors * vector_multiplier
        self.shares = [math.floor(share) for share in self.shares]
        absolute_shares = [abs(share) for share in self.shares]
        values = absolute_shares * prices
        total_trade_value = values.sum()

        order_trades = []

        if type == "close":
            order_string = "Close all trades"
        else:

            for symbol in self.symbols:

                price = self.open_data.loc[date][symbol]
                quantity = self.shares[self.index]  # Here I subtract 1 share to ensure we have enough cash to cover the cost

                order_trades.append({

                    "date": date,
                    "symbol": symbol,
                    "quantity": quantity,

                })

                self.index = self.index + 1

            order_string = " * Open Trades * " + "\n"

            for trade in order_trades:

                order_string = order_string + f"{trade['symbol']}: {trade['quantity']} @ tomorrow's open" + "\n"



        self.order = {
            "date": date,
            "vectors": vectors,
            "type": type, #This is either open to open trades or close to close trades,
            "order_string": order_string,
            "order_trades": order_trades
        }


    def open_trades(self, date, vectors, order_date):

        use_closing_prices_of_previous_day = False

        index = 0

        self._message(f"*** Opening Trades on {date} ***")

        trade_value = 0

        cash_available_to_trade = self.cash * self.max_allowable_allocation

        absolute_vectors = [abs(x) for x in vectors]

        if use_closing_prices_of_previous_day:
            # Here if we want to use closing values of the previous day vs opening prices today
            # previous_day = datetime.datetime.strptime(date, "%Y-%m-%d")
            # days = datetime.timedelta(1)
            # new_date = date - days
            # new_date = new_date.strftime("%Y-%m-%d")

            prices = self.data.loc[order_date][self.symbols]

        else:
            prices = self.open_data.loc[date][self.symbols]




        vector_prices = absolute_vectors * prices
        self.vector_prices = vector_prices
        vector_prices_sum = vector_prices.values.sum()
        self.vector_prices_sum = vector_prices_sum

        vector_multiplier = cash_available_to_trade / vector_prices_sum
        shares = vectors * vector_multiplier
        shares = [math.floor(share) for share in shares]
        absolute_shares = [abs(share) for share in shares]
        values = absolute_shares * prices
        total_trade_value = values.sum()

        self._message(f"Vector Prices: {vector_prices}")
        self._message(f"Vector Prices Sum: {vector_prices_sum}")
        self._message(f"Vector Multiplier: {vector_multiplier}")
        self._message(f"Shares: {shares}")
        self._message(f"Absolute Shares: {absolute_shares}")
        self._message(f"Values: {values}")

        self._message(f"Cash: {self.cash}")
        self._message(f"Total Trade Value: {total_trade_value}")

        for symbol in self.symbols:

            price = self.open_data.loc[date][symbol][0]

            quantity = shares[index] # Here I subtract 1 share to ensure we have enough cash to cover the cost

            purchase_cost = price * quantity

            self._message(f"Buying {quantity} shares of {symbol} at {price}")
            self._message(f"Total Cost: {purchase_cost}")

            self.trade_date = date
            self.trade_symbol = symbol
            self.trade_quantity = quantity
            self.trade_price = price
            self.trade_order_date = order_date

            trade = Trade({

                "date": date,
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "order_date": order_date

            })



            trade_value = trade_value + (self.open_data.loc[date][symbol])

            self.fees = trade.open_trade_fee + self.fees

            self.cash = self.cash - trade.open_trade_fee


            self._message(f"Cash: {self.cash}")

            if quantity < 0:
                self.cash = self.cash + purchase_cost
                self.open_trade_value = self.open_trade_value + (-shares[index] * price)
                # print(f"Cash Before Purchase: {self.cash}, Purchase Cost: {purchase_cost}")

            else:
                self.cash = self.cash - purchase_cost
                self.open_trade_value = self.open_trade_value + (shares[index] * price)
                # print(f"Cash Before Purchase: {self.cash}, Purchase Cost: {purchase_cost}")

            if self.cash < 0:
                # sys.exit("Cash cannot be less than zero")
                pass

            self._message(f"Cash: {self.cash}")
            self._message(f"Open Trade Value: {self.open_trade_value}")

            self.trades.append(trade)
            index = index + 1

        self.equity = self.cash + self.open_trade_value

        self._message(f"Equity: {self.equity}")


    def close_trades(self, date):


        index = 0

        self._message("*** Closing Trades ***")

        self._message(f"Cash on Hand Before Closing: {self.cash}")

        total_trade_value = 0

        for trade in self.trades:

            price = self.open_data.loc[date][trade.symbol][0]

            closed_profit = trade.close_trade({
                "date": date,
                "price": price
            })

            self._message(f"Selling {trade.quantity} shares of {trade.symbol} for {price} with profit of {closed_profit}")

            self.cash = self.cash - trade.close_trade_fee

            self.fees = trade.close_trade_fee + self.fees

            self.trade_ledger = self.trade_ledger.append(trade.__dict__, ignore_index=True)

            if trade.quantity > 0:

                added_cash = (trade.close_price * trade.quantity)
                self.cash = self.cash + added_cash

            else:

                added_cash = (trade.open_price * -trade.quantity) + closed_profit
                self.cash = self.cash + added_cash

            total_trade_value = total_trade_value + added_cash

            index = index + 1

        self._message(f"Total Trade Value: {total_trade_value}")

        self.open_trade_value = 0

        self.trades = []

    def _message(self, message):

        if self.verbose:
            print(message)


    def get_annualized_return(self):

        if self.trading_day == 0:
            self.annualized_return = 0
        else:
            self.annualized_return = ((1 + self.cumulative_return)**(365/self.trading_day)) - 1

        return self.annualized_return




    def get_max_drawdown_backtester_2(self):
        # We are going to use a trailing 252 trading day window
        window = 252

        # Calculate the max drawdown in the past window days for each day in the series.
        # Use min_periods=1 if you want to let the first 252 days data have an expanding window
        Roll_Max = self.model_dataset.loc[self.start_date:self.os_end_date]['equity'].rolling(window, min_periods=1).max()
        Daily_Drawdown = self.model_dataset.loc[self.start_date:self.os_end_date]['equity'] / Roll_Max - 1.0

        # Next we calculate the minimum (negative) daily drawdown in that window.
        # Again, use min_periods=1 if you want to allow the expanding window
        Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()

        return -min(Max_Daily_Drawdown)



    def get_max_drawdown(self):

        # We are going to use a trailing 252 trading day window
        window = 252

        # Calculate the max drawdown in the past window days for each day in the series.
        # Use min_periods=1 if you want to let the first 252 days data have an expanding window
        Roll_Max = self.model_dataset['equity'].rolling(window, min_periods=1).max()
        Daily_Drawdown = self.model_dataset['equity'] / Roll_Max - 1.0

        # Next we calculate the minimum (negative) daily drawdown in that window.
        # Again, use min_periods=1 if you want to allow the expanding window
        Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()

        return -min(Max_Daily_Drawdown)

    def print_summary(self):

        # self.total_trades = 0
        # self.total_winning_trades = 0
        # self.total_losing_trades = 0
        # self.gross_profit = 0
        # self.gross_loss = 0
        # self.net_profit = 0
        # self.average_winning_trade = 0
        # self.average_losing_trade = 0
        # self.percentage_winning_trades = 0




        data = [["Annualized Return", self.annualized_return, "Cumulative Return", self.cumulative_return],
                ["Net Profit", self.net_profit,],
                ['Gross Profit', self.gross_profit, 'Gross Loss', self.gross_loss],
                ['Reward Risk Ratio', self.reward_risk_ratio, 'Equity Smoothness', self.equity_smoothness],
                ['Return Over Max Drawdown', self.return_over_max_drawdown, 'Sharpe Ratio', self.sharpe_ratio],

                ]
        print(tabulate(data, headers=["Item", "value", "Item", "Value"]))

        print("****** Summary *******")
        print(f"Annualized Return: {round(self.annualized_return * 100, 2)}%")
        print(f"Max Drawdown: {round(self.max_drawdown * 100, 1)}%")
        print(f"Total Trading Days: {self.total_days}")
        print(f"Start Equity: {round(self.trading_df['equity'][0], 2)}")
        print(f"End Equity: {round(self.trading_df['equity'][-1], 2)}")
        print(f"Profits: {round(self.profits)}")
        print(f"Cumulative Return: {round(self.cumulative_return * 100, 2)}%")
        print(f"Fees: {round(self.fees, 2)}")
        print(f"Total Trades: {len(self.trade_ledger)}")
        print("Open Trades")
        for trade in self.trades:

            print(f"{trade.quantity} shares of {trade.symbol} purchased on {trade.open_date} at {round(trade.open_price, 2)} currently priced at {round(trade.close_price,2)}")

        print("Open Orders: ")
        if self.order is None:
            print("No open orders")
        else:

            print("************************************")
            print("********** ORDER NOTICE ************")

            print(f"{self.order['order_string']}")



        self.trading_df[['equity']].plot(figsize=(15, 10))
        self.trading_df[['hurst_exponent']].plot(figsize=(15, 10))
        self.trading_df[['p_value']].plot(figsize=(15, 10))
        self.trading_df[['z_score']].plot(figsize=(15, 10))
        self.trading_df[['lower_open', 'synthetic_equity', 'sma', 'upper_open']].plot(figsize=(15, 10))

        # self.trading_df[['slow_rolling_p_value', 'fast_rolling_p_value']].plot(figsize=(15, 10))

        plt.show()



    def get_equity_regression_results(self):

        self.model_dataset.insert(0, 'day', range(0, len(self.model_dataset)))

        y = self.model_dataset['equity']
        x = self.model_dataset['day']
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()

        if results.bse['day'] == 0:
            self.reward_risk_ratio = 0
        else:
            self.reward_risk_ratio = results.params['day'] / results.bse['day']

        self.equity_smoothness = math.sqrt(results.ssr / (results.nobs - 2))

        self.r_squared_adj = results.rsquared_adj




    def get_hurst_exponent(self, time_series, max_lag=20):
        """Returns the Hurst Exponent of the time series"""
        lags = range(2, max_lag)
        # variances of the lagged differences
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        # calculate the slope of the log plot -> the Hurst Exponent
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]


    def run_backtest_up_to_today(self, print_summary=False):

        self.force_download = True

        self.os_end_date = self._get_last_day_of_stock_data(self.symbols[0])

        # self.os_end_date = '2022-5-19'

        self.reset()

        self._message(f"Max ALlowable DD: {self.max_allowable_drawdown}")

        self.run_backtest_2(print_summary=print_summary)


    def live_trade(self, start_date, verbose=False, print_summary=True):

        self.force_download = True

        self.start_date = start_date

        self.is_end_date = start_date

        self.live_trade_start_date = start_date

        self.os_end_date = self._get_last_day_of_stock_data(self.symbols[0])

        self.reset()

        self._message(f"Max ALlowable DD: {self.max_allowable_drawdown}")

        self.run_backtest_2(print_summary=print_summary ,max_allowable_drawdown=self.max_allowable_drawdown, verbose=verbose, live_trading=True)

        self.print_live_trade_orders()


    def print_live_trade_orders(self):

        print("************** Live Trade Orders *****************")

        if len(self.trade_ledger) == 0:

            print("No live trade orders")

        else:
           display(self.trade_ledger[self.trade_ledger['open_date'] > self.live_trade_start_date])

        print("*************** Open Orders ******************")

        display(self.broker.orders)

        #
        # for trade in self.trades:
        #     print(
        #         f"{trade.quantity} shares of {trade.symbol} purchased on {trade.open_date} at {round(trade.open_price, 2)}")

    def _get_last_day_of_stock_data(self, symbol):

        fm = FileManager()

        print(f"Symbol: {symbol}, start_date: {self.start_date}, end_date: {None}")

        fm.get_data(symbol=symbol, start_date=self.start_date, end_date=None)

        last_datetime = fm.data.iloc[-1].name.strftime('%Y-%m-%d')

        print(f"Last_datetime: {last_datetime}")

        return last_datetime

    def _get_open_trades_df(self):

        self.open_trades_df = pd.DataFrame()

        for trade in self.trades:

            self.open_trades_df = self.open_trades_df.append(trade.__dict__, ignore_index=True)















