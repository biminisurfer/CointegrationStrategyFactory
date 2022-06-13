import pandas as pd
from FileManager import FileManager
import math
import sys

class Broker:

    def __init__(self, data):

        # Set variables
        self._commission = data['commission']
        self.symbols = data['symbols']
        self.start_date = data['start_date']
        self.initial_vectors = data['initial_vectors']
        self._max_allowable_allocation = data['max_allowable_allocation']

        self._price_data = pd.DataFrame()

        self.orders = pd.DataFrame(columns={'order_date', 'symbol', 'shares', 'trade_index', 'executed'})
        self.trades = pd.DataFrame()
        self.verbose = False
        self.bar = None

        self._get_price_data()

    def _message(self, message):

        if self.verbose:

            print(message)

    def _get_price_data(self):

        fm = FileManager()

        for symbol in self.symbols:

            fm.get_data(symbol, self.start_date, None)

            self._price_data[f'{symbol}_open'] = fm.data['Open']
            self._price_data[f'{symbol}_close'] = fm.data['Close']

    def set_orders_as_executed(self):

        self.orders.drop(self.orders.index, inplace=True)

    def get_open_orders(self):

        return self.orders[self.orders['executed'] == 0]

    def get_executed_orders(self):

        return self.orders[self.orders['executed'] == 1]

    def get_stock_price(self, symbol, date):

        return {
            'open': self._price_data.loc[date][f'{symbol}_open'],
            'close': self._price_data.loc[date][f'{symbol}_close']
        }

    def execute_orders(self, execution_date):

        open_orders = self.get_open_orders()

        for item in open_orders.iterrows():

            order_index = item[0]

            order = item[1]

            price = self._price_data.loc[execution_date][f'{order.symbol}_open']

            if order.trade_index is None:

                self.trades = self.trades.append({

                    'order_date': order.order_date,
                    'trade_open_date': execution_date,
                    'symbol': order.symbol,
                    'shares': order.shares,
                    'open_price': price,
                    'trade_close_price': None,
                    'trade_close_date': None

                }, ignore_index=True)

            else:

                self.trades.at[order.trade_index, 'trade_close_price'] = price
                self.trades.at[order.trade_index, 'trade_close_date'] = execution_date

            self.trades['trade_open_date'] = pd.to_datetime(self.trades['trade_open_date'])

            self.orders.at[order_index, 'executed'] = 1

    def get_open_trades(self):

        if len(self.trades) == 0:

            return self.trades

        else:

            return self.trades[self.trades['trade_close_price'].isnull()]


    def get_closed_trades(self):

        return self.trades[self.trades['trade_close_price'].isnull() == False]

    def place_orders_to_close_all_positions(self, order_date):

        open_trades = self.get_open_trades()

        for row in open_trades.iterrows():

            trade_index = row[0]
            trade = row[1]

            self.orders = self.orders.append({

                'order_date': order_date,
                'symbol': trade.symbol,
                'shares': -trade.shares,
                'executed': 0,
                'trade_index': trade_index,

            }, ignore_index=True)

    def execute_orders_to_close_positions(self, date):

        open_orders = self.get_open_orders()

        for index, order in open_orders.iterrows():

            price = self.get_stock_price(order['symbol'], date)

            self.trades.at[order['trade_index'], 'trade_close_price'] = price['open']

    def place_orders_for_market_open(self, position_signal, order_date, total_cash_available):

        if position_signal == 0:
            sys.exit("The position signal must be +1 or -1 for this method. Use close all positions method to close positions")

        self._message(f"Cash on hand to open position: {total_cash_available}")

        self.bar = self._price_data.loc[order_date]

        index = 0
        total_abs_value = 0
        for symbol in self.symbols:
            vector = self.initial_vectors[index] * position_signal

            # Here we use the close price of the order date to quantify the number of shares to buy the following day
            # Note we do not yet know what the open price is
            close_price = self.bar[f"{symbol}_close"]

            value = vector * close_price
            abs_open_value = abs(value)
            total_abs_value = total_abs_value + abs_open_value
            self._message(f"Close price for {symbol} is {close_price}, Vector is : {vector}, Value is: {value}")
            index = index + 1

        self._message(f"Total abs open value: {total_abs_value}")

        self._total_cash_available = total_cash_available

        self._total_abs_value = total_abs_value

        vector_multiplier = (total_cash_available * self._max_allowable_allocation) / total_abs_value

        self._message(f"Vector multiplier: {vector_multiplier}")

        index = 0
        total_purchase_cost = 0

        for symbol in self.symbols:

            self._vector_multiplier = vector_multiplier
            self._position_signal = position_signal

            shares = math.floor(self.initial_vectors[index] * vector_multiplier) * position_signal

            # Here we set the algo to purchase at least one share so if the shares are zero we set to the sign
            # of the vector times the position signal

            if shares == 0:

                if self.initial_vectors[index] < 0:

                    shares = -1 * position_signal

                elif self.initial_vectors[index] > 0:

                    shares = position_signal

            open_price = self.bar[f"{symbol}_open"]
            purchase_cost = abs(shares * open_price)
            total_purchase_cost = total_purchase_cost + purchase_cost
            self._message(
                f"Purchase {shares} shares of {symbol} at open price of {open_price} for value of {purchase_cost}")

            self.orders = self.orders.append({

                'order_date': order_date,
                'symbol': symbol,
                'shares': shares,
                'executed': 0,
                'trade_index': None,

            }, ignore_index=True)


            # self.orders['order_date'] = pd.to_datetime(self.orders['order_date'])

            index = index + 1

        self._message(f"Total purchase cost: {total_purchase_cost}")

        if total_purchase_cost > total_cash_available:

            print(f"Broker Model: Total purchase cost of {total_purchase_cost} exceeds cash of {total_cash_available}")


















