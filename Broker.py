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

        self.orders = pd.DataFrame(columns={'order_date', 'symbol', 'shares'})
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

    def clear_orders(self):

        self.orders.drop(self.orders.index, inplace=True)

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

        vector_multiplier = (total_cash_available * self._max_allowable_allocation) / total_abs_value

        self._message(f"Vector multiplier: {vector_multiplier}")

        index = 0
        total_purchase_cost = 0

        for symbol in self.symbols:
            shares = math.floor(self.initial_vectors[index] * vector_multiplier) * position_signal
            open_price = self.bar[f"{symbol}_open"]
            purchase_cost = abs(shares * open_price)
            total_purchase_cost = total_purchase_cost + purchase_cost
            self._message(
                f"Purchase {shares} shares of {symbol} at open price of {open_price} for value of {purchase_cost}")

            self.orders = self.orders.append({

                'order_date': order_date,
                'symbol': symbol,
                'shares': shares,
                'commission': self._commission,

            }, ignore_index=True)

            self.orders['order_date'] = pd.to_datetime(self.orders['order_date'])

            index = index + 1

        self._message(f"Total purchase cost: {total_purchase_cost}")

        if total_purchase_cost > total_cash_available:
            sys.exit(f"Broker Model: Total purchase cost of {total_purchase_cost} exceeds cash of {total_cash_available}, bugging out")


















