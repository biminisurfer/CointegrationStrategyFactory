class Trade:

    def __init__(self, data):

        self.order_date = data['order_date']
        self.open_date = data['date']
        self.symbol = data['symbol']
        self.quantity = data['quantity']
        self.open_price = self._is_valid_price(data['price'])

        self.close_price = None
        self.is_open = True
        self.close_date = None
        self.open_profit = 0
        self.closed_profit = 0
        self.fees = True
        self.open_trade_fee = 0
        self.close_trade_fee = 0
        self.charge_fee("open")



    def _is_valid_qty(self, quantity):

        if quantity < 0:
            raise ValueError("Quantity must be positive.")
        return quantity

    def _is_valid_price(self, price):

        if price < 0:
            raise ValueError("Price must be positive.")
        return price

    def update_trade(self, price):

        self.close_price = price

        if self.quantity >= 0:
            self.open_profit = (self.close_price - self.open_price) * self.quantity
        else:
            self.open_profit = -(self.close_price - self.open_price) * self.quantity

        return self.open_profit

    def close_trade(self, data):

        self.close_date = data['date']
        self.close_price = data['price']
        self.is_open = False
        self.open_profit = 0

        if self.quantity > 0:
            self.closed_profit = (self.close_price - self.open_price) * self.quantity
            #
            # print("QUantity More than Zero")
            # print(f"Close Price: {self.close_price}, Open Price: {self.open_price}, Quantity: {self.quantity}")
            # print(f"Closed Profit: {self.closed_profit}")

        else:
            self.closed_profit = (self.open_price - self.close_price) * -self.quantity
            # print("Quantity Less than Zero")
            # print(f"Close Price: {self.close_price}, Open Price: {self.open_price}, Quantity: {self.quantity}")
            # print(f"Closed Profit: {self.closed_profit}")

        self.charge_fee("close")
        return self.closed_profit

    def charge_fee(self, trade_type):

        if self.fees:
            cost_per_share = 0.0035
            minimum_order_cost = 0.35
            maximum_order_cost_percent = 0.01

        else:

            cost_per_share = 0
            minimum_order_cost = 0
            maximum_order_cost_percent = 0

        fee = 0

        fee = abs(self.quantity) * cost_per_share

        if fee < minimum_order_cost:
            fee = minimum_order_cost

        if trade_type == "open":

            max_price = abs(self.quantity) * self.open_price

            if fee > max_price:
                fee = max_price

            self.open_trade_fee = fee

        if trade_type == "close":
            self.close_trade_fee = fee

            max_price = abs(self.quantity) * self.close_price

            if fee > max_price:
                fee = max_price

            self.close_trade_fee = fee


