import pandas as pd
import datetime


class Stocks:

    def __init__(self, after_date):

        self._after_date = datetime.datetime.strptime(after_date, '%Y-%m-%d')

        self.sp_500 = []

        self._get_sp500_stocks()

    def _get_sp500_stocks(self):

        if self._after_date is not None:

            self.sp_500 = pd.read_csv('sp_500_list.csv')

            self.sp_500['Date first added'] = pd.to_datetime(self.sp_500['Date first added'])

            self.sp_500 = self.sp_500[self.sp_500['Date first added'] < self._after_date]

        else:

            self.sp_500 = pd.read_csv('sp_500_list.csv')

            self.sp_500['Date first added'] = pd.to_datetime(self.sp_500['Date first added'])

