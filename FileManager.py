import sys

import yfinance as yf
import pandas as pd
from os.path import exists
from datetime import datetime


class FileManager():

    def __init__(self, root_dir='stocks'):

        self._root_dir = root_dir
        self.data = None

    def _download_symbol_data(self, symbol):

        data = yf.download(symbol)

        data.to_csv(f'{self._root_dir}/{symbol}.csv')

    def get_data(self, symbol, start_date, end_date, force_download=False):

        if not self._check_file_exists(symbol) or force_download:

            self._download_symbol_data(symbol)

        self.data = pd.read_csv(f'{self._root_dir}/{symbol}.csv', index_col='Date', parse_dates=True)

        if len(self.data) == 0:

            print(f"File Manager Error: No data for {symbol}. Trying to download one more time")

            self._download_symbol_data(symbol)

            self.data = pd.read_csv(f'{self._root_dir}/{symbol}.csv', index_col='Date', parse_dates=True)

            if len(self.data) == 0:

                print(f"File Manager Error: Still no data for {symbol}")

        # If the date is not none then we use an end date, otherwise we bring the last date
        if end_date is not None:

            self._check_dates_exist(symbol, start_date, end_date)

            self.data = self.data[start_date:end_date]

        else:

            self.data = self.data[start_date:]


    def _check_file_exists(self, symbol):

        return exists(f'{self._root_dir}/{symbol}.csv')

    def _check_dates_exist(self, symbol, start, end):

        if len(self.data) == 0:
            sys.exit(f"{symbol} has no data")

        first_date_from_data = self.data.iloc[0].name
        last_date_from_data = self.data.iloc[-1].name

        start_datetime = datetime.strptime(start, '%Y-%m-%d')
        end_datetime = datetime.strptime(end, '%Y-%m-%d')

        errors = []

        if start_datetime < first_date_from_data:

            errors.append(f'The dataset does not go back far enough for this start time. Start Time: {start}, earliest date in the dataset is {first_date_from_data}')

        if end_datetime > last_date_from_data:

            print('The dataset ends before your end date, downloading file again')

            self._download_symbol_data(symbol)

            self.data = pd.read_csv(f'{self._root_dir}/{symbol}.csv', index_col='Date', parse_dates=True)

        if len(errors) > 0:

            sys.exit(errors)


