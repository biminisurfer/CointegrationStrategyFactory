import pickle
import pandas as pd
import matplotlib.pyplot as plt
from FileManager import FileManager

from IPython.display import display

'''
This class loads the CointegrationStrategyFactory and then uses the data to create a live test
'''

class CointegrationLiveTester():

    def __init__(self, data):

        self.live_test_start_date = data['live_test_start_date']
        self._filename = data['filename']
        self._max_backtests = data['max_backtests']

        self.is_end_date = None
        self.backtests = []
        self.data = pd.DataFrame()
        self.open_trades = pd.DataFrame()
        self.trade_ledger = pd.DataFrame()
        self.backtest_df = pd.DataFrame()
        self.symbols = []
        self.price_df = pd.DataFrame()

        self._load_sf_from_file()
        self._filter_non_stopped_out_strategies()
        # self._combine_strategies()


        self._get_backtest_df()
        self._get_symbols_from_backtests()
        self._create_price_df()




    def _load_sf_from_file(self):

        file_pi2 = open(f'savedStrategies/{self._filename}.obj', 'rb')
        self.sf = pickle.load(file_pi2)
        self.is_end_date = self.sf._in_sample_end_date

    def _filter_non_stopped_out_strategies(self):

        for bt in self.sf._best_iteration_backtests:

            if not bt._stop_loss_triggered:

                self.backtests.append(bt)

    def _combine_strategies(self):

        index = 0

        for bt in self.backtests:
            self.data[f'bt_{index}_equity'] = bt.model_dataset['equity']

            index = index + 1

        self.data['equity'] = self.data[self.data.columns].sum(axis=1)


    def plot_combined_strategies(self):

        self._combine_strategies()

        self.data[:self.is_end_date]['equity'].plot(figsize=(15,10))
        self.data[self.is_end_date:]['equity'].plot(figsize=(15,10))
        plt.show()

    def print_summaries(self):

        for bt in self.backtests:

            bt._print_backtest2_summary()

    def _get_backtest_df(self):

        for bt in self.backtests:

            self.backtest_df = self.backtest_df.append(bt.__dict__, ignore_index=True)


    def _get_symbols_from_backtests(self):

        live_test_symbols = []

        for symbol_array in self.backtest_df.symbols:

            for symbol in symbol_array:

                if symbol not in live_test_symbols:
                    live_test_symbols.append(symbol)

        self.symbols = live_test_symbols

    def _create_price_df(self):

        price_df = pd.DataFrame()

        fm = FileManager()

        for symbol in self.symbols:
            fm.get_data(symbol, self.live_test_start_date, None)

            price_df[f'{symbol}_open'] = fm.data['Open']
            price_df[f'{symbol}_close'] = fm.data['Close']

        self.price_df = price_df

    def _filter_backtests(self):

        self.backtest_df = self.backtest_df.sort_values('reward_risk_ratio', ascending=False).head(self._max_backtests)

        backtests = []

        for index, bt in self.backtest_df.iterrows():
            print(index)

            backtests.append(self.backtests[index])

        self.backtests = backtests

    def _get_open_trade_value(self):

        pass

    def run(self):

        self._filter_backtests()


        for bt in self.backtests:

            bt.run_backtest_up_to_today()

            bt.live_trade(self.live_test_start_date, verbose=True)

            self.open_trades = self.open_trades.append(bt.open_trades_df, ignore_index=True)

            if len(bt.trade_ledger) > 0:

                self.trade_ledger = self.trade_ledger.append(bt.trade_ledger[bt.trade_ledger['close_date'] >= self.live_test_start_date], ignore_index=True)

            # if not bt._stop_loss_triggered:
            #     self.backtest_df = self.backtest_df.append(bt.__dict__, ignore_index=True)


        print('\n\n\n')
        print('************************************** Live Closed Trades ****************************************************')

        display(self.trade_ledger)

        print('************************************** Live Open Trades ****************************************************')

        display(self.open_trades)



