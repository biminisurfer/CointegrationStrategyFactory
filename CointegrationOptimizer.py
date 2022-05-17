import pandas as pd
import matplotlib.pyplot as plt
from CointegrationBacktester import CointegrationBacktester
import itertools
from tabulate import tabulate
from IPython.display import display


class CointegrationOptimizer():

    def __init__(self, optimization_parameter_list, show_iteration_details=False):

        self._optimization_parameter_list = optimization_parameter_list
        self._optimization_input_array = []
        self._create_optimization_input_array()
        self._strategy_results = []
        self._backtest = CointegrationBacktester()
        self._equity_df = pd.DataFrame()
        self._show_iteration_details = show_iteration_details
        self._optimization_results = pd.DataFrame()
        self._y_axis_plots = ['reward_risk_ratio', 'annualized_return', 'return_over_max_drawdown', 'sharpe_ratio']
        self._x_axis_plots = []
        self._strategies = []

        self._combined_daily_returns_mean = 0
        self._combined_daily_returns_std = 0
        self._combined_sharpe_ratio = 0
        self._combined_daily_sharpe_ratio = 0

    def get_optimization_input_array(self):

        return self._optimization_input_array

    def _create_optimization_input_array(self):

        optimizer_inputs = []

        for parameter_name in self._optimization_parameter_list:

            input_array = []

            for parameter_values in self._optimization_parameter_list[parameter_name]:
                input_array.append(parameter_values)

            optimizer_inputs.append(input_array)

        input_combination = list(itertools.product(*optimizer_inputs))

        optimization_input_array = []

        for strategy_parameters in input_combination:

            index = 0

            input_dict = {}

            for key in self._optimization_parameter_list:
                input_dict[key] = strategy_parameters[index]

                index = index + 1

            optimization_input_array.append(input_dict)

        self._optimization_input_array = optimization_input_array

        print(f"***** Total Iterations Found: {len(self._optimization_input_array)} *****")

    def run_optimization(self, plot_results=True):

        iteration = 0

        self._backtest = CointegrationBacktester()

        for strategy_input in self._optimization_input_array:


            print(f"-------- Iteration: {iteration} ---------")

            if self._show_iteration_details:

                for input_name in strategy_input:
                    print(f"{input_name}: {strategy_input[input_name]}")

            self._backtest.configure(strategy_input, download_new_data=True)

            self._backtest.run_backtest_2(print_summary=plot_results, verbose=False)

            self._strategies.append(self._backtest)

            self._equity_df[iteration] = self._backtest.model_dataset['equity']

            iteration = iteration + 1

            print("")

            self._store_results()

            if self._show_iteration_details:
                self._print_results()

        self._equity_df['total'] = self._equity_df[self._equity_df.columns].sum(axis=1)

        self._perform_summary_calculations()

        if plot_results:
            self._plot_results()


    def _perform_summary_calculations(self):

        self._equity_df['daily_return'] = self._equity_df['total'].pct_change()

        self._combined_daily_returns_mean = self._equity_df['daily_return'].mean()
        self._combined_daily_returns_std = self._equity_df['daily_return'].std()

        if self._combined_daily_returns_std == 0:
            self._combined_sharpe_ratio = 0
            self._combined_daily_sharpe_ratio = 0

        else:
            self._daily_sharpe_ratio = self._combined_daily_returns_mean / self._combined_daily_returns_std
            self._combined_sharpe_ratio = self._combined_daily_sharpe_ratio * (252 ** 0.5)

        self._get_annualized_return()


    def _get_annualized_return(self):

        start = self._optimization_parameter_list['start_date'][0]
        end = self._optimization_parameter_list['os_end_date'][0]

        self._cumulative_return = (self._equity_df['total'].iloc[-1] / self._equity_df['total'].iloc[0] - 1)

        self._total_trading_days = len(self._equity_df[start:end])

        self._annualized_return = ((1 + self._cumulative_return) ** (365 / self._total_trading_days)) - 1

        return self._annualized_return

    def _plot_results(self):

        print("Equity Results")
        columns_excluding_total = self._equity_df.columns[0:-2]
        self._equity_df[columns_excluding_total].plot(figsize=(15, 10))
        plt.show()

        print("Combined Equity Curves")
        self._equity_df['total'].plot(figsize=(15, 10))

        plt.show()

        self._x_axis_plots = []

        for parameter in self._optimization_parameter_list:

            if parameter == 'stock_input':
                continue

            if len(self._optimization_parameter_list[parameter]) > 1:
                self._x_axis_plots.append(parameter)

                print(f"******** {parameter} *********")

                for y_axis in self._y_axis_plots:

                    self._optimization_results.plot.scatter(x=parameter, y=y_axis)
                    plt.show()


        self._optimization_results.style.set_table_styles([dict(selector="th", props=[('max-width', '10px')])])
        print(self._optimization_results[['reward_risk_ratio', 'annualized_return', 'max_drawdown', 'sharpe_ratio']+self._x_axis_plots])

        print("____________________________ Combined Strategies _________________________________")
        print("")

        data = [["Annualized Return", self._annualized_return, "Cumulative Return", self._cumulative_return],
                # ["Net Profit", self.net_profit, ],
                # ['Gross Profit', self.gross_profit, 'Gross Loss', self.gross_loss],
                # ['Reward Risk Ratio', self.reward_risk_ratio, 'Equity Smoothness', self.equity_smoothness],
                # ['Return Over Max Drawdown', self.return_over_max_drawdown],

                ]
        print(tabulate(data, headers=["Item", "value", "Item", "Value"]))

        # print("")
        # print("____________________________ Hypothesis Test _________________________________")
        # print("")
        #
        # data = [
        #     ['Daily Sharpe Ratio', self.daily_sharpe_ratio, 'Sharpe Ratio', self.sharpe_ratio],
        #     ['Number of Days', len(self.model_dataset), 'Sharpe Ratio Critical Value',
        #      self.sharpe_ratio_critical_value],
        #     ['Sharpe Ratio P Value:', self.sharpe_ratio_p_value],
        # ]
        # print(tabulate(data, headers=["Item", "value", "Item", "Value"]))
        #

    def _store_results(self):

        symbol_string = ''

        symbol_string = "-".join(self._backtest.symbols)

        self._optimization_results = self._optimization_results.append({

            "symbols": self._backtest.symbols,
            "symbol_string": symbol_string,
            "z_score_lookback": self._backtest.z_score_lookback,
            "z_score_open_threshold": self._backtest.z_score_open_threshold,
            "z_score_close_threshold": self._backtest.z_score_close_threshold,
            "annualized_return": self._backtest.annualized_return,
            "sharpe_ratio": self._backtest.sharpe_ratio,
            "max_drawdown": self._backtest.max_drawdown,
            "vector_lookback": self._backtest.vector_lookback,
            "k_ar_diff": self._backtest.k_ar_diff,
            "return_over_max_drawdown": self._backtest.return_over_max_drawdown,
            "equity_smoothness": self._backtest.equity_smoothness,
            "reward_risk_ratio": self._backtest.reward_risk_ratio,
            "r_squared_adj": self._backtest.r_squared_adj,
            "total_trades": self._backtest.total_trades,
            "hurst_exponent_limit": self._backtest.hurst_exponent_limit,
            "max_days_in_trade": self._backtest.max_days_in_trade,
            "profits": self._backtest.profits,
            "half_life_multiplier": self._backtest.half_life_multiplier,
            "fixed_average_price": self._backtest.fixed_average_price,
            "use_fixed_average_price": self._backtest.use_fixed_average_price,
            "sharpe_ratio_p_value": self._backtest.sharpe_ratio_p_value,
            "initial_vectors": self._backtest.initial_vectors,
            "use_kalman_filter": self._backtest.use_kalman_filter,

        }, ignore_index=True)


    def _print_results(self):

        display(self._optimization_results[[
            'sharpe_ratio',
            'annualized_return',
            'equity_smoothness',
            'max_drawdown',
            'return_over_max_drawdown',
            'reward_risk_ratio',
            'symbols',
            'total_trades',
            'vector_lookback',
            'z_score_close_threshold',
            'z_score_open_threshold',
            'z_score_lookback',
        ]].sort_values('reward_risk_ratio', ascending=False).head(20))


