
from FileManager import FileManager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from IPython.display import display
from Stocks import Stocks
from hurst import compute_Hc, random_walk




class ResultRegressionTesting:

    def __init__(self):

        self._stop_trading = False

        self._default_stock_list = {

            'utilities': ['AES', 'LNT', 'AEE', 'AEP', 'AWK', 'ATO', 'CNP', 'CMS', 'ED', 'D', 'DTE', 'DUK', 'EIX', 'ETR',
                          'EVRG', 'ES', 'EXC', 'FE', 'NEE', 'NI', 'NRG', 'PNW', 'PPL', 'PEG', 'SRE', 'SO', 'WEC',
                          'XEL'],
            'micro_energy': ['AE', 'ALTO', 'BKEP', 'DLNG', 'EGY', 'GPP', 'LEU', 'NC', 'REI', 'USDP'],
            'micro_metals_mining': ['GORO', 'MSB', 'SYNL', 'ZEUS'],
            'industrials': ['MMM', 'AOS', 'ALK' ,'ALLE' ,'AAL' ,'AME' ,'BA' ,'CHRW' ,'CAT' ,'CTAS' ,'CPRT' ,'CSX' ,'CMI' ,'DE',
                            'DAL' ,'DOV' ,'ETN' ,'EMR' ,'EFX' ,'EXPD' ,'FAST' ,'FDX' ,'FTV' ,'FBHS' ,'GD' ,'GE' ,'HON' ,'HWM'
                            ,'HII' ,'IEX',
                            'ITW' ,'JBHT' ,'J' ,'JCI' ,'LHX' ,'LDOS' ,'LMT' ,'MAS' ,'NLSN' ,'NSC' ,'NOC' ,'ODFL' ,'PH' ,'PNR',
                            'PWR' ,'RTX' ,'RSG' ,'RHI' ,'ROK' ,'ROL' ,'ROP' ,'SNA' ,'LUV' ,'SWK' ,'TXT' ,'TT' ,'TDG' ,'UNP'
                            ,'UAL' ,'UPS' ,'URI',
                            'VRSK' ,'GWW' ,'WAB' ,'WM' ,'XYL'],
            'energy': ['APA' ,'BKR' ,'CVX' ,'COP' ,'CTRA' ,'DVN' ,'FANG' ,'EOG' ,'XOM' ,'HAL' ,'HES' ,'MRO' ,'MPC' ,'OXY'
                       ,'OKE' ,'PSX',
                       'PXD' ,'SLB' ,'VLO' ,'WMB'],
            'communication_services': ['ATVI' ,'GOOG' ,'T' ,'CHTR' ,'CMCSA' ,'DISCA' ,'DISCK' ,'DISH' ,'EA' ,'FB' ,'IPG',
                                       'LYV' ,'LUMN' ,'MTCH' ,'NFLX' ,'NWSA' ,'NWS' ,'OMC' ,'TMUS' ,'TTWO' ,'DIS' ,'TWTR'
                                       ,'VZ'],
            'consumer_discretionary': ['AAP' ,'AMZN' ,'APTV' ,'AZO' ,'BBWI' ,'BBY' ,'BKNG' ,'BWA' ,'CZR' ,'KMX' ,'CCL' ,'CMG'
                                       ,'DHI' ,'DRI',
                                       'DG' ,'DLTR' ,'DPZ' ,'EBAY' ,'ETSY' ,'EXPE' ,'F' ,'GPS' ,'GRMN' ,'GM' ,'GPC' ,'HBI'
                                       ,'HAS' ,'HLT',
                                       'HD' ,'LVS' ,'LEG' ,'LEN' ,'LKQ' ,'LOW' ,'MAR' ,'MCD' ,'MGM' ,'MHK' ,'NWL' ,'NKE'
                                       ,'NCLH' ,'NVR' ,'ORLY',
                                       'PENN' ,'POOL' ,'PHM' ,'PVH' ,'RL' ,'ROST' ,'RCL' ,'SBUX' ,'TPR' ,'TGT' ,'TSLA' ,'TJX'
                                       ,'TSCO' ,'ULTA',
                                       'UAA' ,'UA' ,'VFC' ,'WHR' ,'WYNN' ,'YUM' ,],
            'consumer_staples': ['ADM' ,'MO' ,'CPB' ,'CHD' ,'CLX' ,'KO' ,'CL' ,'CAG' ,'STZ' ,'COST' ,'EL' ,'GIS' ,'HRL' ,'SJM'
                                 ,'K' ,'KMB' ,'KHC' ,'KR' ,'LW',
                                 'MKC' ,'TAP' ,'MDLZ' ,'MNST' ,'PEP' ,'PM' ,'PG' ,'SYY' ,'HSY' ,'TSN' ,'WBA' ,'WMT' ,],
            'real_estate': ['ARE' ,'AMT' ,'AVB' ,'BXP' ,'CBRE' ,'CCI' ,'DLR' ,'DRE' ,'EQIX' ,'EQR' ,'ESS' ,'EXR',
                            'FRT' ,'PEAK' ,'HST' ,'IRM' ,'KIM' ,'MAA' ,'PLD' ,'PSA' ,'O' ,'REG' ,'SBAC' ,'SPG' ,'UDR' ,'VTR',
                            'VNO' ,'WELL' ,'WY'],
            'materials': ['APD' ,'ALB' ,'AMCR' ,'AVY' ,'BLL' ,'CF' ,'DD' ,'EMN' ,'FMC' ,'FCX' ,'IFF' ,'IP',
                          'LIN' ,'LYB' ,'MLM' ,'NEM' ,'NUE' ,'PKG' ,'PPG' ,'SEE' ,'SHW' ,'MOS' ,'VMC' ,'WRK'],
            'commodities': ["CORN", "SOYB", "CPER", "GLD", "WEAT", "SLV"],
            'financials': ['AFL' ,'ALL' ,'AXP' ,'AIG' ,'AMP' ,'AON' ,'AJG' ,'AIZ' ,'BAC' ,'BLK' ,'BK' ,'BRO' ,'COF' ,'CBOE'
                           ,'SCHW' ,'CB',
                           'CINF' ,'C' ,'CFG' ,'CME' ,'CMA' ,'DFS' ,'RE' ,'FITB' ,'FRC' ,'BEN' ,'GL' ,'GS' ,'HBAN' ,'ICE'
                           ,'IVZ' ,'JPM',
                           'KEY' ,'LNC' ,'L' ,'MTB' ,'MKTX' ,'MMC' ,'MET' ,'MCO' ,'MS' ,'MSCI' ,'NDAQ' ,'NTRS' ,'PBCT' ,'PNC'
                           ,'PFG' ,'PGR' ,'PRU' ,'RJF',
                           'RF' ,'SPGI' ,'STT' ,'SIVB' ,'SYF' ,'TROW' ,'HIG' ,'TRV' ,'TFC' ,'USB' ,'WRB' ,'WFC' ,'ZION' ,],
            'barge_companies': ['AEP', 'NBR', 'TDW', 'TRGP', 'CNX', 'KEX', 'GBX', 'MPC', 'MATX'],
            'test': ['AAL' ,'ALK' ,'ALLE' ,'AOS' ,'DAL' ,'DE' ,'DOV' ,'ETN' ,'FAST' ,'FTV' ,'GE' ,'IEX' ,'JCI' ,'LDOS' ,'LUV'
                     ,'NOC' ,'PCAR' ,'SWK' ,'UAL' ,'URI' ,'WAB'],
            'food': ['CF', 'MOS', 'ADM'],
            'global_developed_markets': ['VEA', 'IEFA', 'EFA', 'BNDX', 'VCIT', 'VCSH', 'IXUS',
                                         'SCHF', 'VT', 'IGSB', 'ACWI', 'EFV', 'MINT', 'GDX', 'SPDW',
                                         'IGIB', 'FLOT', 'SRLN', 'EFG', 'SMH', 'GUNR', 'VSS', 'FNDF'],


        }

    def _check_if_within_allowable_max_drawdown(self, equity_log_returns_drawdown, max_allowable_drawdown):

        '''
        This method determines if we are within the allowable max drawdown
        :param equity_log_returns_drawdown:
        :param max_allowable_drawdown:
        :return: Here we return a 1 if we are within allowable max drawdown and a zero if othersise
        '''

        if max_allowable_drawdown is None:
            return 1

        if equity_log_returns_drawdown < max_allowable_drawdown:
            return 0
        else:
            return 1


    def _get_bollinger_entry_signal_mean_reversion(self, bollinger_upper,
                                                   bollinger_lower,
                                                   mean,
                                                   adj_close,
                                                   abs_mad,
                                                   abs_mad_rolling,
                                                   abs_mad_threshold,
                                                   ):


        if np.isnan(abs_mad_rolling):

            return 0

        if abs_mad_threshold is not None:

            if abs_mad > abs_mad_threshold:

                return 0

        if abs_mad > abs_mad_rolling:

            return 0

        else:

            if adj_close > bollinger_upper:

                return -1

            elif adj_close < bollinger_lower:

                return 1

            else:

                return 0

    def _get_bollinger_exit_signal_mean_reversion(self, bollinger_upper, bollinger_lower, mean, adj_close, entry_signal):

        # This will return a zero if we want to exit, otherwise it will return a 1 to stay in the position

        if entry_signal == 1:
            # If the signal was to go long and the adj close is above the mean we exit

            if adj_close > mean:
                # If
                return 0
            else:
                return 1

        elif entry_signal == -1:

            if adj_close < mean:
                return 0
            else:
                return 1

        else:
            return 1

    def _get_bollinger_entry_signal_momentum(self, bollinger_upper, bollinger_lower, mean, adj_close):

        if adj_close > bollinger_upper:

            return 1

        elif adj_close < bollinger_lower:

            return -1

        else:

            return 0

    def _get_max_drawdown(self, series, window):

        # Calculate the max drawdown in the past window days for each day in the series.
        # Use min_periods=1 if you want to let the first 252 days data have an expanding window
        roll_max = series.rolling(window, min_periods=1).max()
        daily_drawdown = series / roll_max - 1.0

        # Next we calculate the minimum (negative) daily drawdown in that window.
        # Again, use min_periods=1 if you want to allow the expanding window
        max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

        return max_daily_drawdown.min()



    def _get_equity_regression_results(self, df):

        df.insert(0, 'day', range(0, len(df)))

        y = df['equity']
        x = df['day']
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
            'equity_smoothness': equity_smoothness,
            'reward_risk_ratio': reward_risk_ratio,
            'r_squared_adj': r_squared_adj
        }

    def _determine_if_trade_took_place(self, previous_signal, previous_two_signals):

        if previous_signal == previous_two_signals:

            return 0

        else:

            return 1

    def _adjust_equity_for_max_allowable_drawdown(self, row, last_equity):

        if len(last_equity) == 0:

            return row['equity']

        if row['continue_trading'] == 1:

            return row['equity']

        else:

            return last_equity[0]

    def run_iterations(self, input_data):

        start_date = input_data['start_date']
        end_date = input_data['end_date']

        stocks = input_data['stocks']

        bollinger_lookback_array = input_data['bollinger_lookback_array']
        bollinger_std_array = input_data['bollinger_std_array']
        normal_std_threshold_array = input_data['normal_std_threshold_array']
        plot_equity_log_returns = input_data['plot_equity_log_returns']
        print_header_info = input_data['print_header_info']
        print_results = input_data['print_results']
        mad_long_array = input_data['mad_long_array']
        mad_short_array = input_data['mad_short_array']
        mad_signal_rolling = input_data['mad_signal_rolling']
        strategy_type = input_data['strategy_type']
        starting_cash = input_data['starting_cash']
        abs_mad_threshold_array = input_data['abs_mad_threshold_array']
        abs_mad_rolling_input_array = input_data['abs_mad_rolling_input_array']
        max_allowable_drawdown = input_data['max_allowable_drawdown']

        fm = FileManager(ignore_errors=True)

        results = pd.DataFrame()

        iteration = 0

        df_array = []

        for stock in stocks:

            for bollinger_lookback in bollinger_lookback_array:

                for bollinger_std in bollinger_std_array:

                    for normal_std_threshold in normal_std_threshold_array:

                        for abs_mad_threshold in abs_mad_threshold_array:

                            for abs_mad_rolling_input in abs_mad_rolling_input_array:

                                for mad_short in mad_short_array:

                                    for mad_long in mad_long_array:

                                        if print_header_info:
                                            print(f"Iteration: {iteration}")
                                            print(f"Stock: {stock}")
                                            print(f"Bollinger Lookback: {bollinger_lookback}")
                                            print(f"Bollinger Std: {bollinger_std}")
                                            print(f"Std Normal Threshold: {normal_std_threshold}")
                                            print(f"Absolute MAD Threshold: {abs_mad_threshold}")
                                            print(f"Absolute MAD Rolling Input: {abs_mad_rolling_input}")
                                            print(f"MAD Short: {mad_short}")
                                            print(f"MAD Long: {mad_long}")
                                            print(f"Max Allowable Drawdown: {max_allowable_drawdown}")

                                        got_data = fm.get_data(symbol=stock, start_date=start_date, end_date=end_date)

                                        if not got_data:
                                            print("Going to continue")
                                            continue

                                        df = fm.data
                                        df['mean'] = df['Adj Close'].rolling(bollinger_lookback).mean()
                                        df['std'] = df['Adj Close'].rolling(bollinger_lookback).std()
                                        df['bollinger_upper'] = df['mean'] + (df['std'] * bollinger_std)
                                        df['bollinger_lower'] = df['mean'] - (df['std'] * bollinger_std)
                                        df['returns'] = df['Adj Close'].pct_change()
                                        df['log_returns'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
                                        df['cum_log_returns'] = df['log_returns'].cumsum()
                                        df['std_normalized'] = df['std'] / df['mean']
                                        df['std_normalized_rolling'] = df['std_normalized'].rolling(bollinger_lookback).mean()
                                        df['below_threshold'] = (df['std_normalized_rolling'] < normal_std_threshold).astype(int)
                                        df['mad_long_sma'] = df['Adj Close'].rolling(mad_long).mean()
                                        df['mad_short_sma'] = df['Adj Close'].rolling(mad_short).mean()
                                        df['mad'] = df['mad_short_sma'] - df['mad_long_sma']
                                        df['mad_signal'] = df['mad'].rolling(mad_signal_rolling).mean()
                                        df['abs_mad'] = df['mad'].abs()
                                        df['abs_mad_rolling'] = df['abs_mad'].rolling(abs_mad_rolling_input).mean()

                                        if strategy_type == 'mean_reversion':
                                            df['entry_signal'] = df.apply(lambda row: self._get_bollinger_entry_signal_mean_reversion(
                                                row['bollinger_upper'],
                                                row['mean'],
                                                row['bollinger_lower'],
                                                row['Adj Close'],
                                                row['abs_mad'],
                                                row['abs_mad_rolling'],
                                                abs_mad_threshold
                                            ), axis=1)



                                            # df['exit_signal'] = df.apply(lambda row: self._get_bollinger_exit_signal_mean_reversion(
                                            #     row['bollinger_upper'],
                                            #     row['mean'],
                                            #     row['bollinger_lower'],
                                            #     row['Adj Close'],
                                            #     row['entry_signal'],
                                            #
                                            # ), axis=1)
                                            #
                                        #
                                        # elif strategy_type == 'momentum':
                                        #     df['entry_signal'] = df.apply(lambda row: self._get_bollinger_entry_signal_momentum(
                                        #         row['bollinger_upper'],
                                        #         row['mean'],
                                        #         row['bollinger_lower'],
                                        #         row['Adj Close'],
                                        #     ), axis=1)

                                        else:
                                            print("No strategy selected")

                                        #                     df['signal'] = df['entry_signal'] * df['exit_signal']
                                        df['signal'] = df['entry_signal']

                                        df['trade_occured'] = df.apply(
                                            lambda row: self._determine_if_trade_took_place(row.shift(1)['signal'], row.shift(2)['signal']),
                                            axis=1)

                                        df['equity_log_returns_daily'] = (df['log_returns'] * df['signal'].shift(1))

                                        df['equity_log_returns'] = (df['log_returns'] * df['signal'].shift(1)).cumsum()

                                        '''
                                        The code gets a bit confusing here as I need the equity log returns to determine the drawdown
                                        of the strategy, however the drawdown will determine the future equity log returns. 
                                        Therefore I first calculate the pure equity log returns and then apply the stop loss
                                        parameters to the equity log returns to simulate stopping out if the drawdown exceeds max
                                        allowable
                                        '''

                                        df['equity_log_returns_drawdown'] = (df['equity_log_returns'] - df['equity_log_returns'].expanding().max()).cummin()

                                        df['continue_trading'] = df.apply(lambda row: self._check_if_within_allowable_max_drawdown(
                                            row['equity_log_returns_drawdown'],
                                            max_allowable_drawdown
                                        ), axis=1)

                                        df['equity'] = (np.exp(df['equity_log_returns']) * starting_cash)
                                        df['equity'] = df['equity'].fillna(starting_cash)

                                        '''
                                        Here we get the last equity that was allowed to trade based on the continue trading
                                        column in the dataframe
                                        '''
                                        last_equity = df[(df['continue_trading'] == 0) & df['continue_trading'].shift(1) == 1]['equity']

                                        '''
                                        Now here we adjust the equity to be equal to the last equity series
                                        '''
                                        df['equity'] = df.apply(
                                            lambda row: self._adjust_equity_for_max_allowable_drawdown(row, last_equity),
                                            axis=1)

                                        profit = df.iloc[-1]['equity'] - df.iloc[0]['equity']
                                        total_return = profit / starting_cash

                                        max_drawdown = self._get_max_drawdown(df['equity'], len(df['equity']))

                                        if max_drawdown == 0:
                                            return_over_max_drawdown = 0
                                        else:
                                            return_over_max_drawdown = total_return / -max_drawdown

                                        regression_results = self._get_equity_regression_results(df)

                                        std_normalized_mean = df.std_normalized.mean()

                                        annualized_return = (total_return + 1) ** (365 / len(df['equity'])) - 1

                                        total_trades = df['trade_occured'].sum()

                                        daily_log_return_std = df['equity_log_returns_daily'].std()

                                        # H, c, data = compute_Hc(df['Adj Close'].values)

                                        if df['continue_trading'].min() == 0:
                                            stop_loss_triggered = True
                                        else:
                                            stop_loss_triggered = False

                                        result = {

                                            'stock': stock,
                                            'bollinger_lookback': bollinger_lookback,
                                            'bollinger_std': bollinger_std,
                                            'total_return': total_return,
                                            'mad_long': mad_long,
                                            'mad_short': mad_short,
                                            'mad_signal_rolling': mad_signal_rolling,
                                            'abs_mad_threshold': abs_mad_threshold,
                                            'abs_mad_rolling_input': abs_mad_rolling_input,
                                            'annualized_return': annualized_return,
                                            'max_drawdown': max_drawdown,
                                            'return_over_max_drawdown': return_over_max_drawdown,
                                            'equity_smoothness': regression_results['equity_smoothness'],
                                            'reward_risk_ratio': regression_results['reward_risk_ratio'],
                                            'std_normalized_mean': std_normalized_mean,
                                            'total_trades': total_trades,
                                            'daily_log_return_std': daily_log_return_std,
                                            # 'hurst_exponent': H,
                                            'stop_loss_triggered': stop_loss_triggered,

                                        }

                                        if print_results:
                                            print(" **** Results ****")
                                            print(f"Total Return: {total_return}")
                                            print(f"Annualized Return: {annualized_return}")
                                            print(f"Max Drawdown: {max_drawdown}")
                                            print(f"Return Over Max Drawdown: {return_over_max_drawdown}")
                                            print(f"Equity Smoothness: {regression_results['equity_smoothness']}")
                                            print(f"Reward Risk Ratio: {regression_results['reward_risk_ratio']}")
                                            print(f"Standard Deviation Normalized Mean: {std_normalized_mean}")
                                            print(f"Total Trades: {total_trades}")
                                            print(f"Stop Loss Triggered: {stop_loss_triggered}")

                                        results = results.append(result, ignore_index=True)



                                        # df[['bollinger_upper', 'mean', 'bollinger_lower', 'Adj Close']].plot(figsize=(15,10))
                                        # df[['mad_short_sma', 'mad_long_sma', 'mad', 'mad_signal']].plot(figsize=(15,10))
                                        # df[['mad', 'mad_signal']].plot(figsize=(15,10))
                                        #
                                        # if plot_equity_log_returns:
                                        #     df[['equity_log_returns']].plot(figsize=(15, 10))

                                            # plt.show()

                                        #                 df[['std_normalized_rolling']].plot(figsize=(15,10))

                                        if plot_equity_log_returns:

                                            df[['equity']].plot(figsize=(15, 10))

                                            df[['equity_log_returns']].plot(figsize=(15, 10))

                                            df['abs_mad_rolling'] = df['abs_mad_rolling'].fillna(0)

                                            df[['abs_mad_rolling', 'abs_mad']].plot(figsize=(15, 10))

                                        plt.show()

                                        iteration = iteration + 1

                                        df_array.append(df)


        results.index.name = 'iteration'

        #     results.plot.scatter(x='bollinger_lookback', y='total_return')

        if print_results:
            display(results.sort_values('return_over_max_drawdown', ascending=False))

        return results, df_array




