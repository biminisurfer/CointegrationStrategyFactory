from CointegrationStrategyFactory import CointegrationStrategyFactory

# micro_energy
# utilities
# energy
# consumer_discretionary
# industrials
# communication_services


data = {

    'z_score_lookback': [75],
    'in_sample_start_date': '2021-10-1',
    'in_sample_end_date': '2022-4-17',
    'out_sample_end_date': '2022-5-17',
    'number_of_stocks': 4,
    'starting_cash': 4000,
    'reward_risk_ratio': 50,
    'minimum_number_of_trades': 15,
    'sector': 'industrials',
    'hurst_exponent_max': .5,
    'p_value_max': 0.05,
    'jt_pass_criteria': 0.95,
    'drawdown_filter_multiplier': 1.15,
    'max_allowable_optimization_combos': 5000,
    'z_score_open_threshold': [1],
    'z_score_close_threshold': [0],

}

sf = CointegrationStrategyFactory(data)
sf.run()
sf.save('LiveTradeStrategy001')
sf.print_is_summary()
sf.print_os_summary()

sf._best_iterations.plot.scatter(y='reward_risk_ratio', x='z_score_lookback')
sf._best_iterations.plot.scatter(y='reward_risk_ratio', x='z_score_open_threshold')
sf._best_iterations.plot.scatter(y='reward_risk_ratio', x='z_score_close_threshold')
