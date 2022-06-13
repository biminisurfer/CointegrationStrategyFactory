from CointegrationLiveTester import CointegrationLiveTester

data = {

    'live_test_start_date': '2022-5-19',
    'filename': 'IndustrialsLowerThreshold',
    'max_backtests': 5,
    'max_allowable_allocation': 0.7,
    'initial_cash': 4000,
    'description': 'This is the first real live trade that we are starting on 2022-05-18'
}

lt = CointegrationLiveTester(data)

lt.run(print_backtest_summary=False)
lt.plot_combined_strategies()
lt.perform_t_test(alpha=0.05, verbose=True)
lt.perform_monte_carlo(total_trading_days=60, simulations=100, plot_results=True)
lt.print_summary()
