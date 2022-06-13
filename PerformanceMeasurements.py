class PerformanceMeasurements:

    def get_max_drawdown(self, series, window):
        # Calculate the max drawdown in the past window days for each day in the series.
        # Use min_periods=1 if you want to let the first 252 days data have an expanding window
        roll_max = series.rolling(window, min_periods=1).max()
        daily_drawdown = series / roll_max - 1.0

        # Next we calculate the minimum (negative) daily drawdown in that window.
        # Again, use min_periods=1 if you want to allow the expanding window
        max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

        return max_daily_drawdown.min()

