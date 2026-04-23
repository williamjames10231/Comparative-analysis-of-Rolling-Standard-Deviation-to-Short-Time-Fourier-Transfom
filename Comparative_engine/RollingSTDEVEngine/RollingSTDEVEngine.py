import pandas as pd
import math

class RollingSTDEVEngine:
    def __init__(self, given_data, window_size=100):
        self._data : pd.DataFrame = given_data
        self._window_size = window_size

    def run_engine(self):
        stddev = self._data["Log Returns"].rolling(window=self._window_size).std().dropna()
        stddev_dates = self._data.index
        stddev_treatment = pd.Series(stddev, index=stddev_dates, name="STDDEV").dropna()
        return stddev_treatment