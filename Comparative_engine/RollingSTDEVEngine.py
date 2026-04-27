import pandas as pd
import math

class RollingSTDEVEngine:
    def __init__(
        self, 
        given_data : pd.DataFrame, 
        window_interval : int
    ) -> None:
        self._data : pd.DataFrame = given_data
        self._window_interval = window_interval

    @property
    def window_interval(
        self
    ) -> int:
        return self._window_interval
    
    @window_interval.setter
    def window_interval(
        self,
        value : int 
    ) -> None:
        self._window_interval = value

    def run_engine(
        self
    ) -> pd.Series:
        stddev = self._data["Log Returns"].rolling(window=self._window_interval).std().dropna()
        stddev_dates = self._data.index
        stddev_treatment = pd.Series(stddev, index=stddev_dates, name="STDDEV").dropna()
        return stddev_treatment