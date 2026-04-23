import yfinance as yf
import numpy as np

class DataLoader:
    def __init__(self):
        self._ticker = "PSEI.PS"
        self._data = None
        self._start_date = "2017-01-01"
        self._end_date = "2019-12-31"

    def load_data(self):
        self._data = yf.download(
            self._ticker,
            self._start_date,
            self._end_date,
            interval="1d"
        )["Close"].dropna()

        self._data["Log Returns"] = np.log(self._data / self._data.shift(1))
        return self._data