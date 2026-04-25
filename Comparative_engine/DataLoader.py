import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(
        self
    )-> None:
        self._ticker = "PSEI.PS"
        self._data = None
        self._start_date = "2017-01-01"
        self._end_date = "2019-12-31"

    def load_data(
        self
    ) -> None:
        self._data = yf.download(
            self._ticker,
            self._start_date,
            self._end_date,
            interval="1d"
        )["Close"].dropna()

        self._data["Log Returns"] = np.log(self._data / self._data.shift(1))
        return self._data
    
    def visualize_data_log_returns(
        self
    ) -> None:
        self._data["Log Returns"].plot()
        plt.title("Log Returns of PSEI")

        plt.show()

    def visualize_data_raw_returns(
        self
    ) -> None:
        self._data["PSEI.PS"].plot()
        plt.title("Raw closing Prices of PSEI")
        plt.show()