import numpy as np
import pandas as pd
from scipy.signal import spectrogram

class STFTEngine:
    def __init__(
        self, 
        given_data : pd.DataFrame, 
        window_interval : int
    ) -> None:
        self._fs = 1
        self._nperseg = window_interval
        self._noOverlap = self._nperseg - 1
        self._data = given_data

    @property
    def window_interval(
        self
    ) -> int:
        return self._nperseg
    
    @window_interval.setter
    def window_interval(
        self,
        value : int 
    ) -> None:
        self._nperseg = value
        self._noOverlap = self._nperseg - 1

    def run_engine(
        self
    ) -> pd.Series:
        f,t, Sxx = spectrogram(
           self._data["Log Returns"],
            fs=self._fs,
            window="hamming",
            nperseg=self._nperseg,
            noverlap=self._noOverlap,
            mode="magnitude"
        )

        stft_power = np.sum(Sxx**2, axis=0)
        end_indices = np.arange(len(stft_power)) + (self._nperseg - 1)

        end_dates = self._data.index[end_indices]

        stft_correct = pd.Series(stft_power, index=end_dates, name='stft_power')
        return stft_correct
