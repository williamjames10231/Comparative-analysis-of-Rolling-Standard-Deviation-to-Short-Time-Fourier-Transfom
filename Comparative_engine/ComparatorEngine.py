import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, cohen_kappa_score

from Comparative_engine.RollingSTDEVEngine import RollingSTDEVEngine
from Comparative_engine.STFTEngine import STFTEngine
from Comparative_engine.DataLoader import DataLoader

class ComparatorEngine:
    def __init__(
        self,
        loader : DataLoader
    ) -> None:
        default_window = 2

        self._rolling_stdev_engine = RollingSTDEVEngine(
            loader,
            default_window
        )

        self._rolling_stft_engine = STFTEngine(
            loader,
            default_window
        )

    def trial_single_window(
        self,
        interval : int
    )-> dict[str, pd.DataFrame]:
        self._rolling_stdev_engine.window_interval = interval
        self._rolling_stft_engine.window_interval = interval

        treated_stdev = self._rolling_stdev_engine.run_engine()
        treated_stft = self._rolling_stft_engine.run_engine()
        
        aggregate : pd.DataFrame = pd.DataFrame({
            "STDEV": treated_stdev,
            "STFT": treated_stft
        }).dropna()

        thresholds : tuple[list[float], list[float]] = self.volatility_threshold_maker(aggregate)
        self.volatility_regime_classifier(aggregate, thresholds[0], thresholds[1])
        self.normalize_aggregate(aggregate)

        results_summary : dict[str, pd.DataFrame] = {
            "aggregate": aggregate,
            "thresholds": pd.DataFrame({
                "STDEV": thresholds[0],
                "STFT": thresholds[1]
            }, index=["Low", "High"])
        }

        return results_summary


    def volatility_threshold_maker(
        self,
        aggregate : pd.DataFrame
    ) -> tuple[list[float], list[float]]:
        stdev_threshold = [
            aggregate["STDEV"].quantile(0.3),
            aggregate["STDEV"].quantile(0.7)
        ]

        stft_threshold = [
            aggregate["STFT"].quantile(0.3),
            aggregate["STFT"].quantile(0.7)
        ]

        return stdev_threshold, stft_threshold
    
    def volatility_regime_classifier(
        self,
        aggregate : pd.DataFrame,
        stdev_threshold : list[float],
        stft_threshold : list[float]
    ) -> pd.Series:
        
        aggregate["STDEV_Regime"] = pd.cut(
            aggregate["STDEV"],
            bins=[-np.inf, stdev_threshold[0], stdev_threshold[1], np.inf],
            labels=["Low", "Medium", "High"]
        )

        aggregate["STFT_Regime"] = pd.cut(
            aggregate["STFT"],
            bins=[-np.inf, stft_threshold[0], stft_threshold[1], np.inf],
            labels=["Low", "Medium", "High"]
        )

        return aggregate
    
    def normalize_aggregate(
        self,
        aggregate : pd.Series
    ) -> pd.Series:
        aggregate["Rolling_STDEV_Normalized"] = self.normalize_series(aggregate["STDEV"])
        aggregate["STFT_Normalized"] = self.normalize_series(aggregate["STFT"])
        return aggregate

    def normalize_series(
        self,
        series : pd.Series
    ) -> pd.Series:
        normalized_series = (series - series.min()) / (series.max() - series.min())
        return normalized_series