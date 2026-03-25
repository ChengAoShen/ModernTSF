"""Base dataset class for forecasting tasks."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ForecastingDataset(ABC, Dataset):
    """Base class for forecasting datasets.

    Parameters
    ----------
    root_path : str
        Root directory for the dataset.
    data_path : str
        Dataset file name.
    size : tuple[int, int, int]
        Sequence length, label length, prediction length.
    flag : str, optional
        Split flag: "train", "val", or "test".
    features : str, optional
        Feature mode ("M", "S", "MS").
    target : str, optional
        Target column name.
    split_ratio : tuple[float, float, float], optional
        Train/val/test split ratios.
    scale : bool, optional
        Whether to scale features.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "S",
        target: str = "OT",
        split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
        scale: bool = True,
    ):
        super().__init__()
        self.file_path = os.path.join(root_path, data_path)
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.scale = scale
        self.scaler = None
        self.data, self.time_stamp = self._read_data(
            flag, features, target, split_ratio, scale
        )

    def __len__(self) -> int:
        """Return number of windows available in the split."""
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> Tuple:
        """Return one input/target window and optional timestamps.

        Parameters
        ----------
        index : int
            Window start index.

        Returns
        -------
        tuple
            (input_series, output_series, input_stamp, output_stamp)
        """
        input_start = index
        input_end = input_start + self.seq_len
        output_start = input_end - self.label_len
        output_end = input_end + self.pred_len

        input_series = self.data[input_start:input_end]
        output_series = self.data[output_start:output_end]

        if self.time_stamp is not None:
            input_stamp = self.time_stamp[input_start:input_end]
            output_stamp = self.time_stamp[output_start:output_end]
        else:
            input_stamp, output_stamp = None, None

        return input_series, output_series, input_stamp, output_stamp

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data using the fitted scaler.

        Parameters
        ----------
        data : np.ndarray
            Scaled data.

        Returns
        -------
        np.ndarray
            Unscaled data.
        """
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)

    def _build_time_stamp(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Generate time feature matrix from a date column.

        Parameters
        ----------
        df_raw : pandas.DataFrame
            Raw dataframe containing a "date" column.

        Returns
        -------
        np.ndarray
            Time feature matrix with year/month/day/weekday/hour/minute.
        """
        df_stamp = pd.DataFrame()
        df_stamp["date"] = pd.to_datetime(df_raw["date"])
        df_stamp["year"] = df_stamp.date.dt.year
        df_stamp["month"] = df_stamp.date.dt.month
        df_stamp["day"] = df_stamp.date.dt.day
        df_stamp["weekday"] = df_stamp.date.dt.weekday
        df_stamp["hour"] = df_stamp.date.dt.hour
        df_stamp["minute"] = df_stamp.date.dt.minute
        df_stamp = df_stamp.drop(["date"], axis=1).values
        return df_stamp

    def _get_borders(
        self,
        flag: str,
        split_ratio: tuple[float, float, float],
        num_samples: int,
    ) -> Tuple[int, int]:
        """Compute slice borders for the requested split.

        Parameters
        ----------
        flag : str
            Split flag: "train", "val", or "test".
        split_ratio : tuple[float, float, float]
            Train/val/test split ratios.
        num_samples : int
            Total number of samples in the dataset.

        Returns
        -------
        tuple[int, int]
            Start and end indices for the split.
        """
        flag_map = {"train": 0, "val": 1, "test": 2}
        idx = flag_map[flag]
        total_ratio = sum(split_ratio)
        cum_ratios = [
            sum(split_ratio[: i + 1]) / total_ratio for i in range(len(split_ratio))
        ]
        border1 = (
            int(cum_ratios[idx - 1] * num_samples) - self.seq_len if idx > 0 else 0
        )
        border2 = int(cum_ratios[idx] * num_samples)
        return border1, border2

    @abstractmethod
    def _read_data(
        self,
        flag: str,
        features: str,
        target: str,
        split_ratio: tuple[float, float, float],
        scale: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read data for a split and return series and timestamps.

        Parameters
        ----------
        flag : str
            Split flag: "train", "val", or "test".
        features : str
            Feature mode ("M", "S", "MS").
        target : str
            Target column name.
        split_ratio : tuple[float, float, float]
            Train/val/test split ratios.
        scale : bool
            Whether to scale features.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Series data and timestamp features.
        """
        raise NotImplementedError
