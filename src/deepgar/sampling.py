"""Sample preparation: loading, splitting, normalising and windowing the data."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from deepgar.paths import PROCESSED_DATA_DIR

Scaler = dict[str, pd.Series]


def created_naive_forecast(
    dataset: ArrayLike, window_size: int, prediction_step: int = 1, target_index: int = 0
) -> NDArray[Any]:
    """Create a naive (last observed value) forecast."""
    dataset = np.array(dataset)
    y = []
    for i in range(0, len(dataset) - window_size - prediction_step + 1):
        end = i + window_size
        if len(dataset.shape) > 1:
            _y = dataset[end - 1 : end + prediction_step - 1, target_index]
        else:
            _y = dataset[end - 1 : end + prediction_step - 1]
        y.append(_y)
    return np.array(y)


def import_data(name: str) -> pd.DataFrame:
    """Load a csv file from the processed data folder by file name."""
    path = PROCESSED_DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    print(path)
    return pd.read_csv(path, index_col=0)


def df_split(df: pd.DataFrame, split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset chronologically; ``split`` is the train fraction."""
    split_time = int(len(df) * split)
    train_df = df[0:split_time]
    test_df = df[split_time:]
    return train_df, test_df


def normalize_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, Scaler]:
    """Standardise both sets using the train set mean and standard deviation."""
    train_mean = train_df.mean()
    train_std = train_df.std()

    scaler = {"mean": train_mean, "std": train_std}

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, test_df, scaler


def create_windowed_1st_step(
    dataset: ArrayLike, window_size: int, prediction_step: int, target_index: int
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Create a rolling window dataset."""
    dataset = np.array(dataset)
    x = []
    y = []
    for i in range(0, len(dataset) - window_size - prediction_step + 1):
        begin = i
        end = i + window_size
        _x = dataset[begin:end]
        if len(dataset.shape) > 1:
            _y = dataset[end : end + prediction_step, target_index]
        else:
            _y = dataset[end : end + prediction_step]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)


def create_1st_step_y(
    quantiles: Sequence[float], y_train: NDArray[Any], y_test: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Repeat the target column once per quantile."""
    y_traink = y_train[:,]
    for _ in range(len(quantiles) - 1):
        y_traink = np.concatenate((y_traink, y_train[:,]), axis=1)

    y_testk = y_test[:,]
    for _ in range(len(quantiles) - 1):
        y_testk = np.concatenate((y_testk, y_test[:,]), axis=1)
    return y_traink, y_testk


def sample_creation(
    df: pd.DataFrame, config: dict[str, Any], prediction_step: int = 1
) -> tuple[dict[str, Any], Scaler]:
    """Split, normalise and window the data into a modelling sample."""
    quantiles = config["QUANTILES"]
    split = config["SPLIT"]
    window_size = config["WINDOW_SIZE"]
    target_index = 0

    train_df, test_df = df_split(df, split)
    train_df, test_df, scaler = normalize_data(train_df, test_df)

    batch_size = train_df.shape[0] - 1 - window_size

    x_train, y_train = create_windowed_1st_step(
        train_df, window_size, prediction_step, target_index
    )
    x_test, y_test = create_windowed_1st_step(test_df, window_size, prediction_step, target_index)

    y_traink, y_testk = create_1st_step_y(quantiles, y_train, y_test)

    data = {
        "TRAIN_DF": train_df,
        "TEST_DF": test_df,
        "X_TRAIN": x_train,
        "Y_TRAIN": y_train,
        "X_TEST": x_test,
        "Y_TEST": y_test,
        "Y_TRAINK": y_traink,
        "Y_TESTK": y_testk,
        "BATCH_SIZE": batch_size,
    }

    return data, scaler
