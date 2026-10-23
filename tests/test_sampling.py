import numpy as np
import pandas as pd
import pytest

from deepgar.sampling import (
    create_1st_step_y,
    create_windowed_1st_step,
    created_naive_forecast,
    df_split,
    import_data,
    invert_normalize,
    normalize_data,
    sample_creation,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Random quarterly-indexed frame with the target in the first column."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(size=(20, 3)),
        columns=["GDP", "A", "B"],
        index=[f"20{y:02d}Q1" for y in range(20)],
    )


def test_df_split_is_chronological(frame: pd.DataFrame) -> None:
    """The split keeps time order and honours the train fraction."""
    train, test = df_split(frame, 0.75)
    assert len(train) == 15
    assert len(test) == 5
    assert train.index[-1] < test.index[0]


def test_normalize_roundtrip(frame: pd.DataFrame) -> None:
    """Normalising then inverting the target recovers the original values."""
    train, test = df_split(frame, 0.75)
    train_n, test_n, scaler = normalize_data(train, test)
    np.testing.assert_allclose(train_n.mean(), 0, atol=1e-12)
    np.testing.assert_allclose(invert_normalize(scaler, test_n["GDP"]), test["GDP"])


def test_windowing_targets_next_step() -> None:
    """Each window is paired with the target value that follows it."""
    data = np.arange(10, dtype=float).reshape(5, 2)
    x, y = create_windowed_1st_step(data, window_size=2, prediction_step=1, target_index=0)
    assert x.shape == (3, 2, 2)
    np.testing.assert_array_equal(y.squeeze(), [4.0, 6.0, 8.0])


def test_naive_forecast_is_last_observed_value() -> None:
    """The naive forecast is the last target value in each window."""
    data = np.arange(10, dtype=float).reshape(5, 2)
    naive = created_naive_forecast(data, window_size=2)
    np.testing.assert_array_equal(naive.squeeze(), [2.0, 4.0, 6.0])


def test_1st_step_y_repeats_target_per_quantile() -> None:
    """The target column is repeated once per quantile."""
    y = np.array([[1.0], [2.0]])
    y_traink, y_testk = create_1st_step_y([0.05, 0.5, 0.95], y, y)
    assert y_traink.shape == (2, 3)
    np.testing.assert_array_equal(y_testk[:, 2], [1.0, 2.0])


def test_sample_creation_shapes(frame: pd.DataFrame) -> None:
    """The modelling sample has the expected array shapes and batch size."""
    config = {"QUANTILES": [0.05, 0.25, 0.75, 0.95], "SPLIT": 0.75, "WINDOW_SIZE": 1}
    data, _ = sample_creation(frame, config)
    assert data["X_TRAIN"].shape == (14, 1, 3)
    assert data["Y_TRAINK"].shape == (14, 4)
    assert data["BATCH_SIZE"] == 13


def test_import_data_missing_file() -> None:
    """Loading a non-existent dataset raises instead of returning an empty list."""
    with pytest.raises(FileNotFoundError):
        import_data("does_not_exist.csv")
