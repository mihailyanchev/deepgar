"""The experimental setup used in the paper (as in notebooks 02 and 04)."""

import contextlib
import io
from typing import Any

from deepgar.sampling import Scaler, import_data, sample_creation

COUNTRIES = ("Bulgaria", "Estonia", "Lithuania", "Romania")

DATA_FILES = {
    "Bulgaria": "data_Bulgaria_14.07.2022.csv",
    "Estonia": "data_Estonia_14.07.2022.csv",
    "Lithuania": "data_Lithuania_19.07.2022.csv",
    "Romania": "data_Romania_14.04.2022_13.00.19.csv",
}

# Predictors per country; GDP (the target) must come first.
PREDICTORS = {
    "Bulgaria": ["GDP", "SENTIMENT", "SOFIX", "US_BOND_YIELD"],
    "Estonia": ["GDP", "SENTIMENT", "OMXTGI_CLOSE"],
    "Lithuania": ["GDP", "SENTIMENT", "LT_BOND_YIELD"],
    "Romania": ["GDP", "SENTIMENT", "US_BOND_YIELD"],
}

QUANTILES = [0.05, 0.25, 0.75, 0.95]

MODEL_PARAM: dict[str, Any] = {
    "SPLIT": 0.75,
    "WINDOW_SIZE": 1,
    "1_COMMON_LSTM_NUM": 128,
    "2_COMMON_LSTM_NUM": 128,
    "1_IND_LSTM_NUM": 64,
    "1_IND_DENSE_NUM": 32,
    "1_IND_DENSE_ACTIVATION": "relu",
    "1_OUT_DENSE_ACTIVATION": None,
    "1_LEARNING_RATE": 0.001,
    "PRED_STEP": 1,
    "QUANTILES": QUANTILES,
}

DEEPGAR_EPOCHS = {"Bulgaria": 125, "Estonia": 95, "Lithuania": 95, "Romania": 125}
DEEPGAR_PATIENCE = 50
LINEAR_SIMULTANEOUS_EPOCHS = 1000
BENCHMARK_EPOCHS = 200  # AR(1) and the non-simultaneous quantile regressions
BENCHMARK_PATIENCE = 20


def load_sample(
    country: str, columns: list[str] | None = None
) -> tuple[dict[str, Any], Scaler, dict[str, Any]]:
    """Load a country's dataset and build the modelling sample.

    Returns the sample, the target scaler and the model config with
    ``NUM_FEATURES`` set. ``columns`` defaults to the country's predictors.
    """
    param = dict(MODEL_PARAM)
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = import_data(DATA_FILES[country])
    dataset = dataset[columns or PREDICTORS[country]]
    data, scaler = sample_creation(dataset, param, prediction_step=param["PRED_STEP"])
    param["NUM_FEATURES"] = data["TRAIN_DF"].shape[1]
    return data, scaler, param
