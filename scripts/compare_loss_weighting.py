"""Compare DeepGaR trained with equal loss weights against the paper's dynamic scheme.

The published results were produced with an equally weighted sum of the two
losses because of a bug (see ``deepgar.models.deepgar_skew``). This script
retrains the model for every country with both schemes over several seeds and
evaluates them as in ``notebooks/02_deepgar_skew.ipynb``.

Usage:
    uv run python scripts/compare_loss_weighting.py [--seeds 1234 1 2 3 4]
"""

import argparse
from typing import Any

import numpy as np
import pandas as pd
import properscoring as ps
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from deepgar.models.deepgar_skew import model_deepgar_skew
from deepgar.models.weighting import EQUAL_WEIGHTING, PAPER_DYNAMIC_WEIGHTING, LossWeighting
from deepgar.paths import TABLES_DIR
from deepgar.sampling import Scaler, import_data, sample_creation

# Model configuration, data files, predictors and epochs as in notebook 02.
PARAM: dict[str, Any] = {
    "SPLIT": 0.75,
    "WINDOW_SIZE": 1,
    "1_COMMON_LSTM_NUM": 128,
    "1_IND_LSTM_NUM": 64,
    "1_IND_DENSE_NUM": 32,
    "1_IND_DENSE_ACTIVATION": "relu",
    "1_OUT_DENSE_ACTIVATION": None,
    "1_LEARNING_RATE": 0.001,
    "PRED_STEP": 1,
    "QUANTILES": [0.05, 0.25, 0.75, 0.95],
}
DATA_FILES = {
    "Bulgaria": "data_Bulgaria_14.07.2022.csv",
    "Estonia": "data_Estonia_14.07.2022.csv",
    "Lithuania": "data_Lithuania_19.07.2022.csv",
    "Romania": "data_Romania_14.04.2022_13.00.19.csv",
}
PREDICTORS = {
    "Bulgaria": ["GDP", "SENTIMENT", "SOFIX", "US_BOND_YIELD"],
    "Estonia": ["GDP", "SENTIMENT", "OMXTGI_CLOSE"],
    "Lithuania": ["GDP", "SENTIMENT", "LT_BOND_YIELD"],
    "Romania": ["GDP", "SENTIMENT", "US_BOND_YIELD"],
}
EPOCHS = {"Bulgaria": 125, "Estonia": 95, "Lithuania": 95, "Romania": 125}
PATIENCE = 50
N_SAMPLES = 200

SCHEMES = {"equal (published)": EQUAL_WEIGHTING, "dynamic 95/5 (paper)": PAPER_DYNAMIC_WEIGHTING}


def _percent(x: Any, scaler: Scaler) -> Any:
    """Undo the normalisation of the target and express it in percent."""
    return (np.asarray(x) * scaler["std"].iloc[0] + scaler["mean"].iloc[0]) * 100


def run(country: str, scheme: str, weighting: LossWeighting, seed: int) -> dict[str, Any]:
    keras.utils.set_random_seed(seed)
    param = dict(PARAM)
    dataset = import_data(DATA_FILES[country])[PREDICTORS[country]]
    data, scaler = sample_creation(dataset, param, prediction_step=param["PRED_STEP"])
    param["NUM_FEATURES"] = data["TRAIN_DF"].shape[1]

    _, _, history, model = model_deepgar_skew(
        param, data, EPOCHS[country], PATIENCE, weighting=weighting, seed=seed, verbose=0
    )
    epochs_trained = len(history.history["loss"])
    switch_epoch = int(EPOCHS[country] * weighting.switch_at) + 2  # first epoch on final weights
    switched = weighting.final is not None and epochs_trained >= switch_epoch

    m, s, a, t = (np.squeeze(v).astype(np.float32) for v in model([data["X_TEST"], data["Y_TEST"]]))
    real_y = _percent(data["Y_TEST"], scaler).squeeze()

    # Point forecast and interval from the predicted distribution (notebook 02, cells 22-23).
    dist = tfp.distributions.SinhArcsinh(loc=m, scale=s, skewness=a, tailweight=t)
    median = _percent(dist.quantile(0.5), scaler)
    lower = _percent(dist.quantile(0.05), scaler)
    upper = _percent(dist.quantile(0.95), scaler)

    # CRPS of the predictive distribution in percent. The original notebooks
    # omitted the *100 on the scale (see README); that value is kept for reference.
    std = np.float32(scaler["std"].iloc[0])
    loc_pct = _percent(m, scaler).astype(np.float32)

    def crps(scale: Any) -> float:
        sampler = tfp.distributions.SinhArcsinh(loc=loc_pct, scale=scale, skewness=a, tailweight=t)
        samples = np.asarray(sampler.sample(N_SAMPLES, seed=seed))
        return float(ps.crps_ensemble(real_y, samples.T).mean())

    return {
        "country": country,
        "scheme": scheme,
        "seed": seed,
        "epochs_trained": epochs_trained,
        "weights_switched": switched,
        "RMSE": root_mean_squared_error(real_y, median),
        "MAE": mean_absolute_error(real_y, median),
        "CRPS (original, mis-scaled)": crps(s * std),
        "CRPS": crps(s * std * 100),
        "90% interval coverage": float(np.mean((real_y >= lower) & (real_y <= upper))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1, 2, 3, 4])
    args = parser.parse_args()

    tf.config.experimental.enable_op_determinism()

    rows = []
    for country in DATA_FILES:
        for scheme, weighting in SCHEMES.items():
            for seed in args.seeds:
                row = run(country, scheme, weighting, seed)
                print(
                    f"{country:<10} {scheme:<21} seed={seed:<5} "
                    f"RMSE={row['RMSE']:.3f} MAE={row['MAE']:.3f} "
                    f"CRPS={row['CRPS']:.3f} epochs={row['epochs_trained']}"
                )
                rows.append(row)

    runs = pd.DataFrame(rows)
    metrics = [
        "RMSE",
        "MAE",
        "CRPS (original, mis-scaled)",
        "CRPS",
        "90% interval coverage",
    ]
    summary = runs.groupby(["country", "scheme"], sort=False)[metrics].agg(["mean", "std"])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(TABLES_DIR / "loss_weighting_runs.csv", index=False)
    summary.to_csv(TABLES_DIR / "loss_weighting_summary.csv")

    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summary)


if __name__ == "__main__":
    main()
