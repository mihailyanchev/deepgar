"""Recompute the paper's model comparison table with the corrected evaluation.

Trains DeepGaR (equal loss weighting, as published), AR(1) and the two linear
quantile regression benchmarks for every country over several seeds, and
scores the test-set forecasts with RMSE, MAE, CRPS and 90% interval coverage.
The original evaluation errors are described in the README (Errata).

Usage:
    uv run python scripts/recompute_comparison.py [--seeds 1234 1 2 3 4] [--countries Bulgaria]
"""

import argparse
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras

from deepgar.evaluation import Forecast, score, to_percent
from deepgar.experiment import (
    BENCHMARK_EPOCHS,
    BENCHMARK_PATIENCE,
    COUNTRIES,
    DEEPGAR_EPOCHS,
    DEEPGAR_PATIENCE,
    LINEAR_SIMULTANEOUS_EPOCHS,
    QUANTILES,
    load_sample,
)
from deepgar.models import deepgar_skew
from deepgar.models.benchmarks import (
    ar1_forecast,
    linear_non_simultaneous_forecast,
    linear_simultaneous_forecast,
)
from deepgar.paths import TABLES_DIR

N_SAMPLES = 200  # forecast samples per quarter for CRPS, as in the notebooks
MODELS = ["DeepGaR", "AR(1)", "Linear simultaneous QR", "Linear non-simultaneous QR"]
METRICS = ["RMSE", "MAE", "CRPS", "90% interval coverage"]


def run(country: str, seed: int) -> list[dict[str, Any]]:
    """Train and score all four models for one country and seed."""
    data, scaler, param = load_sample(country)
    observed = to_percent(data["Y_TEST"], scaler)
    rng = np.random.default_rng(seed)
    forecasts: dict[str, tuple[Forecast, int]] = {}

    keras.utils.set_random_seed(seed)
    *_, model = deepgar_skew.model_deepgar_skew_v2(
        param, data, DEEPGAR_EPOCHS[country], DEEPGAR_PATIENCE, seed=seed, verbose=0
    )
    forecasts["DeepGaR"] = (deepgar_skew.forecast(model, data, scaler, N_SAMPLES, seed), 0)

    keras.utils.set_random_seed(seed)
    ar_data, ar_scaler, _ = load_sample(country, columns=["GDP"])
    forecasts["AR(1)"] = (
        ar1_forecast(ar_data, ar_scaler, BENCHMARK_EPOCHS, BENCHMARK_PATIENCE, N_SAMPLES, rng),
        0,
    )

    keras.utils.set_random_seed(seed)
    forecasts["Linear simultaneous QR"] = linear_simultaneous_forecast(
        param, data, scaler, LINEAR_SIMULTANEOUS_EPOCHS, seed, N_SAMPLES, rng
    )

    keras.utils.set_random_seed(seed)
    forecasts["Linear non-simultaneous QR"] = linear_non_simultaneous_forecast(
        data, scaler, QUANTILES, BENCHMARK_EPOCHS, BENCHMARK_PATIENCE, N_SAMPLES, rng
    )

    return [
        {"country": country, "model": name, "seed": seed, "invalid_sst_fits": invalid}
        | score(observed, fc)
        for name, (fc, invalid) in forecasts.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--countries", nargs="+", default=list(COUNTRIES), choices=COUNTRIES)
    parser.add_argument("--output-prefix", default="model_comparison")
    args = parser.parse_args()

    tf.config.experimental.enable_op_determinism()
    # The SST least-squares fits explore invalid parameters along the way; failed
    # fits are counted in the `invalid_sst_fits` column instead.
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sstudentt")
    warnings.filterwarnings("ignore", message="Number of calls to function has reached maxfev")

    rows = []
    for country in args.countries:
        for seed in args.seeds:
            start = time.perf_counter()
            results = run(country, seed)
            rows.extend(results)
            crps = ", ".join(f"{r['model']}={r['CRPS']:.3f}" for r in results)
            print(f"{country:<10} seed={seed:<5} CRPS: {crps} ({time.perf_counter() - start:.0f}s)")

    runs = pd.DataFrame(rows)
    runs["model"] = pd.Categorical(runs["model"], categories=MODELS, ordered=True)
    summary = runs.groupby(["country", "model"], observed=True)[METRICS].agg(["mean", "std"])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(TABLES_DIR / f"{args.output_prefix}_runs.csv", index=False)
    summary.to_csv(TABLES_DIR / f"{args.output_prefix}_summary.csv")

    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summary)
        print("\nInvalid SST fits (quarters):", int(runs["invalid_sst_fits"].sum()))


if __name__ == "__main__":
    main()
