"""Compare the PyTorch port of DeepGaR with the TensorFlow model.

Trains the PyTorch model for every country over the seeds used by
``recompute_comparison.py`` and scores it the same way. The TensorFlow results
are read from ``model_comparison_runs.csv`` rather than retrained.

Usage:
    uv run python scripts/compare_torch_port.py [--seeds 1234 1 2 3 4] [--countries Bulgaria]
"""

import argparse
import time

import pandas as pd
import torch

from deepgar.evaluation import score, to_percent
from deepgar.experiment import COUNTRIES, DEEPGAR_EPOCHS, DEEPGAR_PATIENCE, load_sample
from deepgar.models import deepgar_skew_torch
from deepgar.paths import TABLES_DIR

N_SAMPLES = 200  # forecast samples per quarter for CRPS, as in recompute_comparison.py
METRICS = ["RMSE", "MAE", "CRPS", "90% interval coverage"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--countries", nargs="+", default=list(COUNTRIES), choices=COUNTRIES)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)

    rows = []
    for country in args.countries:
        data, scaler, param = load_sample(country)
        observed = to_percent(data["Y_TEST"], scaler)
        for seed in args.seeds:
            start = time.perf_counter()
            *_, history, model = deepgar_skew_torch.model_deepgar_skew(
                param, data, DEEPGAR_EPOCHS[country], DEEPGAR_PATIENCE, seed=seed, verbose=0
            )
            fc = deepgar_skew_torch.forecast(model, data, scaler, N_SAMPLES, seed)
            row = {"country": country, "backend": "PyTorch", "seed": seed} | score(observed, fc)
            rows.append(row | {"epochs_trained": len(history.history["loss"])})
            print(
                f"{country:<10} seed={seed:<5} RMSE={row['RMSE']:.3f} MAE={row['MAE']:.3f} "
                f"CRPS={row['CRPS']:.3f} ({time.perf_counter() - start:.1f}s)"
            )

    tf_runs = pd.read_csv(TABLES_DIR / "model_comparison_runs.csv")
    tf_runs = tf_runs[(tf_runs["model"] == "DeepGaR") & tf_runs["country"].isin(args.countries)]
    tf_runs = tf_runs[tf_runs["seed"].isin(args.seeds)].assign(backend="TensorFlow")

    runs = pd.concat([tf_runs, pd.DataFrame(rows)], ignore_index=True)
    columns = ["country", "backend", "seed", *METRICS, "epochs_trained"]
    summary = runs.groupby(["country", "backend"])[METRICS].agg(["mean", "std"])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    runs[columns].to_csv(TABLES_DIR / "torch_port_runs.csv", index=False)
    summary.to_csv(TABLES_DIR / "torch_port_summary.csv")

    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summary)


if __name__ == "__main__":
    main()
