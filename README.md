# deepgar

Model repository for the DeepGaR model used in the paper: *Deep Growth-at-Risk Model: Nowcasting the 2020 Pandemic Lockdown Recession in Small Open Economies*.

## Repository layout

```
├── src/deepgar/            # installable Python package
│   ├── paths.py            # data/results locations, resolved from the repo root
│   ├── preprocessing.py    # builds per-country datasets from data/interim
│   ├── sampling.py         # train/test split, normalisation, rolling windows
│   ├── losses.py           # tilted quantile loss and distribution losses
│   ├── plotting.py         # training and distribution charts
│   └── models/
│       ├── deepgar_skew.py      # DeepGaR model (LSTM + SinhArcsinh density)
│       └── linear_benchmark.py  # linear quantile regression benchmark
├── notebooks/              # analysis notebooks that produce the paper's results
├── data/
│   ├── raw/                # original downloads (Eurostat, market data)
│   ├── interim/            # cleaned quarterly series
│   └── processed/          # per-country modelling datasets
├── results/
│   ├── figures/
│   └── tables/
└── tests/
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                      # create .venv with all dependencies (incl. dev + notebook tools)
uv run pre-commit install    # optional: run the checks below on every commit
```

To run the notebooks, register the environment as a Jupyter kernel once and select **Python (deepgar)**:

```bash
uv run python -m ipykernel install --sys-prefix --name deepgar --display-name "Python (deepgar)"
uv run jupyter lab
```

The model-diagram cells (`plot_model`) also need the Graphviz system package, e.g. `brew install graphviz` on macOS.

## Usage

Rebuild the per-country datasets from `data/interim`:

```bash
uv run deepgar-preprocess
```

This writes `data_<country>_<timestamp>.csv` files to `data/processed/`. The committed datasets for Bulgaria, Estonia and Lithuania were extended by hand with the country-specific indicators in `data/raw/country_specific/`.

The models are trained and evaluated in the notebooks:

| Notebook | Purpose |
| --- | --- |
| `01_gdp_distribution_charts` | Observed GDP growth distributions |
| `02_deepgar_skew` | DeepGaR model and AR(1) comparison |
| `03_deepgar_skew_explainability` | LIME explanations of the DeepGaR model |
| `04_quantile_linear_benchmark` | Linear quantile regression benchmarks |

## Errata

Several bugs in the original analysis code were found and fixed in 2026. The committed result tables in `results/tables/` (other than `loss_weighting_*` and `model_comparison_*`, which were produced with the fixed code) and figures in `results/figures/` are the **original outputs and are left unchanged**. The notes below describe how they are affected.

### 1. Loss weighting was never applied

The paper describes a dynamic weighting of the two DeepGaR losses: 95% on the tilted loss and 5% on the distribution loss for the first 90% of the training epochs, then 5% / 95% for the rest. A bug in the original code meant this scheme was never applied, so **the published results were produced with an equally weighted (50/50) sum of the two losses for all epochs**.

`deepgar.models.deepgar_skew` now implements the weighting correctly. `EQUAL_WEIGHTING` is the default, so the published results stay reproducible. The scheme described in the paper is available as `PAPER_DYNAMIC_WEIGHTING`. To compare the two:

```bash
uv run python scripts/compare_loss_weighting.py
```

This writes `results/tables/loss_weighting_runs.csv` and `results/tables/loss_weighting_summary.csv`. Over 10 seeds, the difference between the schemes is small and mixed across countries, and much smaller than DeepGaR's margin over the AR(1) benchmark.

### 2. Unit errors in the probabilistic evaluation

The models work on standardised GDP growth, and results are reported in percent: `(x * std + mean) * 100`. Several evaluation steps mixed these units.

| Quantity | Bug | Effect on the published values |
| --- | --- | --- |
| **CRPS** (Continuous Ranked Probability Score), all models | The predictive distribution's scale was converted with `* std` but without `* 100`, while its location was in percent. In the linear simultaneous benchmark, the *skewness* parameter was also used as the scale. | The predictive spread was ~100× too narrow, so the reported CRPS is essentially the mean absolute error of the location forecast, not a probabilistic score. Reported CRPS values are too high and do not reflect the shape of the distributions. Affects the `CRPS` column in `dqpr_ar1_summary.*` and `lsqe_lnqe_summary.csv`. |
| **Probability of recession**, all models | The threshold `-0.001` was evaluated on the standardised scale. | The column is P(growth < training-sample mean growth), roughly P(growth < 0.8–0.9%), not P(growth < 0). For example, Bulgaria 2017Q3 is reported at 0.44 although the 5% quantile of the same forecast is +0.09%. Affects `results_deepgar_*.csv`, `results_benchmark_*.csv` and `summary_results.xlsx`. |
| **Predictive score** | The density was evaluated in standardised units. | The values are densities per standard deviation of growth, not per percentage point; rankings within a country are unaffected. |
| **Standard Deviation** column (scale parameter) | Reported in standardised units, while the location is in percent. | Not comparable with the location, quantile and IQR columns. |

The notebooks now compute all of these in percent. Recession is defined as q-o-q growth below −0.1% (`RECESSION_THRESHOLD`, the original −0.001 read as a growth rate).

RMSE, MAE, the 5%/95% quantiles, the IQR and the quantile bands in the figures were computed correctly and are **not** affected.

### 3. Other fixes

- **AR(1) predictive spread:** the residual standard deviation was computed as input minus forecast instead of target minus forecast. This affects the AR(1) CRPS.
- **Linear non-simultaneous benchmark:**
  - The 5% and 95% quantiles were swapped, so the ".95 quantile" was the 5% quantile and vice versa.
  - Its CRPS used the location of the *simultaneous* benchmark's distributions, because the location was recomputed only after the CRPS cell.
- **Point forecasts:** RMSE now uses the median forecast for every model, like MAE. Previously the linear non-simultaneous benchmark (and DeepGaR in notebook 03) used the distribution's mean for RMSE and the median for MAE.

### 4. Corrected model comparison

The comparison table was recomputed with all of the fixes above, for all four countries and models:

```bash
uv run python scripts/recompute_comparison.py
```

Each model was trained with 10 seeds (1234 and 1–9), using the paper's setup and the published equal loss weighting for DeepGaR. The script writes `results/tables/model_comparison_runs.csv` (one row per run) and `results/tables/model_comparison_summary.csv`.

The table shows published values (single runs, from `dqpr_ar1_summary.xlsx`) and corrected values (mean ± standard deviation over the 10 seeds). All metrics are in percentage points of quarter-on-quarter GDP growth, and lower is better. The best corrected value per country is in bold.

| Country | Model | RMSE published | RMSE corrected | MAE published | MAE corrected | CRPS published | CRPS corrected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bulgaria | DeepGaR | 1.70 | **1.52 ± 0.05** | 0.88 | **0.95 ± 0.05** | 1.07 | **0.84 ± 0.03** |
|  | AR(1) | 2.39 | 2.31 ± 0.07 | 1.22 | 1.18 ± 0.03 | 1.21 | 1.06 ± 0.04 |
|  | Linear simultaneous QR | 2.68 | 2.45 ± 0.43 | 1.36 | 1.33 ± 0.17 | 1.79 | 1.13 ± 0.20 |
|  | Linear non-simultaneous QR | 2.05 | 2.39 ± 0.38 | 1.79 | 1.32 ± 0.15 | 1.63 | 1.09 ± 0.16 |
| Estonia | DeepGaR | 1.64 | **1.30 ± 0.04** | 0.81 | **0.83 ± 0.02** | 1.02 | **0.69 ± 0.02** |
|  | AR(1) | 1.92 | 1.89 ± 0.03 | 1.11 | 1.13 ± 0.02 | 1.10 | 0.97 ± 0.02 |
|  | Linear simultaneous QR | 2.35 | 1.99 ± 0.51 | 1.53 | 1.17 ± 0.20 | 1.66 | 0.94 ± 0.20 |
|  | Linear non-simultaneous QR | 1.60 | 1.91 ± 0.50 | 1.02 | 1.13 ± 0.23 | 1.36 | 0.88 ± 0.19 |
| Lithuania | DeepGaR | 1.62 | **1.16 ± 0.04** | 0.50 | **0.63 ± 0.05** | 0.76 | **0.59 ± 0.03** |
|  | AR(1) | 2.23 | 2.10 ± 0.20 | 1.04 | 0.96 ± 0.11 | 1.03 | 1.01 ± 0.07 |
|  | Linear simultaneous QR | 2.07 | 2.23 ± 0.51 | 0.97 | 1.23 ± 0.25 | 0.91 | 1.00 ± 0.20 |
|  | Linear non-simultaneous QR | 1.52 | 1.89 ± 0.23 | 0.76 | 1.05 ± 0.12 | 0.91 | 0.86 ± 0.12 |
| Romania | DeepGaR | 2.07 | **1.72 ± 0.09** | 1.03 | **1.02 ± 0.03** | 1.06 | **0.82 ± 0.03** |
|  | AR(1) | 2.99 | 2.93 ± 0.06 | 1.42 | 1.40 ± 0.02 | 1.41 | 1.22 ± 0.02 |
|  | Linear simultaneous QR | 2.65 | 2.61 ± 0.55 | 1.39 | 1.32 ± 0.21 | 1.67 | 1.09 ± 0.20 |
|  | Linear non-simultaneous QR | 1.85 | 2.36 ± 0.50 | 1.12 | 1.21 ± 0.18 | 1.32 | 0.99 ± 0.14 |

**Findings**

- **DeepGaR is the best model on RMSE, MAE and CRPS in all four countries.** On CRPS it beats each benchmark in 8–10 of the 10 seeds (paired Wilcoxon tests, which compare models seed by seed: p ≤ 0.02 for every country and benchmark).
- **DeepGaR's advantage is clearer than published.** In the published RMSE, the linear non-simultaneous benchmark beat DeepGaR in Estonia, Lithuania and Romania. Over 10 seeds it does not. The linear benchmarks vary a lot between seeds (RMSE ± 0.23–0.55), so the published single runs were favourable draws; switching RMSE from the mean to the median forecast also contributes. These effects cannot be separated.
- **CRPS values are lower for every model** once the predictive spread is in the right units. DeepGaR's margin over the benchmarks is larger than published.
- **DeepGaR's 90% intervals are too narrow in some countries.** They cover 69% of outcomes in Bulgaria and 81–84% in Lithuania and Romania (95% in Estonia); see the `90% interval coverage` column in the summary file. On a test set of only 18–21 quarters dominated by the 2020 shock, this should be read with caution.

**Caveats**

- **Not a pure before/after comparison.** Corrected and published values also differ in library versions (TensorFlow 2.21 vs 2.7–2.9) and in the number of runs, so differences cannot be attributed to the bug fixes alone.
- **Some SST fits are invalid.** The SST fitting procedure (unchanged from the paper) returns invalid parameters for 58 of the 1,560 benchmark quarter-forecasts (3.7%), all in the two linear benchmarks. These are counted in the `invalid_sst_fits` column.

## Development

```bash
uv run ruff check            # lint
uv run ruff format           # format
uv run mypy                  # static type checking
uv run bandit -c pyproject.toml -r src   # security checks
uv run pytest                # tests
```

The same checks run in CI on every push and pull request.

> **Note:** the models still use Keras 2 through the `tf-keras` package. Migrating to Keras 3 is planned.
