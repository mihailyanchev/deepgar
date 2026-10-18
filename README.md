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

Several bugs in the original analysis code were found and fixed in 2026. The committed result tables in `results/tables/` (other than `loss_weighting_*`) and figures in `results/figures/` are the **original outputs and are left unchanged**. The notes below describe how they are affected.

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

With the fixes, DeepGaR still has a lower CRPS than AR(1) in single-run checks (Bulgaria 0.80 vs 1.11, Romania 0.87 vs 1.24). Recomputing the full table for all countries and models is pending.

### 3. Other fixes

- **AR(1) predictive spread:** the residual standard deviation was computed as input minus forecast instead of target minus forecast. This affects the AR(1) CRPS.
- **Linear non-simultaneous benchmark:** the 5% and 95% quantiles were swapped, so the ".95 quantile" was the 5% quantile and vice versa.

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
