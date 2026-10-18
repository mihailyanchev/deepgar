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

## Note on the loss weighting

The paper describes a dynamic weighting of the two DeepGaR losses: 95% on the tilted loss and 5% on the distribution loss for the first 90% of the training epochs, then 5% / 95% for the rest. A bug in the original code meant this scheme was never applied, so **the published results were produced with an equally weighted (50/50) sum of the two losses for all epochs**.

`deepgar.models.deepgar_skew` now implements the weighting correctly. `EQUAL_WEIGHTING` is the default, so the published results stay reproducible. The scheme described in the paper is available as `PAPER_DYNAMIC_WEIGHTING`. To compare the two:

```bash
uv run python scripts/compare_loss_weighting.py
```

This writes `results/tables/loss_weighting_runs.csv` and `results/tables/loss_weighting_summary.csv`.

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
