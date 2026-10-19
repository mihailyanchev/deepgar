import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sstudentt import SST

from deepgar.evaluation import Forecast, crps, scale_to_percent, score, to_percent
from deepgar.models.benchmarks import fit_sst, sst_forecast

SCALER = {"mean": pd.Series([0.008]), "std": pd.Series([0.01])}
QUANTILES = [0.05, 0.25, 0.75, 0.95]


def test_unit_conversion() -> None:
    """Values are shifted and scaled to percent; scales are only scaled."""
    np.testing.assert_allclose(to_percent([0.0, 1.0], SCALER), [0.8, 1.8])
    np.testing.assert_allclose(scale_to_percent([1.0], SCALER), [1.0])


def test_crps_matches_closed_form_for_normal() -> None:
    """Sample-based CRPS approximates the closed-form CRPS of a normal forecast."""
    rng = np.random.default_rng(0)
    y = np.array([0.3, -1.2])
    # Kept small: without numba, properscoring's ensemble CRPS is quadratic in the samples.
    samples = rng.normal(0.0, 1.0, size=(2_000, 2))
    exact = np.mean(y * (2 * stats.norm.cdf(y) - 1) + 2 * stats.norm.pdf(y) - 1 / np.sqrt(np.pi))
    assert crps(y, samples) == pytest.approx(exact, abs=0.03)


def test_crps_of_a_point_forecast_is_absolute_error() -> None:
    """With no spread, CRPS reduces to the absolute error."""
    samples = np.full((100, 2), 1.0)
    assert crps(np.array([0.0, 3.0]), samples) == pytest.approx(1.5)


def test_score() -> None:
    """Point metrics use the median; coverage counts outcomes inside the 90% interval."""
    fc = Forecast(
        median=np.array([1.0, 2.0]),
        lower=np.array([0.0, 2.5]),
        upper=np.array([2.0, 3.0]),
        samples=np.full((10, 2), 1.0),
    )
    result = score(np.array([1.0, 4.0]), fc)
    assert result["MAE"] == pytest.approx(1.0)
    assert result["RMSE"] == pytest.approx(np.sqrt(2.0))
    assert result["90% interval coverage"] == pytest.approx(0.5)


def test_fit_sst_recovers_quantiles() -> None:
    """The fitted SST reproduces the quantiles it was fitted to."""
    target = SST(0.2, 1.1, 1.3, 6.0).q(QUANTILES)
    params = fit_sst(target, QUANTILES)
    np.testing.assert_allclose(SST(*params).q(QUANTILES), target, atol=1e-4)


def test_sst_forecast_is_in_percent() -> None:
    """The SST forecast's quantiles equal the standardised quantiles converted to percent."""
    standardised = np.array([SST(0.0, 1.0, 1.2, 5.0).q(QUANTILES)])
    fc, invalid = sst_forecast(standardised, QUANTILES, SCALER, 10, np.random.default_rng(0))
    assert invalid == 0
    np.testing.assert_allclose(fc.lower, to_percent(standardised[:, 0], SCALER), atol=1e-3)
    np.testing.assert_allclose(fc.upper, to_percent(standardised[:, 3], SCALER), atol=1e-3)
