"""Benchmark models from the paper (notebooks 02 and 04).

* AR(1): linear regression on lagged GDP growth with a normal predictive
  distribution whose spread is the training residual standard deviation.
* Linear quantile regressions, estimated simultaneously (one model, tilted
  loss with a crossing penalty) or separately per quantile. A skewed Student-t
  (SST) distribution is fitted to each quarter's four predicted quantiles.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import tf_keras as keras
from numpy.typing import NDArray
from scipy import optimize, stats
from sstudentt import SST

from deepgar.evaluation import Forecast, scale_to_percent, to_percent
from deepgar.losses import tilted_loss
from deepgar.models.linear_benchmark import model_linear
from deepgar.sampling import Scaler

SST_INITIAL_ESTIMATES = np.array([1.0, 1.0, 1.0, 3.0])


def _fit_dense(
    data: dict[str, Any], loss: str | Callable[..., Any], epochs: int, patience: int
) -> keras.Model:
    """One linear layer trained on (X_TRAIN, Y_TRAIN) with early stopping."""
    model = keras.Sequential([keras.layers.Dense(1, activation="linear")])
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(0.001))
    model.fit(
        data["X_TRAIN"],
        data["Y_TRAIN"],
        epochs=epochs,
        batch_size=1,
        verbose=0,
        validation_split=0.2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)],
    )
    return model


def ar1_forecast(
    data: dict[str, Any],
    scaler: Scaler,
    epochs: int,
    patience: int,
    n_samples: int,
    rng: np.random.Generator,
) -> Forecast:
    """AR(1) forecast. ``data`` must be built from the GDP column only."""
    model = _fit_dense(data, "mse", epochs, patience)
    pred_test = np.asarray(model.predict(data["X_TEST"], verbose=0)).squeeze()
    pred_train = np.asarray(model.predict(data["X_TRAIN"], verbose=0)).squeeze()
    # Residuals are target minus forecast (the original used input minus forecast).
    sigma = float(scale_to_percent(np.std(data["Y_TRAIN"].squeeze() - pred_train), scaler))

    loc = to_percent(pred_test, scaler)
    z = stats.norm.ppf(0.95)
    return Forecast(
        median=loc,
        lower=loc - z * sigma,
        upper=loc + z * sigma,
        samples=rng.normal(loc, sigma, size=(n_samples, loc.size)),
    )


def fit_sst(quantile_values: Sequence[float], quantiles: Sequence[float]) -> NDArray[np.float64]:
    """Fit SST (mu, sigma, nu, tau) to a set of quantiles by least squares."""

    def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(SST(*params).q(quantiles) - np.asarray(quantile_values))

    params, _ = optimize.leastsq(residuals, SST_INITIAL_ESTIMATES)
    return np.asarray(params, dtype=np.float64)


def sst_forecast(
    quantile_preds: NDArray[np.float64],
    quantiles: Sequence[float],
    scaler: Scaler,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[Forecast, int]:
    """Fit an SST to each row of standardised quantile predictions.

    Returns the forecast in percent and the number of quarters whose fitted
    parameters are invalid (sigma <= 0, nu <= 0, tau <= 2 or non-finite).
    """
    params = np.array([fit_sst(row, quantiles) for row in quantile_preds])
    mu, sigma, nu, tau = params.T
    invalid = int(np.sum(~np.isfinite(params).all(axis=1) | (sigma <= 0) | (nu <= 0) | (tau <= 2)))

    # SST is a location-scale family in (mu, sigma), so converting the
    # parameters converts the whole distribution to percent.
    dist = SST(to_percent(mu, scaler), scale_to_percent(sigma, scaler), nu, tau)
    forecast = Forecast(
        median=np.asarray(dist.q(0.5)),
        lower=np.asarray(dist.q(0.05)),
        upper=np.asarray(dist.q(0.95)),
        samples=np.asarray(dist.q(rng.random((n_samples, len(mu))))),
    )
    return forecast, invalid


def linear_simultaneous_forecast(
    param: dict[str, Any],
    data: dict[str, Any],
    scaler: Scaler,
    epochs: int,
    seed: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[Forecast, int]:
    """Linear quantile regression with all quantiles estimated jointly."""
    _, y_test_dist, _, _ = model_linear(param, data, epochs, seed=seed, verbose=0)
    return sst_forecast(np.squeeze(y_test_dist), param["QUANTILES"], scaler, n_samples, rng)


def linear_non_simultaneous_forecast(
    data: dict[str, Any],
    scaler: Scaler,
    quantiles: Sequence[float],
    epochs: int,
    patience: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[Forecast, int]:
    """Linear quantile regression with one model per quantile."""
    preds = []
    for q in quantiles:
        model = _fit_dense(data, lambda y, f, q=q: tilted_loss(q, y, f), epochs, patience)
        preds.append(np.asarray(model.predict(data["X_TEST"], verbose=0)).squeeze())
    return sst_forecast(np.column_stack(preds), quantiles, scaler, n_samples, rng)
