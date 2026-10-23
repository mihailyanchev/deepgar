"""Framework-independent evaluation of probabilistic GDP growth forecasts.

The models work on standardised growth; everything here is in percent, i.e.
``(x * std + mean) * 100`` for values and ``x * std * 100`` for scales.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import properscoring as ps
from numpy.typing import NDArray

from deepgar.sampling import Scaler


def to_percent(x: Any, scaler: Scaler) -> NDArray[np.float64]:
    """Convert standardised growth values to percent."""
    std = float(scaler["std"].iloc[0])
    mean = float(scaler["mean"].iloc[0])
    return np.asarray((np.asarray(x, dtype=np.float64) * std + mean) * 100)


def scale_to_percent(x: Any, scaler: Scaler) -> NDArray[np.float64]:
    """Convert a standardised scale (spread) parameter to percent; no location shift."""
    return np.asarray(np.asarray(x, dtype=np.float64) * float(scaler["std"].iloc[0]) * 100)


@dataclass(frozen=True)
class Forecast:
    """A probabilistic forecast for each test quarter, in percent.

    ``samples`` has shape ``(n_samples, n_quarters)``.
    """

    median: NDArray[np.float64]
    lower: NDArray[np.float64]  # 5% quantile
    upper: NDArray[np.float64]  # 95% quantile
    samples: NDArray[np.float64]


def crps(observed: NDArray[np.float64], samples: NDArray[np.float64]) -> float:
    """Mean CRPS (Continuous Ranked Probability Score) of a sample-based forecast."""
    return float(np.mean(ps.crps_ensemble(observed, samples.T)))


def score(observed: Any, forecast: Forecast) -> dict[str, float]:
    """Point and probabilistic accuracy of a forecast. Lower is better except coverage."""
    y = np.asarray(observed, dtype=np.float64).squeeze()
    error = y - forecast.median
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "CRPS": crps(y, forecast.samples),
        "90% interval coverage": float(np.mean((y >= forecast.lower) & (y <= forecast.upper))),
    }
