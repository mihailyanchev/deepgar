"""DeepGaR in PyTorch: a port of ``deepgar.models.deepgar_skew``.

The port mirrors the TensorFlow model so that the two are interchangeable:

Two details of how Keras trains the TensorFlow model matter here, because
DeepGaR's gradients are small (roughly 1e-7 to 1e-4), so Adam's epsilon is not
negligible:

* the per-sample loss is registered with ``add_loss`` unreduced, so Keras
  differentiates its sum over the batch (it logs the mean);
* Keras adds epsilon before Adam's bias correction (see ``KerasAdam``).

A Keras LSTM also has one bias where PyTorch's has two; only one is trained.

Weights use PyTorch's default initialisation rather than Keras's. Because the
weights stay close to where they start, the port is less accurate than the
TensorFlow model; see the README and ``scripts/compare_torch_port.py``.

The quantile heads read the LSTM output at the last time step. With
``WINDOW_SIZE = 1``, as in the paper, this is what the TensorFlow model does.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from deepgar.evaluation import Forecast, scale_to_percent, to_percent
from deepgar.models.weighting import EQUAL_WEIGHTING, LossWeighting
from deepgar.sampling import Scaler

# Keras's K.epsilon(): the floor in the tilted loss and Adam's epsilon.
KERAS_EPSILON = 1e-7

_ASINH_2 = math.asinh(2.0)


class SinhArcsinh:
    """SinhArcsinh-transformed standard normal, parameterised as TFP's ``SinhArcsinh``.

    ``Y = loc + scale * sinh((asinh(Z) + skewness) * tailweight) * 2 / sinh(asinh(2) * tailweight)``
    with ``Z ~ N(0, 1)``.
    """

    def __init__(self, loc: Tensor, scale: Tensor, skewness: Tensor, tailweight: Tensor) -> None:
        self.loc = loc
        self.scale = scale
        self.skewness = skewness
        self.tailweight = tailweight

    def _effective_scale(self) -> Tensor:
        """``scale`` times TFP's normalising factor ``2 / F_0(2)``."""
        return self.scale * 2 / torch.sinh(_ASINH_2 * self.tailweight)

    def _from_normal(self, z: Tensor) -> Tensor:
        f = torch.sinh((torch.asinh(z) + self.skewness) * self.tailweight)
        return self.loc + self._effective_scale() * f

    def quantile(self, p: Tensor | float) -> Tensor:
        """Quantile function; ``p`` broadcasts against the parameters."""
        return self._from_normal(torch.special.ndtri(torch.as_tensor(p, dtype=self.loc.dtype)))

    def cdf(self, y: Tensor | float) -> Tensor:
        """Cumulative distribution function; ``y`` broadcasts against the parameters."""
        u = (torch.as_tensor(y, dtype=self.loc.dtype) - self.loc) / self._effective_scale()
        z = torch.sinh(torch.asinh(u) / self.tailweight - self.skewness)
        cdf: Tensor = torch.special.ndtr(z)
        return cdf

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """``n`` draws, with shape ``(n, *loc.shape)``."""
        z = torch.randn((n, *self.loc.shape), generator=generator, dtype=self.loc.dtype)
        return self._from_normal(z)


def tilted_loss(quantiles: Sequence[float], y: Tensor, f: Tensor) -> Tensor:
    """Per-sample tilted loss over all quantiles, plus the quantile-crossing penalty.

    ``y`` has shape ``(batch,)`` and ``f`` ``(batch, n_quantiles)``. As in
    ``deepgar.losses.multi_tilted_loss_shared_target``, the crossing penalty is
    a batch mean that is added to every sample.
    """
    q = f.new_tensor(quantiles)
    e = y.unsqueeze(1) - f
    tilted = (q * e + torch.clamp(-e, min=KERAS_EPSILON)).sum(dim=1)
    crossing = F.relu(f[:, :-1] - f[:, 1:]).mean(dim=0).sum()
    return tilted + crossing


def distribution_loss(dist: SinhArcsinh, quantiles: Sequence[float], f: Tensor) -> Tensor:
    """Per-sample mean squared distance between ``f`` and the distribution's quantiles."""
    p = f.new_tensor(quantiles).unsqueeze(1)  # (n_quantiles, 1) broadcasts against (batch,)
    return (f - dist.quantile(p).T).pow(2).mean(dim=1)


def _activation(name: str | None) -> nn.Module:
    activations: dict[str | None, type[nn.Module]] = {
        None: nn.Identity,
        "linear": nn.Identity,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name!r}")
    return activations[name]()


class DeepGaRNet(nn.Module):
    """The DeepGaR network, built from the same config dictionary as the TensorFlow model."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.quantiles = list(config["QUANTILES"])
        common = config["1_COMMON_LSTM_NUM"]
        hidden = config["1_IND_LSTM_NUM"]
        dense = config["1_IND_DENSE_NUM"]

        self.common_lstm = nn.LSTM(config["NUM_FEATURES"], common, batch_first=True)
        self.quantile_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(common, hidden),
                nn.Tanh(),
                nn.Linear(hidden, dense),
                _activation(config["1_IND_DENSE_ACTIVATION"]),
                nn.Linear(dense, 1),
                _activation(config["1_OUT_DENSE_ACTIVATION"]),
            )
            for _ in self.quantiles
        )
        self.distribution_lstm = nn.LSTM(common, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, 1)
        self.sigma = nn.Linear(hidden, 1)
        self.skewness = nn.Linear(hidden, 1)
        self.tailweight = nn.Linear(hidden, 1)

        # A Keras LSTM has one bias; PyTorch's has two that are summed. Training
        # both would give each a full Adam step, doubling the bias learning rate.
        for lstm in (self.common_lstm, self.distribution_lstm):
            bias_hh = lstm.get_parameter("bias_hh_l0")
            nn.init.zeros_(bias_hh)
            bias_hh.requires_grad_(False)

    def forward(self, x: Tensor) -> tuple[Tensor, SinhArcsinh]:
        """Quantile predictions ``(batch, n_quantiles)`` and the predictive distribution."""
        seq, _ = self.common_lstm(x)
        last = seq[:, -1]
        quantile_preds = torch.cat([head(last) for head in self.quantile_heads], dim=1)

        h = self.distribution_lstm(seq)[0][:, -1]
        dist = SinhArcsinh(
            loc=self.mu(h).squeeze(1),
            scale=F.elu(self.sigma(h)).squeeze(1) + 1,
            skewness=self.skewness(h).squeeze(1),
            tailweight=F.elu(self.tailweight(h)).squeeze(1) + 1,
        )
        return quantile_preds, dist

    def loss(self, x: Tensor, y: Tensor, weights: tuple[float, float]) -> Tensor:
        """Per-sample weighted (tilted, distribution) loss, with shape ``(batch,)``."""
        f, dist = self(x)
        alpha, beta = weights
        return alpha * tilted_loss(self.quantiles, y, f) + beta * distribution_loss(
            dist, self.quantiles, f
        )


class KerasAdam(torch.optim.Adam):
    """Adam as tf_keras implements it, with ``epsilon=1e-7``.

    Keras adds epsilon to ``sqrt(v)`` before the bias correction, PyTorch after
    it, so in PyTorch's terms Keras's epsilon is ``1e-7 / sqrt(1 - beta_2**t)``
    at step ``t``: about 3e-6 at the start, decaying towards 1e-7.
    """

    def __init__(self, params: Any, lr: float) -> None:
        super().__init__(params, lr=lr, eps=KERAS_EPSILON)
        self._steps = 0

    def step(self, closure: Any = None) -> Any:
        self._steps += 1
        for group in self.param_groups:
            beta_2 = group["betas"][1]
            group["eps"] = KERAS_EPSILON / math.sqrt(1 - beta_2**self._steps)
        return super().step(closure)


@dataclass
class History:
    """Per-epoch losses, laid out like Keras's ``History`` (``history.history["loss"]``)."""

    history: dict[str, list[float]] = field(default_factory=lambda: {"loss": [], "val_loss": []})
    switched_at_epoch: int | None = None


def _tensor(a: ArrayLike) -> Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def train_deepgar_model(
    model: DeepGaRNet,
    data: dict[str, Any],
    epochs: int,
    patience: int,
    learning_rate: float,
    weighting: LossWeighting = EQUAL_WEIGHTING,
    seed: int = 1234,
    verbose: int = 1,
) -> History:
    """Fit the model as ``keras.Model.fit`` does for the TensorFlow model.

    Each epoch shuffles the training windows into batches of ``BATCH_SIZE``,
    steps on each batch's summed loss, then evaluates the loss on the test set.
    The logged losses are batch means, as Keras reports them. After the epoch
    the loss weights may switch (as ``WeightsUpdater``), and training stops once
    the test loss has not improved for ``patience`` epochs (as ``EarlyStopping``;
    the final weights are kept, not the best ones).
    """
    x_train, y_train = _tensor(data["X_TRAIN"]), _tensor(data["Y_TRAIN"]).reshape(-1)
    x_test, y_test = _tensor(data["X_TEST"]), _tensor(data["Y_TEST"]).reshape(-1)
    optimizer = KerasAdam(model.parameters(), learning_rate)
    generator = torch.Generator().manual_seed(seed)

    weights = weighting.initial
    history = History()
    best, wait = math.inf, 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(x_train), generator=generator).split(data["BATCH_SIZE"]):
            loss = model.loss(x_train[idx], y_train[idx], weights)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            total += loss.sum().item()

        model.eval()
        with torch.no_grad():
            val_loss = model.loss(x_test, y_test, weights).mean().item()
        train_loss = total / len(x_train)
        history.history["loss"].append(train_loss)
        history.history["val_loss"].append(val_loss)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        final = weighting.final
        if (
            final is not None
            and history.switched_at_epoch is None
            and weighting.switches_after(epoch, epochs)
        ):
            weights = final
            history.switched_at_epoch = epoch
            if verbose:
                print(f"Loss weights switched to {final} after epoch {epoch + 1}")

        if val_loss < best:
            best, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break

    return history


def predict(model: DeepGaRNet, x: ArrayLike) -> list[NDArray[np.float32]]:
    """``[mu, sigma, skewness, tailweight]``, each ``(n, 1)``, like the TensorFlow ``predict``."""
    model.eval()
    with torch.no_grad():
        _, dist = model(_tensor(x))
    return [p.unsqueeze(1).numpy() for p in (dist.loc, dist.scale, dist.skewness, dist.tailweight)]


def model_deepgar_skew(
    param: dict[str, Any],
    data: dict[str, Any],
    epochs: int,
    patience: int,
    weighting: LossWeighting = EQUAL_WEIGHTING,
    seed: int = 1234,
    verbose: int = 1,
) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]], History, DeepGaRNet]:
    """Build and train the DeepGaR model with early stopping.

    Same signature and return values as ``deepgar.models.deepgar_skew.model_deepgar_skew``.
    """
    torch.manual_seed(seed)
    model = DeepGaRNet(param)
    history = train_deepgar_model(
        model, data, epochs, patience, param["1_LEARNING_RATE"], weighting, seed, verbose
    )
    return predict(model, data["X_TRAIN"]), predict(model, data["X_TEST"]), history, model


def forecast(
    model: DeepGaRNet,
    data: dict[str, Any],
    scaler: Scaler,
    n_samples: int = 200,
    seed: int | None = None,
) -> Forecast:
    """Predictive SinhArcsinh distribution for the test quarters, in percent."""
    model.eval()
    with torch.no_grad():
        _, dist = model(_tensor(data["X_TEST"]))
    # SinhArcsinh is a location-scale family, so converting location and scale
    # converts the whole distribution to percent.
    pct = SinhArcsinh(
        loc=torch.from_numpy(to_percent(dist.loc.numpy(), scaler)),
        scale=torch.from_numpy(scale_to_percent(dist.scale.numpy(), scaler)),
        skewness=dist.skewness.double(),
        tailweight=dist.tailweight.double(),
    )
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return Forecast(
        median=pct.quantile(0.5).numpy(),
        lower=pct.quantile(0.05).numpy(),
        upper=pct.quantile(0.95).numpy(),
        samples=pct.sample(n_samples, generator).numpy(),
    )
