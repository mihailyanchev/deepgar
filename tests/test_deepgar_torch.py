"""The PyTorch DeepGaR port, checked against TFP and the TensorFlow model."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import tensorflow_probability as tfp
import tf_keras as keras
import torch
from torch import nn

from deepgar.experiment import MODEL_PARAM
from deepgar.models import deepgar_skew
from deepgar.models import deepgar_skew_torch as dgt
from deepgar.models.weighting import LossWeighting
from deepgar.sampling import sample_creation

PARAM = dict(MODEL_PARAM, NUM_FEATURES=3)
DIST_PARAMS = {
    "loc": [0.1, -0.3, 0.0],
    "scale": [0.5, 1.2, 2.0],
    "skewness": [0.4, -0.7, 0.0],
    "tailweight": [1.0, 1.5, 0.8],
}


def _sinh_arcsinh() -> dgt.SinhArcsinh:
    params = {k: torch.tensor(v, dtype=torch.float64) for k, v in DIST_PARAMS.items()}
    return dgt.SinhArcsinh(**params)


def _assert_same_predictions(keras_model: Any, net: dgt.DeepGaRNet, x: np.ndarray) -> None:
    """Both models predict the same distribution parameters for ``x``."""
    expected = keras_model([x, np.zeros((len(x), 1), dtype=np.float32)])
    with torch.no_grad():
        _, dist = net(torch.from_numpy(x))
    ours = [dist.loc, dist.scale, dist.skewness, dist.tailweight]
    for exp, got in zip(expected, ours, strict=True):
        np.testing.assert_allclose(got, np.squeeze(exp, 1), rtol=1e-4, atol=1e-5)


def test_sinh_arcsinh_matches_tfp() -> None:
    """Quantiles and CDF agree with TFP's SinhArcsinh."""
    ours = _sinh_arcsinh()
    theirs = tfp.distributions.SinhArcsinh(**{k: np.array(v) for k, v in DIST_PARAMS.items()})
    p = np.array([[0.05], [0.25], [0.5], [0.75], [0.95]])
    np.testing.assert_allclose(ours.quantile(torch.from_numpy(p)), theirs.quantile(p), rtol=1e-6)
    y = np.linspace(-3, 3, 7)[:, None]
    np.testing.assert_allclose(ours.cdf(torch.from_numpy(y)), theirs.cdf(y), rtol=1e-6, atol=1e-12)


def test_sinh_arcsinh_samples_follow_quantiles() -> None:
    """Empirical quantiles of the samples match the quantile function."""
    dist = _sinh_arcsinh()
    samples = dist.sample(200_000, torch.Generator().manual_seed(0))
    p = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)
    np.testing.assert_allclose(
        torch.quantile(samples, p, dim=0), dist.quantile(p.unsqueeze(1)), atol=0.03
    )


def _copy_keras_weights(keras_model: Any, net: dgt.DeepGaRNet) -> None:
    """Load the weights of a Keras DeepGaR model into the PyTorch network."""
    layers = {layer.name: layer for layer in keras_model.layers}

    def inbound(layer: Any) -> Any:
        return layer.inbound_nodes[0].inbound_layers

    def dense(src: Any, dst: nn.Linear) -> None:
        kernel, bias = src.get_weights()
        dst.weight.copy_(torch.from_numpy(kernel.T))
        dst.bias.copy_(torch.from_numpy(bias))

    def lstm(src: Any, dst: nn.LSTM) -> None:
        # Same gate order (input, forget, cell, output); Keras has a single bias.
        kernel, recurrent, bias = src.get_weights()
        params = dict(dst.named_parameters())
        params["weight_ih_l0"].copy_(torch.from_numpy(kernel.T))
        params["weight_hh_l0"].copy_(torch.from_numpy(recurrent.T))
        params["bias_ih_l0"].copy_(torch.from_numpy(bias))
        params["bias_hh_l0"].zero_()

    with torch.no_grad():
        for q, head in zip(net.quantiles, net.quantile_heads, strict=True):
            out = layers[f"q{int(q * 100)}"]
            hidden = inbound(out)
            first = inbound(hidden)
            linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
            for src, dst in zip([first, hidden, out], linears, strict=True):
                dense(src, dst)
        lstm(inbound(first), net.common_lstm)
        lstm(inbound(layers["mu"]), net.distribution_lstm)
        for name in ("mu", "sigma", "skewness", "tailweight"):
            dense(layers[name], getattr(net, name))


@pytest.fixture
def models() -> tuple[Any, Any, Any, dgt.DeepGaRNet]:
    """A Keras DeepGaR model, its loss-weight variables, and a PyTorch copy."""
    keras_model, alpha, beta = deepgar_skew.QuantileLossModel(PARAM).create_deepgar_model(seed=0)
    net = dgt.DeepGaRNet(PARAM)
    _copy_keras_weights(keras_model, net)
    return keras_model, alpha, beta, net


@pytest.fixture
def batch() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 1, 3)).astype(np.float32)
    y = rng.normal(size=(16, 1)).astype(np.float32)
    return x, y


def test_forward_matches_keras(
    models: tuple[Any, Any, Any, dgt.DeepGaRNet], batch: tuple[np.ndarray, np.ndarray]
) -> None:
    """With the same weights, both models predict the same distribution parameters."""
    keras_model, _, _, net = models
    _assert_same_predictions(keras_model, net, batch[0])


@pytest.mark.parametrize("weights", [(1.0, 1.0), (0.95, 0.05), (0.05, 0.95)])
def test_loss_matches_keras(
    models: tuple[Any, Any, Any, dgt.DeepGaRNet],
    batch: tuple[np.ndarray, np.ndarray],
    weights: tuple[float, float],
) -> None:
    """The weighted loss equals the Keras model's ``add_loss`` value."""
    keras_model, alpha, beta, net = models
    keras.backend.set_value(alpha, weights[0])
    keras.backend.set_value(beta, weights[1])
    x, y = batch
    expected = keras_model.evaluate([x, y], batch_size=len(x), verbose=0)
    with torch.no_grad():
        ours = net.loss(torch.from_numpy(x), torch.from_numpy(y).reshape(-1), weights)
    # Keras reports the batch mean of the per-sample loss.
    assert float(ours.mean()) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("batch_size", [1, 16])
def test_training_steps_match_keras(
    models: tuple[Any, Any, Any, dgt.DeepGaRNet],
    batch: tuple[np.ndarray, np.ndarray],
    batch_size: int,
) -> None:
    """Optimisation steps move both models to the same weights.

    A batch of one is where the summed gradient and Keras's epsilon matter most.
    """
    keras_model, _, _, net = models
    x, y = batch[0][:batch_size], batch[1][:batch_size]
    optimizer = dgt.KerasAdam(net.parameters(), PARAM["1_LEARNING_RATE"])
    for _ in range(3):
        keras_model.train_on_batch([x, y])
        optimizer.zero_grad()
        net.loss(torch.from_numpy(x), torch.from_numpy(y).reshape(-1), (1.0, 1.0)).sum().backward()
        optimizer.step()

    _assert_same_predictions(keras_model, net, batch[0])


@pytest.fixture
def sample() -> dict[str, Any]:
    """A small synthetic modelling sample with three predictors."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(40, 3)), columns=["GDP", "A", "B"])
    data, _ = sample_creation(frame, PARAM)
    return data


def test_training_switches_weights_and_is_deterministic(sample: dict[str, Any]) -> None:
    """The weights switch after the configured epoch, and a seed fixes the run."""
    weighting = LossWeighting(initial=(0.95, 0.05), final=(0.05, 0.95), switch_at=0.5)
    runs = [
        dgt.model_deepgar_skew(PARAM, sample, 12, 100, weighting, seed=3, verbose=0)
        for _ in range(2)
    ]
    _, test_pred, history, _ = runs[0]
    assert history.switched_at_epoch == 7  # first epoch after int(12 * 0.5)
    assert len(history.history["loss"]) == 12
    assert history.history == runs[1][2].history
    assert [p.shape for p in test_pred] == [(len(sample["X_TEST"]), 1)] * 4


def test_early_stopping_stops_after_patience(sample: dict[str, Any]) -> None:
    """Training stops ``patience`` epochs after the best test loss."""
    _, _, history, _ = dgt.model_deepgar_skew(PARAM, sample, 500, 3, seed=0, verbose=0)
    val_loss = history.history["val_loss"]
    assert len(val_loss) < 500
    assert int(np.argmin(val_loss)) == len(val_loss) - 1 - 3
