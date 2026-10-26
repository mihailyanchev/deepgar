"""Loss functions shared by the DeepGaR and benchmark models."""

from collections.abc import Sequence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_keras import backend as K


def ls_quantile_cdf(
    mu: tf.Tensor,
    sigma: tf.Tensor,
    skewness: tf.Tensor,
    tailweight: tf.Tensor,
    quantile_values: tf.Tensor,
    quantiles: Sequence[float],
) -> tf.Tensor:
    """Squared distance between estimated quantiles and SinhArcsinh quantiles."""
    dist = tfp.distributions.SinhArcsinh(
        loc=mu, scale=sigma, skewness=skewness, tailweight=tailweight
    )
    q_val = tf.squeeze(quantile_values)
    q_est = dist.quantile(quantiles)
    return K.mean(tf.math.squared_difference(q_val, q_est), axis=-1)


def multi_tilted_loss_shared_target(
    quantiles: Sequence[float], y: tf.Tensor, f: tf.Tensor
) -> tf.Tensor:
    """Tilted loss for simultaneous quantile generation, with a crossing penalty.

    ``y`` holds a single target column that is compared against every quantile.
    """
    loss = 0

    # quantile loss
    for k, q in enumerate(quantiles):
        e = y - f[:, k]
        loss += K.mean(q * e + K.clip(-e, K.epsilon(), np.inf), axis=-1)

    # crossing loss
    for i in range(len(quantiles) - 1):
        q1 = f[:, i]
        q2 = f[:, i + 1]
        loss += K.mean(K.maximum(0.0, q1 - q2))

    return loss


def multi_tilted_loss(quantiles: Sequence[float], y: tf.Tensor, f: tf.Tensor) -> tf.Tensor:
    """Tilted loss for simultaneous quantile generation, with a crossing penalty.

    ``y`` holds one target column per quantile (see ``create_1st_step_y``).
    """
    loss = 0

    # quantile loss
    for k, q in enumerate(quantiles):
        e = y[:, k] - f[:, k]
        loss += K.mean(q * e + K.clip(-e, K.epsilon(), np.inf), axis=-1)

    # crossing loss
    for i in range(len(quantiles) - 1):
        q1 = f[:, i]
        q2 = f[:, i + 1]
        loss += K.mean(K.maximum(0.0, q1 - q2))

    return loss


def tilted_loss(q: float, y: tf.Tensor, f: tf.Tensor) -> tf.Tensor:
    """Tilted (pinball) loss for a single quantile ``q``."""
    e = y - f
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
