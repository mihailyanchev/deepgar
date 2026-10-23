import numpy as np
import pytest
import tensorflow as tf

from deepgar.losses import multi_tilted_loss, multi_tilted_loss_v2

QUANTILES = [0.05, 0.5, 0.95]


def test_tilted_loss_variants_agree() -> None:
    """Both tilted-loss variants give the same value for equivalent targets."""
    y = tf.constant([[0.1], [0.4], [-0.2]])
    f = tf.constant([[-0.5, 0.0, 0.5], [0.0, 0.3, 0.9], [-0.9, -0.1, 0.2]])
    y_k = tf.tile(y, [1, len(QUANTILES)])
    v1 = multi_tilted_loss(QUANTILES, y_k, f)
    v2 = multi_tilted_loss_v2(QUANTILES, tf.squeeze(y), f)
    assert float(v1) == pytest.approx(float(v2))


def test_crossing_quantiles_are_penalised() -> None:
    """Crossed quantile predictions incur a higher loss than ordered ones."""
    y = tf.constant([0.0, 0.0])
    ordered = tf.constant([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    crossed = tf.constant([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    assert float(multi_tilted_loss_v2(QUANTILES, y, crossed)) > float(
        multi_tilted_loss_v2(QUANTILES, y, ordered)
    )
    assert np.isfinite(float(multi_tilted_loss_v2(QUANTILES, y, ordered)))
