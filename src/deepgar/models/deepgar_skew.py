"""DeepGaR: LSTM quantile network with a SinhArcsinh density head.

Loss weighting
--------------
The model is trained on two losses: the tilted (quantile) loss and the
least-squares loss that fits the SinhArcsinh distribution to the quantiles.

The paper describes a dynamic scheme: weights of 0.95 / 0.05 for the first 90%
of the training epochs, switching to 0.05 / 0.95 for the remainder. Due to a
bug, the original code never applied it:

* the weights were passed to ``compile(loss_weights=...)``, which Keras ignores
  for losses registered through ``add_loss``; and
* the callback rebound Python attributes instead of updating the Keras
  variables, so the weights never switched either.

The published results were therefore produced with an unweighted sum of the
two losses (equal, 50/50 weighting) for all epochs. ``EQUAL_WEIGHTING`` is the
default so the published results remain reproducible; ``PAPER_DYNAMIC_WEIGHTING``
implements the scheme as described in the paper. Both are defined in
``deepgar.models.weighting``.
"""

from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras
from tf_keras import backend as K
from tf_keras import layers

from deepgar.evaluation import Forecast, scale_to_percent, to_percent
from deepgar.losses import ls_quantile_cdf, multi_tilted_loss_shared_target
from deepgar.models.weighting import EQUAL_WEIGHTING, LossWeighting
from deepgar.sampling import Scaler


class WeightedLossSum(layers.Layer):  # type: ignore[misc]
    """Weighted sum of the tilted and distribution losses.

    The weights are non-trainable layer weights, so they are read on every
    training step and can be changed during training with ``K.set_value``.
    (Multiplying by a ``K.variable`` directly in the functional API freezes its
    value into the graph when the model is built.)
    """

    def __init__(self, initial: tuple[float, float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._initial = initial

    def build(self, input_shape: Any) -> None:
        self.alpha = self.add_weight(
            name="tilted_loss_weight",
            shape=(),
            initializer=keras.initializers.Constant(self._initial[0]),
            trainable=False,
        )
        self.beta = self.add_weight(
            name="distribution_loss_weight",
            shape=(),
            initializer=keras.initializers.Constant(self._initial[1]),
            trainable=False,
        )

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        # Per-sample losses are kept unreduced, as in the original implementation.
        loss_q, loss_f = inputs
        return self.alpha * loss_q + self.beta * loss_f


class WeightsUpdater(keras.callbacks.Callback):  # type: ignore[misc]
    """Switch the loss weights near the end of training."""

    def __init__(self, alpha: Any, beta: Any, epochs: int, weighting: LossWeighting) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.total_epochs = epochs
        self.weighting = weighting
        self.switched_at_epoch: int | None = None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        final = self.weighting.final
        if final is None or self.switched_at_epoch is not None:
            return
        if self.weighting.switches_after(epoch, self.total_epochs):
            # Update the variables in place; rebinding the attributes (the
            # original bug) leaves the weights used by the loss unchanged.
            K.set_value(self.alpha, final[0])
            K.set_value(self.beta, final[1])
            self.switched_at_epoch = epoch
            print(f"\nLoss weights switched to {final} after epoch {epoch + 1}")


class QuantileLossModel:
    """Builds the DeepGaR model from a config dictionary."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._num_features = config["NUM_FEATURES"]
        self._quantiles = config["QUANTILES"]
        self._1_common_lstm_layer_num = config["1_COMMON_LSTM_NUM"]
        self._1_ind_lstm_layer_num = config["1_IND_LSTM_NUM"]
        self._1_ind_dense_num = config["1_IND_DENSE_NUM"]
        self._1_ind_dense_activation = config["1_IND_DENSE_ACTIVATION"]
        self._1_out_dense_activation = config["1_OUT_DENSE_ACTIVATION"]
        self._1_lr = config["1_LEARNING_RATE"]

    def create_deepgar_model(
        self, weighting: LossWeighting = EQUAL_WEIGHTING, seed: int = 1234
    ) -> tuple[keras.Model, Any, Any]:
        """Create and compile the model.

        Returns the model and the two loss-weight variables.
        """
        keras.backend.clear_session()

        tf.random.set_seed(seed)

        data_input = keras.Input(shape=(None, self._num_features), name="data_input")

        lstm_origin = layers.LSTM(self._1_common_lstm_layer_num, return_sequences=True)(data_input)

        out_layers = []
        for q in self._quantiles:
            lstm_layer1 = layers.Dense(self._1_ind_lstm_layer_num, activation="tanh")(lstm_origin)
            dense_layer = layers.Dense(
                self._1_ind_dense_num, activation=self._1_ind_dense_activation
            )(lstm_layer1)
            out_layer = layers.Dense(
                1, activation=self._1_out_dense_activation, name=f"q{int(q * 100)}"
            )(dense_layer)
            out_layers.append(out_layer)

        quantiles_layer = tf.concat(out_layers, 1)

        lstm_layer2 = layers.LSTM(self._1_ind_lstm_layer_num)(lstm_origin)

        mu = layers.Dense(1, activation="linear", name="mu")(lstm_layer2)
        sigma = layers.Dense(1, activation=lambda x: tf.nn.elu(x) + 1, name="sigma")(lstm_layer2)
        skewness = layers.Dense(1, activation="linear", name="skewness")(lstm_layer2)
        tailweight = layers.Dense(1, activation=lambda x: tf.nn.elu(x) + 1, name="tailweight")(
            lstm_layer2
        )

        y_real = keras.Input(shape=(1,))
        loss_f = ls_quantile_cdf(mu, sigma, skewness, tailweight, quantiles_layer, self._quantiles)
        loss_q = multi_tilted_loss_shared_target(self._quantiles, y_real, quantiles_layer)

        model = keras.Model(inputs=[data_input, y_real], outputs=[mu, sigma, skewness, tailweight])

        # The weights must multiply the losses directly: `compile(loss_weights=...)`
        # does not apply to losses added with `add_loss`.
        weighted = WeightedLossSum(weighting.initial, name="weighted_loss")
        model.add_loss(weighted([loss_q, loss_f]))

        model.compile(optimizer=keras.optimizers.Adam(self._1_lr))

        return model, weighted.alpha, weighted.beta


def train_deepgar_model(
    model_1: keras.Model,
    data: dict[str, Any],
    epochs: int,
    callbacks: list[keras.callbacks.Callback],
    verbose: int = 1,
) -> tuple[Any, Any, keras.callbacks.History, keras.Model]:
    """Fit the model and predict on the train and test sets."""
    history = model_1.fit(
        [data["X_TRAIN"], data["Y_TRAIN"]],
        validation_data=[[data["X_TEST"], data["Y_TEST"]]],
        epochs=epochs,
        batch_size=data["BATCH_SIZE"],
        verbose=verbose,
        callbacks=callbacks,
    )

    y_train_dist = model_1.predict([data["X_TRAIN"], data["Y_TRAIN"]], verbose=verbose)
    y_test_dist = model_1.predict([data["X_TEST"], data["Y_TEST"]], verbose=verbose)

    return y_train_dist, y_test_dist, history, model_1


def model_deepgar_skew(
    param: dict[str, Any],
    data: dict[str, Any],
    epochs: int,
    patience: int,
    weighting: LossWeighting = EQUAL_WEIGHTING,
    seed: int = 1234,
    verbose: int = 1,
) -> tuple[Any, Any, keras.callbacks.History, keras.Model]:
    """Build and train the DeepGaR model with early stopping.

    The default ``weighting`` reproduces the published results; see the module
    docstring for why it differs from the scheme described in the paper.
    """
    early_stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)
    model_creator_1 = QuantileLossModel(param)
    model_1, alpha, beta = model_creator_1.create_deepgar_model(weighting, seed)
    return train_deepgar_model(
        model_1,
        data,
        epochs,
        [WeightsUpdater(alpha, beta, epochs, weighting), early_stopper],
        verbose=verbose,
    )


def forecast(
    model: keras.Model,
    data: dict[str, Any],
    scaler: Scaler,
    n_samples: int = 200,
    seed: int | None = None,
) -> Forecast:
    """Predictive SinhArcsinh distribution for the test quarters, in percent."""
    m, s, a, t = (
        np.squeeze(np.asarray(v)).astype(np.float32)
        for v in model([data["X_TEST"], data["Y_TEST"]])
    )
    # SinhArcsinh is a location-scale family, so converting location and scale
    # converts the whole distribution to percent.
    dist = tfp.distributions.SinhArcsinh(
        loc=to_percent(m, scaler).astype(np.float32),
        scale=scale_to_percent(s, scaler).astype(np.float32),
        skewness=a,
        tailweight=t,
    )
    return Forecast(
        median=np.asarray(dist.quantile(0.5), dtype=np.float64),
        lower=np.asarray(dist.quantile(0.05), dtype=np.float64),
        upper=np.asarray(dist.quantile(0.95), dtype=np.float64),
        samples=np.asarray(dist.sample(n_samples, seed=seed), dtype=np.float64),
    )
