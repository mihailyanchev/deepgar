"""DeepGaR: LSTM quantile network with a SinhArcsinh density head."""

from typing import Any

import tensorflow as tf
import tf_keras as keras
from tf_keras import backend as K
from tf_keras import layers

from deepgar.losses import ls_quantile_cdf, multi_tilted_loss_v2


class WeightsUpdater(keras.callbacks.Callback):  # type: ignore[misc]
    """Switch the loss weights near the end of training."""

    def __init__(self, alpha: Any, beta: Any, epochs: int) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.total_epochs = epochs
        self.coef = self.alpha / self.total_epochs

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if epoch > (int(self.total_epochs * 0.9)):
            self.alpha = 0.05
            self.beta = 0.95

        print("\nLoss_weights used:", K.get_value(self.alpha), K.get_value(self.beta))


class QuantileLossModel:
    """Builds the DeepGaR model from a config dictionary."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._num_features = config["NUM_FEATURES"]
        self._quantiles = config["QUANTILES"]
        self._1_common_lstm_layer_num = config["1_COMMON_LSTM_NUM"]
        self._2_common_lstm_layer_num = config["2_COMMON_LSTM_NUM"]
        self._1_ind_lstm_layer_num = config["1_IND_LSTM_NUM"]
        self._1_ind_dense_num = config["1_IND_DENSE_NUM"]
        self._1_ind_dense_activation = config["1_IND_DENSE_ACTIVATION"]
        self._1_out_dense_activation = config["1_OUT_DENSE_ACTIVATION"]
        self._1_lr = config["1_LEARNING_RATE"]
        self._pred_step = config["PRED_STEP"]

    def create_deepgar_skew_v2(self) -> tuple[keras.Model, Any, Any]:
        """Create and compile the model.

        Returns the model and the two loss-weight variables.
        """
        keras.backend.clear_session()

        tf.random.set_seed(1234)

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
        loss_q = multi_tilted_loss_v2(self._quantiles, y_real, quantiles_layer)

        model = keras.Model(inputs=[data_input, y_real], outputs=[mu, sigma, skewness, tailweight])

        model.add_loss([loss_q, loss_f])

        alpha = K.variable(0.95)
        beta = K.variable(0.05)

        model.compile(optimizer=keras.optimizers.Adam(self._1_lr), loss_weights=[alpha, beta])

        return model, alpha, beta


def train_tfp_model_norm2(
    model_1: keras.Model,
    data: dict[str, Any],
    epochs: int,
    callbacks: list[keras.callbacks.Callback],
) -> tuple[Any, Any, keras.callbacks.History, keras.Model]:
    """Fit the model and predict on the train and test sets."""
    history = model_1.fit(
        [data["X_TRAIN"], data["Y_TRAIN"]],
        validation_data=[[data["X_TEST"], data["Y_TEST"]]],
        epochs=epochs,
        batch_size=data["BATCH_SIZE"],
        verbose=1,
        callbacks=callbacks,
    )

    y_train_dist = model_1.predict([data["X_TRAIN"], data["Y_TRAIN"]])
    y_test_dist = model_1.predict([data["X_TEST"], data["Y_TEST"]])

    return y_train_dist, y_test_dist, history, model_1


def model_deepgar_skew_v2(
    param: dict[str, Any], data: dict[str, Any], epochs: int, patience: int
) -> tuple[Any, Any, keras.callbacks.History, keras.Model]:
    """Build and train the DeepGaR model with early stopping."""
    early_stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)
    model_creator_1 = QuantileLossModel(param)
    model_1, alpha, beta = model_creator_1.create_deepgar_skew_v2()
    return train_tfp_model_norm2(
        model_1, data, epochs, [WeightsUpdater(alpha, beta, epochs), early_stopper]
    )
