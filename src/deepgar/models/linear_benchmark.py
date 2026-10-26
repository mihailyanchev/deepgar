"""Linear quantile regression benchmark trained with the tilted loss."""

from typing import Any

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers

from deepgar.losses import multi_tilted_loss


class QuantileLossModel:
    """Builds the linear quantile benchmark from a config dictionary."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._num_features = config["NUM_FEATURES"]
        self._quantiles = config["QUANTILES"]
        self._1_out_dense_activation = config["1_OUT_DENSE_ACTIVATION"]
        self._1_lr = config["1_LEARNING_RATE"]

    def create_linear_model(self, seed: int = 1234) -> keras.Model:
        """Create and compile one linear output per quantile."""
        keras.backend.clear_session()

        tf.random.set_seed(seed)

        data_input = keras.Input(shape=(None, self._num_features), name="data_input")

        out_layers = []
        for q in self._quantiles:
            out_layer = layers.Dense(
                1, activation=self._1_out_dense_activation, name=f"q{int(q * 100)}"
            )(data_input)
            out_layers.append(out_layer)

        final_output = tf.concat(out_layers, 1)

        model = keras.Model(inputs=[data_input], outputs=[final_output])

        model.compile(
            optimizer=keras.optimizers.Adam(self._1_lr),
            loss=lambda y, f: multi_tilted_loss(self._quantiles, y, f),
            metrics=["mae", "mse"],
        )

        return model


def train_1step_model(
    model_1: keras.Model, data: dict[str, Any], epochs: int, verbose: int = 2
) -> tuple[Any, Any, keras.callbacks.History, keras.Model]:
    """Fit the model and predict on the train and test sets."""
    history = model_1.fit(
        {"data_input": data["X_TRAIN"]},
        {"tf.concat": data["Y_TRAINK"]},
        validation_data=(data["X_TEST"], data["Y_TESTK"]),
        epochs=epochs,
        batch_size=data["BATCH_SIZE"],
        verbose=verbose,
    )

    y_train_dist = model_1.predict(data["X_TRAIN"], verbose=verbose)
    y_test_dist = model_1.predict(data["X_TEST"], verbose=verbose)

    return y_train_dist, y_test_dist, history, model_1


def model_linear(
    param: dict[str, Any],
    data: dict[str, Any],
    epochs: int,
    seed: int = 1234,
    verbose: int = 2,
) -> tuple[Any, Any, keras.callbacks.History, keras.Model]:
    """Build and train the linear quantile benchmark."""
    model_creator_1 = QuantileLossModel(param)
    model_1 = model_creator_1.create_linear_model(seed)
    return train_1step_model(model_1, data, epochs, verbose=verbose)
