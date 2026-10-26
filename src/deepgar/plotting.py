"""Chart helpers for training diagnostics."""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def loss_figure(history1: Any) -> tuple[Figure, Axes]:
    """Plot training and validation loss per epoch."""
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot()
    loss = history1.history["loss"]
    val_loss = history1.history["val_loss"]
    epochs = range(len(loss))
    ax1.plot(epochs, loss, "b")
    ax1.plot(epochs, val_loss, "g")
    ax1.set_title(
        "Training and Validation Loss - Simultaneous Quantiles Generation & Crossing Loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(["Training loss", "Validation loss"])

    return fig1, ax1
