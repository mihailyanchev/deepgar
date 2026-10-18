"""Chart helpers for training diagnostics and predicted distributions."""

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import Cycler, cycler
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure

QUALITATIVE_CMAPS = {
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
}


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


def train_dist_figure(
    y_train_dist: Any, quantiles: Any, data: dict[str, Any]
) -> tuple[Figure, Axes]:
    """Plot the observed train target against each predicted quantile."""
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot()
    plt.plot(data["Y_TRAIN"].squeeze(), label="GDP", color="purple")
    for i, p in enumerate(y_train_dist.transpose()):
        ax1.plot(p, label=f"{int(np.flip(quantiles)[i] * 100)}th Quantile")
    ax1.margins(0.01, 0.1)
    ax1.grid(color="black", linestyle="-", linewidth=0.5, alpha=0.1)
    ax1.legend()

    return fig1, ax1


def test_dist_figure(y_test_dist: Any, quantiles: Any, data: dict[str, Any]) -> tuple[Figure, Axes]:
    """Plot the observed test target against each predicted quantile."""
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot()
    plt.plot(data["Y_TEST"].squeeze(), label="GDP", color="purple")
    for i, p in enumerate(y_test_dist.transpose()):
        plt.plot(p, label=f"{int(np.flip(quantiles)[i] * 100)}th Quantile")

    plt.margins(0.05, 0.1)
    plt.grid(color="black", linestyle="-", linewidth=0.5, alpha=0.1)
    plt.legend()

    return fig1, ax1


def get_cycle(
    cmap: str | Colormap, N: int | None = None, use_index: bool | str = "auto"
) -> Cycler[str, Any]:
    """Build a colour cycler with ``N`` colours from a colormap."""
    if isinstance(cmap, str):
        if use_index == "auto":
            use_index = cmap in QUALITATIVE_CMAPS
        cmap = mpl.colormaps[cmap]
    if not N:
        N = cmap.N
    if use_index == "auto":
        if cmap.N > 100 or isinstance(cmap, LinearSegmentedColormap):
            use_index = False
        elif isinstance(cmap, ListedColormap):
            use_index = True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color", cmap(ind))
    colors = cmap(np.linspace(0, 1, N))
    return cycler("color", colors)
