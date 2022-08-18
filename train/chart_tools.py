'''
Script containing functions
for chart creation
including for mlflow
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm

def loss_figure(history1):
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot()
    loss = history1.history['loss']
    val_loss = history1.history['val_loss']
    epochs=range(len(loss))
    #plt.figure(figsize=(10, 6))
    ax1.plot(epochs, loss, 'b')
    ax1.plot(epochs, val_loss, 'g')
    ax1.set_title('Training and Validation Loss - Simultaneous Quantiles Generation & Crossing Loss')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(["Training loss", "Validation loss"])
    
    return fig1, ax1

def train_dist_figure(y_train_dist, quantiles, data):
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot()
    N = 6
    #ax1.rcParams["axes.prop_cycle"] = get_cycle("winter", N)
    plt.plot(data['Y_TRAIN'].squeeze(), label='GDP', color='purple')
    for i, p in enumerate(y_train_dist.transpose()):
        ax1.plot(p, label='{}th Quantile'.format(int(np.flip(quantiles)[i]*100)))
    ax1.margins(0.01, 0.1)
    ax1.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.1)
    ax1.legend()
    
    return fig1, ax1

def test_dist_figure(y_test_dist, quantiles, data):
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot()
    #plt.set_title('Test Set - Actual and Distribution')
    plt.plot(data['Y_TEST'].squeeze(), label='GDP', color='purple')
    for i, p in enumerate(y_test_dist.transpose()):
        plt.plot(p, label='{}th Quantile'.format(int(np.flip(quantiles)[i]*100)))

    plt.margins(0.05, 0.1)
    plt.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.1)
    plt.legend()

    return fig1, ax1

def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(0,1,N))
        return cycler("color",colors)
