import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import disentangled.dataset.utils as utils

"""Disable toolbar in matplotlib"""
matplotlib.rcParams["toolbar"] = "None"

plt.rcParams["figure.figsize"] = [2 * i for i in plt.rcParams["figure.figsize"]]

def batch_to_grid(images, rows=4, cols=8):
    images = images[: rows * cols]
    cols = np.array_split(images, rows, axis=0)

    return np.stack(cols, axis=0)


def results(target, encoded, rows=4, cols=8, **kwargs):
    target = batch_to_grid(target, int(rows / 2), cols)
    encoded = batch_to_grid(encoded, int(rows / 2), cols)

    target_rows = np.split(target, target.shape[0], axis=0)
    encoded_rows = np.split(encoded, encoded.shape[0], axis=0)

    image_rows = [
        element for pair in zip(target_rows, encoded_rows) for element in pair
    ]
    show_grid(np.concatenate(image_rows, axis=0), **kwargs)


def show_grid(images, title=None):
    rows, cols = images.shape[:2]

    fig, axes = plt.subplots(rows, cols)

    for idx in np.ndindex(images.shape[:2]):
        axes[idx].imshow(np.squeeze(images[idx]), cmap="gray")
        axes[idx].axis("off")
        axes[idx].axis("tight")
        axes[idx].set_aspect("equal", adjustable="box")

    w, h = plt.figaspect(rows / cols)
    fig.set_size_inches(w, h)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.01, hspace=0.01)

    if title is not None:
        fig.canvas.set_window_title(title)

    # ax.imshow(image, cmap='gray')
    # ax.axis('tight')

    plt.show()


def show(image, title=None):
    fig, axes = plt.subplots(1, 1)

    axes.imshow(np.squeeze(image), cmap="gray")
    axes.axis("off")
    axes.axis("tight")
    axes.set_aspect("equal", adjustable="box")

    w, h = plt.figaspect(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.01, hspace=0.01)

    if title is not None:
        fig.canvas.set_window_title(title)

    plt.show()
