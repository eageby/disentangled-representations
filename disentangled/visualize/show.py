import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gin

import disentangled.dataset.utils as utils
import disentangled.utils 

"""Disable toolbar in matplotlib"""
matplotlib.rcParams["toolbar"] = "None"

plt.rcParams["figure.figsize"] = [2 * i for i in plt.rcParams["figure.figsize"]]


def batch_to_grid(images, rows, cols):
    images = images[: rows * cols]
    cols = np.array_split(images, rows, axis=0)

    return np.stack(cols, axis=0)



def comparison(target, encoded, rows, cols, **kwargs):
    """Shows input and output, alternating by row, input first."""
    target = batch_to_grid(target, int(rows / 2), cols)
    encoded = batch_to_grid(encoded, int(rows / 2), cols)

    target_rows = np.split(target, target.shape[0], axis=0)
    encoded_rows = np.split(encoded, encoded.shape[0], axis=0)

    image_rows = [
        element for pair in zip(target_rows, encoded_rows) for element in pair
    ]
    show_grid(np.concatenate(image_rows, axis=0), **kwargs)

def grid(target, rows, cols, **kwargs):
    show_grid(batch_to_grid(target, rows, cols), **kwargs)
        

def show_grid(images, title=None):
    rows, cols = images.shape[:2]

    if (rows == 1 and cols == 1):
        show(images)
        return

    fig, axes = plt.subplots(rows, cols)
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    elif cols == 1:
        axes = np.expand_dims(axes, 1)
           
    for idx in np.ndindex(images.shape[:2]):
        axes[idx].imshow(np.squeeze(images[idx]), cmap="gray")
        axes[idx].axis("off")
        axes[idx].axis("tight")
        axes[idx].set_aspect("equal", adjustable="box")

    w, h = plt.figaspect(rows / cols)
    pixel_width = images.shape[2]
    pixel_heigth = images.shape[3]

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.01, hspace=0.01)

    dpi = float(fig.get_dpi())
    fig.set_size_inches(cols*dpi/pixel_width, rows*dpi/pixel_heigth)

    output(plt, show_plot=gin.REQUIRED)


def show(image):
    fig, axes = plt.subplots(1, 1)

    axes.imshow(np.squeeze(image), cmap="gray")
    axes.axis("off")
    axes.axis("tight")
    axes.set_aspect("equal", adjustable="box")

    w, h = plt.figaspect(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.01, hspace=0.01)

    output(plt, show_plot=gin.REQUIRED)

@gin.configurable(whitelist=['show_plot', 'filename', 'format'])
def output(plot, show_plot, filename=None, format='png'):
    if filename is not None:
        save_dir = disentangled.utils.get_data_path().resolve() / 'images' / (filename + '.' + format)
        save_dir.parent.mkdir(exist_ok=True, parents=True)
        plot.savefig(save_dir, format=format)

    if show_plot:
        plot.show()
