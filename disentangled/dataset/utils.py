__all__ = ["binary", "get_image", "numpy"]

import numpy as np
import tensorflow as tf


def binary(element):
    """Binarizes the element in the dataset.
    
    Requires that the dataset is populated with only one element.
    Recommended usage is to call get_image first, as dataset.map(get_image).map(binary).

    Args:
        element: Element to be mapped

    Returns:
        Binary quantized element
    """

    return tf.quantization.fake_quant_with_min_max_args(
        tf.cast(element, tf.float32), min=0, max=1
    )


def get_image(element):
    """Gets the image feature of the element in a dataset."""

    return tf.cast(element["image"], tf.float32)


def numpy(dataset):
    """Converts a dataset to a numpy array.

    Requires that the dataset is populated with only one element.
    Batch dimension at axis 0.

    Args:
        dataset (tf.data.Dataset): An instance of a dataset.

    Returns:
        np.array
    """

    return np.asarray(list(dataset.as_numpy_iterator()))


def normalize_uint8(x):
    """Normalizes data in uint8 range [0,255] to [0,1]"""
    if isinstance(x, dict):
        x["image"] /= 255
        return x

    return x / 255
