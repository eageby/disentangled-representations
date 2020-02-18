__all__ = ['binary', 'get_image', 'numpy']

import numpy as np
import tensorflow as tf


def binary(element):
    return tf.quantization.fake_quant_with_min_max_args(
        tf.cast(element, tf.float32), min=0, max=1
    )

def get_image(element):
    return tf.cast(element["image"], tf.float32)


def numpy(dataset):
    return np.asarray(list(data.as_numpy_iterator()))
