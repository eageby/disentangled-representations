import itertools
import resource
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from . import serialize, utils

# Handles Too many open files error
# https://github.com/tensorflow/datasets/issues/1441
_, _high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (_high, _high))


def get(name):
    return getattr(sys.modules[__name__], name)


class Dataset:
    """Abstract Base Class for datasets, interfacing the tensorflow_datasets API"""

    @classmethod
    def load(cls, split="train") -> tf.data.Dataset:
        """Loads an instance of the tensorflow dataset.

        Args:
            split:  From tensorflow_datasets doc https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetBuilder#as_dataset.
                    Which subset(s) of the data to read.
                    If None (default), returns all splits in a dict <key: tfds.Split, value: tf.data.Dataset>.
        Returns:
            tf.data.Dataset
        """

        cls._builder.download_and_prepare()

        return cls._builder.as_dataset(split=split)

    @classmethod
    def version(cls):
        """Returns the version used."""

        return cls.__version__

    @classmethod
    def info(cls):
        """Returns information about the dataset. """

        return cls._builder.info


class MNIST(Dataset):
    """MNIST Dataset"""

    _name = "mnist"
    __version__ = "3.0.0"

    _builder = tfds.builder("{}:{}".format(_name, __version__))

    @classmethod
    def pipeline(cls, split="train", batch_size=128):
        return (
            cls.load(split)
            .map(utils.get_image)
            .map(utils.normalize_uint8)
            .batch(batch_size)
        )


class Shapes3d(Dataset):
    """Shapes3d Dataset"""

    _name = "shapes3d"
    __version__ = "2.0.0"

    _builder = tfds.builder("{}:{}".format(_name, __version__))

    factors = [
        "label_floor_hue",
        "label_wall_hue",
        "label_object_hue",
        "label_scale",
        "label_shape",
        "label_orientation",
    ]
    num_values_per_factor = [10, 10, 10, 8, 4, 15]

    @classmethod
    def pipeline(cls, batch_size=64, prefetch_batches=10):
        return (
            cls.load()
            .map(utils.get_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size)
            .map(
                utils.normalize_uint8, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .prefetch(prefetch_batches)
        )

    @classmethod
    def as_image_label(cls):
        return (
            cls.load()
            .map(cls.label_map)
            .map(
                utils.normalize_uint8, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        )

    @classmethod
    def label_map(cls, element):
        labels = tf.convert_to_tensor([element[f]
                                       for f in cls.factors], dtype=tf.uint8)

        return {"image": element["image"], "label": labels}

    class ordered:
        @classmethod
        def load(cls):
            return serialize.read(serialize.raw_datasets.Shapes3d).map(
                utils.normalize_uint8, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
