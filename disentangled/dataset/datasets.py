import itertools
import resource
import sys

import gin
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

    @staticmethod
    def pipeline(split="train", batch_size=128):
        return (
            MNIST.load(split)
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

    @staticmethod
    @gin.configurable(module="Shapes3d")
    def pipeline(batch_size, prefetch_batches, num_parallel_calls, shuffle=None):
        dataset = (
            Shapes3d.load()
            .map(utils.get_image, num_parallel_calls=num_parallel_calls)
            .batch(batch_size)
            .map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls)
            .prefetch(prefetch_batches)
            )
    
        if shuffle is None:
            return dataset

        return shuffle(dataset)

    @staticmethod
    @gin.configurable(module='Shapes3d')
    def supervised(num_parallel_calls, shuffle=None):
        dataset = (
            Shapes3d.load()
            .map(Shapes3d.label_map, num_parallel_calls=num_parallel_calls)
            .map(
                utils.normalize_uint8, num_parallel_calls=num_parallel_calls
            )
        )

        if shuffle is None:
            return dataset

        return shuffle(dataset)


    @staticmethod
    def label_map(element):
        labels = tf.convert_to_tensor(
            [element[f] for f in Shapes3d.factors], dtype=tf.uint8
        )

        return {"image": element["image"], "label": labels}

    class ordered:
        @classmethod
        def load(cls):
            return serialize.read(serialize.raw_datasets.Shapes3d).map(
                utils.normalize_uint8, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
