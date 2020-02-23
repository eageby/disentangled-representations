import sys

import tensorflow as tf
import tensorflow_datasets as tfds

from . import utils


def get_dataset(name):
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

        return cls._builder.as_dataset(split=split)

    @classmethod
    def show(cls, **kwargs):
        """Shows examples of the dataset"""

        return tfds.show_examples(
            cls.info, cls._builder.as_dataset(split="train"), **kwargs
        )

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
    def pipeline(cls, batch_size=128):
        return (
            cls.load().map(dataset.utils.get_image).map(dataset.utils.binary).batch(128)
        )


class Shapes3d(Dataset):
    """Shapes3d Dataset"""

    _name = "shapes3d"
    __version__ = "2.0.0"

    _builder = tfds.builder("{}:{}".format(_name, __version__))

    @classmethod
    def pipeline(cls, batch_size=128):
        return (
            cls.load().batch(batch_size).map(utils.get_image).map(utils.normalize_uint8)
        )
