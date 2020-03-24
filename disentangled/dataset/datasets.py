import resource
import sys
from pathlib import Path
import numpy as np
import h5py

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio

from . import utils

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
            #as_dataset.
            split:  From tensorflow_datasets doc https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetBuilder
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
    num_values_per_factor = [ 10, 10, 10, 8, 4, 15 ]

    @classmethod
    def pipeline(cls, split="train", batch_size=64, prefetch_batches=5):
        return (
            cls.load()
            .batch(batch_size)
            .prefetch(prefetch_batches)
            .map(utils.get_image)
            .map(utils.normalize_uint8)
        )

    @classmethod
    def as_image_label(cls):
        return cls.load().map(cls.label_map)

    @classmethod
    def label_map(cls, element):
        labels = tf.convert_to_tensor([element[f]
                                       for f in cls.factors], dtype=tf.uint8)

        return {"image": element["image"] / 255, "label": labels}

class Shapes3d_ordered(Shapes3d):
    path = tf.keras.utils.get_file(
        '3dshapes.h5', 'https://storage.googleapis.com/3d-shapes/3dshapes.h5')

    images = h5py.File(path, 'r')['images']

    def load(self):
        images = tfio.IODataset.from_hdf5(self.path, '/images')
        labels = tfio.IODataset.from_hdf5(self.path, '/labels')

        def make_dict(image, label):
            dataset = {}
            dataset['image'] = image
            dataset['label'] = label

            return dataset

        return tf.data.Dataset.zip((images, labels)).map(make_dict, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def get_index(self, factors):
        """ Converts factors to indices in range(num_data)
        Args:
        factors: np array shape [6,batch_size].
                 factors[i]=factors[i,:] takes integer values in
                 range(_NUM_VALUES_PER_FACTOR[self.factors[i]]).

        Returns:
        indices: np array shape [batch_size].
        """
        indices = 0
        base = 1

        for factor, _ in reversed(list(enumerate(self.factors))):
            indices += factors[factor] * base
            base *= self.num_values_per_factor[factor]

        return indices

    def batch_indices(self, batch_size):
        def inner(fixed_factor, fixed_factor_value):
            factors = [np.random.choice(self.num_values_per_factor[f], batch_size) for f, _ in enumerate(self.factors)]
            factors[fixed_factor] = fixed_factor_value

            return self.get_index(factors)
        return inner

    def create_index(self, batches, batch_size):
        fixed_factors = np.random.choice(len(self.factors), batches)
        fixed_factors_value = [np.random.choice(self.num_values_per_factor[i]) for i in fixed_factors]
        
        idx = [self.batch_indices(batch_size)(i,j) for i,j in zip(fixed_factors, fixed_factors_value)]
        return tf.data.Dataset.from_tensor_slices(idx)


    @classmethod
    def read(cls):
        data = h5py.File(cls.path, 'r')
        def inner(idx):
            import pdb;pdb.set_trace()
            return data['images'][idx]
        return inner 
