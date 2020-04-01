import resource
import sys
import itertools
from pathlib import Path
import numpy as np
import h5py

import tensorflow as tf
import tensorflow_datasets as tfds

from . import utils
from . import serialize

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

    @classmethod
    def pipeline(cls, split="train", batch_size=64, prefetch_batches=10):
        return (
            cls.load()
            .map(utils.get_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size)
            .map(utils.normalize_uint8, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .prefetch(prefetch_batches)
        )

    class Ordered:
        factors = [
            "label_floor_hue",
            "label_wall_hue",
            "label_object_hue",
            "label_scale",
            "label_shape",
            "label_orientation",
        ]
        num_values_per_factor = [ 10, 10, 10, 8, 4, 15 ]

        path = tf.keras.utils.get_file(
            '3dshapes.h5', 'https://storage.googleapis.com/3d-shapes/3dshapes.h5')

        @classmethod
        def get_index(cls, factors):
            """ Converts factors to indices in range(num_data)
            Args:
            factors: np array shape [6,batch_size].
                     factors[i]=factors[i,:] takes integer values in
                     range(_NUM_VALUES_PER_FACTOR[cls.factors[i]]).

            Returns:
            indices: np array shape [batch_size].
            """
            indices = 0
            base = 1

            for factor, _ in reversed(list(enumerate(cls.factors))):
                indices += factors[factor] * base
                base *= cls.num_values_per_factor[factor]

            return indices

        @classmethod
        def batch_indices(cls, batch_size):
            factors = [np.random.choice(cls.num_values_per_factor[f], batch_size) for f, _ in enumerate(cls.factors)]

            fixed_factor = np.random.choice(len(cls.factors))
            fixed_factor_value = np.random.choice(cls.num_values_per_factor[fixed_factor])
            factors[fixed_factor] = fixed_factor_value
            return cls.get_index(factors), fixed_factor, fixed_factor_value

        @classmethod
        def generator(cls, batch_size):
            for _ in itertools.count():
                idx, factor, value = cls.batch_indices(batch_size)
                idx = np.sort(idx) 
                yield idx, factor, value 

        @classmethod
        def read(cls):
            data = h5py.File(cls.path, 'r', swmr=True)['images']
            
            def wrapper(idx, factor, value):
                def inner(idx, factor, value):
                    idx = np.asarray(idx)
                    
                    im = []
                    for i in idx:
                        im.append(data[i])
                    im = np.stack(im) 
                    return im / 255,  factor,  value
                return tf.py_function(inner, inp=[idx, factor, value], Tout=(tf.float32, tf.int64, tf.int64)) 

            return wrapper

        @classmethod
        def create(cls, batch_size):
            def dict_map(image, factor, value):
                image.set_shape((None, 64,64,3))
                factor.set_shape(())
                value.set_shape(())
                return {'image': image, 'factor':factor, 'factor_value': value}

            index = tf.data.Dataset.from_generator(cls.generator, (tf.int64, tf.int64, tf.int64), output_shapes=((None), (), ()), args=(batch_size,))
            dataset = index.map(cls.read())

            return dataset.map(dict_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        @staticmethod
        def example(element):
            features = {'image': serialize.bytes_feature(element['image']),
                        'factor': serialize.int64_feature(element['factor']),
                        'factor_value': serialize.int64_feature(element['factor_value'])}

            return tf.train.Example(features=tf.train.Features(feature=features))
        
        @staticmethod
        def parse_example(example):
            image_feature_description = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'factor': tf.io.FixedLenFeature([], tf.int64),
                'factor_value': tf.io.FixedLenFeature([], tf.int64),
            }
            return tf.io.parse_single_example(example, image_feature_description)
