import itertools
import sys
from pathlib import Path

import disentangled.dataset.serialize.utils as utils
import disentangled.utils
import gin
import h5py
import numpy as np
import tensorflow as tf


def get(name):
    return getattr(sys.modules[__name__], name)


class Shapes3d:
    factors = [
        "label_floor_hue",
        "label_wall_hue",
        "label_object_hue",
        "label_scale",
        "label_shape",
        "label_orientation",
    ]
    num_values_per_factor = [10, 10, 10, 8, 4, 15]

    data_base_dir = disentangled.utils.get_data_path()

    path = None

    @staticmethod
    @gin.configurable(module="disentangled.dataset.serialize.raw_datasets.Shapes3d")
    def get_serialized_path():
        return disentangled.utils.get_data_path() / "serialized" / "Shapes3d.tfrecords"

    def get_file_path():
        file_path = disentangled.utils.get_data_path() / "downloads/3dshapes.h5"
        file_path.parent.mkdir(exist_ok=True, parents=True)
        return file_path

    @staticmethod
    def get_index(factors):
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

        for factor, _ in reversed(list(enumerate(Shapes3d.factors))):
            indices += factors[factor] * base
            base *= Shapes3d.num_values_per_factor[factor]

        return indices

    @staticmethod
    def batch_indices(batch_size):
        factors = [
            np.random.choice(Shapes3d.num_values_per_factor[f], batch_size)
            for f, _ in enumerate(Shapes3d.factors)
        ]

        fixed_factor = np.random.choice(len(Shapes3d.factors))
        fixed_factor_value = np.random.choice(
            Shapes3d.num_values_per_factor[fixed_factor]
        )
        factors[fixed_factor] = fixed_factor_value

        return Shapes3d.get_index(factors), fixed_factor, fixed_factor_value

    @staticmethod
    def generator(batch_size):
        for _ in itertools.count():
            idx, factor, value = Shapes3d.batch_indices(batch_size)
            idx = np.sort(idx)
            yield idx, factor, value

    @staticmethod
    def read():
        data = h5py.File(Shapes3d.path, "r", swmr=True)["images"]

        def wrapper(idx, factor, value):
            def inner(idx, factor, value):
                idx = np.asarray(idx)

                im = []

                for i in idx:
                    im.append(data[i])
                im = np.stack(im)

                return im, factor, value

            return tf.py_function(
                inner, inp=[idx, factor, value], Tout=(tf.float32, tf.int64, tf.int64)
            )

        return wrapper

    @staticmethod
    @gin.configurable(module="disentangled.dataset.serialize.raw_datasets.Shapes3d")
    def create(batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        if Shapes3d.path is None:
            Shapes3d.path = tf.keras.utils.get_file(
                str(Shapes3d.get_file_path().resolve()),
                "https://storage.googleapis.com/3d-shapes/3dshapes.h5",
            )

        def dict_map(image, factor, value):
            image.set_shape((None, 64, 64, 3))
            factor.set_shape(())
            value.set_shape(())

            return {"image": image, "factor": factor, "factor_value": value}

        index = tf.data.Dataset.from_generator(
            Shapes3d.generator,
            (tf.int64, tf.int64, tf.int64),
            output_shapes=((None), (), ()),
            args=(batch_size,),
        )
        dataset = index.map(Shapes3d.read())

        return dataset.map(dict_map, num_parallel_calls=num_parallel_calls)

    @staticmethod
    def example(element):
        features = {
            "image": utils.bytes_feature(element["image"]),
            "factor": utils.int64_feature(element["factor"]),
            "factor_value": utils.int64_feature(element["factor_value"]),
        }

        return tf.train.Example(features=tf.train.Features(feature=features))

    @staticmethod
    def parse_example(example):
        image_feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "factor": tf.io.FixedLenFeature([], tf.int64),
            "factor_value": tf.io.FixedLenFeature([], tf.int64),
        }

        return tf.io.parse_single_example(example, image_feature_description)

    @staticmethod
    def set_shapes(element):
        element["image"].set_shape((None, 64, 64, 3))
        element["factor"].set_shape(())
        element["factor_value"].set_shape(())

        return element
