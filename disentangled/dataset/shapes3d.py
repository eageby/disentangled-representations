import gin
import tensorflow as tf
import tensorflow_datasets as tfds
import h5py
import numpy as np

import disentangled.utils
from . import utils
from ._dataset import Dataset

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
    def pipeline(batch_size=64, prefetch_batches=1, num_parallel_calls=tf.data.experimental.AUTOTUNE, shuffle=None):
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
    @gin.configurable(module="Shapes3d")
    def supervised(num_parallel_calls=tf.data.experimental.AUTOTUNE, shuffle=None):
        dataset = (
            Shapes3d.load()
            .map(Shapes3d.label_map, num_parallel_calls=num_parallel_calls)
            .map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls)
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

    @staticmethod
    @gin.configurable(module="Shapes3d")
    def ordered(chunk_size=1000, prefetch_batches=1, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        fname = 'shapes3d.h5'
        file = disentangled.utils.get_data_path()/ 'downloads' / fname
        # file = tf.keras.utils.get_file(str(path), 'https://storage.cloud.google.com/3d-shapes/3dshapes.h5')
        
        def generator():
            with h5py.File(file, 'r') as data:
                chunk_idx = np.arange(0, len(data['images']), chunk_size)
                for i in range(len(chunk_idx)-1):
                    start = chunk_idx[i]
                    end = chunk_idx[i+1]
                    for im in data["images"][start:end]:
                        yield im   

        return tf.data.Dataset.from_generator(
                generator,
                tf.uint8,
                tf.TensorShape([64,64,3])
                ).map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls).prefetch(prefetch_batches)
gin.constant('Shapes3d.num_values_per_factor', Shapes3d.num_values_per_factor)
