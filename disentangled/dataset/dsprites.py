import gin
import tensorflow as tf
import tensorflow_datasets as tfds
import h5py
import numpy as np

import disentangled.utils
from . import utils
from ._dataset import Dataset

class DSprites(Dataset):
    """dSprites Dataset"""

    _name = "dsprites"
    __version__ = "2.0.0"

    _builder = tfds.builder("{}:{}".format(_name, __version__))

    factors = [
        'label_orientation',
        'label_scale',
        'label_shape',
        'label_x_position',
        'label_y_position',
    ]
    num_values_per_factor = [40, 6, 3, 32, 32]

    @staticmethod
    @gin.configurable(module="DSprites")
    def pipeline(batch_size, prefetch_batches=1, num_parallel_calls=tf.data.experimental.AUTOTUNE, shuffle=None):
        dataset = (
            DSprites.load()
            .map(utils.get_image, num_parallel_calls=num_parallel_calls)
            .batch(batch_size)
            .prefetch(prefetch_batches)
        )

        if shuffle is None:
            return dataset

        return shuffle(dataset)

    @staticmethod
    @gin.configurable(module="DSprites")
    def ordered(chunk_size=1000, prefetch_batches=1, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        fname = 'dsprites.hdf5'
        file = disentangled.utils.get_data_path()/ 'downloads' / fname
        # file = tf.keras.utils.get_file(str(path), 'https://storage.cloud.google.com/3d-shapes/3dshapes.h5')
        
        def generator():
            with h5py.File(file, 'r') as data:
                chunk_idx = np.arange(0, len(data['imgs'])+1, chunk_size)
                chunk_idx = np.concatenate([chunk_idx, [len(data['imgs'])]])
                for i in range(len(chunk_idx)-1):
                    start = chunk_idx[i]
                    end = chunk_idx[i+1]
                    for im in data["imgs"][start:end]:
                        yield np.expand_dims(im,axis=-1)

        return tf.data.Dataset.from_generator(
                generator,
                tf.uint8,
                tf.TensorShape([64,64,1]),
                ).map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls).prefetch(prefetch_batches)

    @staticmethod
    @gin.configurable(module="DSprites")
    def supervised(num_parallel_calls=tf.data.experimental.AUTOTUNE, shuffle=None):
        dataset = (
            DSprites.load()
            .map(utils.image_float32, num_parallel_calls=num_parallel_calls)
            .map(DSprites.label_map, num_parallel_calls=num_parallel_calls)
        )

        if shuffle is None:
            return dataset

        return shuffle(dataset)


    @staticmethod
    def label_map(element):
        labels = tf.convert_to_tensor(
            [element[f] for f in DSprites.factors], dtype=tf.uint8
        )

        return {"image": element["image"], "label": labels}

gin.constant('DSprites.num_values_per_factor', DSprites.num_values_per_factor)
