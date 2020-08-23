import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from . import utils
from ._dataset import Dataset

class CelebA(Dataset):
    """CelebA Dataset"""

    _name = "celeb_a"
    __version__ = "2.0.0"

    _builder = tfds.builder("{}:{}".format(_name, __version__))

    @staticmethod
    def resize(x):
        return tf.image.resize(x, [64, 64])

    @staticmethod
    @gin.configurable(module="CelebA")
    def pipeline(batch_size, prefetch_batches=1, num_parallel_calls=tf.data.experimental.AUTOTUNE, shuffle=None):
        dataset = (
            CelebA.load()
            .map(utils.get_image, num_parallel_calls=num_parallel_calls)
            .map(CelebA.resize, num_parallel_calls=num_parallel_calls)
            .batch(batch_size)
            .map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls)
            .prefetch(prefetch_batches)
        )

        if shuffle is None:
            return dataset

        return shuffle(dataset)
