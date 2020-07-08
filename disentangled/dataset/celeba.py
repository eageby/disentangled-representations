import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from . import utils
from ._dataset import Dataset

class CelebA(Dataset):
    """CelebA Dataset"""

    _name = "celeb_a"
    __version__ = "2.0.1"

    _builder = tfds.builder("{}:{}".format(_name, __version__))

    factors = [
        'lefteye_x',
        'lefteye_y',
        'leftmouth_x',
        'leftmouth_y',
        'nose_x',
        'nose_y',
        'righteye_x',
        'righteye_y',
        'rightmouth_x',
        'rightmouth_y'
    ]
    num_values_per_factor = [10, 10, 10, 8, 4, 15]

    @staticmethod
    @gin.configurable(module="CelebA")
    def pipeline(batch_size, prefetch_batches=1, num_parallel_calls=tf.data.experimental.AUTOTUNE, shuffle=None):
        dataset = (
            CelebA.load()
            .map(utils.get_image, num_parallel_calls=num_parallel_calls)
            .batch(batch_size)
            .map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls)
            .prefetch(prefetch_batches)
        )

        if shuffle is None:
            return dataset

        return shuffle(dataset)

    @staticmethod
    @gin.configurable(module="CelebA")
    def supervised(num_parallel_calls, shuffle=None):
        dataset = (
            CelebA.load()
            .map(CelebA.label_map, num_parallel_calls=num_parallel_calls)
            .map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls)
        )

        if shuffle is None:
            return dataset

        return shuffle(dataset)

    @staticmethod
    def label_map(element):
        labels = tf.convert_to_tensor(
            [element[f] for f in CelebA.factors], dtype=tf.uint8
        )

        return {"image": element["image"], "label": labels}

gin.constant('CelebA.num_values_per_factor', CelebA.num_values_per_factor)
