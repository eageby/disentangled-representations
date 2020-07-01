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
        "label_floor_hue",
        "label_wall_hue",
        "label_object_hue",
        "label_scale",
        "label_shape",
        "label_orientation",
    ]
    num_values_per_factor = [10, 10, 10, 8, 4, 15]

    @staticmethod
    @gin.configurable(module="CelebA")
    def pipeline(batch_size, prefetch_batches, num_parallel_calls, shuffle=None):
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
        raise NotImplementedError
        labels = tf.convert_to_tensor(
            [element[f] for f in CelebA.factors], dtype=tf.uint8
        )

        return {"image": element["image"], "label": labels}

    class ordered:
        @staticmethod
        @gin.configurable(module='CelebA.ordered')
        def load(num_parallel_calls):
            raise NotImplementedError
            return serialize.read(
                serialize.raw_datasets.CelebA, num_parallel_calls
            ).map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls)
