import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from . import serialize, utils
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
    def pipeline(batch_size, prefetch_batches, num_parallel_calls, shuffle=None):
        dataset = (
            DSprites.load()
            .map(utils.get_image, num_parallel_calls=num_parallel_calls)
            .batch(batch_size)
            .map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls)
            .prefetch(prefetch_batches)
        )

        if shuffle is None:
            return dataset

        return shuffle(dataset)

    @staticmethod
    @gin.configurable(module="DSprites")
    def supervised(num_parallel_calls, shuffle=None):
        dataset = (
            DSprites.load()
            .map(DSprites.label_map, num_parallel_calls=num_parallel_calls)
            .map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls)
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

    class ordered:
        @staticmethod
        @gin.configurable(module='DSprites.ordered')
        def load(num_parallel_calls, shuffle=None):

            dataset = serialize.read(
                serialize.raw_datasets.DSprites, num_parallel_calls
            ).map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls)

            if shuffle is not None:
                return shuffle(dataset)

            return dataset

