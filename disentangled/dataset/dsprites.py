import gin
import tensorflow as tf
import tensorflow_datasets as tfds

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
