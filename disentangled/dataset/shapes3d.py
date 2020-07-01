import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from . import serialize, utils
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
    def supervised(num_parallel_calls, shuffle=None):
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

    class ordered:
        @staticmethod
        @gin.configurable(module='Shapes3d.ordered')
        def load(num_parallel_calls, shuffle=None):
            dataset = serialize.read(
                serialize.raw_datasets.Shapes3d, num_parallel_calls
            ).map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls)
            
            if shuffle is None:
                return dataset

            return shuffle(dataset)


gin.constant('Shapes3d.num_values_per_factor', Shapes3d.num_values_per_factor)
