import gin
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import disentangled.dataset.utils as utils
import disentangled.utils
from disentangled.dataset._dataset import Dataset

class CelebA(Dataset):
    """CelebA Dataset"""

    _name = "celeb_a"
    __version__ = "2.0.0"

    _builder = tfds.builder("{}:{}".format(_name, __version__))
    
    @staticmethod
    @gin.configurable(module="CelebA")
    def pipeline(batch_size, prefetch_batches=1, num_parallel_calls=tf.data.experimental.AUTOTUNE, shuffle=None):
        dataset = (
            CelebA.load(split="train+test+validation")
            .map(utils.get_image, num_parallel_calls=num_parallel_calls)
            .repeat()
            .batch(batch_size)
            .map(utils.normalize_uint8, num_parallel_calls=num_parallel_calls)
            .prefetch(prefetch_batches)
        )

        if shuffle is None:
            return dataset

        return shuffle(dataset)

    @staticmethod
    def attribute_distribution():
        def get_attribute(elem):
            return elem['attributes']
        initial = tf.zeros((1,41), dtype=tf.int64)
        def reduce_func(old_state, element):
            element = tf.cast(tf.expand_dims(tf.stack([i for i in element.values()]), axis=0), tf.int64)
            counter = tf.ones((1,1), dtype=tf.int64)
            old_state += tf.concat([counter, element], axis=1)
            return old_state


        dataset = CelebA.load(split="train+test+validation")
        header = ','.join(['total'] + [i for i in dataset.element_spec['attributes'].keys()])
        n_true = dataset.map(get_attribute).reduce(initial, reduce_func)
        idx = tf.expand_dims(tf.range(n_true.shape[1], dtype=tf.int64), axis=0)
        data = np.concatenate([idx, n_true], axis=0).T
        np.savetxt(disentangled.utils.get_data_path() / "CelebA_labels.csv", data, delimiter=',', header=header, fmt='%d')
