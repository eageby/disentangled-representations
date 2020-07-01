import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from . import utils
from ._dataset import Dataset

class MNIST(Dataset):
    """MNIST Dataset"""

    _name = "mnist"
    __version__ = "3.0.1"


    @staticmethod
    def pipeline(split="train", batch_size=128):
        return (
            MNIST.load(split)
            .map(utils.get_image)
            .map(utils.normalize_uint8)
            .batch(batch_size)
        )

