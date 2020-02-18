import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset:
    @classmethod
    def load(cls, split="train"):
        return cls._builder.as_dataset(split=split)

    @classmethod
    def show(cls,**kwargs):
        cls._builder.download_and_prepare()
        return tfds.show_examples(
            cls.info, cls._builder.as_dataset(split="train"), **kwargs
        )

    @classmethod
    def version(cls):
        return cls.__version__

    @classmethod
    def info(cls):
        return cls._builder.info


class MNIST(Dataset):
    _name = "mnist"
    __version__ = "3.0.0"

    _builder = tfds.builder("{}:{}".format(_name, __version__))

class Shapes3d(Dataset):
    _name = "shapes3d"
    __version__ = "2.0.0"

    _builder = tfds.builder("{}:{}".format(_name, __version__))
