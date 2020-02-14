import tensorflow as tf
import tensorflow_datasets as tfds

_module = None


def map_auto(element):
    x = tf.cast(element["image"], tf.float32)

    return x, x


def load(split="train"):
    return _module._builder.as_dataset(split=split)


def show(**kwargs):
    return tfds.show_examples(
        _module.info, _module._builder.as_dataset(split="train"), **kwargs
    )


def version():
    return _module._version


def info():
    return _module._builder.info
