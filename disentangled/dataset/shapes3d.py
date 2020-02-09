import tensorflow_datasets as tfds
import matplotlib.pyplot

_name = "shapes3d"
_version = "2.0.0"

_builder = tfds.builder("{}:{}".format(_name, _version))

info = _builder.info

def load(split='train'):
    return _builder.as_dataset(split=split)

def show(**kwargs):
    return tfds.show_examples(info, _builder.as_dataset(split='train'), **kwargs)

