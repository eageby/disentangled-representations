import sys 
import tensorflow_datasets as tfds

from . import core
core._module = sys.modules[__name__]
from .core import *

_name = "mnist"
__version__ = "3.0.0"

_builder = tfds.builder("{}:{}".format(_name, __version__))

_builder.download_and_prepare()
