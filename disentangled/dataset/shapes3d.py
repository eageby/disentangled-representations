import sys 
import tensorflow_datasets as tfds

from . import core
core._module = sys.modules[__name__]
from .core import *

_name = "shapes3d"
_version = "2.0.0"

_builder = tfds.builder("{}:{}".format(_name, _version))
