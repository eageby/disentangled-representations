import sys
import tensorflow as tf

from .vae import *

__all__ = ["betavae_mnist", "betavae_shapes3d", "factorvae_shapes3d", "beta_tcvae_shapes3d", "sparsevae_shapes3d", "get"]

def get(name):
    return getattr(sys.modules[__name__], name)
