import sys

from . import betavae

__all__ = ['betavae_mnist', 'betavae_shapes3d', 'get']

def get(name):
    return getattr(sys.modules[__name__], name)


betavae_mnist = betavae.Conv_32_1(latents=32)
betavae_shapes3d = betavae.Conv_64_3(latents=32)
