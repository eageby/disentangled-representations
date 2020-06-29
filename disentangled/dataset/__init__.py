import sys
from .mnist import MNIST as mnist
from .shapes3d import Shapes3d 
from .dsprites import DSprites
from .celeba import CelebA 

__all__ = ['Shapes3d', 'DSprites']

def get(name):
    return getattr(sys.modules[__name__], name)
