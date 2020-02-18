import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import disentangled.dataset.utils as utils

"""Disable toolbar in matplotlib"""
matplotlib.rcParams['toolbar'] = 'None'

def normalize(images):
    max_ = images.max(axis=(1,2,3))
    return images/np.expand_dims(max_, axis=(1,2,3))
    
def _row(data):
    data = np.split(data, data.shape[0], axis=0)
    return np.squeeze(np.concatenate(data, axis=2), axis =0)

def results(target, encoded, rows=4, cols=8):
    if isinstance(target, tf.data.Dataset):
        target = utils.numpy(target)

    if isinstance(encoded, tf.data.Dataset):
        encoded = utils.numpy(encoded)
         
    # target = normalize(target)
    # encoded = normalize(encoded)

    target_rows = [_row(target[cols*i:cols*(i+1)]) for i in range(int(rows/2))]
    encoded_rows = [_row(encoded[cols*i:cols*(i+1)]) for i in range(int(rows/2))]
    
    image_rows = [element for pair in zip(target_rows, encoded_rows) for element in pair]
    show(np.concatenate(image_rows, axis=0))

def show(image):
    image = np.squeeze(image)
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    ax.imshow(image, cmap='gray')
    ax.axis('off')
    ax.axis('tight')
    plt.show()
