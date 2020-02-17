import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['toolbar'] = 'None'

def stack(dataset, rows, cols):
    if isinstance(dataset, tf.data.Dataset):
        dataset = dataset.take(rows*cols)
        map_ = lambda x: x['image']
        dataset = np.asarray(list(dataset.map(map_).as_numpy_iterator()))
    image_rows = [np.concatenate(dataset[cols*i:cols*(i+1)], axis=1) for i in range(rows)]

    return np.concatenate(image_rows, axis=0)

def show(image):
    image = np.squeeze(image)
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    ax.imshow(image, cmap='gray')
    ax.axis('off')
    ax.axis('tight')
    plt.show()
