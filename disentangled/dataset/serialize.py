from pathlib import Path

import disentangled.dataset
import disentangled.utils
import tensorflow as tf
import tqdm


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""

    if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_image(element):
    element['image'] = tf.io.serialize_tensor(element['image'])

    return element

def parse_image(element):
    element['image'] = tf.io.parse_tensor(element['image'], tf.float32)
    return element


def write(dataset, path, batches=1300):
    data = dataset.create(10).take(batches).map(serialize_image)
    progress = disentangled.utils.TrainingProgress(data, total=batches)

    progress.write('Serializing dataset to {}'.format(path))
    with tf.io.TFRecordWriter(str(path)) as writer:
        for batch in progress:
            ex = dataset.example(batch)
            writer.write(ex.SerializeToString())

def read(dataset, path):
    data = tf.data.TFRecordDataset(path)  
    return data.map(dataset.parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

if __name__ == '__main__':
    # write(disentangled.dataset.shapes3d.Ordered, './data', batches=10)
    data = read(disentangled.dataset.shapes3d.Ordered, './data')
