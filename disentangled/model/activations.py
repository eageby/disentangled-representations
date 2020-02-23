import tensorflow as tf

def relu1(x, **kwargs):
    return tf.nn.relu6(x*6, **kwargs) / 6
