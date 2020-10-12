import disentangled.dataset
import disentangled.model.distributions as dist
import disentangled.model
import disentangled.utils
import disentangled.metric.utils
import numpy as np
import gin
import tensorflow as tf


@gin.configurable(module="disentangled.metric")
def loglikelihood(model, dataset):
    target = dataset.take(1).as_numpy_iterator().next()
    mean = tf.reshape(model.call(target)[0], (target.shape[0], -1))
    bernoulli = dist.Bernoulli()
    return tf.reduce_mean(
        tf.reduce_sum(
            bernoulli.log_likelihood(tf.reshape(target, (target.shape[0],-1)), mean)
            , axis=1),
        axis=0,
    )

