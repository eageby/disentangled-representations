import disentangled.dataset
import disentangled.model
import disentangled.utils
import disentangled.metric.utils
import numpy as np
import gin
import tensorflow as tf


@gin.configurable(module="disentangled.metric")
def collapsed(model, dataset, tolerance):
    empirical_var = disentangled.metric.utils.representation_variance(
        model, dataset)
    intact_idx = np.where(empirical_var > tolerance)[0]

    return empirical_var.shape[0] - len(intact_idx)
