import disentangled.dataset as dataset
import disentangled.model.distributions as dist
import disentangled.model.networks as networks
import disentangled.model.utils
import disentangled.utils as utils
import gin
import numpy as np
import tensorflow as tf

# TODO GIN CONFIGURABLE
def occurences(data):
    """Counts occurences of values in data

    Args:
        data: ∊ ℝ (N , K) = (batch_size, factors)

    Returns:
        (tf.RaggedTensor) ∊ ℝ (K, (A))
     """
    data = tf.cast(data, tf.int32)
    _, K = data.shape

    occurences = tf.ragged.stack(
        [tf.math.bincount(data[:, factor], dtype=tf.float32)
         for factor in range(K)],
        axis=0,
    )

    return occurences

# @tf.function
def estimate_marginal_entropy(samples, encoding_dist, *encoding_parameters):
    """Estimates marginal entropy
    H(z) = sum_z( 1/N sum_x (q(z|x)) log ( 1/N sum_x(q(z|x)) ) ) ∊ ℝ [D]

    Args:
        samples: z ∼ q(z|x) ∊ ℝ (N, D)
        encoding_dist: q(z|x)
        *encoding_parameters: list of parameters ∊ ℝ [N]
    Returns:
        (tf.Tensor)  ∊ ℝ [D]
    """
    N, D = tf.unstack(tf.cast(samples.shape, tf.float32))  # Number of latent dims
    samples = tf.transpose(samples)  # ∊ ℝ [D, N]

    encoding_parameters = tf.stack(encoding_parameters, axis=2)
    n_params = encoding_parameters.shape[2]

    # log q(z_j|x_n) ∊ ℝ [N, N, D]
    log_qzx_matrix = tf.reshape(
        encoding_dist.log_likelihood(
            tf.broadcast_to(tf.reshape(samples, (1, D, N)), (N, D, N)),
            *tf.unstack(
                tf.broadcast_to(
                    tf.reshape(encoding_parameters, (N, D, 1, n_params)),
                    (N, D, N, n_params),
                ),
                axis=3,
            )
        ),
        (N, N, D),
    )

    # H(z) = sum_z( 1/N sum_x (q(z|x)) log ( 1/N sum_x(q(z|x)) ) ) ∊ ℝ [D]
    log_qz = tf.reduce_logsumexp(log_qzx_matrix - tf.math.log(N), axis=0)

    return -tf.reduce_mean(log_qz, axis=0)

    return -tf.reduce_sum(tf.math.exp(log_qz) * log_qz, axis=0)

def ragged_logsumexp(values, **kwargs):
    """logsumexp with support for tf.RaggedTensor"""
    return tf.math.log(tf.reduce_sum(tf.math.exp(values), **kwargs)) 

# @tf.function
def estimate_conditional_entropy(
        samples, log_pv, log_pxv, encoding_dist, *encoding_parameters):
    """Calculates conditional entropy
    H(z| v) = - sum(p(z,v) log p(z|v)) ∊ ℝ [D, K]

    Args:
        samples: z ∼ q(z|x) ∊ ℝ [N, D]
        log_pv: log p(v) ∊ ℝ [K, (A) ]
        log_pxv: log p(x|v) ∊ ℝ [K, (A)]
        encoding_dist: q(z|x)
        *encoding_parameters: list of parameters ∊ ℝ [N]
    Returns: 
        (tf.Tensor) H(z| v) ∊ ℝ [D, K]
    """
    N, D = tf.unstack(tf.cast(samples.shape, tf.float32))  # Batch size, latent_dims
    K = tf.cast(log_pv.shape[0], tf.float32)  # Factors

    # q(z|x) ∊ ℝ [N, D]
    log_qzx = tf.reshape(
            encoding_dist.log_likelihood(samples, *encoding_parameters),
            (N, D, 1, 1)
    )

   # log_pzv = log_pxv + log_pv + tf.reduce_logsumexp(log_qzx, axis=0)

    # p(z|v) = sum_x(q(z|x)p(x|v)) ∊ ℝ [D, K, (A)]
    log_pzv_cond = ragged_logsumexp(log_pxv + log_qzx, axis=0)

    # p(z,v) = sum_x (q(z|x) p(x|v) p(v)) = p(z|v)p(v) ∊ ℝ [D, K, (A)]
    log_pzv = log_pzv_cond + tf.expand_dims(log_pv, 0)

    return -tf.reduce_mean(log_pzv_cond, axis=2).to_tensor()

    # H(z|v) = - sum(p(z,v) log p(z|v)) ∊ ℝ [D, K]
    return -tf.reduce_sum(tf.math.exp(log_pzv) * log_pzv_cond, axis=2).to_tensor()


# @tf.function
def estimate_factor_statistics(labels):
    """Estimates entropy and prior and conditioned prob
    Args:
        labels: Factor values for samples ∊ ℝ [N, D]

    Returns:
        (tf.RaggedTensor) p(v) ∊ ℝ [K, (A) ]
        (tf.RaggedTensor) p(x|v) ∊ ℝ [K, (A)]
        (tf.Tensor) H(v_k) ∊ ℝ [K]
    """
    N, K = tf.unstack(tf.cast(labels.shape, tf.float32))  # Batch size, Factors

    # #{v_k = a}
    factor_occurences = occurences(labels)

    # p(v_k = a) = #{v_k = a} / N
    # p(v) ∊ ℝ (K, (A) ) 
    log_pv = tf.math.log(factor_occurences) - tf.math.log(N) 

    # p(v=a|x=b) = 1
    # p(x|v=a) = p(v|x) / #{v=a} ∊ ℝ [K, (A)]
    log_pxv = -tf.math.log(factor_occurences)

    # factor_possibilites = tf.cast(factor_occurences.row_lengths(axis=1), tf.float32)
    # log_pxv = -tf.math.log(factor_occurences) - tf.expand_dims(tf.math.log(factor_possibilites), 1)

    p_x = tf.reduce_sum(tf.math.exp(log_pxv+ log_pv))

    # H(v_k) = - sum_a(p(v_k=a) log(p(v_k=a)) ∊ ℝ [K]
    entropy = -tf.reduce_sum(tf.math.exp(log_pv) * log_pv, axis=1)

    return entropy, log_pv, log_pxv


# @tf.function
def normalized_mutual_information(labels, samples, encoding_dist, *encoding_parameters):
    """Calculates normalized mutual information
    I_n(z_j; v_k) = H(z_j) - H(z_j | v_k) / H(v_k) ∊ ℝ [D, K]

    Args:
        labels: Factor values for samples ∊ ℝ (N, D)
        samples: z ∼ q(z|x) ∊ ℝ (N, D)
        encoding_dist: q(z|x)
        *encoding_parameters: list of parameters ∊ ℝ [N]

    Returns:
        (tf.Tensor) I_n(z_j; v_k)∊ ℝ [D, K]
    """
    N, K = tf.unstack(tf.cast(labels.shape, tf.float32))  # Batch size, number factors
    factor_entropy, log_pv, log_pxv = estimate_factor_statistics(labels)
    conditional_entropy = estimate_conditional_entropy(
        samples, log_pv, log_pxv, encoding_dist, *encoding_parameters
    )
    # H(z_j) = ∊ ℝ [D, ]
    marginal_entropy = estimate_marginal_entropy(
        samples, encoding_dist, *encoding_parameters
    )
    # I(z_j; v_k) = H(z_j) - H(z_j | v_k) ∊ ℝ [D, K]
    mutual_information = tf.expand_dims(
        marginal_entropy, 1) - conditional_entropy
    return mutual_information / factor_entropy

# @tf.function
def mutual_information_gap_batch(labels, samples, encoding_dist, *encoding_parameters):
    """Estimates mutual information gap (MIG)
    1/K sum_{k=1}^K 1/H(v_k) (I(z_j[k]; v_k) - max_{j !=j[k]} I(z_j;v_k))

    Args:
        labels: Factor values for samples ∊ ℝ (N, D)
        samples: z ∼ q(z|x) ∊ ℝ (N, D)
        encoding_dist: q(z|x)
        *encoding_parameters: list of parameters ∊ ℝ [N]

    Returns:
        (tf.Tensor) ∊ ℝ []
    """
    # I_n(z_j; v_k) = (H(z_j) - H(z_j | v_k)) / H(v_k) ∊ ℝ [D, K]
    nmi = normalized_mutual_information(
        labels, samples, encoding_dist, *encoding_parameters
    )
    nmi = tf.sort(nmi, axis=0, direction="DESCENDING")
    # ∊ ℝ [K]
    mig = nmi[0, :] - nmi[1, :]
    return tf.reduce_mean(mig)

@gin.configurable("mutual_information_gap", module="disentangled.metric")
def mutual_information_gap(
        model,
        dataset,
        points,
        batch_size,
        encoding_dist,
        num_values_per_factor,
        progress_bar=True,
):
    dataset = dataset.take(points).batch(batch_size, drop_remainder=True)
    n_batches = points // batch_size

    if progress_bar:
        progress = disentangled.utils.TrainingProgress(
            dataset, total=n_batches)
        progress.write("Calculating MIG")
    else:
        progress = dataset

    mig = tf.TensorArray(dtype=tf.float32, size=n_batches)

    for i, batch in enumerate(progress):
        labels = tf.cast(batch["label"], tf.int32)

        # z ∼ q(z|x) ∊ ℝ (N, D)
        encoding_parameters = model.encode(batch["image"])
        samples = model.sample(*encoding_parameters, training=True)

        mig_batch = mutual_information_gap_batch(
            labels, samples, encoding_dist, *encoding_parameters
        )

        mig = mig.write(i, mig_batch)

    breakpoint()

    return tf.reduce_mean(mig.stack())
