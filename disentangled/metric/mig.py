import disentangled.dataset as dataset
import disentangled.model.distributions as dist
import disentangled.model.networks as networks
import disentangled.model.utils 
import disentangled.utils as utils
import gin
import numpy as np
import tensorflow as tf

# @tf.function
def estimate_entropy_batch(samples, encoding_dist, encoding_parameters):
    """Estimates marginal entropy
    H(z) = sum_z( 1/N sum_x (q(z|x)) log ( 1/N sum_x(q(z|x)) ) ) ∊ ℝ [D]

    Args:
        samples: z ∼ q(z|x) ∊ ℝ (N, D)
        encoding_dist: q(z|x)
        encoding_parameters: e.g. mean, log_var ∊ ℝ [N,n_params]
    Returns:
        (tf.Tensor)  ∊ ℝ [D]
    """
    N, D = tf.unstack(tf.cast(samples.shape, tf.float32))  # Number of latent dims
    samples = tf.transpose(samples)  # ∊ ℝ [D, N]

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

    # H(z) = 1/N sum_z( 1/N sum_x (q(z|x))) ∊ ℝ [D]
    log_qz = tf.reduce_logsumexp(log_qzx_matrix - tf.math.log(N), axis=0)
    return -tf.reduce_mean(log_qz, axis=0)

def estimate_entropy(dataset, encoding_dist, total=None, progress_bar=True):
    entropy = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    if progress_bar:
        dataset = utils.TrainingProgress(dataset, total=total)

    for i, batch in enumerate(dataset):
        samples, params = tf.split(batch, [1,-1], axis=2)
        samples = tf.squeeze(samples)
        entropy_batch = estimate_entropy_batch(samples, encoding_dist, params)
        entropy = entropy.write(i, entropy_batch)
    
    estimated_entropy = tf.reduce_mean(entropy.stack(), axis=0)
    entropy.mark_used()
    return estimated_entropy

@gin.configurable("mutual_information_gap", module="disentangled.metric")
def mutual_information_gap(
        model,
        dataset,
        batch_size,
        encoding_dist,
        num_values_per_factor,
        progress_bar=True,
):
    print("MIG")
    N = tf.math.reduce_prod(num_values_per_factor)

    dataset = dataset.batch(batch_size)
    encoded = disentangled.metric.utils.encode_dataset(model, dataset).cache()
    # H(z_j) = ∊ ℝ [D, ]
    marginal_entropy = estimate_entropy(encoded, encoding_dist, total=int(N//batch_size), progress_bar=progress_bar)

    D = encoded.element_spec.shape[1] 

    params = encoded.reduce(tf.zeros((0, *encoded.element_spec.shape[1:])), lambda x,y: tf.concat([x,y], axis=0))
    params = tf.reshape(params, [*num_values_per_factor,D , -1])
    # params = tf.zeros((*num_values_per_factor, 10, 3))

    conditional_entropy_array = tf.TensorArray(tf.float32, size=len(num_values_per_factor))
    for factor, num_values in enumerate(num_values_per_factor):
        conditional_entropy_factor = tf.TensorArray(tf.float32, size=num_values)
        for j in range(num_values):
            params_factor = tf.reshape(tf.gather(params, j, axis=factor), (N // num_values, D, -1))
            factor_set = tf.data.Dataset.from_tensor_slices(params_factor).batch(batch_size)
            conditional_entropy_factor.write(j, estimate_entropy(factor_set, encoding_dist, total=int(N // num_values //batch_size), progress_bar=progress_bar))

        
        conditional_entropy_array.write(factor, tf.reduce_mean(conditional_entropy_factor.stack(), axis=0))

    # H(z|v) ∊ ℝ [D, K]
    conditional_entropy = tf.transpose(conditional_entropy_array.stack())

    # H(v_k) ∊ ℝ [K]
    factor_entropy = tf.math.log(tf.cast(num_values_per_factor[-1], tf.float32))

    # I(z_j; v_k) = H(z_j) - H(z_j | v_k) ∊ ℝ [D, K]
    mutual_information = tf.expand_dims(
        marginal_entropy, 1) - conditional_entropy

    # I_n(z_j; v_k) = (H(z_j) - H(z_j | v_k)) / H(v_k) ∊ ℝ [D, K]
    normalized_mutual_information = mutual_information / factor_entropy
    normalized_mutual_information = tf.sort(normalized_mutual_information, axis=0, direction="DESCENDING")

    # ∊ ℝ [K]
    mig = normalized_mutual_information[0, :] - normalized_mutual_information[1, :]
    breakpoint()
    return tf.reduce_mean(mig)
