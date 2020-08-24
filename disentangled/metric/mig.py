import disentangled.dataset as dataset
import disentangled.model.distributions as dist
import disentangled.model.networks as networks
import disentangled.model.utils 
import disentangled.utils as utils
import gin
import numpy as np
import tensorflow as tf

# @tf.function
def estimate_entropy_batch(samples, encoding_dist, *encoding_parameters):
    """Estimates marginal entropy
    H(z) = sum_z( 1/N sum_x (q(z|x)) log ( 1/N sum_x(q(z|x)) ) ) ∊ ℝ [D]

    Args:
        samples: z ∼ q(z|x) ∊ ℝ (N, D)
        encoding_dist: q(z|x)
        *encoding_parameters: list of parameters ∊ ℝ [N,]
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

    # H(z) = 1/N sum_z( 1/N sum_x (q(z|x))) ∊ ℝ [D]
    log_qz = tf.reduce_logsumexp(log_qzx_matrix - tf.math.log(N), axis=0)
    return -tf.reduce_mean(log_qz, axis=0)

def estimate_entropy(dataset, encoding_dist, total=None, progress_bar=True):
    entropy = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    if progress_bar:
        dataset = utils.TrainingProgress(dataset, total=total)

    for i, batch in enumerate(dataset):
        samples, params = batch
        entropy_batch = estimate_entropy_batch(samples, encoding_dist, *params)
        entropy = entropy.write(i, entropy_batch)
    
    return tf.reduce_mean(entropy.stack(), axis=0)
        

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

    taken = 0
    conditional_entropy = tf.TensorArray(tf.float32, size=len(num_values_per_factor))
    for i, num_values in enumerate(num_values_per_factor):
        n_samples = int(N / num_values)
        factor_set = encoded.unbatch().skip(taken).take(n_samples).batch(batch_size)
        taken += n_samples
        conditional_entropy_factor = estimate_entropy(factor_set, encoding_dist, total=int(n_samples//batch_size), progress_bar=progress_bar) / num_values 

        conditional_entropy.write(i, conditional_entropy_factor)


    # H(z|v) ∊ ℝ [D, K]
    conditional_entropy = tf.transpose(conditional_entropy.stack())

    # H(v_k) ∊ ℝ [K]
    factor_entropy = tf.math.log(tf.cast(num_values_per_factor, tf.float32))

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

