import math
import disentangled.model.distributions as dist

import tensorflow as tf

_TOLERANCE = 1e-7

class _Objective(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(_Objective, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, *args):
        return self.objective(*[self.flatten(i) for i in args])


class BetaVAE(_Objective):
    def __init__(self, beta=1, gaussian=False, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.beta = beta
        self.gaussian = gaussian

    @tf.function
    def objective(self, target, x_mean, x_log_var, z_mean, z_log_var):
        if self.gaussian:
            log_likelihood = \
                tf.reduce_mean(
                        tf.reduce_sum(
                            dist.Gaussian.log_likelihood(dist.Gaussian, target, x_mean, x_log_var)
                        , axis=1)
                , axis=0)
        else:
            log_likelihood = \
                tf.reduce_mean(
                        tf.reduce_sum(
                            dist.Bernoulli.log_likelihood(dist.Bernoulli, target, x_mean)
                        , axis=1)
                , axis=0)

        prior_z_dist = dist.Gaussian(mean=0., log_var=0.)
        kld = tf.reduce_mean(prior_z_dist.kld(z_mean, z_log_var))

        self.add_metric(-log_likelihood, aggregation="mean",
                        name="-loglikelihood")
        self.add_metric(kld, aggregation="mean", name="kld")

        return -log_likelihood + self.beta * kld


class FactorVAE(_Objective):
    def __init__(self, gamma, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def objective(
        self, target, x_mean, x_log_var, z_mean, z_log_var, discriminator_probability
    ):

        log_likelihood = \
                tf.reduce_mean(
                    tf.reduce_sum(
                        dist.Bernoulli.log_likelihood(dist.Bernoulli, target, x_mean)
                        , axis=1)
                    , axis=0)

        prior_z_dist = dist.Gaussian(mean=0., log_var=0.)
        kld = tf.reduce_mean(prior_z_dist.kld(z_mean, z_log_var), axis=0)

        kld_discriminator = tf.reduce_mean(
            tf.math.log(
                discriminator_probability
                / ((1 - discriminator_probability) + _TOLERANCE)
            )
        )

        self.add_metric(-log_likelihood, aggregation="mean",
                        name="-loglikelihood")
        self.add_metric(kld, aggregation="mean", name="kld1")
        self.add_metric(kld_discriminator, aggregation="mean", name="kld2")

        return -log_likelihood + kld + self.gamma * kld_discriminator

    @staticmethod
    def discriminator(p_z, p_permuted):
        return -tf.reduce_mean(
            tf.math.log(p_z + _TOLERANCE) +
            tf.math.log(p_permuted + _TOLERANCE)
        )


class Beta_TCVAE(_Objective):
    def __init__(self, beta, dataset_size=1000, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.dataset_size = dataset_size

    def objective(self, target, x_mean, x_log_var, z,  z_mean, z_log_var):
        log_nm = tf.math.log(float(target.shape[0] * self.dataset_size))

        log_px = tf.reduce_sum(
            dist.Bernoulli.log_likelihood(dist.Bernoulli, target, x_mean)
            , axis=1)

        prior_z = dist.Gaussian(mean=0., log_var=0.)
        log_pz = tf.reduce_sum(
            prior_z.log_likelihood(z)
            , axis=1)

        # E_q(z|x)[ log q(z|x) ]
        log_qzx = tf.reduce_sum(
            dist.Gaussian.log_likelihood(dist.Gaussian, z, z_mean, z_log_var)
            , axis=1)

        log_qz_matrix = dist.Gaussian.log_likelihood(
                dist.Gaussian, z, tf.expand_dims(z_mean, axis=1), tf.expand_dims(z_log_var, axis=0)
        )

        log_qz_prod = tf.reduce_sum(
            tf.reduce_logsumexp(log_qz_matrix, axis=1) - log_nm, axis=1)

        log_qz = tf.reduce_logsumexp(
                tf.reduce_sum(log_qz_matrix, axis=2)
                , axis=1) - log_nm

        mutual_information = log_qzx - log_qz
        total_correlation = log_qz - log_qz_prod
        kld = log_qz_prod - log_pz

        self.add_metric(-log_px, aggregation="mean", name="-loglikelihood")
        self.add_metric(mutual_information, aggregation="mean", name="info")
        self.add_metric(kld, aggregation="mean", name="kld")
        self.add_metric(total_correlation, aggregation="mean", name="tc")

        return -tf.reduce_mean(
            log_px - mutual_information - self.beta * total_correlation - kld, axis=0
        )


class SparseVAE(_Objective):
    def __init__(self, beta=1, gamma=1e-4, **kwargs):
        super(SparseVAE, self).__init__(**kwargs)
        self.beta = beta
        self.gamma = gamma

    @tf.function
    def objective(self, target, x_mean, x_log_var, z_mean, z_log_var, l1):
        breakpoint()
        log_likelihood = loglikelihood_bernoulli(target, x_mean)

        kld = kld_laplacian(z_mean, z_log_var)

        self.add_metric(-log_likelihood, aggregation="mean",
                        name="-loglikelihood")
        self.add_metric(kld, aggregation="mean", name="kld")

        return -log_likelihood + self.beta * kld + self.gamma * l1
