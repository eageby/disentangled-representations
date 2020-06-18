import math
import disentangled.model.distributions as dist
import gin
import tensorflow as tf

_TOLERANCE = 1e-7

class _Objective(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(_Objective, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, model, *args):
        return self.objective(model, *[self.flatten(i) for i in args])


@gin.configurable('BetaVAE', module='objectives')
class BetaVAE(_Objective):
    @tf.function
    @gin.configurable(module='objectives.BetaVAE')
    def objective(self, model, target, x_mean, x_log_var, z, z_mean, z_log_var, beta=gin.REQUIRED):
        log_likelihood = \
            tf.reduce_mean(
                    tf.reduce_sum(
                        model.output_dist.log_likelihood(target, x_mean, x_log_var)
                    , axis=1)
            , axis=0)

        kld = tf.reduce_mean(
                tf.reduce_sum(
                    model.prior_dist.kld(z_mean, z_log_var)
                    , axis=1)
                , axis=0)

        self.add_metric(-log_likelihood, aggregation="mean",
                        name="-loglikelihood")
        self.add_metric(kld, aggregation="mean", name="kld")

        return -log_likelihood + beta * kld


class FactorVAE(_Objective):
    def objective(
        self, model, target, x_mean, x_log_var, z_mean, z_log_var, discriminator_probability
    ):

        log_likelihood = \
                tf.reduce_mean(
                    tf.reduce_sum(
                        model.output_dist.log_likelihood(target, x_mean, x_log_var)
                        , axis=1)
                    , axis=0)

        kld = tf.reduce_mean(
                tf.reduce_sum(
                    model.prior_dist.kld(z_mean, z_log_var)
                    , axis=1)
                , axis=0)

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

        return -log_likelihood + kld + model.gamma * kld_discriminator

    @staticmethod
    def discriminator(p_z, p_permuted):
        return -tf.reduce_mean(
            tf.math.log(p_z + _TOLERANCE) +
            tf.math.log(p_permuted + _TOLERANCE)
        )


class Beta_TCVAE(_Objective):
    def objective(self, model, target, x_mean, x_log_var, z, z_mean, z_log_var):
        log_nm = tf.math.log(float(target.shape[0] * model.dataset_size))

        log_px = tf.reduce_sum(
            model.output_dist.log_likelihood(target, x_mean, x_log_var)
            , axis=1)

        log_pz = tf.reduce_sum(
            model.prior_dist.log_likelihood(z)
            , axis=1)

        log_qzx = tf.reduce_sum(
            model.prior_dist.log_likelihood(z, z_mean, z_log_var)
            , axis=1)

        log_qz_matrix = model.prior_dist.log_likelihood(
                z, tf.expand_dims(z_mean, axis=1), tf.expand_dims(z_log_var, axis=0)
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
            log_px - mutual_information - model.beta * total_correlation - kld, axis=0
        )


class SparseVAE(_Objective):
    @tf.function
    def objective(self, model, target, x_mean, x_log_var, z, z_mean, z_log_var):
        log_likelihood = \
            tf.reduce_mean(
                    tf.reduce_sum(
                        model.output_dist.log_likelihood(target, x_mean, x_log_var)
                    , axis=1)
            , axis=0)

        kld = tf.reduce_mean(
                tf.reduce_sum(
                    model.prior_dist.kld(z_mean, z_log_var)
                    , axis=1)
                , axis=0)


        weight_regularizer = tf.reduce_sum(tf.stack(model.losses, axis=0), axis=0)
        
        self.add_metric(-log_likelihood, aggregation="mean",
                        name="-loglikelihood")
        self.add_metric(kld, aggregation="mean", name="kld")
        self.add_metric(weight_regularizer, aggregation="mean", name="l1")

        return -log_likelihood + model.beta * kld + model.gamma * weight_regularizer
