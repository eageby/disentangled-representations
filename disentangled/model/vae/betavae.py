import tensorflow as tf
from .vae import VAE

import disentangled.model.networks as networks
import disentangled.utils
import disentangled.model.objectives as objectives

class BetaVAE(VAE):
    def train(self, data, learning_rate, iterations=100, **kwargs):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        progress = disentangled.utils.TrainingProgress(data.take(int(iterations)), total=int(iterations))
        for batch in progress:        
            with tf.GradientTape() as tape:
                z_mean, z_log_var = self.encode(batch)
                z = self.sample(z_mean, z_log_var, training=True)
               
                x_mean, x_log_var = self.decode(z)
            
                loss = self.objective(batch, x_mean, x_log_var, z_mean, z_log_var)
                self.add_loss(lambda: loss)

                try: 
                    tf.debugging.check_numerics(loss, message='loss')
                except:
                    import pdb;pdb.set_trace()
 
            # Discriminator weights are assigned as not trainable in init
            grad = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grad, self.trainable_variables))

            progress.update(self)
            progress.log(interval=5e4)


class betavae_mnist(BetaVAE):
    def __init__(self, latents, beta, **kwargs):
        super().__init__(
            # Encoder
            f_phi=networks.conv_2,
            f_phi_mean= tf.keras.layers.Dense( latents, activation=None),
            f_phi_log_var = tf.keras.layers.Dense( latents, activation=None),
            # Decoder
            f_theta=networks.conv_2_transpose,
            f_theta_mean=tf.keras.layers.Conv2DTranspose(
                1, kernel_size=(3, 3), strides=(1, 1), activation="sigmoid"
            ),
            f_theta_log_var=tf.keras.layers.Conv2DTranspose(
                1, kernel_size=(3, 3), strides=(1, 1), activation=None
            ),
            objective=objectives.BetaVAE(gaussian=False, beta=beta),
            latents=latents
        )

class betavae_shapes3d(BetaVAE):
    def __init__(self, latents, beta, **kwargs):
        super().__init__(
            # Encoder
            f_phi=networks.conv_4,
            f_phi_mean= tf.keras.layers.Dense( latents, activation=None),
            f_phi_log_var = tf.keras.layers.Dense( latents, activation=None),
            # Decoder
            f_theta=networks.conv_4_transpose,
            f_theta_mean=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation="sigmoid"
            ),
            f_theta_log_var=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation=None
            ),
            objective=objectives.BetaVAE(gaussian=False, beta=beta),
            latents=latents
        )
