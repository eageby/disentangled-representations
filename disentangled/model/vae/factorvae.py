import tqdm
import numpy as np
import tensorflow as tf
from .vae import VAE

import disentangled.model.networks as networks
import disentangled.model.objectives as objectives
import disentangled.utils 
 
class FactorVAE(VAE):
    def __init__(
        self,
        f_phi,
        f_theta,
        f_theta_mean,
        f_theta_log_var,
        discriminator,
        latents,
        gamma,
        **kwargs
    ):
        super().__init__(
            # Encoder
            f_phi=f_phi,
            f_phi_mean= tf.keras.layers.Dense( latents, activation=None),
            f_phi_log_var = tf.keras.layers.Dense( latents, activation=None),
            # Decoder
            f_theta=f_theta,
            f_theta_mean=f_theta_mean,
            f_theta_log_var=f_theta_log_var,
            objective=objectives.FactorVAE(gamma),
            latents=latents,
            **kwargs
        )

        self.discriminator_net = discriminator 
        self.discriminator_net.trainable = False
    
    def build(self, input_shape):
        super().build(input_shape)
        self.discriminator_net.build((None, self.latents))

    def call(self, x):
        z = self.encode(x)[0]
        return self.decode(z)

    def discriminator(self, z):
        probabilities = self.discriminator_net(z)
        return tf.split(probabilities, 2, axis=-1)[0]

    @staticmethod
    def permute_dims(representation):
        representation = np.array(representation)
        for j in range(representation.shape[1]):
            permutation_index = np.random.permutation(representation.shape[0])
            representation[:, j] = representation[permutation_index, j]
    
        return representation         

    def train(self, data, optimizer, iterations=100):
        optimizer_theta = tf.keras.optimizers.Adam(learning_rate=1e-4)
        optimizer_psi = tf.keras.optimizers.Adam(learning_rate=1e-5)

        data = data.as_numpy_iterator()

        progress = disentangled.utils.TrainingProgress(range(iterations))
        for _ in progress:        
            with tf.GradientTape() as tape:
                batch_theta = data.next()
                z_mean, z_log_var = self.encode(batch_theta)
                z = self.sample(z_mean, z_log_var, training=True)

                p_z = self.discriminator(z)
               
                x_mean, x_log_var = self.decode(z)
            
                loss_theta = self.objective(batch_theta, x_mean, x_log_var, z_mean, z_log_var, p_z)
                self.add_loss(lambda: loss_theta)

            # Discriminator weights are assigned as not trainable in init
            grad_theta = tape.gradient(loss_theta, self.trainable_variables)
            optimizer_theta.apply_gradients(zip(grad_theta, self.trainable_variables))

            # Updating Discriminator
            with tf.GradientTape() as tape:
                batch_psi = data.next()
                z_mean, z_log_var = self.encode(batch_psi)
                z = self.sample(z_mean, z_log_var, training=True)
                
                z_permuted = self.permute_dims(z)

                p_permuted = self.discriminator(z_permuted)
                    
                loss_psi = -tf.reduce_mean( 
                    tf.math.log(p_z) + tf.math.log(p_permuted)
                )

            grad_psi = tape.gradient(loss_psi, self.discriminator_net.variables)
            optimizer_psi.apply_gradients(zip(grad_psi, self.discriminator_net.variables))

            progress.update(self) 

class factorvae_shapes3d(FactorVAE):
    def __init__(self, latents, gamma):
        super().__init__(
            f_phi = networks.conv_4,
            f_theta = networks.conv_4_transpose,
            f_theta_mean=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation="sigmoid"
            ),
            f_theta_log_var=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation=None
            ),
            latents=latents,
            gamma=gamma,
            discriminator=tf.keras.Sequential([
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
        )