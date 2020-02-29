import tensorflow as tf

import disentangled.model.objectives as objectives
import disentangled.model.networks as networks
import disentangled.model.activations as activations


class BetaVAE(tf.keras.Model):
    def __init__(self, latents=32, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.latents = latents

        self.flatten = tf.keras.layers.Flatten()
 
        self.f_phi = None             
        self.activations_representation = (None, None, None)
        
        self.f_theta = None
        self.f_theta_mean = None
        self.f_theta_log_var = None

        self.objective = objectives.BetaVAE(gaussian=False)

    def build(self, input_shape):
        self.f_phi_mean = tf.keras.layers.Dense(self.latents, activation=self.activations_representation[0])
        self.f_phi_log_var = tf.keras.layers.Dense(self.latents, activation=self.activations_representation[1])
    
        intermediate_shape = self.f_phi.compute_output_shape(input_shape)[1:]
        self.f_theta_dense = tf.keras.layers.Dense(tf.reduce_prod(intermediate_shape), activation=self.activations_representation[2])
        self.reshape_theta = tf.keras.layers.Reshape(intermediate_shape, name="ReshapeTheta")
            
        self.reshape_output = tf.keras.layers.Reshape(input_shape[1:], name="ReshapeOutput")

        super().build(input_shape)

    def sample(self, mean, log_var, training):
        if not training:
            return mean

        noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0)

        return mean + tf.exp(0.5 * log_var) * noise

    @tf.function
    def encode(self, x, training=False):
        x = self.f_phi(x)
        x = self.flatten(x)
        z_mean = self.f_phi_mean(x)
        z_log_var = self.f_phi_log_var(x)
        return z_mean, z_log_var

    @tf.function
    def decode(self, x):
        x = self.f_theta_dense(x)
        x = self.reshape_theta(x)
        x = self.f_theta(x)
        x_mean = self.f_theta_mean(x)
        x_log_var = self.f_theta_log_var(x)
        return self.reshape_output(x_mean), x_log_var

    def call(self, target, training=False):
        z_mean, z_log_var = self.encode(target)
        z = self.sample(z_mean, z_log_var, training)
        x_mean, x_log_var = self.decode(z)
        self.add_loss(self.objective((target, x_mean, x_log_var, z_mean, z_log_var)))

        return x_mean, z, target

class Mlp_mnist(BetaVAE):
    def __init__(self, latents, gaussian, **kwargs):
        super().__init__(latents, **kwargs)
        
        self.f_phi = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(400, activation='relu')
            ])

        self.activations_representation = (None, None, 'relu')
        
        self.f_theta = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(400, activation='relu')
            ])

        self.f_theta_mean = tf.keras.layers.Dense(28*28, activation='sigmoid')
        self.f_theta_log_var = tf.keras.layers.Dense(28*28, activation=None)
        
        self.objective = objectives.BetaVAE(gaussian=False)

class Conv_64_3(BetaVAE):
    def __init__(self, latents, **kwargs):
        super().__init__(latents, **kwargs)

        self.f_phi = networks.conv_4

        self.activations_representation = (None, None, 'relu')
        
        self.f_theta = networks.conv_4_transpose

        self.f_theta_mean = tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation=None
            )
        self.f_theta_log_var = tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation=None
            )

class Conv_32_1(BetaVAE):
    def __init__(self, latents, **kwargs):
        super().__init__(latents, **kwargs)

        self.f_phi = networks.conv_2

        self.activations_representation = (None, None, 'relu')
        
        self.f_theta = networks.conv_2_transpose

        self.f_theta_mean = tf.keras.layers.Conv2DTranspose(
                1, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid'
            )
        self.f_theta_log_var = tf.keras.layers.Conv2DTranspose(
                1, kernel_size=(3, 3), strides=(1, 1), activation=None
            )


