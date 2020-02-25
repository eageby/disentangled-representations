import tensorflow as tf
import numpy as np

import disentangled.dataset as dataset
import disentangled.model.utils as modelutils
import disentangled.visualize as vi
import disentangled.model.betavae as betavae
import disentangled.model.objectives as objectives

tf.random.set_seed(10)

model = betavae.BetaVAE()
objective = objectives.BetaVAE(gaussian=True)

data = dataset.mnist.pipeline(batch_size=128)

flatten = tf.keras.layers.Flatten()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

epochs = 5
for epoch in range(epochs):
    for batch in data: 
        with tf.GradientTape() as tape:
            x_train = batch
            z_mean, z_log_var = model.encode(x_train)
            z = model.sample(z_mean, z_log_var, training=True)
            x_mean, x_log_var = model.decoder(z)
            log_likelihood = objective.log_likelihood(flatten(x_train), flatten(x_mean), flatten(x_log_var) )
            kld = objective.kld(z_mean, z_log_var)
            loss = -log_likelihood + kld

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print("Epoch: {}/{}".format(epoch+1, epochs))
    print("Log Likelihood: {}, KLD: {}, Total Loss: {}".format(log_likelihood, kld, loss))

vi.results(batch, model(batch))
