import disentangled.dataset as dataset

from disentangled.model.vae import *
import disentangled.model.objective as objective
import tensorflow as tf

import disentangled.visualize as vi

data = dataset.mnist.load().map(dataset.utils.get_image).map(dataset.utils.binary).batch(128)
model= VAE(Encoder(20), Representation(), Decoder('sigmoid'), objective.BetaVAE(gaussian=False))

model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4))
model.fit(data, epochs=5)

estimate = model.predict(data, steps=50)

vi.stack(estimate, 5, 10)
