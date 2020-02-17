import disentangled.dataset.shapes3d as shapes3d
import disentangled.dataset.mnist as mnist
from disentangled.model.mlp import *
from disentangled.model.vae import *
import tensorflow as tf

import disentangled.visualize as vi

data = mnist.load().map(mnist.binary).batch(128)

model= VAE(Encoder, Representation, Decoder)

model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4))
model.fit(data, epochs=5)

estimate = model.predict(data, steps=50)

vi.show(vi.stack(estimate, 5, 10))
