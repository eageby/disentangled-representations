import disentangled.dataset as dataset

from disentangled.model.vae import *
import disentangled.model.objective as objective
import tensorflow as tf

import disentangled.visualize as vi

def mnist():
    data = dataset.mnist.load().map(dataset.utils.get_image).batch(128)
    model= VAE(Encoder(20), Representation(), Decoder('relu'), objective.BetaVAE(gaussian=True))

    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4))
    model.fit(data, epochs=5)

    estimate, representation, target = model.predict(data, steps=1)

    vi.results(target, estimate, 5, 10)

def shapes3d():
    data = dataset.shapes3d.load().take(128000).map(dataset.utils.get_image).map(dataset.utils.normalize_uint8).batch(128)
    model= VAE(Encoder(20), Representation(), Decoder('sigmoid'), objective.BetaVAE(gaussian=True))

    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4))
    model.fit(data, epochs=5)

    estimate, representation, target = model.predict(data, steps=1)
   
    vi.results(target, estimate, 5, 10)

shapes3d()
