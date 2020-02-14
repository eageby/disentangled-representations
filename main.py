import disentangled.dataset.shapes3d as shapes3d
import disentangled.dataset.mnist as mnist
from disentangled.model.mlp import *
from disentangled.model.vae import *
import tensorflow as tf

import disentangled.visualize as vi


def main():
    data = mnist.load()
    binary = lambda x: tf.quantization.fake_quant_with_min_max_args(tf.cast(x['image'], tf.float32), min=0, max=1)
    data = data.map(binary)
    automap = lambda val: (val, val)
    data = data.batch(100).map(automap)

    # data = shapes3d.load()
    # data = data.batch(256).map(shapes3d.map_auto)

    model = VAE(BetaVAE(gaussian=False))

    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), loss=model.loss)
    model.fit(data, epochs=5) 

    estimate = model.predict(data, steps=50)
    vi.show(vi.stack(estimate, 5, 10))


if __name__ == "__main__":
    main()
