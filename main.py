import tensorflow as tf

import disentangled.dataset as dataset
import disentangled.model.models as models
import disentangled.model.utils as modelutils
import disentangled.visualize as vi


def shapes3d():
    data = dataset.mnist.pipeline(batch_size=64).take(500)
    model = models.MLP(latents=32)  

    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-2))
    model.fit(data, epochs=5)

    estimate, representation, target = model.predict(data, steps=1)
    vi.results(target, estimate, 5, 10)
    modelutils.save(model, 'shapes3d')


def continue_training():
    model = modelutils.load('shapes3d')
    data = dataset.shapes3d.pipeline(batch_size=256).take(500)
    model.fit(data, epochs=1)
    modelutils.save(model, 'shapes3d')

    estimate, representation, target = model.predict(data, steps=1)
    vi.results(target, estimate, 5, 10)

shapes3d()
