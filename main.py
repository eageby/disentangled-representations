import tensorflow as tf

import disentangled.dataset as dataset
import disentangled.model.models as models
import disentangled.model.utils as modelutils
import disentangled.visualize as vi


import disentangled.model.betavae as bv 

def shapes3d():
    tf.random.set_seed(10)
    data = dataset.shapes3d.pipeline(batch_size=64).take(1000)
    model = bv.Conv_64_3(32)

    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4))

    model.fit(data, epochs=5)
    modelutils.save(model, 'shapes3d')

    estimate, representation, target = model.predict(data, steps=1)
    vi.results(target, estimate, 5, 10)

def mnist():
    tf.random.set_seed(10)
    data = dataset.mnist.pipeline(batch_size=128)
    model = bv.Conv_32_1(32)

    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4))

    model.fit(data, epochs=5)

    estimate, representation, target = model.predict(data, steps=1)
    vi.results(target, estimate, 5, 10)

def continue_training():
    model = modelutils.load('shapes3d')
    data = dataset.shapes3d.pipeline(batch_size=256).take(500)
    model.fit(data, epochs=1)
    modelutils.save(model, 'shapes3d')

    estimate, representation, target = model.predict(data, steps=1)
    vi.results(target, estimate, 5, 10)

shapes3d()
