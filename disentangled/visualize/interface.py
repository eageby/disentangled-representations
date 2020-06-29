import gin
from . import show

@gin.configurable(module='disentangled.visualize')
def reconstructed(model, dataset, rows, cols):
    reconstructed, representation, target = model.predict(dataset, steps=1)
    show.grid(reconstructed, rows, cols)

@gin.configurable(module='disentangled.visualize')
def data(dataset, rows, cols):
    target = dataset.as_numpy_iterator().next()
    show.grid(target, rows, cols)

@gin.configurable(module='disentangled.visualize')
def comparison(model, dataset, rows, cols):
    reconstructed, representation, target = model.predict(dataset, steps=1)
    show.comparison(reconstructed, target, rows, cols)
