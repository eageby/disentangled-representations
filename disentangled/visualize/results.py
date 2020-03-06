from . import show 

def results(model, dataset, rows, cols):
    model.compile('adam', loss=lambda _, __: 0.0)

    reconstructed, representation, target = model.predict(dataset.pipeline(),steps=10)
    
    show.results(target, reconstructed, rows, cols)
