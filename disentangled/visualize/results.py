from . import show 

def results(model, dataset, rows, cols):
    reconstructed, representation, target = model.predict(dataset.pipeline(),steps=10)
    
    show.results(target, reconstructed, rows, cols)
