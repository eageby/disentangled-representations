import click
from pathlib import Path

import disentangled.training

@click.command()
@click.argument('model_name', type=click.Choice(['betavae_mnist', 'betavae_shapes3d'], case_sensitive=True))
@click.argument('dataset', type=click.Choice(['MNIST', 'Shapes3d'], case_sensitive=True))
@click.option('--optimizer','-o', type=str)
@click.option('--learning_rate','-l', type=float)
@click.option('--batch_size','-b', type=int)
@click.option('--iterations','-i', type=float)
@click.option('--save/--no-save','-s', default=True)
@click.option('--directory', '-d', type=click.Path(writable=True), default=Path('./models'), show_default=True)
@click.option('--show_default', '-D', is_flag=True)
@click.option('visual', '--show/--no-show','-v', default=True)

def train(model_name, dataset, save, directory, visual, show_default, **kwargs):
    if show_default:
        disentangled.training.print_default(model_name)
        return

    model = disentangled.training.train(
        model_name, dataset, hyperparameters=kwargs, show_results=visual
    )  

    if save:
        disentangled.model.save(model, model_name, dir_=directory)

train()
