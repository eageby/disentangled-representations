import click
from pathlib import Path

import disentangled.training
import disentangled.model
import disentangled.visualize.latentspace
import disentangled.utils
import disentangled.hyperparameters


@click.group(chain=True, context_settings={'help_option_names':['-h','--help']})
@click.argument('model', type=click.Choice(['betavae_mnist', 'betavae_shapes3d'], case_sensitive=True))
@click.option('--no-gpu', is_flag=True) 
@click.pass_context
def cli(ctx, model, no_gpu):
    if no_gpu:
        disentangled.utils.disable_gpu()
         
    mnist_models = ['betavae_mnist']
    shapes3d_models = ['betavae_shapes3d']
    
    if model in mnist_models:
        dataset = 'MNIST'
    else: 
        dataset = 'Shapes3d'
    
    ctx.ensure_object(dict)

    ctx.obj['model_name'] = model
    ctx.obj['model'] = disentangled.model.get(model)
    ctx.obj['dataset_name'] = dataset
    ctx.obj['dataset'] = disentangled.dataset.get(dataset) 

@cli.command()
@click.option('--optimizer','-o', type=str)
@click.option('--learning_rate','-l', type=float)
@click.option('--batch_size','-b', type=int)
@click.option('--iterations','-i', type=float)
@click.option('--save/--no-save','-s', default=True)
@click.option('--overwrite', is_flag=True, default=False)
@click.option('--directory', '-d', type=click.Path(writable=True), default=Path('./models'), show_default=True)
@click.option('--show_default', '-D', is_flag=True)
@click.pass_context
def train(ctx, save, overwrite, directory, show_default, **kwargs):
    """train model"""
    if show_default:
        disentangled.hyperparameters.print_default(ctx.obj['model_name'])
        return

    hyperparameters = disentangled.hyperparameters.complete(ctx.obj['model_name'], kwargs)
    disentangled.hyperparameters.print_(ctx.obj['model_name'], hyperparameters, ctx.obj['dataset_name'])

    model = disentangled.training.train(
        ctx.obj['model'], ctx.obj['dataset'], hyperparameters=hyperparameters
    )  

    if save:
        disentangled.model.save(model, ctx.obj['model_name'], dir_=directory, overwrite=overwrite)

    ctx.obj['model'] = model

@cli.command()
@click.option('--rows', '-r', type=int, default=4)
@click.option('--cols', '-c', type=int, default=8)
@click.pass_context
def results(ctx, rows, cols):
    """Compare reconstruction with target"""
    disentangled.visualize.results(ctx.obj['model'], ctx.obj['dataset'], rows, cols)

@cli.command('latentspace')
@click.pass_context
def latentspace(ctx):
    """Latent space traversal visual"""
    disentangled.visualize.latentspace.traversal_1d(ctx.obj['model'], ctx.obj['dataset'])

cli()
