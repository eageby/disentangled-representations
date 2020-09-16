import click
import gin
import disentangled
import disentangled.dataset
import disentangled.utils
import disentangled.metric.utils
import disentangled.training
import disentangled.visualize

from disentangled.cli.utils import parse, gin_options, visual_options, add_gin, _MODELS, _DATASETS, DatasetGroup

@click.group(cls=DatasetGroup)
@gin_options
@click.argument('dataset', type=click.Choice(_DATASETS.copy()))
@click.pass_context
def dataset(ctx, dataset, **kwargs):
    ctx.ensure_object(dict)
    ctx.obj['dataset'] = dataset
    gin.bind_parameter('%HP_SWEEP_VALUES', None)

@dataset.command()
@click.pass_context
def prepare(ctx):
    if ctx.obj['dataset'] == 'all':
        prepare_datasets = _DATASETS
    else:
        prepare_datasets = [ctx.obj['dataset']]
        
    for dataset in prepare_datasets:
        disentangled.dataset.get(dataset).load()

@dataset.command()
@click.option('plot', '--plot/--no-plot', is_flag=True, default=True)
@gin_options
@visual_options
@click.pass_context
def examples(ctx, filename, rows, cols, plot, **kwargs):
    dataset = ctx.obj['dataset']

    add_gin(ctx, 'config', ['evaluate/dataset/{}.gin'.format(dataset)])
    parse(ctx)
    
    with gin.unlock_config():
        gin.bind_parameter('disentangled.visualize.show.output.show_plot', plot)
        gin.bind_parameter('disentangled.visualize.show.output.filename', filename)
        if rows is not None:
            gin.bind_parameter('disentangled.visualize.data.rows', rows)
        if cols is not None:
            gin.bind_parameter('disentangled.visualize.data.cols', cols)

    dataset = disentangled.dataset.get(dataset).pipeline()

    disentangled.visualize.data(dataset, rows=gin.REQUIRED, cols=gin.REQUIRED)

@dataset.command()
@click.option('--batch_size', type=int, default=64)
@click.option('--verbose', '-v', is_flag=True, default=False)
@gin_options
@visual_options
@click.pass_context
def fixed(ctx, batch_size, filename, rows, cols, plot, verbose, **kwargs):
    dataset = ctx.obj['dataset']
    add_gin(ctx, 'config', ['evaluate/dataset/{}.gin'.format(dataset)])
    parse(ctx)
    
    with gin.unlock_config():
        gin.bind_parameter('disentangled.visualize.show.output.show_plot', plot)
        gin.bind_parameter('disentangled.visualize.show.output.filename', filename)
        if rows is not None:
            gin.bind_parameter('disentangled.visualize.fixed_factor_data.rows', rows)
        if cols is not None:
            gin.bind_parameter('disentangled.visualize.fixed_factor_data.cols', cols)

    num_values_per_factor = disentangled.dataset.get(dataset).num_values_per_factor    
    dataset = disentangled.dataset.get(dataset).supervised()

    fixed, _ = disentangled.metric.utils.fixed_factor_dataset(dataset, batch_size, num_values_per_factor)

    disentangled.visualize.fixed_factor_data(fixed, rows=gin.REQUIRED, cols=gin.REQUIRED, verbose=verbose)

@dataset.command()
def labels():
    disentangled.dataset.CelebA.attribute_distribution()
