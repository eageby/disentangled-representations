import click
import gin
import disentangled
import disentangled.utils
import disentangled.training

from disentangled.cli.utils import add_gin, gin_options, parse

@click.command('train')
@click.argument('model')
@click.option('--overwrite', is_flag=True, default=False)
@click.option('--print-operative', is_flag=True, default=False)
@gin_options
@click.pass_context
def train(ctx, model, config, gin_param, gin_file, overwrite=False, print_operative=False):
    add_gin(ctx, 'config', ['train/{}.gin'.format(model)], insert=True)

    if overwrite is True:
        add_gin(ctx, 'gin_param' ['disentangled.model.utils.save.overwrite=True'])
    parse(ctx)

    disentangled.training.run_training(model=gin.REQUIRED, dataset=gin.REQUIRED, iterations=gin.REQUIRED)

    if print_operative:
        print(gin.operative_config_str())
