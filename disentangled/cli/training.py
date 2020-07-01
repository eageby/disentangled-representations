import click
import gin
import disentangled
import disentangled.utils
import disentangled.training

from disentangled.cli.evaluate import add_gin, gin_options, parse

@click.command('train')
@click.argument('model')
@gin_options
@click.option('--overwrite', is_flag=True, default=False)
@click.option('--print-operative', is_flag=True, default=False)
@click.pass_context
def train(ctx, model, config, gin_param, gin_file, overwrite, print_operative):
    parse(ctx)
    disentangled.training.run_training(model=gin.REQUIRED, dataset=gin.REQUIRED, iterations=gin.REQUIRED, overwrite=overwrite)

    if print_operative:
        print(gin.operative_config_str())
