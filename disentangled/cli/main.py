import click

import disentangled.utils
import gin

from disentangled.cli.evaluate import evaluate, gin_options, add_gin
from disentangled.cli.training import train
from disentangled.cli.dataset import dataset

@click.group(
    context_settings=dict(
        help_option_names=["-h", "--help"],
    ),
)

@gin_options
@click.pass_context
def cli(ctx, **kwargs):
    add_gin(ctx, 'config', ['config.gin'])

cli.add_command(evaluate)
cli.add_command(train)
cli.add_command(dataset)
