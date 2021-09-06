import itertools

import click
import disentangled.utils
import gin
from disentangled.cli.utils import gin_options, add_gin

from disentangled.cli.dataset import dataset
from disentangled.cli.evaluate import evaluate
from disentangled.cli.experiment import experiment
from disentangled.cli.training import train

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@gin_options
@click.pass_context
def cli(ctx, **kwargs):
    """Train and evaluate disentangled representation learning models."""
    add_gin(ctx, "config", ["config.gin"])

cli.add_command(evaluate)
cli.add_command(train)
cli.add_command(dataset)
cli.add_command(experiment)
