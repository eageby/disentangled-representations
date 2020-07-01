import click

import disentangled.utils
import gin

from disentangled.cli.evaluate import evaluate
from disentangled.cli.training import train

@click.group(
    context_settings=dict(
        help_option_names=["-h", "--help"],
    ),
)

def cli():
    pass

cli.add_command(evaluate)
cli.add_command(train)
