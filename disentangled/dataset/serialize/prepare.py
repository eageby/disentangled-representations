from pathlib import Path
import click
import disentangled.dataset.serialize.raw_datasets as raw_datasets
from disentangled.dataset.serialize import write
import disentangled.utils

@click.command(
    context_settings=dict(
        help_option_names=["-h", "--help"],
        ignore_unknown_options=True,
        allow_extra_args=True
    )
)
@click.argument('dataset', type=click.Choice(raw_datasets.__all__, case_sensitive=False))
@click.option('--gpu/--no-gpu', 'gpu', default=False)
@click.option('--batches', default=1300, type=int)
@click.option('--batch_size', default=100, type=int)
@click.option('--overwrite', is_flag=True, default=False)
@click.pass_context

def cli(ctx,gpu, dataset, **kwargs):
    if not gpu:
        disentangled.utils.disable_gpu()
    write(raw_datasets.get(dataset), **kwargs)
