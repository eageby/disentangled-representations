from pathlib import Path

import gin
import click
import disentangled.dataset.serialize.raw_datasets as raw_datasets
import disentangled.utils
from disentangled.dataset.serialize import write


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--config", "-c", multiple=True)
@click.option("--gin-param", "--gin-parameter", "-p", multiple=True)
@click.option("--gin-file", "-f", multiple=True)
@click.option("--overwrite/--no-write", is_flag=True, default=None)
def main(config, gin_file, gin_param, overwrite):
    for c in config:
        disentangled.utils.parse_config_file(c)

    gin.parse_config_files_and_bindings(gin_file, gin_param)

    breakpoint()
    disentangled.dataset.serialize.write(
        dataset=gin.REQUIRED, batches=gin.REQUIRED, overwrite=overwrite
    )

main()
