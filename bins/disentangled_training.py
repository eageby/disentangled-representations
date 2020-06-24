from pathlib import Path

import click
import disentangled.utils
import disentangled
import gin

@click.command()
@click.option("--config", "-c", multiple=True)
@click.option('--gin-param',"--gin-parameter",  "-p", multiple=True)
@click.option("--gin-file", "-f", multiple=True)

def main(config, gin_file, gin_param):
    for c in config:
        disentangled.utils.parse_config_file(c)

    gin.parse_config_files_and_bindings(gin_file, gin_param)

    disentangled.training.run_training(model=gin.REQUIRED, dataset=gin.REQUIRED)


