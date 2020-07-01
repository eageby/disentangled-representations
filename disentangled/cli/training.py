import click
import gin
import disentangled
import disentangled.utils
import disentangled.training

import itertools

_METHODS = ["FactorVAE", "BetaVAE", "BetaTCVAE", "BetaSVAE"]
_DATASETS = ["DSprites", "Shapes3d"]
_MODELS = [i for i in itertools.product(_METHODS, _DATASETS)]


@click.command('train')
@click.argument('model')
@click.option("--config", "-c", multiple=True)
@click.option('--gin-param',"--gin-parameter",  "-p", multiple=True)
@click.option("--gin-file", "-f", multiple=True)
@click.option('--overwrite', is_flag=True, default=False)
@click.option('--print-operative', is_flag=True, default=False)
def train(model, config, gin_param, gin_file, overwrite, print_operative):
    disentangled.utils.parse_config_file('train/' + model + '.gin')

    for c in config:
        disentangled.utils.parse_config_file(c)

    gin.parse_config_files_and_bindings(gin_file, gin_param)
    disentangled.training.run_training(model=gin.REQUIRED, dataset=gin.REQUIRED, iterations=gin.REQUIRED, overwrite=overwrite)

    if print_operative:
        print(gin.operative_config_str())

