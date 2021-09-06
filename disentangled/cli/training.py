import click
import gin
import disentangled
import disentangled.utils
import disentangled.training

from disentangled.cli.utils import add_gin, gin_options, parse, path_callback

@click.command('train')
@click.argument('model')
@click.option('--path', type=click.Path(dir_okay=True, writable=True), default=None, callback=path_callback, help='Specify path to save model to.')
@click.option('--checkpoint', type=click.Path(dir_okay=True, readable=True), default=None, callback=path_callback, help='Path to previously trained model to continue training.')
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing saved model.')
@click.option('--print-operative', is_flag=True, default=False, help='Print operative config.')
@gin_options
@click.pass_context
def train(ctx, model, config, gin_param, gin_file, path, checkpoint, overwrite=False, print_operative=False):
    """Interface for training models by using syntax MODEL/DATASET.
    Training is executed with default config values found in config/train directory. 
    See config files or print operative configuration for configurable parameters.

    \b
    Example of modifying default config:
        disentangled train BetaVAE/DSprites -p iterations=1000

    \b
    Available Models:
        BetaVAE
        BetaSVAE
        BetaTCVAE
        FactorVAE
    Available Datasets:
        Shapes3d
        DSprites
        CelebA
    """
    add_gin(ctx, 'config', ['train/{}.gin'.format(model)], insert=True)

    if overwrite is True:
        add_gin(ctx, 'gin_param' ['disentangled.model.utils.save.overwrite=True'])
    parse(ctx)

    disentangled.training.run_training(model=gin.REQUIRED, dataset=gin.REQUIRED, iterations=gin.REQUIRED, path=path, checkpoint=checkpoint)

    if print_operative:
        print(gin.operative_config_str())
