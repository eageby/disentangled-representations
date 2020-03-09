from pathlib import Path

import click

import disentangled.hyperparameters
import disentangled.model
import disentangled.training
import disentangled.utils
import disentangled.visualize.latentspace

_MODELS = ["betavae_mnist", "betavae_shapes3d"]
_DATASETS = {("betavae_mnist"): "MNIST", ("betavae_shapes3d"): "Shapes3d"}


@click.group(chain=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "model", type=click.Choice(_MODELS, case_sensitive=True),
)
@click.option("--no-gpu", is_flag=True, default=False)
@click.option(
    "--directory",
    "-d",
    type=click.Path(writable=True),
    default=Path("./models"),
    show_default=True,
)
@click.pass_context
def cli(ctx, model, no_gpu, directory):
    if no_gpu:
        disentangled.utils.disable_gpu()

    dataset = next(val for key, val in _DATASETS.items() if model in key)

    ctx.ensure_object(dict)
    ctx.obj['directory'] = directory

    ctx.obj["model_name"] = model

    if ctx.invoked_subcommand == 'train':
        ctx.obj["model"] = disentangled.model.get(model)
    else:
        ctx.obj["model"] = disentangled.model.utils.load(model, directory)

    ctx.obj["dataset_name"] = dataset
    ctx.obj["dataset"] = disentangled.dataset.get(dataset)


@cli.command()
@click.option("--learning_rate", "-l", type=float)
@click.option("--batch_size", "-b", type=int)
@click.option("--iterations", "-i", type=float)
@click.option("--save/--no-save", "-s", default=True)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--show_default", "-D", is_flag=True)
@click.pass_context
def train(ctx, save, overwrite, show_default, **kwargs):
    """train model"""

    if show_default:
        disentangled.hyperparameters.print_default(ctx.obj["model_name"])

        return

    hyperparameters = disentangled.hyperparameters.complete(
        ctx.obj["model_name"], kwargs
    )
    disentangled.hyperparameters.print_(
        ctx.obj["model_name"], hyperparameters, ctx.obj["dataset_name"]
    )

    model = disentangled.training.train(
        disentangled.model.get(ctx.obj["model_name"]), ctx.obj["dataset"], **hyperparameters
    )

    if save:
        disentangled.model.save(
            model, ctx.obj["model_name"], path=ctx.obj['directory'], overwrite=overwrite
        )

    ctx.obj["model"] = model


@cli.command()
@click.option("--rows", "-r", type=int, default=4)
@click.option("--cols", "-c", type=int, default=8)
@click.pass_context
def results(ctx, **kwargs):
    """Compare reconstruction with target"""
    disentangled.visualize.results(ctx.obj["model"], ctx.obj["dataset"], **kwargs)


@cli.command()
@click.option("--shuffle", is_flag=True, default=False)
@click.option("--steps", type=int, default=41)
@click.option("--sample", "-s", type=int, default=0)
@click.option("--dimensions", type=int, default=10)
@click.option("--offset", "-o", type=int, default=0)
@click.pass_context
def latent1d(ctx, **kwargs):
    """Latent space traversal in 1D"""
    disentangled.visualize.latentspace.traversal_1d(
        ctx.obj["model"], ctx.obj["dataset"], **kwargs
    )


@cli.command()
@click.option("--shuffle", is_flag=True, default=False)
@click.option("--steps", type=int, default=20)
@click.option("--sample", "-s", type=int, default=0)
@click.option("--offset", "-o", type=int, default=0)
@click.pass_context
def latent2d(ctx, **kwargs):
    """Latent space traversal in 2D"""
    disentangled.visualize.latentspace.traversal_2d(
        ctx.obj["model"], ctx.obj["dataset"], **kwargs
    )


cli()
