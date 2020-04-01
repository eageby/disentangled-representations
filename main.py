from pathlib import Path

import click

import disentangled.hyperparameters
import disentangled.model
import disentangled.training
import disentangled.metric
import disentangled.utils
import disentangled.visualize.latentspace

_MODELS = ["factorvae", "betavae"]
_DATASETS = ["MNIST", "Shapes3d"]


def load(ctx): 
    if ctx.obj['model'] is None: 
        ctx.obj['model'] = disentangled.model.utils.load(ctx.obj['model_name'], ctx.obj['directory'])
    return ctx

@click.group(
    chain=True,
    context_settings=dict(
        help_option_names=["-h", "--help"],
        ignore_unknown_options=True,
        allow_extra_args=True
    ),
)
@click.argument(
    "model", type=click.Choice(_MODELS, case_sensitive=False),
)
@click.argument(
    "dataset", type=click.Choice(_DATASETS, case_sensitive=False),
)
@click.option("--no-gpu", is_flag=True, default=False)
@click.option("--no-info", is_flag=True, default=False)
@click.option(
    "--directory",
    "-d",
    type=click.Path(writable=True),
    default=Path("./models"),
    show_default=True,
)
@click.pass_context
def cli(ctx, model, dataset, no_gpu, no_info, directory):
    disentangled.utils.config()

    if no_gpu:
        disentangled.utils.disable_gpu()
    if no_info:
        disentangled.utils.disable_info_output(2)

    ctx.ensure_object(dict)
    ctx.obj["directory"] = directory

    ctx.obj["model_name"] = "_".join([model, dataset]).lower()

    ctx.obj["model"] = None

    ctx.obj["dataset_name"] = dataset
    ctx.obj["dataset"] = disentangled.dataset.get(dataset)


@cli.command(cls=disentangled.utils.AcceptAllCommand)
@click.option("--learning_rate", "-l", type=float)
@click.option("--batch_size", "-b", type=int)
@click.option("--iterations", "-i", type=float)
@click.option("--save/--no-save", "-s", default=True)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--show_default", "-D", is_flag=True, default=False)
@click.pass_context
def train(ctx, save, overwrite, show_default, **kwargs):
    """train model"""
    ctx.obj["model"] = disentangled.model.get(ctx.obj["model_name"])

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
        disentangled.model.get(ctx.obj["model_name"]),
        ctx.obj["dataset"],
        **hyperparameters
    )

    if save:
        disentangled.model.save(
            model, ctx.obj["model_name"], path=ctx.obj["directory"], overwrite=overwrite
        )

    ctx.obj["model"] = model


@cli.command()
@click.option("--rows", "-r", type=int, default=4)
@click.option("--cols", "-c", type=int, default=8)
@click.pass_context
def results(ctx, **kwargs):
    """Compare reconstruction with target"""
    ctx = load(ctx)
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
    ctx = load(ctx)
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
    ctx = load(ctx)
    disentangled.visualize.latentspace.traversal_2d(
        ctx.obj["model"], ctx.obj["dataset"], **kwargs
    )

@cli.command()
@click.option("training_votes", "--train", type=int, default=500)
@click.option("test_votes", "--test", type=int, default=800)
@click.option("--batch_size", type=int, default=100)
@click.pass_context
def metric(ctx, batch_size, **kwargs):
    ctx = load(ctx)
    error_rate = disentangled.metric.metric_factorvae(ctx.obj['model'], ctx.obj['dataset'].ordered.create(batch_size), **kwargs)

    print("Error Rate: {:%}".format(error_rate))
