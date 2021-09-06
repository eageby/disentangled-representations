import functools
import itertools

import click
import disentangled
import disentangled.metric
import disentangled.model.utils
import disentangled.training
import disentangled.utils
import gin
from disentangled.cli.utils import _MODELS, add_gin, gin_options, visual_options, parse


@click.group()
@click.argument("model", type=click.Choice(_MODELS))
@click.option('--path', type=click.Path(exists=True, readable=True, dir_okay=True), default=None)
@gin_options
@click.pass_context
def evaluate(ctx, model, path, **kwargs):
    """Interface for evaluating models.
    Quantitative and qualitative evaluation methods are provided.
    """
    ctx.obj["model"] = disentangled.model.utils.load(path, model)
    ctx.obj["model_str"] = model
    method, dataset = model.split("/")

    ctx.obj["dataset"] = disentangled.dataset.get(dataset)
    ctx.obj["method_str"] = method
    ctx.obj["dataset_str"] = dataset

    add_gin(ctx, "gin_param", ["HP_SWEEP_VALUES=None"])
    add_gin(ctx, "gin_param", ["log_metric.path='{}'".format(path)])
    add_gin(ctx, "config", ["evaluate/evaluate.gin"])
    add_gin(ctx, "config", ["evaluate/dataset/" + dataset + ".gin"])
    add_gin(ctx, "config", ["evaluate/model/" + method + ".gin"])


@evaluate.command()
@gin_options
@click.pass_context
def gini_index(ctx, **kwargs):
    """Quantitative sparsity metric of representation."""
    add_gin(ctx, "config", ["metric/gini.gin"])
    parse(ctx, set_seed=True)

    metric = disentangled.metric.gini_index(
        ctx.obj["model"],
        dataset=gin.REQUIRED,
        points=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        tolerance=gin.REQUIRED,
    )
    disentangled.metric.log_metric(
        metric, name=ctx.obj["model_str"], metric_name=gin.REQUIRED
    )

@evaluate.command()
@gin_options
@click.pass_context
def collapsed(ctx, **kwargs):
    """Number of latent dimensions collapsed to prior."""
    add_gin(ctx, "config", ["metric/collapsed.gin"])
    parse(ctx, set_seed=True)

    metric = disentangled.metric.collapsed(
        ctx.obj["model"],
        dataset=gin.REQUIRED,
        tolerance=gin.REQUIRED,
    )
    disentangled.metric.log_metric(
        metric, name=ctx.obj["model_str"], metric_name=gin.REQUIRED
    )

@evaluate.command()
@gin_options
@click.pass_context
def loglikelihood(ctx, **kwargs):
    """The logarithmic likelihood of the input given the representation."""
    add_gin(ctx, "config", ["metric/loglikelihood.gin"])
    parse(ctx, set_seed=True)

    metric = disentangled.metric.loglikelihood(
        ctx.obj["model"],
        dataset=gin.REQUIRED,
    )
    breakpoint()
    disentangled.metric.log_metric(
        metric, name=ctx.obj["model_str"], metric_name=gin.REQUIRED
    )

@evaluate.command()
@gin_options
@click.pass_context
def mig(ctx, **kwargs):
    """Mutual Information GAP.
    Quantitative disentanglement metric."""
    add_gin(ctx, "config", ["metric/mig.gin"])
    parse(ctx, set_seed=True)

    metric = disentangled.metric.mutual_information_gap(
        ctx.obj["model"],
        dataset=gin.REQUIRED,
        encoding_dist=gin.REQUIRED,
    )
    disentangled.metric.log_metric(
        metric, metric_name=gin.REQUIRED, name=ctx.obj["model_str"]
    )

# @evaluate.command()
# @gin_options
# @click.pass_context
# def mig_batch(ctx, **kwargs):
#     add_gin(ctx, "config", ["metric/mig_batch.gin"])
#     parse(ctx, set_seed=True)

#     metric = disentangled.metric.mutual_information_gap_batch(
#         ctx.obj["model"],
#         dataset=gin.REQUIRED,
#         encoding_dist=gin.REQUIRED,
#     )
#     disentangled.metric.log_metric(
#         metric, metric_name=gin.REQUIRED, name=ctx.obj["model_str"]
#     )

@evaluate.command()
@gin_options
@click.pass_context
def dmig(ctx, **kwargs):
    """Discrete Mutual Information Gap.
    Quantitative disentanglement metric."""
    add_gin(ctx, "config", ["metric/dmig.gin"])
    parse(ctx, set_seed=True)

    metric = disentangled.metric.discrete_mutual_information_gap(
        ctx.obj["model"],
        dataset=gin.REQUIRED,
        points=gin.REQUIRED,
        batch_size=gin.REQUIRED
    )
    disentangled.metric.log_metric(
        metric, metric_name=gin.REQUIRED, name=ctx.obj["model_str"]
    )


@evaluate.command()
@gin_options
@click.pass_context
def factorvae_score(ctx, **kwargs):
    """The FactorVAE-score.
    Quantitative disentanglement metric."""
    add_gin(ctx, "config", ["metric/factorvae_score.gin"])
    parse(ctx, set_seed=True)

    metric = disentangled.metric.factorvae_score(
        ctx.obj["model"],
        dataset=gin.REQUIRED,
        training_points=gin.REQUIRED,
        test_points=gin.REQUIRED,
        tolerance=gin.REQUIRED,
    )
    disentangled.metric.log_metric(
        metric, name=ctx.obj["model_str"], metric_name=gin.REQUIRED
    )


@evaluate.command()
@gin_options
@visual_options
@click.pass_context
def visual(ctx, rows, cols, plot, filename, **kwargs):
    """Qualitative evaluation of output."""
    parse(ctx, set_seed=True)

    with gin.unlock_config():
        gin.bind_parameter(
            "disentangled.visualize.show.output.show_plot", plot)

        if filename is not None:
            gin.bind_parameter(
                "disentangled.visualize.show.output.filename", filename)

        if rows is not None:
            gin.bind_parameter(
                "disentangled.visualize.reconstructed.rows", rows)

        if cols is not None:
            gin.bind_parameter(
                "disentangled.visualize.reconstructed.cols", cols)

    dataset = ctx.obj["dataset"].pipeline()
    disentangled.visualize.reconstructed(
        ctx.obj["model"], dataset, rows=gin.REQUIRED, cols=gin.REQUIRED
    )


@evaluate.command()
@gin_options
@visual_options
@click.pass_context
def visual_compare(ctx, rows, cols, plot, filename, **kwargs):
    """Qualitative evaluation of output in comparison to input."""
    parse(ctx, set_seed=True)
    with gin.unlock_config():
        gin.bind_parameter(
            "disentangled.visualize.show.output.show_plot", plot)

        if filename is not None:
            gin.bind_parameter(
                "disentangled.visualize.show.output.filename", filename)

        if rows is not None:
            gin.bind_parameter("disentangled.visualize.comparison.rows", rows)

        if cols is not None:
            gin.bind_parameter("disentangled.visualize.comparison.cols", cols)

    dataset = ctx.obj["dataset"].pipeline()
    disentangled.visualize.comparison(
        ctx.obj["model"], dataset, rows=gin.REQUIRED, cols=gin.REQUIRED
    )


@evaluate.command()
# @visual_options
@click.option("--filename")
@click.option("--rows", type=int)
@click.option("--cols", type=int)
@click.option("plot", "--plot/--no-plot", is_flag=True, default=True)
@gin_options
@click.pass_context
def latent1d(ctx, rows, cols, plot, filename, **kwargs):
    """Latent space traversal in 1D."""
    add_gin(ctx, "config", ["evaluate/visual/latent1d.gin"])
    parse(ctx, set_seed=True)

    with gin.unlock_config():
        gin.bind_parameter(
            "disentangled.visualize.show.output.show_plot", plot)

        if filename is not None:
            gin.bind_parameter(
                "disentangled.visualize.show.output.filename", filename)

        if rows is not None:
            gin.bind_parameter(
                "disentangled.visualize.traversal1d.dimensions", rows)

        if cols is not None:
            gin.bind_parameter(
                "disentangled.visualize.traversal1d.steps", cols)

    dataset = ctx.obj["dataset"].pipeline()
    disentangled.visualize.traversal1d(
        ctx.obj["model"],
        dataset,
        dimensions=gin.REQUIRED,
        offset=gin.REQUIRED,
        skip_batches=gin.REQUIRED,
        steps=gin.REQUIRED,
    )


@evaluate.command()
# @visual_options
@click.option("--filename")
@click.option("--rows", type=int)
@click.option("--cols", type=int)
@click.option("plot", "--plot/--no-plot", is_flag=True, default=True)
@gin_options
@click.pass_context
def latent2d(ctx, rows, cols, plot, filename, **kwargs):
    """Latent space traversal in 2D."""
    add_gin(ctx, "config", ["evaluate/visual/latent2d.gin"])
    parse(ctx, set_seed=True)

    with gin.unlock_config():
        gin.bind_parameter(
            "disentangled.visualize.show.output.show_plot", plot)
        gin.bind_parameter(
            "disentangled.visualize.show.output.filename", filename)

        if rows is not None:
            gin.bind_parameter(
                "disentangled.visualize.traversal2d.rows", rows)

        if cols is not None:
            gin.bind_parameter(
                "disentangled.visualize.traversal2d.cols", cols)

    dataset = ctx.obj["dataset"].pipeline()
    disentangled.visualize.traversal2d(ctx.obj["model"], dataset)


@evaluate.command()
@gin_options
@click.pass_context
def gui(ctx, **kwargs):
    """Qualitiative interactive evaluation of disentanglement."""
    add_gin(ctx, "config", ["evaluate/visual/gui.gin"])
    parse(ctx, set_seed=True)
    dataset = ctx.obj["dataset"].pipeline()

    disentangled.visualize.gui(ctx.obj["model"], dataset)
