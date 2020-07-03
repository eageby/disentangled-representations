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
@gin_options
@click.pass_context
def evaluate(ctx, model, **kwargs):
    ctx.obj["model"] = disentangled.model.utils.load(model)
    ctx.obj["model_str"] = model
    method, dataset = model.split("/")

    ctx.obj["dataset"] = disentangled.dataset.get(dataset)
    ctx.obj["method_str"] = method
    ctx.obj["dataset_str"] = dataset

    add_gin(ctx, "config", ["evaluate/evaluate.gin"])
    add_gin(ctx, "config", ["evaluate/dataset/" + dataset + ".gin"])


@evaluate.command()
@gin_options
@click.pass_context
def gini_index(ctx, **kwargs):
    add_gin(ctx, "config", ["metric/gini.gin"])
    parse(ctx)

    metric = disentangled.metric.gini_index(
        ctx.obj["model"],
        dataset=gin.REQUIRED,
        samples=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        tolerance=gin.REQUIRED,
    )
    disentangled.metric.log_metric(
        metric, name=ctx.obj["model_str"], metric_name=gin.REQUIRED
    )


@evaluate.command()
@gin_options
@click.pass_context
def mig(ctx, **kwargs):
    add_gin(ctx, "config", ["metric/mig.gin"])
    parse(ctx)

    metric = disentangled.metric.mutual_information_gap(
        ctx.obj["model"],
        dataset=gin.REQUIRED,
        batches=gin.REQUIRED,
        batch_size=gin.REQUIRED,
    )
    disentangled.metric.log_metric(
        metric, metric_name=gin.REQUIRED, name=ctx.obj["model_str"]
    )


@evaluate.command()
@gin_options
@click.pass_context
def factorvae_score(ctx, **kwargs):
    add_gin(ctx, "config", ["metric/factorvae_score.gin"])
    parse(ctx)

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
@visual_options
@gin_options
@click.pass_context
def visual(ctx, rows, cols, plot, filename, **kwargs):
    parse(ctx)

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
@visual_options
@gin_options
@click.pass_context
def visual_compare(ctx, rows, cols, plot, filename, **kwargs):
    parse(ctx)
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
@visual_options
@gin_options
@click.pass_context
def latent1d(ctx, rows, cols, plot, filename, **kwargs):
    """Latent space traversal in 1D"""
    add_gin(ctx, "config", ["evaluate/visual/latent1d.gin"])
    parse(ctx)

    with gin.unlock_config():
        gin.bind_parameter(
            "disentangled.visualize.show.output.show_plot", plot)

        if filename is not None:
            gin.bind_parameter(
                "disentangled.visualize.show.output.filename", filename)

        if rows is not None:
            gin.bind_parameter(
                "disentangled.visualize.traversal1d.rows", rows)

        if cols is not None:
            gin.bind_parameter(
                "disentangled.visualize.traversal1d.cols", cols)

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
@visual_options
@gin_options
@click.pass_context
def latent2d(ctx, rows, cols, plot, filename, **kwargs):
    """Latent space traversal in 2D"""
    add_gin(ctx, "config", ["evaluate/visual/latent2d.gin"])
    parse(ctx)

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
    parse(ctx)
    dataset = ctx.obj["dataset"].pipeline()

    disentangled.visualize.gui(ctx.obj["model"], dataset)
