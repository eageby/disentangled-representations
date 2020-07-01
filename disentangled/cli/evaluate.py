import functools
import itertools

import click
import disentangled
import disentangled.metric
import disentangled.model.utils
import disentangled.training
import disentangled.utils
import gin

import itertools
_METHODS = ["FactorVAE", "BetaVAE", "BetaTCVAE", "BetaSVAE"]
_DATASETS = ["DSprites", "Shapes3d"]
_MODELS = ['/'.join(i) for i in itertools.product(_METHODS, _DATASETS)]


def gin_options(func):
    @functools.wraps(func)
    @click.option("--config", "-c", multiple=True, callback=add_gin)
    @click.option(
        "--gin-param", "--gin-parameter", "-p", multiple=True, callback=add_gin
    )
    @click.option("--gin-file", "-f", multiple=True, callback=add_gin)
    def _(*args, **kwargs):
        return func(*args, **kwargs)

    return _


def add_gin(ctx, param, value):
    ctx.ensure_object(dict)

    if not isinstance(param, str):
        param = param.name

    if param not in ctx.obj.keys():
        ctx.obj[param] = []

    ctx.obj[param] += list(value)


def parse(ctx):
    for config in ctx.obj["config"]:
        disentangled.utils.parse_config_file(config)

    gin.parse_config_files_and_bindings(
        ctx.obj["gin_file"], ctx.obj["gin_param"], finalize_config=True
    )


@click.group()
@click.argument("model", type=click.Choice(_MODELS))
@gin_options
@click.pass_context
def evaluate(ctx, model, **kwargs):
    ctx.obj["model"] = disentangled.model.utils.load(model)
    ctx.obj["model_str"] = model
    method, dataset = model.split("/")

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
        training_votes=gin.REQUIRED,
        test_votes=gin.REQUIRED,
        tolerance=gin.REQUIRED,
    )
    disentangled.metric.log_metric(
        metric, name=ctx.obj["model_str"], metric_name=gin.REQUIRED
    )
