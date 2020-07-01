import itertools

import click
import disentangled
import disentangled.metric
import disentangled.model.utils
import disentangled.training
import disentangled.utils
import gin

@click.group()
@click.argument("model")
@click.option("--config", "-c", multiple=True)
@click.option('--gin-param',"--gin-parameter",  "-p", multiple=True)
@click.option("--gin-file", "-f", multiple=True)
@click.pass_context
def evaluate(ctx, model, config, gin_param, gin_file):
    ctx.ensure_object(dict)
    ctx.obj["model"] = disentangled.model.utils.load(model)
    ctx.obj["model_str"] = model

    method, dataset = model.split('/')

    disentangled.utils.parse_config_file('evaluate/evaluate.gin')
    disentangled.utils.parse_config_file('evaluate/dataset/'+dataset + '.gin')

    for c in config:
        disentangled.utils.parse_config_file(c)

    gin.parse_config_files_and_bindings(gin_file, gin_param)

@evaluate.command()
@click.pass_context
def gini_index(ctx):
    with gin.unlock_config():
        disentangled.utils.parse_config_file("metric/gini.gin")

    metric = disentangled.metric.gini_index(
        ctx.obj["model"],
        dataset=gin.REQUIRED,
        samples=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        tolerance=gin.REQUIRED,
    )
    disentangled.metric.log_metric(metric,
            name=ctx.obj['model_str'],
            metric_name=gin.REQUIRED
            )


@evaluate.command()
@click.pass_context
def mig(ctx):
    with gin.unlock_config():
        disentangled.utils.parse_config_file("metric/mig.gin")

    metric = disentangled.metric.mutual_information_gap(
        ctx.obj["model"],
        dataset=gin.REQUIRED,
        batches=gin.REQUIRED,
        batch_size=gin.REQUIRED,
    )
    disentangled.metric.log_metric(metric,
            metric_name=gin.REQUIRED,
            name=ctx.obj['model_str']
            )

@evaluate.command()
@click.pass_context
def factorvae_score(ctx):
    with gin.unlock_config():
        disentangled.utils.parse_config_file("metric/factorvae_score.gin")

    metric = disentangled.metric.factorvae_score(
        ctx.obj["model"],
        dataset=gin.REQUIRED,
        training_votes=gin.REQUIRED,
        test_votes=gin.REQUIRED,
        tolerance=gin.REQUIRED,
    )
    disentangled.metric.log_metric(metric,
            name=ctx.obj['model_str'],
            metric_name=gin.REQUIRED,
            )
