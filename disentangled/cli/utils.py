import click
import functools
import itertools
import disentangled.utils
import gin

_METHODS = ["FactorVAE", "BetaVAE", "BetaTCVAE", "BetaSVAE"]
_DATASETS = ["DSprites", "Shapes3d"]
_MODELS = ["/".join(i) for i in itertools.product(_METHODS, _DATASETS)]


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

def visual_options(func):
    @functools.wraps(func)
    @click.option("--filename")
    @click.option("--rows", type=int)
    @click.option("--cols", type=int)
    @click.option("plot", "--plot/--no-plot", is_flag=True, default=True)
    def _(*args, **kwargs):
        return func(*args, **kwargs)

    return _


def add_gin(ctx, param, value, insert=False):
    ctx.ensure_object(dict)

    if not isinstance(param, str):
        param = param.name

    if param not in ctx.obj.keys():
        ctx.obj[param] = []

    if insert:
        for v in value:
            ctx.obj[param].insert(0, v)
    else:
        ctx.obj[param] += list(value)


def parse(ctx):
    for config in ctx.obj["config"]:
        disentangled.utils.parse_config_file(config)

    gin.parse_config_files_and_bindings(
        ctx.obj["gin_file"], ctx.obj["gin_param"], finalize_config=True
    )