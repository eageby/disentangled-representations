from decorator import decorator
import click
import subprocess
import sys
import gin
import disentangled.utils

from disentangled.cli.utils import _MODELS, gin_options

def write_completion(model):
    data_dir = disentangled.utils.get_data_path()
    path = data_dir/ 'experiment' / model / 'experiment.complete'

    path.parent.mkdir(exist_ok=True, parents=True)
    path.touch(exist_ok=True)

    with open(path, 'w') as file:
        file.write("True")
        

@decorator
@gin.configurable
def random_seed_sweep(func, *args, random_seed_list=None, **kwargs):
    for seed in random_seed_list:
        calling_args = list(args).copy()
        calling_args += ['--gin-param', 'RANDOM_SEED={}'.format(seed)]

        returncode = func(*calling_args, **kwargs)
        if returncode == 1:
            breakpoint()
            return

    write_completion(args[1])

@decorator
@gin.configurable
def hyperparameter_sweep(func, *args, n_values, **kwargs):
    for index in range(n_values):
        calling_args = list(args).copy()
        calling_args += ['--gin-param', 'HP_INDEX={}'.format(index)]
        return func(*calling_args, **kwargs)

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("model", type=click.Choice(_MODELS))
@click.pass_context
@random_seed_sweep
@hyperparameter_sweep
def experiment(ctx, model, *args, **kwargs):
    dataset = model.split('/')[1]
    
    calling_args = ['disentangled', 'train', model]
    calling_args += ['--config', 'experiment/experiment.gin']
    calling_args += ['--config', 'experiment/full_metric.gin']
    calling_args += ['--config', 'evaluate/dataset/{}.gin'.format(dataset)]
    calling_args += args
    calling_args += ctx.args

    print(' '.join(calling_args))

    completed = subprocess.run(calling_args, stdout=sys.stdout)
    return completed.returncode

disentangled.utils.parse_config_file('experiment/experiment.gin')
