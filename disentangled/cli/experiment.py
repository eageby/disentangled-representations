from decorator import decorator
import click
import subprocess
import sys
import gin
import disentangled.utils

from disentangled.cli.utils import _MODELS, gin_options

def write_completion(model, random_seed_index, hp_index):
    data_dir = disentangled.utils.get_data_path()
    path = data_dir/ 'experiment' / model / 'HP{}'.format(hp_index) / 'RS{}'.format(random_seed_index) / 'experiment.complete'

    path.parent.mkdir(exist_ok=True, parents=True)
    path.touch(exist_ok=True)
    
    with open(path, 'w') as file:
        file.write("True")
        

@decorator
@gin.configurable
def hyperparameter_sweep(func, *args, n_values, hp_index=[], **kwargs):
    if not hp_index:
        hp_index = range(n_values)

    for index in hp_index:
        calling_args = list(args).copy()
        calling_args += ['--gin-param', 'HP_INDEX={}'.format(index)]
        func(*calling_args, hp_index=index, **kwargs)

@decorator
@gin.configurable
def random_seed_sweep(func, *args, hp_index, random_seed_list=None, random_seed_index=[], log=False, **kwargs):
    if random_seed_index:
        random_seed_list = [(i, v) for i, v in enumerate(random_seed_list) if i in random_seed_index]
    else:
        random_seed_list = [(i, v) for i, v in enumerate(random_seed_list)]

    for i, seed in random_seed_list:
        calling_args = list(args).copy()
        calling_args += ['--gin-param', 'RANDOM_SEED={}'.format(seed)]

        returncode = func(*calling_args, **kwargs)
        if returncode == 0 and log:
            write_completion(args[1], i, hp_index)


@click.command()
@click.argument("model", type=click.Choice(_MODELS))
@click.option('--random-seed-index', '-r', multiple=True, type=int)
@click.option('--hp-index', '-h', multiple=True, type=int)
@click.option('--log', is_flag=True, default=False)
@click.option('--dry-run', '-n', is_flag=True, default=False)
@click.pass_context
@hyperparameter_sweep
@random_seed_sweep
def experiment(ctx, model, *args, dry_run, **kwargs):
    """Interface for running full scale experiment.
    The experiment is run with random seed and hyperparameter sweeps.
    """
    dataset = model.split('/')[1]
    
    calling_args = ['disentangled', 'train', model]
    calling_args += ['--config', 'experiment/experiment.gin']
    calling_args += ['--config', 'experiment/full_metric.gin']
    calling_args += ['--config', 'evaluate/dataset/{}.gin'.format(dataset)]
    calling_args += args
    calling_args += ctx.args

    print(' '.join(calling_args))

    if dry_run:
        return 0

    completed = subprocess.run(calling_args, stdout=sys.stdout)
    return completed.returncode


disentangled.utils.parse_config_file('experiment/experiment.gin')
