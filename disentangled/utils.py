import os
from contextlib import contextmanager
from pathlib import Path

import click
import gin
import tensorflow as tf
import tqdm
from decouple import config


@gin.configurable
def get_data_path(data_path=None):
    if data_path is None:
        return Path(config("DISENTANGLED_REPRESENTATIONS_DIRECTORY"))
    return Path(data_path)


@gin.configurable
def get_logs_path(prefix=None, name=None, suffix=None, hyperparameter_index=None, random_seed=None):
    base = get_data_path() / "logs"

    for i in [prefix, name, suffix]:
        if i is not None:
            base /= i

    if hyperparameter_index is not None:
        base /= 'HP{}'.format(hyperparameter_index)

    if random_seed is not None:
        base /= 'RS{}'.format(random_seed)


    base.mkdir(exist_ok=True, parents=True)

    subdir = [b for b in base.iterdir()]

    if len(subdir) == 0:
        counter = 0
    else:
        counter = max([int(b.parts[-1]) for b in subdir])

    return base / str(counter + 1)


def get_config_path():
    return Path(__file__).resolve().parent / "config"


@contextmanager
def config_path():
    old = Path(".")
    os.chdir(get_config_path())
    try:
        yield
    finally:
        os.chdir(old)


def parse_config_file(path):
    with config_path():
        gin.parse_config_file(path)


def markdownify_operative_config_str(string):
    """Convert an operative config string to markdown format.
        Pasted from future release of Gin-config"""

    def process(line):
        """Convert a single line to markdown format."""
        if not line.startswith("#"):
            return "    " + line

        line = line[2:]
        if line.startswith("===="):
            return ""

        if line.startswith("None"):
            return "    # None."

        if line.endswith(":"):
            return "#### " + line

        return line

    output_lines = []

    for line in string.splitlines():
        procd_line = process(line)

        if procd_line is not None:
            output_lines.append(procd_line)

    return "\n".join(output_lines)


@gin.configurable
class OperativeConfigCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(OperativeConfigCallback, self).__init__()
        self.writer = tf.summary.create_file_writer(logdir=str(log_dir / "train"))

    def on_train_begin(self, batch):
        with self.writer.as_default():
            config = markdownify_operative_config_str(gin.operative_config_str())
            tf.summary.text("operative_config", config, step=0)

    def on_train_end(self, batch):
        with self.writer.as_default():
            config = markdownify_operative_config_str(gin.operative_config_str())
            tf.summary.text("operative_config", config, step=batch)

class TrainingProgress(tqdm.tqdm):
    def __init__(self, iterable, **kwargs):

        left_bar = "    {n_fmt}/{total_fmt} ["
        right_bar = "] - ETA: {remaining} -{rate_inv_fmt}{postfix}"
        bar_format = left_bar + "{bar}" + right_bar

        super().__init__(
            iterable=iterable,
            bar_format=bar_format,
            ascii=".>>=",
            unit="it",
            dynamic_ncols=True,
            position=0,
            **kwargs
        )

    def update(self, logs, interval=10):
        if self.n % interval == 0:
            self.postfix = "Loss: {loss:.2f}, ".format(loss=logs['loss']) + ", ".join(
                key + ": " + "{:.2f}".format(logs[key]) for key in logs.keys() if key is not 'loss'
            )
            self.refresh()

@gin.configurable
def dataset(get, dataset, supervised=None, ordered=None):
    if get == 'dataset':
        return dataset
    elif get == 'supervised':
        return supervised
    elif get == 'ordered':
        return ordered

@gin.configurable
def model(model):
    return model

@gin.configurable
def hyperparameter(default, values, index=None, get_index=False):
    if get_index:
        return index
    elif index is None:
        return default
    else:
        return values[index]
