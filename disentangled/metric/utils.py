import csv
import gin
import numpy as np
import tensorflow as tf
import functools
import disentangled.utils
from decorator import decorator


@gin.configurable
class MetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, metric_fn, metric_name, log_dir, interval=1):
        self.metric_fn = metric_fn
        self.interval = interval
        self.metric_name = metric_name
        self.writer = tf.summary.create_file_writer(
            logdir=str(log_dir / "train"))
        self.last_batch_write = 0

    def _write(self, batch):
        if batch == self.last_batch_write:
            return

        self.last_batch_write = batch
        metric_result = self.metric_fn(model=self.model)

        with self.writer.as_default():
            tf.summary.scalar(self.metric_name, metric_result, step=batch)

    def on_train_batch_end(self, batch, logs=None):
        tf.summary.experimental.set_step(batch)

        if ((batch % self.interval) == 0 and batch != 0) or (
            batch - self.last_batch_write > self.interval
        ):
            self._write(batch)

    def on_train_end(self, logs=None):
        self._write(tf.summary.experimental.get_step())

@gin.configurable
def log_metric(metric, metric_name, name=None, print_=False):
    result = metric()
    if print_:
        print("Metric {}: {:.2f}".format(metric_name, result))
  
    if name is not None:
        path = disentangled.utils.get_data_path() / 'metric' / metric_name / (name + '.data')
        path.parent.mkdir(exist_ok=True, parents=True)
        path.touch(exist_ok=True)
        with open(path, 'w') as file:
            file.write("{}".format(result))

    return result

def intact_dimensions_kld(model, data, tolerance, prior_dist, subset=None, progress_bar=True):
    kld = []

    if subset is not None:
        data = data.take(subset)

    if progress_bar:
        progress = disentangled.utils.TrainingProgress(data, total=subset)
        progress.write("Calculating Collapsed Latent Dimensions")
    else:
        progress = data

    for batch in progress:
        mean, log_var = model.encode(batch)

        kld.append(
            tf.reduce_mean(
                prior_dist.kld(mean, log_var), axis=0
            )
        )

    kld = tf.reduce_mean(tf.stack(kld, axis=0), axis=0)
    idx = np.where(kld > tolerance)[0]
    if progress_bar:
        progress.write("{} collapsed dimensions".format(kld.shape[-1] - len(idx)))

    return idx
