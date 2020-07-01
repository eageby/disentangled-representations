import csv
import gin
import numpy as np
import tensorflow as tf
import functools
import disentangled.utils
from decorator import decorator

@gin.configurable(module="disentangled.metric", blacklist=['model', 'data'])
def representation_variance(model, data, samples, batch_size, progress_bar=True):

    data = data.take(samples).batch(batch_size)

    if progress_bar:
        data = disentangled.utils.TrainingProgress(data, total=int(tf.math.ceil(samples/batch_size)))
        data.write("Calculating Empirical Variance")

    all_var = None
    for batch in data:
        var = tf.math.reduce_variance(model.encode(batch["image"]), axis=0)
        if all_var is None:
            all_var = var
        else:
            all_var = tf.concat([all_var, var], axis=0)

    return tf.math.reduce_mean(all_var, axis=0)


@gin.configurable(module="disentangled.metric", blacklist=['dataset'])
def fixed_factor_dataset(dataset, batch_size, num_values_per_factor, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    n_factors = dataset.element_spec['label'].shape[0]
    factor_set = tf.data.Dataset.range(n_factors).shuffle(n_factors).repeat()
    
    def map_to_batch(fixed_factor):
        fixed_factor_value = tf.cast(tf.random.uniform(shape=(), maxval=tf.gather(num_values_per_factor,fixed_factor), dtype=tf.int32), tf.uint8)

        def add_factor_data(element):
            element['factor'] = tf.cast(fixed_factor, dtype=tf.uint8)
            element['factor_value'] = fixed_factor_value
            return element

        return dataset.filter(lambda x: tf.equal(tf.gather(x['label'], fixed_factor), fixed_factor_value)).batch(batch_size).map(add_factor_data).take(1)

    return factor_set.interleave(map_to_batch, num_parallel_calls=num_parallel_calls)

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
    if print_:
        print("Metric {}: {:.2f}".format(metric_name, metric))
  
    if name is not None:
        path = disentangled.utils.get_data_path() / 'metric' / metric_name / (name + '.data')
        path.parent.mkdir(exist_ok=True, parents=True)
        path.touch(exist_ok=True)
        with open(path, 'w') as file:
            file.write("{}".format(metric))
    
