import gin
import tensorflow as tf


@gin.configurable
class MetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, metric_fn, metric_name, log_dir, interval=1):
        self.metric_fn = metric_fn
        self.interval = interval
        self.metric_name = metric_name
        self.writer = tf.summary.create_file_writer(logdir=str(log_dir / "train"))

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.interval == 0:
            metric_result = self.metric_fn(model=self.model)

            with self.writer.as_default():
                tf.summary.scalar(self.metric_name, metric_result, step=batch)
