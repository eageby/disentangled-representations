import gin
import tensorflow as tf


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
