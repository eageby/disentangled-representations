import click
import disentangled.dataset
import disentangled.utils
import tensorflow as tf
import tqdm
import gin


def serialize_image(element):
    element["image"] = tf.io.serialize_tensor(element["image"])

    return element


def parse_image(element):
    element["image"] = tf.io.parse_tensor(element["image"], tf.float32)

    return element


@gin.configurable(module="disentangled.dataset.serialize")
def write(dataset, batches, overwrite=False, **kwargs):
    data = dataset.create().take(batches).map(serialize_image)

    path = dataset.get_serialized_path()
    path.parent.mkdir(exist_ok=True)

    if path.exists() and (
        overwrite
        or click.confirm("Do you want to overwrite {}".format(path), abort=True)
    ):
        path.unlink()

    progress = disentangled.utils.TrainingProgress(data, total=batches)
    progress.write("Serializing dataset to {}".format(path.resolve()))
    progress.write(
        "batches: {}, {}".format(
            batches, ",".join(["{}: {}".format(k, v) for k, v in kwargs.items()])
        )
    )
    with tf.io.TFRecordWriter(str(path)) as writer:
        for batch in progress:
            ex = dataset.example(batch)
            writer.write(ex.SerializeToString())


def read(dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    data = tf.data.TFRecordDataset(str(dataset.get_serialized_path()))

    return (
        data.map(dataset.parse_example, num_parallel_calls=num_parallel_calls)
        .map(parse_image, num_parallel_calls=num_parallel_calls)
        .map(dataset.set_shapes, num_parallel_calls=num_parallel_calls)
    )
