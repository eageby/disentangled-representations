import disentangled.dataset
import disentangled.utils
import tensorflow as tf
import tqdm


def serialize_image(element):
    element["image"] = tf.io.serialize_tensor(element["image"])

    return element


def parse_image(element):
    element["image"] = tf.io.parse_tensor(element["image"], tf.float32)

    return element


def write(dataset, batches=1300, **kwargs):
    data = dataset.create(**kwargs).take(batches).map(serialize_image)
    progress = disentangled.utils.TrainingProgress(data, total=batches)

    dataset.serialized_path.parent.mkdir(exist_ok=True)

    progress.write("Serializing dataset to {}".format(dataset.serialized_path.resolve()))
    progress.write("batches: {}, {}".format(batches, ','.join(["{}: {}".format(k,v) for k,v in kwargs.items()])))
    with tf.io.TFRecordWriter(str(dataset.serialized_path)) as writer:
        for batch in progress:
            ex = dataset.example(batch)
            writer.write(ex.SerializeToString())


def read(dataset):
    data = tf.data.TFRecordDataset(str(dataset.serialized_path))

    return (
        data.map(
            dataset.parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(dataset.set_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    )
