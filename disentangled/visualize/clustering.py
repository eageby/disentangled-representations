import disentangled.utils
import gin
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import umap


@gin.configurable
def clustering(model, dataset, batches, batch_size):
    dataset = dataset.batch(batch_size).take(batches)

    reducer = umap.UMAP()

    for batch in dataset:
        mean, log_var = model.encode(batch["image"])
        samples = model.sample(mean, log_var)
        embedding_samples = reducer.fit_transform(samples)

        n_labels = batch["label"].shape[-1]
        n = np.ceil(np.sqrt(n_labels))

        for i in range(n_labels):
            ax = plt.subplot(n, n, i + 1)
            ax.scatter(
                embedding_samples[:, 0], embedding_samples[:, 1], c=batch["label"][:, i]
            )
            ax.set_title("Label {}".format(i))

        plt.suptitle("Samples")
        plt.show()

        embedding_samples = reducer.fit_transform(batch["label"])
        for i in range(n_labels):
            ax = plt.subplot(n, n, i + 1)
            ax.scatter(
                embedding_samples[:, 0], embedding_samples[:, 1], c=batch["label"][:, i]
            )
            ax.set_title("Label {}".format(i))

        plt.suptitle("Labels")
        plt.show()


if __name__ == "__main__":
    disentangled.utils.parse_config_file("clustering.gin")
    clustering(
        model=gin.REQUIRED,
        dataset=gin.REQUIRED,
        batches=gin.REQUIRED,
        batch_size=gin.REQUIRED,
    )
