import sys

import tensorflow as tf

import disentangled.dataset
import disentangled.model

__all__ = ["train"]


def train(model_name: str, dataset_name: str, hyperparameters: dict) -> None:
    tf.random.set_seed(10)

    data = disentangled.dataset.get(dataset_name)
    model = disentangled.model.get(model_name)

    hyperparameters = complete_hyperparameters(model_name, hyperparameters)
    print_parameters(model_name, hyperparameters, dataset_name)

    optimizer = tf.keras.optimizers.get(optimizer_identifier(hyperparameters))
    model.compile(optimizer)
    model.fit(data.pipeline().repeat(), steps_per_epoch=hyperparameters['iterations'])

    return model


def complete_hyperparameters(model_name: str, hyperparameters: dict) -> dict:
    hyperparameters = {
        key: hyperparameters[key]

        for key in hyperparameters

        if hyperparameters[key] is not None
    }
    hyperparameters_default = get_default(model_name)
    hyperparameters_default.update(hyperparameters)

    return hyperparameters_default


def get_default(model_name: str) -> dict:
    return getattr(sys.modules[__name__], model_name)


def print_default(model_name: str) -> None:
    print_parameters(model_name, get_default(model_name))


def print_parameters(model_name: str, parameters: dict, dataset: str = None):
    line = "\u2500" * 30

    message = """{line}
{name}
{line}
Iterations:     {iterations:.0e}
Batch Size:     {batch_size}
Optimizer:      {optimizer} 
Learning Rate:  {learning_rate:.1e}
{line}
""".format(
        line=line, name=model_name, **parameters
    )
    if dataset:
        message += """Dataset:        {dataset}
{line}""".format(
            line=line, dataset=dataset
        )

    print(message)


def optimizer_identifier(parameters: dict) -> dict:
    valid_config_fields = [
        "learning_rate",
        "beta_1",
        "beta_2",
        "epsilon",
        "amsgrad",
        "name",
    ]

    config = {key: parameters[key] for key in parameters if key in valid_config_fields}

    return {"class_name": parameters["optimizer"], "config": config}


betavae_mnist = {
    "iterations": 2e4,
    "batch_size": 128,
    "optimizer": "adam",
    "learning_rate": 1e-3,
}

betavae_shapes3d = {
    "iterations": 5e5,
    "batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 1e-4,
}
