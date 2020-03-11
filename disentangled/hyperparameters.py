import sys


def complete(model_name: str, hyperparameters: dict) -> dict:
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
    message = """{}
Default Hyperparameters""".format(
        "\u2500" * 40
    )
    print(message)
    print_(model_name, get_default(model_name))


def print_(model_name: str, parameters: dict, dataset: str = None):
    line = "\u2500" * 40

    message = """{line}
Model:          {name}
{line}
Iterations:     {iterations:.0e}
Batch Size:     {batch_size}
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


betavae_mnist = {
    "iterations": 2e4,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "beta": 4,
    "latents": 32,
}

betavae_shapes3d = {
    "iterations": 5e5,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "beta": 4,
    "latents": 32,
}
