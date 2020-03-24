import sys

def complete(model_name: str, hyperparameters: dict) -> dict:
    hyperparameters = {
        key: value for key, value in hyperparameters.items() if value is not None
    }
     
    hyperparameters_default = get_default(model_name)

    for key in hyperparameters:
        if key not in hyperparameters_default:
            raise ValueError("{} is not a valid hyperparameter. See default hyperparameters for {}.".format(key, model_name))

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

    parameter_str = "\n".join(["{:16.16}   {}".format(k,v) for k,v in parameters.items()])

    message = """{line}
Model              {name}
{line}
{parameters}
{line}
""".format(
        line=line, name=model_name, parameters=parameter_str
    )
    if dataset:
        message += """Dataset            {dataset}
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
