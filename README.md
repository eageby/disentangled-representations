# Disentangled Representations 
Full code for implementations and testing described in the thesis [Introducing Sparsity into the Current Landscape of Disentangled Representation Learning](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-292929).

Web-enabled demo and summary can be accessed [here.](https://eageby.github.io/disentangled-representations/).

# Command Line Interface (CLI)
## Main interface
```console
[Usage](Usage): disentangled [OPTIONS] COMMAND [ARGS]...

  Train and evaluate disentangled representation learning models.

Options:
  -c, --config TEXT               Add gin-configuration.
  -p, --gin-param, --gin-parameter TEXT
                                  Add gin-config parameter.
  -f, --gin-file TEXT             Specify gin-config file.
  -h, --help                      Show this message and exit.

Commands:
  dataset     Interface for viewing and preparing datasets.
  evaluate    Interface for evaluating models.
  experiment  Interface for running full scale experiment.
  train       Interface for training models by using syntax MODEL/DATASET.
  ```
