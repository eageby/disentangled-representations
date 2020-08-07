#!/usr/bin/env python

import functools
import itertools
import pprint
from datetime import datetime
from pathlib import Path

import numpy as np
from decouple import config

N_MODELS = 4
N_DATASETS = 2
N_HP = 5
N_RS = 5

N_CHAR_SEP = 60
SEPARATORS = {"light": N_CHAR_SEP * u"\u2500", "heavy": N_CHAR_SEP * u"\u2501"}


def get_index(data, elem):
    idx = []

    for i, d in enumerate(data):
        if elem[i] in d:
            idx.append(d.index(elem[i]))
        else:
            idx.append(False)

    return idx


def main():
    status = np.zeros((N_MODELS, N_DATASETS, N_HP, N_RS))

    path = Path(config("DISENTANGLED_REPRESENTATIONS_DIRECTORY")) / \
        "experiment"

    completed = [i.parent.parts[-4:] for i in path.glob("**/*complete")]

    def extract(j): return sorted(list(set([i[j] for i in completed])))
    models, datasets, hyperparameter, random_seed = map(extract, range(4))

    for i in completed:
        idx = get_index([models, datasets, hyperparameter, random_seed], i)
        status[idx[0], idx[1], idx[2], idx[3]] = 1

    print(SEPARATORS["heavy"])
    print(datetime.now())
    print(SEPARATORS["light"])
    total = np.mean(status)
    print("Total  {:.0%}".format(total))

    print(SEPARATORS["light"])
    print(*["{:<15}".format(i) for i in models])
    model_status = np.mean(status, axis=(1, 2, 3))
    print(*["{:<15.0%}".format(i) for i in model_status])
    print(SEPARATORS["light"])

    print(*["{:<15}".format(i) for i in datasets])
    model_status = np.mean(status, axis=(0, 2, 3))
    print(*["{:<15.0%}".format(i) for i in model_status])

    prev_m = -1

    for m, d in itertools.product(range(N_MODELS), range(N_DATASETS)):
        if m != prev_m:
            print(SEPARATORS["light"])
        print("{:<30}".format("/".join([models[m], datasets[d]])), end="")
        print("{:<15.0%}".format(np.mean(status, axis=(2, 3))[m, d]))
        prev_m = m

    print(SEPARATORS["heavy"])


main()
