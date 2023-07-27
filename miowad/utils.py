import os
import random

import numpy as np


def shuffle(x: np.ndarray, y: np.ndarray):
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return x[p], y[p]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    os.environ['PYTHONSEEDHASH'] = str(seed)
