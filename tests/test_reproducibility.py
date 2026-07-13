import random

import numpy as np
import torch

from reproducibility import seed_everything


def test_seed_everything_repeats_python_numpy_and_torch():
    first_meta = seed_everything(2023)
    first = (random.random(), np.random.random(), torch.rand(4))
    second_meta = seed_everything(2023)
    second = (random.random(), np.random.random(), torch.rand(4))
    assert first_meta["seed"] == second_meta["seed"] == 2023
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
