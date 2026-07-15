"""Lightweight reproducibility helpers shared by experiment entry points."""

import os
import random

import dgl
import numpy as np
import torch


def seed_everything(seed: int) -> dict:
    """Seed Python, NumPy, PyTorch/CUDA and DGL with one experiment seed."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    if hasattr(dgl, "random") and hasattr(dgl.random, "seed"):
        dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Disable TF32 for cuDNN to avoid "Unable to find a valid cuDNN algorithm"
    # on Ada Lovelace / Hopper GPUs with certain cuDNN + PyTorch 1.12 combinations.
    torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": seed,
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": True,
        "cuda_seeded": bool(torch.cuda.is_available()),
        "dgl_seeded": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
