import os
import sys
import torch
import random
import numpy as np


def add_parent_path(level=1):

    script_path = os.path.realpath(sys.argv[0])
    parent_dir = os.path.dirname(script_path)

    for _ in range(level):
        parent_dir = os.path.dirname(parent_dir)

    sys.path.insert(0, parent_dir)


def add_parent_paths(levels=[1,2]):
    for level in levels:
        add_parent_path(level=level)


def set_seeds(seed, cuda_deterministic=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False