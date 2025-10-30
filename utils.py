import random
import torch
import numpy as np
import os


def setup_seed(seed=42, determinism=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    if determinism:
        print('Use deterministic algorithms')
        torch.use_deterministic_algorithms(True)
    else:
        print('Do not Use deterministic algorithms')
        torch.use_deterministic_algorithms(False)