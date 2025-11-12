import random

import numpy as np
import torch
from torch.utils.data import dataloader
from functools import partial

def _set_seed(worker_id: int, base_seed: int) -> None:
    s = base_seed + worker_id
    np.random.seed(s)
    random.seed(s)

def get_loader(cfg, dataset, seed, **kwargs):
    worker_init = partial(_set_seed, base_seed=42+seed)
    n_workers = cfg.task.data.num_workers
    return {'train': torch.utils.data.DataLoader(dataset['train'], shuffle=True, num_workers=n_workers, worker_init_fn=worker_init, **kwargs),
            'val': torch.utils.data.DataLoader(dataset['val'], shuffle=False, num_workers=n_workers, worker_init_fn=worker_init, **kwargs),
            'test': torch.utils.data.DataLoader(dataset['test'], shuffle=False, num_workers=n_workers, worker_init_fn=worker_init, **kwargs)
    }