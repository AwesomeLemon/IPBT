from algo.base import BaseAlgo
import copy
import time
from copy import deepcopy
from pathlib import Path

import GPy
import numpy as np
import pandas as pd
import ray
import torch
import yaml

from algo.pbt import PBT
from algo.pbt_utils import save_explored_ckpt_to_path
from utils import solution_history_to_str, save_yaml


class BayesianMixin(BaseAlgo):
    def __init__(self, cfg, search_space, task, **__):
        super().__init__(cfg, search_space, task)
        self.data = pd.DataFrame()
        self.trial_ids = {}  # current trial ids
        self.trial_id_counter = 0

        if self.cfg.general.continue_auto:
            self.data = pd.read_csv(Path(self.exp_dir) / 'data.csv',
                                    float_precision='round_trip')  # the code relies on df.groupby, which needs the exact float values.
            self.trial_ids, self.trial_id_counter = yaml.safe_load(open(self.exp_dir / 'active_trial_ids.yaml'))

    def update_bayesian_dataset(self, fitnesses):
        raise NotImplementedError

    def save_state(self):
        super().save_state()
        self.data.to_csv(self.exp_dir / 'data.csv', index=False)
        save_yaml((self.trial_ids, self.trial_id_counter), self.exp_dir / 'active_trial_ids.yaml')

