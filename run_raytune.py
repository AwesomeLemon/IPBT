import copy
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray.air._internal.util import is_nan_or_inf
from ray.tune import ExperimentAnalysis
from ray.tune.experiment import Trial
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import BasicVariantGenerator
import dill as pickle
import numpy.random

import ray
from ray import train, tune

from search_space.cs import ConfigSpaceSearchSpace
from utils import set_random_seeds, setup_logging

class TaskTrainable(tune.Trainable):
    def setup(self, config, cfg=None, task=None):
        self.config = config # HP values sampled by Ray
        self.cfg = cfg # My config
        self.task = pickle.loads(task)

        self.seed = config['seed']
        set_random_seeds(self.seed)

        tb_dir = None
        self.prepared_task_vars = self.task.prepare(self.seed, config, None, tb_dir, None)
        self.t = 0

    def step(self):
        prepped = self.prepared_task_vars
        prepped['seed'] = self.seed + self.t
        prepped['solution'] = self.config
        prepped = self.task.prepare_with_new_seed(prepped)
        prepped['t'] = self.t
        prepped['t_step'] = self.cfg.algo.t_step
        prepped['to_cpu'] = False
        out, updated_dicts = self.task.train_and_eval(prepped)
        self.t += self.cfg.algo.t_step
        self.prepared_task_vars = self.task._load_states(self.prepared_task_vars, updated_dicts)
        out = {"obj": out['fitness'], "tick": self.t}

        if self.task.scheduler and self.task.scheduler.endswith('_restart'):
            schedule_step_size = self.config['T_0']
            T_mult = self.config['T_mult']
            T_cur = self.t // self.cfg.algo.t_step
            T_max = self.cfg.algo.t_max // self.cfg.algo.t_step
            T_to_check = schedule_step_size
            is_before_cosine_restart = False
            while T_to_check <= T_max:
                if T_to_check == T_cur:
                    is_before_cosine_restart = True
                    break

                schedule_step_size *= T_mult
                T_to_check += schedule_step_size
            out['should_checkpoint'] = is_before_cosine_restart

        return out

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "ckpt.pt")
        task_dict = self.task._get_dict_to_save(self.prepared_task_vars)
        d = {'task_dict': task_dict, 't': self.t, 'config': self.config}
        torch.save(d, checkpoint_path)

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "ckpt.pt")
        d = torch.load(checkpoint_path)
        self.t = d['t']
        task_dict = d['task_dict']
        self.prepared_task_vars = self.task._load_states(self.prepared_task_vars, task_dict)

    def test(self):
        '''
        Used to test the best checkpoint, not a part of the Ray Trainable API
        '''
        prepped = self.prepared_task_vars
        prepped['seed'] = self.seed
        prepped['solution'] = self.config
        prepped = self.task.prepare_with_new_seed(prepped)
        prepped['t'] = 0
        prepped['t_step'] = 0
        prepped['tensorboard_dir'] = None
        out = self.task.eval(prepped, ['test'])
        return out


def search_space_to_tune_space(ss: ConfigSpaceSearchSpace):
    out = {}
    for i in range(ss.n_vars):
        hp_name = ss.cs.get_hyperparameter_by_idx(i)
        hp = ss.get_hp_by_name(hp_name)
        if isinstance(hp, UniformFloatHyperparameter):
            bounds = ss.get_hp_bounds_numerical(i)
            out[hp_name] = tune.uniform(*bounds)
        elif isinstance(hp, UniformIntegerHyperparameter):
            lower, upper = ss.get_hp_bounds_numerical(i)
            out[hp_name] = tune.randint(lower, upper + 1) # upper is inclusive in ConfigSpace, exclusive in Ray
        else:
            raise NotImplementedError(hp)

    return out

def get_best_trial_at_restarts(
    exp_analysis: ExperimentAnalysis,
    metric: Optional[str] = None,
    mode: Optional[str] = None,
    filter_nan_and_inf: bool = True,
) -> Optional[Trial]:
    """Retrieve the best trial object at points before the LR schedule restarts.
    """
    if len(exp_analysis.trials) == 1:
        return exp_analysis.trials[0]

    metric = exp_analysis._validate_metric(metric)
    mode = exp_analysis._validate_mode(mode)

    best_trial = None
    best_metric_score = None

    trial_id_to_df = exp_analysis._fetch_trial_dataframes()

    for trial in exp_analysis.trials:
        if metric not in trial.metric_analysis:
            continue

        trial_df = trial_id_to_df[trial.trial_id]

        # select only the entries corresponding to checkpoints
        trial_df = trial_df[trial_df['should_checkpoint'] | trial_df['done']]

        if len(trial_df) == 0 or (trial.checkpoint is None):
            # no checkpoints
            continue

        if mode == 'max':
            metric_score = trial_df[metric].max()
            df_entry_id = trial_df[metric].argmax()
        elif mode == 'min':
            metric_score = trial_df[metric].min()
            df_entry_id = trial_df[metric].argmin()

        if filter_nan_and_inf and is_nan_or_inf(metric_score):
            continue

        path = Path(trial.checkpoint.path)
        new_subdir_name = f'checkpoint_{df_entry_id:06d}'
        trial.checkpoint.path = str(path.parent / new_subdir_name)

        trial.metrics_at_correct_step = dict(trial_df.iloc[df_entry_id])

        if best_metric_score is None:
            best_metric_score = metric_score
            best_trial = trial
            continue

        if (mode == "max") and (best_metric_score < metric_score):
            best_metric_score = metric_score
            best_trial = trial
        elif (mode == "min") and (best_metric_score > metric_score):
            best_metric_score = metric_score
            best_trial = trial

    return best_trial

class CumulativeBudgetStopper(tune.Stopper):
    def __init__(self, t_step, t_max, max_ticks):
        self.should_stop = False
        self.t_step = t_step
        self.t_max = t_max
        self.tick_cum_max = max_ticks
        self.tick_cum = 0

    def __call__(self, trial_id, result):
        self.tick_cum += self.t_step
        result['tick_cum'] = self.tick_cum
        return self.should_stop or result["tick"] >= self.t_max

    def stop_all(self):
        return self.should_stop or self.tick_cum >= self.tick_cum_max


@hydra.main(version_base=None, config_path="config",
            config_name='raytune_cls_0001'
            )
def main(cfg: DictConfig) -> None:
    ray.init(address=cfg.server.ray_address)
    cfg.general.seed_base += cfg.general.seed_offset
    set_random_seeds(cfg.general.seed_base)

    exp_dir = Path(cfg.path.dir_exp)
    exp_dir.mkdir(exist_ok=True, parents=True)

    setup_logging(exp_dir / '_log.txt')
    OmegaConf.save(cfg, exp_dir / 'config.yaml')

    # create file the name of which is cfg.general.exp_desc
    with open(exp_dir / cfg.general.exp_desc, 'w') as f:
        f.write('')

    ss = instantiate(cfg.search_space, seed=cfg.general.seed_base)
    ss_tune = search_space_to_tune_space(ss)
    task = instantiate(cfg.task, search_space=ss, cfg=cfg, _recursive_=False)

    if cfg.algo.name == 'asha':
        search_alg = BasicVariantGenerator(max_concurrent=cfg.algo.max_concurrent_trials)
        scheduler = ASHAScheduler(
            time_attr='tick',
            max_t=cfg.algo.t_max
        )
        num_samples = -1  # custom stopper will stop when the tick budget is exhausted

    elif cfg.algo.name == 'rs':
        search_alg = BasicVariantGenerator(max_concurrent=cfg.algo.max_concurrent_trials)
        scheduler = None
        num_samples = cfg.algo.max_full_trials

    else:
        raise NotImplementedError(cfg.algo.name)

    t_total = cfg.algo.t_max * cfg.algo.max_full_trials
    stopper = CumulativeBudgetStopper(cfg.algo.t_step, cfg.algo.t_max, t_total)

    ray_options = {'cpu': cfg.general.num_cpus, 'gpu': cfg.general.num_gpus}

    serialized_task = pickle.dumps(task)
    tr_w_kwargs = tune.with_parameters(TaskTrainable, cfg=cfg, task=serialized_task)
    tr_w_kwargs_w_resources = tune.with_resources(tr_w_kwargs, ray_options)

    ss_tune_with_seed = copy.deepcopy(ss_tune)
    ss_tune_with_seed['seed'] = tune.sample_from(lambda _: numpy.random.randint(0, 1000000))

    if not cfg.general.continue_auto:
        ckpt_freq = 5
        if task.scheduler and task.scheduler.endswith('_restart'):
            ckpt_freq = 1  # won't save too much because I modified ray source to save only when should_checkpoint is True
        tuner = tune.Tuner(
            tr_w_kwargs_w_resources,
            run_config=train.RunConfig(
                name=str(cfg.general.seed_offset),
                stop=stopper,
                verbose=2,
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_score_attribute="obj",
                    checkpoint_score_order="max",
                    checkpoint_at_end=True,
                    checkpoint_frequency=ckpt_freq,
                    num_to_keep=2,
                ),
                storage_path=Path(cfg.path.logs) / cfg.general.exp_name
            ),
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                metric="obj",
                mode="max",
                num_samples=num_samples,
                reuse_actors=False,
            ),
            param_space=ss_tune_with_seed,
        )
    else:
        tuner = tune.Tuner.restore(
            str(exp_dir.absolute()),
            trainable=tr_w_kwargs_w_resources,
            resume_errored=True,
            param_space=ss_tune_with_seed,
        )

    result_grid = tuner.fit()

    df = pd.concat(result_grid._experiment_analysis.trial_dataframes.values(), axis=0, ignore_index=True)
    df.to_csv(exp_dir / 'results.csv')

    if (task.scheduler is None) or (not task.scheduler.endswith('_restart')):
        scope = cfg.algo.scope_best
        best_result = result_grid.get_best_result(scope=scope)

        best_checkpoint = best_result.checkpoint
        best_config = best_result.config
        best_metrics = best_result.metrics
    else:
        best_trial = get_best_trial_at_restarts(result_grid._experiment_analysis)

        best_checkpoint = best_trial.checkpoint
        print(f'{best_checkpoint.path=}')
        best_config = best_trial.config
        best_metrics_raw = best_trial.metrics_at_correct_step
        best_metrics = {}
        for k, v in best_metrics_raw.items():
            if k.startswith('config/'):
                continue
            tv = type(v)

            if tv == np.int64:
                v = int(v)
            elif tv == np.float64:
                v = float(v)
            elif tv == np.bool_:
                v = bool(v)

            best_metrics[k] = v

    try:
        shutil.copy(Path(best_checkpoint.path) / 'ckpt.pt',
                    exp_dir / 'best_model.pt')

        @ray.remote(num_cpus=1, num_gpus=0.25)
        def _test(tr_w_kwargs, config, best_checkpoint):
            trainable = tr_w_kwargs(config)
            trainable.load_checkpoint(best_checkpoint.path)
            out = trainable.test()
            return out

        ray_options_test = {'num_cpus': cfg.general.num_cpus, 'num_gpus': cfg.general.num_gpus} # remote functions have different param names: cpu -> num_cpus

        f = _test.options(**ray_options_test).remote(tr_w_kwargs, best_config, best_checkpoint)
        res = ray.get(f)

        with open(exp_dir / 'best_info.yaml', 'w') as f:
            yaml.safe_dump({
                'config': best_config,
                'fitness': best_metrics['obj'],
                'metrics': best_metrics,
                'test': res['test']
            }, f)
        print(f'Val: {best_metrics["obj"]:.4f}, Test: {res["test"]:.4f}')

        if 'policy_gif' in res:
            with open(exp_dir / 'policy.webp', 'wb') as f:
                f.write(res['policy_gif'])
    except Exception as e:
        print(f'Error saving best: {e}')

    if cfg.algo.delete_all_ckpts_at_the_end:
        # delete .pt files in all subfolders recursively but avoid the main folder where best is stored
        for subdir in exp_dir.iterdir():
            if subdir.is_dir():
                for pt_file in subdir.rglob('*.pt'):
                    pt_file.unlink()

    print('Success |', datetime.now().strftime('%d/%m/%Y | %H:%M:%S'),
          f' | {cfg.general.exp_name} | seed offset {cfg.general.seed_offset}')


if __name__ == "__main__":
    main()
