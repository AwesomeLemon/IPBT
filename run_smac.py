import copy
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import hydra
import pandas as pd
import ray
import torch
import yaml

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from smac import HyperparameterOptimizationFacade, Scenario

import numpy as np
from ConfigSpace import Configuration

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.hyperband_utils import get_n_trials_for_hyperband_multifidelity
from smac.initial_design.empty_design import EmptyInitialDesign
from smac.intensifier.successive_halving import SuccessiveHalving

from utils import set_random_seeds

class TaskTrainable: # don't inherit from tune.Trainable because wanna use it in plain Ray, not in Ray Tune
    def __init__(self, cfg, task):
        self.cfg = cfg
        self.task = task

        ckpt_dir = Path(cfg.path.dir_ckpt)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = ckpt_dir

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> Tuple[float, dict]:
        config_hash = str(hash(config))
        dir = self.ckpt_dir / config_hash
        dir.mkdir(parents=True, exist_ok=True)

        config_dict = {}
        for k, v in dict(config).items():
            tv = type(v)
            if tv == np.str_:
                v = str(v)
            config_dict[k] = v
        print(f'{config_dict=}')

        self.seed = seed
        set_random_seeds(seed)
        tb_dir = None
        self.prepared_task_vars = self.task.prepare(seed, config_dict, None, tb_dir, None)
        self.t = 0
        self.config = config_dict

        budget_rounded = int(np.round(budget))
        budget = budget_rounded * self.cfg.algo.t_step
        trajectory = []
        obj_best, tick_best = float('-inf'), 0
        while self.t < budget:
            out = self.step()
            tick, obj = out['tick'], out['obj']
            timestamp = time.time()
            trajectory.append((tick, obj, timestamp))
            if out['should_checkpoint']:
                ckpt_name = f'ckpt_{tick}.pt'
                self.save_checkpoint(dir, ckpt_name)
                if obj > obj_best:
                    obj_best = obj
                    tick_best = tick
                    ckpt_name = 'ckpt.pt'
                    self.save_checkpoint(dir, ckpt_name)

        # save last, if not already saved
        if not out['should_checkpoint']:
            ckpt_name = f'ckpt_{tick}.pt'
            self.save_checkpoint(dir, ckpt_name)
            if obj > obj_best:
                obj_best = obj
                tick_best = tick
                ckpt_name = 'ckpt.pt'
                self.save_checkpoint(dir, ckpt_name)

        additional_info = {
            "dir": str(dir.absolute()),
            "trajectory": trajectory,
            "obj_best": obj_best,
            "tick_best": tick_best,
            "budget": budget_rounded,
        }
        return -obj_best, additional_info

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

        if self.task.scheduler.endswith('_restart'):
            schedule_step_size = self.config['T_0']
            T_mult = self.config['T_mult']
            T_cur = self.t // self.cfg.algo.t_step
            T_max = self.cfg.algo.t_max // self.cfg.algo.t_step
            T_to_check = schedule_step_size
            is_before_cosine_restart = False
            # print(f'{schedule_step_size=}, {T_mult=}, {T_cur=}, {T_max=}')
            while T_to_check <= T_max:
                if T_to_check == T_cur:
                    is_before_cosine_restart = True
                    break

                schedule_step_size *= T_mult
                T_to_check += schedule_step_size
            out['should_checkpoint'] = is_before_cosine_restart

        return out

    def save_checkpoint(self, checkpoint_dir, checkpoint_name):
        checkpoint_path = checkpoint_dir / checkpoint_name
        task_dict = self.task._get_dict_to_save(self.prepared_task_vars)
        d = {'task_dict': task_dict, 't': self.t, 'config': self.config}
        torch.save(d, checkpoint_path)

    def load_checkpoint(self, checkpoint_dir, checkpoint_name, config_dict):
        checkpoint_path = checkpoint_dir / checkpoint_name
        d = torch.load(checkpoint_path)
        self.t = d['t']
        task_dict = d['task_dict']
        if not hasattr(self, 'prepared_task_vars'):
            self.prepared_task_vars = self.task.prepare(config_dict['seed'], config_dict, None,
                                                        None, None)
        self.prepared_task_vars = self.task._load_states(self.prepared_task_vars, task_dict)

    def test(self, config_dict):
        prepped = self.prepared_task_vars
        prepped['seed'] = config_dict['seed']
        prepped['solution'] = config_dict
        prepped = self.task.prepare_with_new_seed(prepped)
        prepped['t'] = 0
        prepped['t_step'] = 0
        prepped['tensorboard_dir'] = None
        out = self.task.eval(prepped, ['test'])
        return out

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        # If you need to trick libraries that only behave properly if
        # stdout is a TTY, return True here (or inspect self.streams[0]).
        return False

@hydra.main(version_base=None, config_path="config",
            config_name='smac_cls_0001'
            )
def main(cfg: DictConfig) -> None:
    ray.init(address=cfg.server.ray_address)
    set_random_seeds(cfg.general.seed_base + cfg.general.seed_offset)

    exp_dir = Path(cfg.path.dir_exp)
    exp_dir.mkdir(exist_ok=True, parents=True)

    with open(exp_dir / '_log.txt', "a") as f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            # This Tee prints to both the real stdout and the file
            sys.stdout = Tee(sys.stdout, f)
            sys.stderr = Tee(sys.stderr, f)

            # setup_logging(exp_dir / '_log.txt') # conflicts with SMAC logging => disable
            OmegaConf.save(cfg, exp_dir / 'config.yaml')

            # create file the name of which is cfg.general.exp_desc
            with open(exp_dir / cfg.general.exp_desc, 'w') as f:
                f.write('')

            ss = instantiate(cfg.search_space, seed=cfg.general.seed_base)
            task = instantiate(cfg.task, search_space=ss, cfg=cfg, _recursive_=False)
            trainable = TaskTrainable(cfg, task)

            min_budget = cfg.algo.t_step // cfg.algo.t_step
            max_budget = cfg.algo.t_max // cfg.algo.t_step
            t_total = max_budget * cfg.algo.max_full_trials

            # calculate how many trials we need to exhaust the total optimization budget (in terms of
            # fidelity units)
            # Unfortunately, it is not consistent with what actually happens.
            # Therefore, use an additional stopping criterion tracking the actually used budget.
            n_trials = get_n_trials_for_hyperband_multifidelity(
                total_budget=t_total,
                min_budget=min_budget,
                max_budget=max_budget,
                print_summary=True,
            )

            scenario = Scenario(
                ss.cs,
                n_trials=n_trials,
                min_budget=min_budget,
                max_budget=max_budget,
                total_budget=t_total,
                output_directory=Path(cfg.path.logs),
                name=cfg.general.exp_name,
                seed=cfg.general.seed_base,
                seed_offset=cfg.general.seed_offset
            )

            initial_design = MFFacade.get_initial_design(scenario)

            intensifier = Hyperband(
                scenario,
                seed=(cfg.general.seed_base + cfg.general.seed_offset + 1) * 10,
                incumbent_selection='any_budget'
            )

            ray_options = {'num_cpus': cfg.general.num_cpus, 'num_gpus': cfg.general.num_gpus}

            smac = MFFacade(
                scenario,
                trainable.train,
                initial_design=initial_design,
                intensifier=intensifier,
                overwrite=not cfg.general.continue_auto,
                parallel_runner_type='ray',
                parallel_runner_kwargs={'ray_options': ray_options},
            )

            incumbent = smac.optimize()

            # save best
            print(f'{incumbent=}')
            runhistory = smac.runhistory
            incumbent_id = runhistory.get_config_id(incumbent)

            for incumbent_trial_key in runhistory._data.keys():
                if incumbent_trial_key.config_id == incumbent_id:
                    print(incumbent_trial_key)
                    incumbent_trial_value = runhistory._data[incumbent_trial_key]
                    break

            info = incumbent_trial_value.additional_info
            best_checkpoint_path = Path(info['dir'])
            best_fitness = -incumbent_trial_value.cost
            best_metrics_raw = copy.deepcopy(info)
            best_metrics_raw.update({
                'cost': incumbent_trial_value.cost,
                'time': incumbent_trial_value.time,
                'starttime': incumbent_trial_value.starttime,
                'endtime': incumbent_trial_value.endtime,
            })
            best_metrics = {}
            for k, v in best_metrics_raw.items():
                tv = type(v)

                if tv == np.int64:
                    v = int(v)
                elif tv == np.float64:
                    v = float(v)
                elif tv == np.bool_:
                    v = bool(v)
                elif tv is object:
                    v = str(v)

                best_metrics[k] = v

            try:
                shutil.copy(best_checkpoint_path / 'ckpt.pt',
                            exp_dir / 'best_model.pt')

                config_dict = {}
                for k, v in dict(incumbent).items():
                    tv = type(v)
                    if tv == np.str_:
                        v = str(v)
                    print(f'{v=} {type(v)=}')
                    config_dict[k] = v

                @ray.remote(num_cpus=1, num_gpus=0.25)
                def _test(trainable, config_dict, best_checkpoint_path):
                    trainable.load_checkpoint(best_checkpoint_path, 'ckpt.pt', config_dict)
                    out = trainable.test(config_dict)
                    return out

                config_dict['seed'] = cfg.general.seed_base + cfg.general.seed_offset
                f = _test.options(**ray_options).remote(trainable, config_dict, best_checkpoint_path)
                res = ray.get(f)
                del config_dict['seed']

                with open(exp_dir / 'best_info.yaml', 'w') as f:
                    yaml.safe_dump({
                        'config': config_dict,
                        'fitness': best_fitness,
                        'metrics': best_metrics,
                        'test': res['test']
                    }, f)
                print(f'Val: {best_fitness:.4f}, Test: {res["test"]:.4f}')

                if 'policy_gif' in res:
                    with open(exp_dir / 'policy.webp', 'wb') as f:
                        f.write(res['policy_gif'])

            except Exception as e:
                print(f'Error saving best: {e}')

            # create results.csv
            all_trajectory_points = []
            for trial_value in runhistory._data.values():
                traj = trial_value.additional_info.get('trajectory', [])
                for point in traj:
                    tick_relative, obj, timestamp = point
                    all_trajectory_points.append((obj, timestamp))

            all_trajectory_points = sorted(all_trajectory_points, key=lambda x: x[1])
            all_trajectory_objs = [x[0] for x in all_trajectory_points]

            tick_cum = 0
            records = []
            for obj in all_trajectory_objs:
                tick_cum += cfg.algo.t_step
                records.append(dict(
                    tick_cum=tick_cum,
                    obj=obj
                ))

            df = pd.DataFrame.from_records(records)
            df.to_csv(exp_dir / 'results.csv', index=False)

            if cfg.algo.get('delete_all_ckpts_at_the_end', False):
                for p in Path(cfg.path.dir_ckpt).rglob('*.pt'):
                    p.unlink()

            print('Success |', datetime.now().strftime('%d/%m/%Y | %H:%M:%S'),
                  f' | {cfg.general.exp_name} | seed offset {cfg.general.seed_offset}')


        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr



if __name__ == "__main__":
    main()
