import copy
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import yaml
from matplotlib import pyplot as plt

import utils
from algo.bayes_mixin import BayesianMixin
from algo.pbt import PBT
from algo.pbt_utils import save_explored_ckpt_to_path
from utils import solution_history_to_str, save_yaml

from algo.bgpbt_utils import normalize, copula_standardize, train_gp, MAX_CHOLESKY_SIZE, MIN_CUDA, _Casmo, \
    _get_all_bounds, _adjust_categorical
import gpytorch
import torch
import logging
import pickle


class BGPBT(BayesianMixin, PBT):
    '''
    Bayesian Generational Population-Based Training
    https://arxiv.org/abs/2207.09405

    This is a simplified version of the original code: https://github.com/xingchenwan/bgpbt
    I also referred to https://github.com/facebookresearch/how-to-autorl/blob/main/hydra_plugins/hydra_pbt_sweeper/hydra_bgt.py
    Note: the objective should be maximized but for now I minimize in the bayesian optimization part for simplicity
          (e.g would have to change lcb=>ucb)
    '''

    def __init__(self, cfg, search_space, task, **__):
        self.pop_size_actual = copy.deepcopy(cfg.algo.pop_size)
        cfg.algo.pop_size *= cfg.algo.n_init_mul
        super().__init__(cfg, search_space, task)
        # BG-PBT variables:
        self.config_space = search_space.cs
        self.n_init = cfg.algo.pop_size # it's already multiplied by n_init_mul above
        self.n_init_mul = cfg.algo.n_init_mul
        self.verbose = True
        self.casmo = _Casmo(self.config_space,
                            n_init=self.n_init,
                            max_evals=1,  # Me: default value; also, influences nothing.
                            batch_size=None,  # this will be updated later. batch_size=None signifies initialisation
                            verbose=self.verbose,
                            ard=False,  # Me: default value, the only one used.
                            acq='lcb',  # Me: default value, the only one used.
                            use_standard_gp=False,  # Me: default value, the only one used.
                            time_varying=False)  # Me: default value, but it's overriden below
        self.patience = 15 #https://github.com/xingchenwan/bgpbt/blob/main/hpo/casmo/bgpbt.py#L91
        self.n_fail = 0
        self.n_distills = 0
        self.t_step_is_modified_due_to_restart = False
        self.if_restarting = False
        self.too_long_without_restart = cfg.algo.get('too_long_without_restart', None)
        self.reinit_weights_strategy = cfg.algo.get('reinit_weights_strategy', 'copy')
        self.n_distillation_timesteps = cfg.algo.get('n_distillation_timesteps',
                                                     30_000_000)  # from the BG codebase; whoever survives until the end of SH, will have been trained for this many steps
        self.ckpt_best_dir = self.exp_dir / 'checkpoints_best'
        self.ckpt_best_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.general.continue_auto:
            self.n_distills, self.n_fail, self.if_restarting = yaml.safe_load(open(self.exp_dir / 'bgpbt_state.yaml'))
            self.casmo = pickle.load(open(self.exp_dir / 'casmo.pkl', 'rb'))

    def adjust_step_size(self):
        if self.t_step_is_modified_due_to_restart:
            self.t_step = self.t_step_backup
            self.t_step_is_modified_due_to_restart = False
        super().adjust_step_size()

    def tick(self):
        # 0. if restarting, init hyperparameters by sampling around historically good values
        if self.if_restarting:
            assert self.n_distills > 0, 'if restarting, needs to have done at least one iteration'

            if self.reinit_weights_strategy == 'distill':
                # save teacher hyperparameters (at this point, the population
                # consists of copies of the top quantile)
                self.pop_old = copy.deepcopy(self.pop)

            self.reinit_population_global()

            # set if_restarting to False in _schedule_all_populations, after shrink-perturb

        # 1. evaluate population
        results = self._schedule_all_populations_and_record_time()
        fitnesses = [res_dict['fitness'] for res_dict in results]

        # 2. save new checkpoints & collect names of the old ones to delete
        ckpts_to_delete = self._save_checkpoints(results)

        # 3. log history
        self.extend_result_records_with_many_sorted(results)
        self.extend_fitness_and_solution_history(fitnesses)
        self.update_bayesian_dataset(fitnesses)

        # 4. update population
        self._update_population_and_record_time(fitnesses)

        # 5. delete old checkpoints
        for p in ckpts_to_delete:
            p.unlink()


    def reinit_population_global(self):
        '''
        Based on casmo/bgpbt.py:_generate_initializing_points_ucb
        Also see description in the IPBT paper Section 3.3.Hyperparameters
        '''

        # (1) create S_t, the surrogate on the latest timepoint data.
        ## (1.1) Get the data from self.data
        print('Global reinitialization of the population')
        bounds, bounds_cont, bounds_noncont = _get_all_bounds(self.search_space)

        def _prepare_data(n_distills_target):
            df = self.data.sort_values(by="Time").reset_index(drop=True)
            df = df[df['n_distills'] == n_distills_target]
            df = df.copy()

            df["Reward"] = -df["Reward"]  # minimization
            df["y"] = df.groupby(["Trial"] + list(bounds.keys()))["Reward"].diff()
            df["t_change"] = df.groupby(["Trial"] + list(bounds.keys()))["Time"].diff()
            df = df[df["t_change"] > 0].reset_index(drop=True)
            df["R_before"] = df.Reward - df.y

            df = df[~df.y.isna()].reset_index(drop=True)
            df = df.sort_values(by="Time").reset_index(drop=True)
            df = df.iloc[-100:, :].reset_index(drop=True)

            r_t = df[["R_before", "Time"]]
            hparams = df[bounds.keys()]
            _, hparams, _ = _adjust_categorical(self.search_space, None, hparams, None,
                                                bounds, bounds_cont, bounds_noncont)
            X_raw = pd.concat([hparams, r_t], axis=1).values
            y_raw = np.array(df.y.values)

            int_dims = np.array(self.search_space.get_int_indices())

            limits, limits_hps, limits_reward_time, int_half_ranges = _compute_limits(
                None, X_raw, bounds, int_dims
            )

            X, _, t, _, _, y = _normalize_based_on_limits(
                None, y_raw, None, X_raw,
                int_half_ranges, limits,
                int_dims
            )

            return X, y, t

        X, y, t = _prepare_data(self.n_distills - 1) # self.n_distills has already been incremented

        ## (1.2) train S_t
        gp, _, dtype = _train_gp(self.casmo, X, y, t,
                                 True, self.verbose)
        gp.eval()

        # (2) create auxiliary GP
        ## (2.1) Create the training points
        aux_train_input, aux_train_target = [], []
        for n_distill_cur in range(self.n_distills):
            X, y, t = _prepare_data(n_distill_cur)

            t_current = np.max(t) * np.ones(X.shape[0])
            t_x_current = torch.hstack(
                (torch.tensor(t_current, dtype=dtype).reshape(-1, 1),
                 torch.tensor(X, dtype=dtype))
            )
            pred_ = gp(t_x_current).mean
            # select the x with the best
            pred_np = pred_.detach().numpy()
            best_idx = np.argmin(pred_np)
            aux_train_input.append(X[best_idx, :-1]) # exclude R_before (the last column), as it will not be known for the random configs
            aux_train_target.append(pred_np[best_idx])

        ## (2.2) Fit the auxiliary GP
        aux_train_input = np.array(aux_train_input)
        aux_train_target = np.array(aux_train_target)
        print(f'{aux_train_input.shape=}')
        print(f'{aux_train_input=}')
        print(f'{aux_train_target.shape=}')
        print(f'{aux_train_target=}')
        aux_gp, _, _ = _train_gp(self.casmo, aux_train_input,
                                 aux_train_target,
                                 None, False, self.verbose)
        aux_gp.eval()

        # (3) Create new population
        ## (3.1) generate a bunch of random configs
        random_configs = [self.search_space.sample(self.task, True)
                          for _ in range(10 * self.n_init)]
        random_configs_df = pd.DataFrame(random_configs, columns=self.search_space.get_hp_names())

        _, random_configs_df_adj, _ = _adjust_categorical(self.search_space,
                                              None, random_configs_df, None,
                                              bounds, bounds_cont, bounds_noncont)
        X_raw_random = random_configs_df_adj.values

        int_dims = np.array(self.search_space.get_int_indices())
        limits_hps = np.array(list(bounds.values())).T.astype(np.float32)
        max_is_min = limits_hps[0] == limits_hps[1]
        limits_hps[0][max_is_min] -= 1e-8
        limits_hps[1][max_is_min] += 1e-8
        int_n_values = limits_hps[1, int_dims] - limits_hps[0, int_dims] + 1
        limits_hps[1, int_dims] += 1
        int_half_ranges = 0.5 * (1 / int_n_values)

        X_random = _normalize_X(X_raw_random, int_dims, int_half_ranges, limits_hps)

        # the original code added t but that's incorrect, since the aux surrogate is not time-varying

        ## (3.2) select by the LCB score using the predicted mean + var of the auxiliary GP.
        X_random_torch = torch.tensor(X_random, dtype=dtype)
        pred = aux_gp(X_random_torch)
        pred_mean, pred_std = pred.mean.detach().numpy(), pred.stddev.detach().numpy()
        lcb = pred_mean - 1.96 * pred_std

        if self.reinit_weights_strategy == 'distill':
            n_to_select = self.n_init
        else:
            n_to_select = self.pop_size_actual
        top_config_ids = np.argpartition(np.array(lcb), n_to_select)[:n_to_select].tolist()
        for i_pop_member, i_top_config in enumerate(top_config_ids):
            new_hp_values = random_configs[i_top_config]
            self.pop[i_pop_member] = new_hp_values
            print(f'{i_pop_member=} {new_hp_values=}')


    def save_state(self):
        super().save_state()
        self.data.to_csv(self.exp_dir / 'data.csv', index=False)
        save_yaml((self.trial_ids, self.trial_id_counter), self.exp_dir / 'active_trial_ids.yaml')
        save_yaml((self.n_distills, self.n_fail, self.if_restarting), self.exp_dir / 'bgpbt_state.yaml')
        pickle.dump(self.casmo, open(self.exp_dir / 'casmo.pkl', 'wb'))

    def _save_checkpoints(self, results):
        ckpts_to_delete = []  # first collect, then delete after everything is saved

        for i, res_dict in enumerate(results):
            ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur + self.t_step}.pt'
            torch.save(res_dict['dict_to_save'], ckpt_path)
        if self.delete_old_ckpts:
            for i in range(self.n_init): # potentially there are more ckpts than the current population size
                p = Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur}.pt'
                if p.exists():
                    ckpts_to_delete.append(p)
        return ckpts_to_delete

    def update_bayesian_dataset(self, fitnesses):
        for i, (fitness, p) in enumerate(zip(fitnesses, self.pop)):
            trial_id = self.trial_ids[i]
            lst = [[trial_id, self.t_cur + self.t_step] +
                   copy.deepcopy(p) +
                   [fitness, "bo", self.n_distills]]
            cols = (["Trial", "Time"] +
                    self.search_space.get_hp_names() +
                    ["Reward", "config_source", "n_distills"])
            entry = pd.DataFrame(lst, columns=cols)
            self.data = pd.concat([self.data, entry]).reset_index(drop=True)
            self.data.Trial = self.data.Trial.astype("str")

    def _schedule_all_populations(self):
        futures = []
        futures_fitness_before = []  # GP is built based on difference from previous result => need results at 0

        if self.reinit_weights_strategy == 'distill':
            teacher_ckpts = []
            pop_ckpts_loaded = []
            pop_hps = []

        for i, p in enumerate(self.pop):
            ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur}.pt'
            if self.t_cur == 0:
                self.prepare_initial_ckpt(ckpt_path, p)
                f = self._task_fn_ray.options(**self.ray_options).remote(
                    self.task, 0, p, 0, 0, torch.load(ckpt_path), None, ['val']
                )
                futures_fitness_before.append(f)

            ckpt_loaded = torch.load(ckpt_path)

            if self.if_restarting:
                assert self.t_cur != 0, 'cannot restart at t=0'

                if self.reinit_weights_strategy == 'distill':
                    # currently the checkpoints of the entire population are the copied checkpoints
                    # of the top 25% => just need to save them
                    teacher_ckpts.append(copy.deepcopy(ckpt_loaded))
                    # random reinit
                    fresh_reinit = self.task.get_fresh_model(p)
                    ckpt_loaded['model_state_dict'] = fresh_reinit.state_dict()

                # evaluate the perturbed model before training
                f = self._task_fn_ray.options(**self.ray_options).remote(
                    self.task, 0, p, 0, 0, ckpt_loaded, None, ['val']
                ) # 't' is irrelevant in eval when no tensorboard
                futures_fitness_before.append(f)

            if not (self.if_restarting and self.reinit_weights_strategy == 'distill'):
                seed = self.seed_base * 100 + int(self.t_cur / self.t_max * 1e6) + i
                tb_dir = Path(self.exp_dir) / 'tb' / f'pop_{i}'
                tb_dir.mkdir(parents=True, exist_ok=True)
                f = self._task_fn_ray.options(**self.ray_options).remote(
                    self.task, seed, p, self.t_cur, self.t_step, ckpt_loaded, tb_dir, None
                )
                futures.append(f)
            else:
                pop_ckpts_loaded.append(ckpt_loaded)
                pop_hps.append(p)

        results = ray.get(futures)

        if self.t_cur == 0 or self.if_restarting:
            if self.reinit_weights_strategy == 'distill' and self.if_restarting:
                # do the successive halving like in the BG codebase; some comments are theirs, some are mine.
                print(f'{self.n_init=} {self.pop_size_actual=}')
                s = int(np.ceil(np.log(self.n_init) / np.log(self.pop_size_actual)))
                eta = 2.0  # halving by default -- set anything above 2 for more aggressive elimination
                distill_timestep = 0
                n_distillation_timesteps = self.n_distillation_timesteps
                survivor_indices = list(range(self.n_init))
                survivor_hps = pop_hps
                survivor_ckpts = pop_ckpts_loaded
                teacher_hps = self.pop_old
                for rung in range(s):
                    if rung < s - 1:
                        timesteps_this_rung = int(n_distillation_timesteps * eta ** (rung - s))
                    else:
                        # for the final rung, simply use up the rest
                        timesteps_this_rung = int(n_distillation_timesteps - distill_timestep)

                    print(
                        f"Running SuccessiveHalving Rung={rung + 1}/{s}. Budgeted timestep={timesteps_this_rung}."
                    )
                    futures_distill = []

                    for idx, hps, ckpt, hps_teacher, ckpt_teacher in zip(
                            survivor_indices, survivor_hps, survivor_ckpts, teacher_hps, teacher_ckpts,
                    ):
                        seed = self.seed_base * 100 + int(self.t_cur / self.t_max * 1e6) + idx * 10 + rung
                        tb_dir = Path(self.exp_dir) / 'tb' / f'distill_{idx}'
                        tb_dir.mkdir(parents=True, exist_ok=True)
                        distill_kwargs = {
                            "teacher_hparams": hps_teacher,
                            "teacher_ckpt_loaded": ckpt_teacher,
                            "num_total_distill_steps": n_distillation_timesteps,
                            "num_already_done_distill_steps": distill_timestep
                        }
                        f = self._task_fn_ray.options(**self.ray_options).remote(
                            self.task, seed, hps, self.t_cur + n_distillation_timesteps,
                            timesteps_this_rung, ckpt, tb_dir, None, distill_kwargs
                        )
                        futures_distill.append(f)

                    results_distill = ray.get(futures_distill)
                    distill_timestep += timesteps_this_rung

                    fitnesses_distill = np.array([res_dict['fitness'] for res_dict in results_distill])
                    n_should_survive = int(max(self.pop_size_actual, int(round(len(results_distill) / eta))))

                    survivors_indices_in_results = np.argsort(fitnesses_distill)[-n_should_survive:]
                    # these indices are inconsistent with the original indices that are in range 0-24,
                    # while these may be from higher rungs with eg just 12 current survivors
                    # so I keep track of the original indices.
                    survivor_indices_new = []
                    survivor_hps_new = []
                    survivor_ckpts_new = []
                    teacher_hps_new = []
                    teacher_ckpts_new = []
                    for i, (idx, hps, res, hps_teacher, ckpt_teacher) in enumerate(zip(
                            survivor_indices, survivor_hps, results_distill,
                            teacher_hps, teacher_ckpts
                    )):
                        # will need to return the list of results from _schedule_all_populations
                        if rung == 0:
                            results.append(res)
                        else:
                            results[idx] = res  # update

                        if i not in survivors_indices_in_results:
                            continue

                        survivor_indices_new.append(idx)
                        survivor_hps_new.append(hps)
                        survivor_ckpts_new.append(res['dict_to_save'])
                        teacher_hps_new.append(hps_teacher)
                        teacher_ckpts_new.append(ckpt_teacher)

                    survivor_indices = survivor_indices_new
                    survivor_hps = survivor_hps_new
                    survivor_ckpts = survivor_ckpts_new
                    teacher_hps = teacher_hps_new
                    teacher_ckpts = teacher_ckpts_new

                    print(f"Surviving indices={survivor_indices}.")

                survivors = survivor_indices
            else:
                fitnesses = np.array([res_dict['fitness'] for res_dict in results])
                print(f'{fitnesses=}')
                survivors = np.argsort(fitnesses)[-int(self.pop_size_actual):]

            print(f'{survivors=}')
            new_idx = 0
            pop_new = []
            fitness_history_new = []
            solution_history_new = []

            results_before = ray.get(futures_fitness_before)

            for i in range(len(self.pop)):
                if i not in survivors:
                    continue

                result_before = results_before[i]
                fitness_before = result_before['fitness']
                lst = [[str(self.trial_id_counter), 0] +
                       copy.deepcopy(self.pop[i]) +
                       [fitness_before, "random" if self.t_cur == 0 else 'bo',
                        self.n_distills]]
                cols = (["Trial", "Time"] +
                        self.search_space.get_hp_names() +
                        ["Reward", "config_source", "n_distills"])
                entry = pd.DataFrame(lst, columns=cols)
                self.data = pd.concat([self.data, entry]).reset_index(drop=True)
                self.data.Trial = self.data.Trial.astype("str")
                self.trial_ids[new_idx] = str(self.trial_id_counter)

                # add results record in the ray tune format
                self.extend_result_records(result_before, self.trial_id_counter, self.t_cur, 0)

                self.trial_id_counter += 1

                self.fitness_history[i].append((self.t_cur, fitness_before))
                self.solution_history[i].append((self.t_cur, copy.deepcopy(self.pop[i])))

                pop_new.append(self.pop[i])
                fitness_history_new.append(self.fitness_history[i])
                solution_history_new.append(self.solution_history[i])
                # if i in self.population_history[1]:
                #     population_history_new[1][new_idx] = self.population_history[1][i]
                new_idx += 1

            # need to track the cost of extra trials
            self.t_step_backup = copy.deepcopy(self.t_step)
            if self.reinit_weights_strategy == 'distill' and self.if_restarting:
                # even if we distill, at time 0 we just do 1 step, so we are in the other, "normal" branch.
                # otherwise, just take the max distill time, like the BG codebase.
                self.t_step = n_distillation_timesteps
                self.t_step_is_modified_due_to_restart = True
            else:
                if self.t_cur == 0:
                    self.t_step *= self.n_init_mul # we train n_init_mul times many more networks
                    self.t_step_is_modified_due_to_restart = True
                else:
                    # population is not enlarged if distillation is not used and it's not the initial population
                    self.t_step_is_modified_due_to_restart = False

            if self.if_restarting:
                self.if_restarting = False

            self.pop_size = self.pop_size_actual
            self.pop = pop_new
            self.fitness_history = fitness_history_new
            self.solution_history = solution_history_new
            self.population_history[1][self.t_cur + self.t_step] = copy.deepcopy(self.pop)
            # can't just index results by survivors because results is not a np array
            results = [results[i] for i in sorted(survivors)]

        return results

    def _update_population_and_record_time(self, fitnesses):
        if self.min_steps_before_eval <= self.t_cur < self.t_max - self.t_step:
            st = time.time()
            # 4.1 adjust trust region length before exploiting and exploring
            self.adjust_tr_length(True)

            # 4.2 potentially restart GP - before explore&exploit because exploring makes no sense before restart
            best_fitness = self.data[
                (self.data['n_distills'] == self.n_distills)
            ].Reward.max()

            best_fitness_cur = self.data[
                (self.data.Time == (self.t_cur + self.t_step)) &
                (self.data['n_distills'] == self.n_distills)
            ].Reward.max()

            if best_fitness_cur == best_fitness:
                self.n_fail = 0
            else:
                self.n_fail += 1

            if self.reinit_weights_strategy == 'distill':
                t_after_this_step = self.t_cur + self.t_step
                t_after_distill = t_after_this_step + self.n_distillation_timesteps
                if t_after_distill > self.t_max:
                    print('Prevent restarts because there is no time to distill.')
                    self.patience = float('inf')

            restart_because_too_long = False
            if self.too_long_without_restart:
                t = self.t_cur + self.t_step
                if self.n_distills > 0:
                    t_prev_distill = self.data[self.data['n_distills'] == self.n_distills - 1].Time.max()
                    if self.reinit_weights_strategy == 'distill':
                        t_prev_distill += self.n_distillation_timesteps
                else:
                    t_prev_distill = 0
                diff = t - t_prev_distill
                diff_percent = diff / self.t_max
                print(f'Restart because too long? {diff_percent=:.3f} {t=} {t_prev_distill=}')
                if diff_percent >= self.too_long_without_restart:
                    restart_because_too_long = True

            if self.n_fail >= self.patience or (restart_because_too_long and self.patience != float('inf')):
                self.n_fail = 0
                self.if_restarting = True
                reason = 'Limit of steps without restart reached. ' if (
                    restart_because_too_long) else 'n_fail reached patience. '
                print(reason + 'Restarting GP')
                self._restart()
            else:
                # 4.3 exploit & explore
                self._exploit_and_explore(fitnesses)

            print(f'n_fail: {self.n_fail}')
            self.update_times = pd.concat([self.update_times,
                                           pd.DataFrame({'t': [self.t_cur], 'time': [time.time() - st]})],
                                          ignore_index=True)

    def _exploit_and_explore(self, fitnesses):
        idx_top = np.argsort(fitnesses)[-int(self.quant_top * self.pop_size):]
        idx_bottom = np.argsort(fitnesses)[:int(self.quant_bottom * self.pop_size)]
        # first populate current, only then start adding stuff
        current = []

        for i in range(len(self.pop)):
            if i not in idx_bottom:
                # add to current
                hp_values_cur = np.array(self.pop[i])
                hp_values_cur = hp_values_cur.reshape(1, -1)
                current.append(hp_values_cur)
                continue

        for i in range(len(self.pop)):
            if i not in idx_bottom:
                continue
            # replace bottom
            chosen_idx = np.random.choice(idx_top)
            self.pop[i] = copy.deepcopy(self.pop[chosen_idx])
            # replace history
            self.fitness_history[i] = copy.deepcopy(self.fitness_history[chosen_idx])
            self.solution_history[i] = copy.deepcopy(self.solution_history[chosen_idx])

            # <explore/> #####################################
            new_trial_id = str(self.trial_id_counter)
            current_stacked = np.concatenate(current, axis=0) if len(current) > 0 else None
            new_hp_values = self.explore(self.trial_ids[chosen_idx], new_trial_id, current_stacked)
            self.pop[i] = new_hp_values
            current.append(np.array(new_hp_values).reshape(1, -1))
            self.trial_ids[i] = new_trial_id
            self.trial_id_counter += 1
            ####################################### </explore>

            # replace checkpoint
            ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur + self.t_step}.pt'
            ckpt_path.unlink()
            ckpt_chosen_path = Path(self.cpkt_dir) / f'pop_{chosen_idx}_t{self.t_cur + self.t_step}.pt'
            save_explored_ckpt_to_path(self.task, self.pop[i], ckpt_chosen_path, ckpt_path)
            print(f'Replaced {i}  with {chosen_idx}, '
                  f'unperturbed values {self.pop[chosen_idx]}, perturbed values {self.pop[i]}')

    def explore(self, base_trial_id, new_trial_id, current):
        df = self.data.sort_values(by="Time").reset_index(drop=True)
        bounds_cont = self.search_space.get_bounds_cont(treat_int_as_cont=True)
        bounds_noncont = self.search_space.get_bounds_noncont(treat_int_as_cont=True)
        bounds = {}
        for hp_name in self.search_space.get_hp_names():  # to preserve order
            if hp_name in bounds_cont:
                bounds[hp_name] = bounds_cont[hp_name]
            else:
                # don't normalize categorical hps
                bounds[hp_name] = (0, 1)

        # <diff wrt PB2/> #####################################
        # if a reset happened, we will have only the just-evaluated data, and therefore no diff, so we sample randomly
        df = df[df['n_distills'] == self.n_distills]
        df_check_too_few_data = df.copy()
        df_check_too_few_data["t_change"] = df_check_too_few_data.groupby(["Trial"] + list(bounds.keys()))["Time"].diff()
        df_check_too_few_data = df_check_too_few_data[df_check_too_few_data["t_change"] > 0].reset_index(drop=True)
        if df_check_too_few_data.shape[0] == 0:
            values = self.search_space.sample()
            df = self.data
            new_T = df[df["Trial"] == str(base_trial_id)].iloc[-1, :]["Time"]
            new_Reward = df[df["Trial"] == str(base_trial_id)].iloc[-1, :].Reward

            lst = [[new_trial_id, new_T] + values + [new_Reward, "random", self.n_distills]]
            cols = ["Trial", "Time"] + list(bounds) + ["Reward", "config_source", "n_distills"]
            new_entry = pd.DataFrame(lst, columns=cols)

            self.data = pd.concat([self.data, new_entry]).reset_index(drop=True)
            return values

        df = df.copy()
        df["Reward"] = -df["Reward"]  # minimization
        ####################################### </diff wrt PB2>

        # At this point, df contains only the good n_distill
        # Group by trial ID and hyperparams.
        # Compute change in timesteps and reward.
        df["y"] = df.groupby(["Trial"] + list(bounds.keys()))["Reward"].diff()
        df["t_change"] = df.groupby(["Trial"] + list(bounds.keys()))["Time"].diff()
        # Delete entries without positive change in t. Me: there should be none (because sync)
        df = df[df["t_change"] > 0].reset_index(drop=True)
        df["R_before"] = df.Reward - df.y

        # Normalize the reward change by the update size.
        # For example if trials took diff lengths of time.
        df["y"] = df.y / df.t_change
        df = df[~df.y.isna()].reset_index(drop=True)
        df = df.sort_values(by="Time").reset_index(drop=True)

        # <diff wrt PB2/> 100 last points, not 1000 </diff wrt PB2> ###########
        df = df.iloc[-100:, :].reset_index(drop=True)

        # We need this to know the T and Reward for the weights.
        dfnewpoint = df[(df["Trial"] == str(base_trial_id)) & (df["Time"] == df["Time"].max())]
        if dfnewpoint.empty:
            print('there is a bug somewhere, print the variables')
            print(f'{df=}')
            print(f'{base_trial_id=} {new_trial_id=} {self.t_cur=}')
            # save df as tmp csv for debugging
            df.to_csv(self.exp_dir / 'data_DEBUG.csv', index=False)

        # Now specify the dataset for the GP.
        y = np.array(df.y.values)

        # Metadata we keep -> episodes and reward.
        r_t = df[["R_before", "Time"]]
        hparams = df[bounds.keys()]
        # for categorical hyperparameters, need to go from value to its index
        # (and same for current)
        # (no normalization should be done)
        if current is not None:
            current = pd.DataFrame(current, columns=list(bounds.keys()))
        for hp_noncont in bounds_noncont.keys():
            hparams.loc[:, hp_noncont] = hparams[hp_noncont].apply(
                lambda x: self.search_space.get_idx_by_value(hp_noncont, x)
            )
            dfnewpoint.loc[:, hp_noncont] = dfnewpoint[hp_noncont].apply(
                lambda x: self.search_space.get_idx_by_value(hp_noncont, x)
            )
            if current is not None:
                current.loc[:, hp_noncont] = current[hp_noncont].apply(
                    lambda x: self.search_space.get_idx_by_value(hp_noncont, x)
                )
        if current is not None:
            # current contained categorical values (likely strings) => cont values also became strings
            # => need to convert them back to float
            for hp_cont in bounds_cont.keys():
                current[hp_cont] = current[hp_cont].astype(float)
            current = current.values
        X = pd.concat([hparams, r_t], axis=1).values  # unlike in PB2, concat to the end
        X_best = dfnewpoint.iloc[-1, :][list(bounds.keys()) + ["R_before", "Time"]].values
        r_t_current = np.tile(dfnewpoint[["R_before", "Time"]].values, (current.shape[0], 1))

        current = np.hstack([current, r_t_current]).astype(float)
        int_indices = np.array(self.search_space.get_int_indices())
        new = _select_config(self.casmo, X, y, current, X_best, bounds, int_indices, self.verbose)
        values = [fn_(new_) for fn_, new_ in
                  zip(self.search_space.get_fns_to_convert_from_encoding(treat_int_as_cont=True), new)]

        df["Reward"] = -df["Reward"] # we minimized in BO, but actually we maximize
        new_T = df[df["Trial"] == str(base_trial_id)].iloc[-1, :]["Time"]
        new_Reward = df[df["Trial"] == str(base_trial_id)].iloc[-1, :].Reward

        lst = [[new_trial_id] + [new_T] + values + [new_Reward, "bo",
                                                    self.n_distills]]
        cols = ["Trial", "Time"] + list(bounds) + ["Reward", "config_source", "n_distills"]
        new_entry = pd.DataFrame(lst, columns=cols)

        # Create an entry for the new config, with the reward from the
        # copied agent.
        self.data = pd.concat([self.data, new_entry]).reset_index(drop=True)

        return values

    def adjust_tr_length(self, restart=False):
        """Adjust trust region size -- the criterion is that whether any config sampled by BO outperforms the other config
        sampled otherwise (e.g. randomly, or carried from previous timesteps). If true, then it will be a success or
        failure otherwise."""
        agents = self.data[self.data.Time == (self.t_cur + self.t_step)]
        # get the negative reward
        best_reward = np.max(agents.Reward.values)
        # get the agents selected by Bayesian optimization
        bo_agents = agents[agents.config_source == 'bo']
        if bo_agents.shape[0] == 0:
            return

        # if the best reward is caused by a config suggested by BayesOpt
        if np.max(bo_agents.Reward.values) == best_reward:
            self.casmo.succcount += 1
            self.casmo.failcount = 0
        else:
            self.casmo.failcount += 1
            self.casmo.succcount = 0

        if self.casmo.succcount == self.casmo.succtol:  # Expand trust region
            self.casmo.length = min(
                [self.casmo.tr_multiplier * self.casmo.length, self.casmo.length_max])
            self.casmo.length_cat = min(
                self.casmo.length_cat * self.casmo.tr_multiplier, self.casmo.length_max_cat)
            self.casmo.succcount = 0
            logging.info(f'Expanding TR length to {self.casmo.length}')
        elif self.casmo.failcount == self.casmo.failtol:  # Shrink trust region
            self.casmo.failcount = 0
            self.casmo.length_cat = max(
                self.casmo.length_cat / self.casmo.tr_multiplier, self.casmo.length_min_cat)
            self.casmo.length = max(
                self.casmo.length / self.casmo.tr_multiplier, self.casmo.length_min)
            logging.info(f'Shrinking TR length to {self.casmo.length}')

        if restart and (self.casmo.length <= self.casmo.length_min
                        or self.casmo.length_max_cat <= self.casmo.length_min_cat):
            self._restart()

    def _restart(self):
        print('Restarting!')
        self.n_distills += 1  # this will cause the GP to reset in the next iteration
        self.casmo.length = self.casmo.length_init
        self.casmo.length_cat = self.casmo.length_init_cat
        self.casmo.failcount = self.casmo.succcount = 0

        # save best weight of this iteration & associated info
        individuals_sorted = np.argsort([f[-1][1] for f in self.fitness_history])
        idx_best = int(individuals_sorted[-1])
        t = self.t_cur + self.t_step
        ckpt_path = Path(self.cpkt_dir) / f'pop_{idx_best}_t{t}.pt'
        out_path = self.ckpt_best_dir / f'best_model_{t}.pt'
        shutil.copy(ckpt_path, out_path)

        with open(self.exp_dir / f'best_info_{t}.yaml', 'w') as f:
            yaml.safe_dump({
                'solution': self.pop[idx_best],
                'fitness': self.fitness_history[idx_best][-1][1],
                'fitness_history': self.fitness_history[idx_best],
                'solution_history': self.solution_history[idx_best],
                'solution_id': idx_best,
                't': t
            }, f)

        if self.reinit_weights_strategy == 'distill':
            # replace weights of everything outside top quantile with the top quantile
            # these will be the teachers weights in the distillation

            idx_top = individuals_sorted[-int(self.quant_top * self.pop_size):]
            idx_rest = individuals_sorted[:-int(self.quant_top * self.pop_size)]
            for i in idx_rest:
                chosen_idx = np.random.choice(idx_top)

                self.pop[i] = copy.deepcopy(self.pop[chosen_idx])
                self.fitness_history[i] = copy.deepcopy(self.fitness_history[chosen_idx])
                self.solution_history[i] = copy.deepcopy(self.solution_history[chosen_idx])

                ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{t}.pt'
                ckpt_path.unlink()
                ckpt_chosen_path = Path(self.cpkt_dir) / f'pop_{chosen_idx}_t{t}.pt'
                save_explored_ckpt_to_path(self.task, self.pop[i], ckpt_chosen_path, ckpt_path)

                print(f'Replaced weights of {i}  with {chosen_idx}')

            if self.n_init > self.pop_size_actual:
                for i in range(self.pop_size_actual, self.n_init):
                    chosen_idx = np.random.choice(idx_top)

                    self.pop.append(copy.deepcopy(self.pop[chosen_idx]))
                    self.fitness_history.append(copy.deepcopy(self.fitness_history[chosen_idx]))
                    self.solution_history.append(copy.deepcopy(self.solution_history[chosen_idx]))

                    ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{t}.pt'
                    ckpt_chosen_path = Path(self.cpkt_dir) / f'pop_{chosen_idx}_t{t}.pt'
                    save_explored_ckpt_to_path(self.task, self.pop[i], ckpt_chosen_path, ckpt_path)

                    print(f'Added weights of {i} from {chosen_idx}')

    def save_best(self):
        # save best to yaml
        best_idx = int(np.argmax([self.fitness_history[i][-1][1] for i in range(self.pop_size)]))
        best_fitness = self.fitness_history[best_idx][-1][1]

        # compare with final results of previous restarts
        best_f_prev_restart = float('-inf')
        best_info_prev = None
        for p in self.exp_dir.glob('best_info_*.yaml'):
            with open(p, 'r') as fp:
                info = yaml.safe_load(fp)
                if info['fitness'] > best_f_prev_restart:
                    best_f_prev_restart = info['fitness']
                    best_info_prev = info

        if best_f_prev_restart > best_fitness:
            best_fitness = best_f_prev_restart
            best_idx = best_info_prev['solution_id']

            best = best_info_prev['solution']
            best_sol_history = best_info_prev['solution_history']
            t = best_info_prev['t']
            best_path = Path(self.ckpt_best_dir) / f'best_model_{t}.pt'
            best_fit_history = best_info_prev['fitness_history']
        else:
            best = self.pop[best_idx]
            best_sol_history = self.solution_history[best_idx]
            t = self.t_cur
            best_path = Path(self.cpkt_dir) / f'pop_{best_idx}_t{self.t_cur}.pt'
            best_fit_history = self.fitness_history[best_idx]

        print(f'Best solution: {best_sol_history}')

        fp = (self._task_fn_ray.options(**self.ray_options)
             .remote(self.task, self.seed_base, best, 0, 0,
                     torch.load(best_path),
                     None, ['test']))
        res = ray.get(fp)
        with open(self.exp_dir / 'best_info.yaml', 'w') as fp:
            yaml.safe_dump({'solution': best, 'fitness': best_fitness,
                            'fitness_history': best_fit_history,
                            'solution_history': best_sol_history,
                            'solution_id': best_idx,
                            'test': res['test'],
                            't': t
                            }, fp)
        print(f'Val: {best_fitness:.4f}, Test: {res["test"]:.4f}')

        shutil.copy(best_path,
                    Path(self.exp_dir) / 'best_model.pt')

        if 'policy_gif' in res:
            with open(Path(self.exp_dir) / 'policy.webp', 'wb') as fp:
                fp.write(res['policy_gif'])

        utils.set_plot_style()

        # plot all fitnesses, with the best one highlighted
        plt.figure(figsize=(8, 5))
        for i in range(self.pop_size):
            linewidth = 1
            plt.plot([f[0] for f in self.fitness_history[i]], [f[1] for f in self.fitness_history[i]],
                     linewidth=linewidth)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xlabel('t')
        plt.ylabel('fitness')
        plt.title(f'{self.exp_dir.name}: fitness')
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'fitness_history.png')
        plt.show()
        plt.close()

def _select_config(
        casmo: _Casmo,
        X_raw: np.array,
        yraw: np.array,
        current: np.array,
        X_best_raw: np.array,
        bounds: dict,
        int_dims: np.array,
        verbose: bool
) -> np.ndarray:

    limits, limits_hps, limits_reward_time, int_half_ranges = _compute_limits(
        X_best_raw, X_raw, bounds, int_dims
    )

    X, X_current, t, t_current, x_center, y = _normalize_based_on_limits(
        X_best_raw, yraw, current, X_raw,
        int_half_ranges, limits,
        int_dims
    )

    hypers = {}
    use_time_varying_gp = np.unique(t).shape[0] > 1

    if X_current is not None:  # should be always True in the sync setup (but not in meta-setup)
        # 2. Train a GP conditioned on the *real* data which would give us the fantasised y output for the pending fixed_points
        gp, hypers, dtype = _train_gp(casmo, X, y, t, use_time_varying_gp, verbose)

        # 3. Get the posterior prediction at the fantasised points
        gp.eval()
        with torch.no_grad():
            if use_time_varying_gp:
                t_x_current = torch.hstack(
                    (torch.tensor(t_current, dtype=dtype).reshape(-1, 1), torch.tensor(X_current, dtype=dtype)))
            else:
                t_x_current = torch.tensor(X_current, dtype=dtype)
            pred_ = gp(t_x_current).mean
            y_fantasised = pred_.detach().numpy()
        y = np.concatenate((y, y_fantasised))
        X = np.concatenate((X, X_current), axis=0)
        t = np.concatenate((t, t_current))
        del gp

    # if_restandardize = False # restandardization leads to fantasized points influencing the gp mean, which is bad => avoid.
    # if if_restandardize:
    #     y = copula_standardize(copy.deepcopy(y).ravel())

    next_config = casmo._create_and_select_candidates(X, y, length_cat=casmo.length_cat,
                                                      length_cont=casmo.length,
                                                      hypers=hypers, batch_size=1,
                                                      t=t if use_time_varying_gp else None,
                                                      time_varying=use_time_varying_gp,
                                                      x_center=x_center,
                                                      t_center=t.max(),
                                                      frozen_dims=None,
                                                      frozen_vals=None,
                                                      n_training_steps=100).flatten() #increased from 1 to 100 in case hp loading fails

    next_config = next_config[:-1]  # remove reward

    # print(f'A {next_config=}')
    # convert back...
    # subtract int half ranges
    next_config[int_dims] -= int_half_ranges
    next_config = next_config * (np.max(limits_hps, axis=0) - np.min(limits_hps, axis=0)) + \
                  np.min(limits_hps, axis=0)
    # print(f'B {next_config=}')

    next_config = next_config.astype(np.float32)
    return next_config


def _train_gp(casmo, X, y, t, use_time_varying_gp, verbose):
    if len(X) < MIN_CUDA:
        device, dtype = torch.device("cpu"), torch.float32
    else:
        device, dtype = casmo.device, casmo.dtype
    with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
        X_torch = torch.tensor(X).to(device=device, dtype=dtype)
        # here we replace the nan values with zero, but record the nan locations via the X_torch_nan_mask
        y_torch = torch.tensor(y).to(device=device, dtype=dtype)
        # add some noise to improve numerical stability
        y_torch += torch.randn(y_torch.size()) * 1e-5
        if use_time_varying_gp:
            t_torch = torch.tensor(t).to(device=device, dtype=dtype)
        else:
            t_torch = None

        gp = train_gp(
            configspace=casmo.cs,
            train_x=X_torch,
            train_y=y_torch,
            use_ard=casmo.use_ard,
            num_steps=200,
            time_varying=use_time_varying_gp,
            train_t=t_torch,
            verbose=verbose
        )
        hypers = gp.state_dict()
        gp.eval()
    return gp, hypers, dtype


def _compute_limits(X_best_raw, X_raw, bounds, int_dims):
    limits_hps = np.array(list(bounds.values())).T.astype(np.float32)
    # Me: if min == max, get nan. Therefore, +-eps, same as in PB2-Mix implementation
    max_is_min = limits_hps[0] == limits_hps[1]
    limits_hps[0][max_is_min] -= 1e-8
    limits_hps[1][max_is_min] += 1e-8
    num_f = 2  # reward, time
    X_reward_time = X_raw[:, -num_f:]
    if X_best_raw is not None:
        # for proper normalization in the meta-GP case, need to include X_best into normalization (otherwise it'll have time larger than max)
        X_reward_time = np.concatenate((X_reward_time, X_best_raw[None, -num_f:].astype(float)), axis=0)
    limits_reward_time = np.concatenate(
        (np.max(X_reward_time, axis=0), np.min(X_reward_time, axis=0))
    ).reshape(2, X_reward_time.shape[1])
    # Me: if min == max, get nan. Therefore, +-eps, same as in PB2-Mix implementation.
    # Note that here max is 0, min is 1. This inconsistency is crazy but
    # I keep it to avoid introducing unnecessary changes wrt official implementations.
    max_is_min = limits_reward_time[0] == limits_reward_time[1]
    limits_reward_time[0][max_is_min] += 1e-8
    limits_reward_time[1][max_is_min] -= 1e-8
    limits = np.concatenate((limits_hps, limits_reward_time), axis=1)
    '''
        Need to adjust int dims.
        Without this change, the values are transformed into floats rather weirdly.
        E.g., for an integer hparam in range[1, 4], the value 1 becomes 0, 2 -> 0.33, 3-> 0.67, 4->1
        But this is later used to define bounding box of e.g. +- 0.2, which leads to wrong results.
        Ergo, normalize 1->0, 2->0.25, 3->0.5, 4->0.75 (i.e. add 1 to max value), and add half a range (0.125).
    '''
    int_n_values = limits_hps[1, int_dims] - limits_hps[0, int_dims] + 1
    limits[1, int_dims] += 1
    int_half_ranges = 0.5 * (1 / int_n_values)
    return limits, limits_hps, limits_reward_time, int_half_ranges


def _normalize_based_on_limits(X_best_raw, y_raw, current, X_raw, int_half_ranges, limits,
                               int_dims):
    X = _normalize_X(X_raw, int_dims, int_half_ranges, limits)
    X_current = _normalize_X(current, int_dims, int_half_ranges, limits)
    x_center = _normalize_X(X_best_raw, int_dims, int_half_ranges, limits)

    X, t = _split_time_from_X(X)
    X_current, t_current = _split_time_from_X(X_current)
    if x_center is not None:
        x_center = x_center[:-1]
        # add fake "batch" dimension to x_center
        x_center = x_center.reshape(1, -1)

    y = copula_standardize(copy.deepcopy(y_raw).ravel())  # since don't want to restandardize later, do it here
    return X, X_current, t, t_current, x_center, y

def _normalize_X(X_raw, int_dims, int_half_ranges, limits):
    if X_raw is None:
        return None

    X = normalize(X_raw, limits).astype(float)
    X[..., int_dims] += int_half_ranges
    return X

def _split_time_from_X(X):
    if X is None:
        return None, None
    X, t = X[:, :-1], X[:, -1]
    return X, t