import copy
import random
import time
from pathlib import Path

import jax
import math
import numpy as np
import torch
from tqdm import tqdm

from optimizer import get_optimizer
import torch.utils.tensorboard as tb

from task.ppo_torch import Agent, eval_unroll, train_unroll, sd_map, unroll_first, shuffle_and_batch, \
    train_unroll_efficient, unroll_first_efficient, train_unroll_efficient_combined, LinearScheduleFromBGPBT
from search_space.cs import ConfigSpaceSearchSpace
from task import brax_task_utils
from utils import adjust_optimizer_settings, convert_config_from_logarithmic

from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
import traceback
import gc


class BraxTask:
    def __init__(self, cfg, search_space, **__):
        self.cfg = cfg
        self.search_space: ConfigSpaceSearchSpace = search_space
        self.num_evals_per_step = self.cfg.task.num_evals_per_step # how often to eval for the curves
        self.viz = self.cfg.task.get('viz', False)

        # Hardcoded values from brax/BG-PBT:
        self.num_envs = self.cfg.task.get('num_envs', 2048)
        self.episode_length = 1000
        self.num_minibatches = 32

        self.env_name = self.cfg.task.name
        def _create_env():
            env = envs.create(self.env_name, batch_size=self.num_envs,
                              episode_length=self.episode_length,
                              backend='spring')
            env = gym_wrapper.VectorGymWrapper(env)
            env.seed(cfg.general.seed_base) # seeds different from base will be set in every __call__
            return env
        self.env = _create_env()
        self.env_eval = _create_env()
        self.scheduler = cfg.task.get('scheduler', None)
        print(f'{self.scheduler=}')

    def prepare(self, seed, solution, cpkt_loaded, tensorboard_dir, only_evaluate, distill_kwargs=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tb_writer = None
        if (only_evaluate is None) and (tensorboard_dir is not None):
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = tb.SummaryWriter(tensorboard_dir)

        # create env - within __call__ so that the seed would influence it.
        env = self.env
        env.seed(seed)
        env = torch_wrapper.TorchWrapper(env, device=device)
        # Warm-start
        observation = env.reset()
        action = torch.zeros(env.action_space.shape).to(device)
        env.step(action)
        env_eval = torch_wrapper.TorchWrapper(self.env_eval, device=device)
        env_eval.seed(random.randint(0, 2**32 - 1))

        # get rl values
        rl_config = self.get_rl_vals(solution)
        batch_size = rl_config['batch_size']
        num_update_epochs = rl_config['num_update_epochs']
        unroll_length = rl_config['unroll_length']

        # create agent
        agent = self.create_agent(env, rl_config)
        if cpkt_loaded is not None:
            agent.load_state_dict(cpkt_loaded['model_state_dict'])
        agent.to(device)

        # create teacher, if needed
        teacher = None
        num_already_done_distill_steps = None
        num_total_distill_steps = None
        if distill_kwargs is not None:
            teacher_hparams_encoded = distill_kwargs['teacher_hparams']
            rl_config_teacher = self.get_rl_vals(teacher_hparams_encoded)
            teacher = self.create_agent(env, rl_config_teacher)
            teacher_ckpt_loaded = distill_kwargs['teacher_ckpt_loaded']
            teacher.load_state_dict(teacher_ckpt_loaded['model_state_dict'])
            teacher.to(device)

            num_already_done_distill_steps = distill_kwargs['num_already_done_distill_steps']
            num_total_distill_steps = distill_kwargs['num_total_distill_steps']

        # create optimizer
        solution_optimizer_vals = self.get_optimizer_vals(solution)
        optimizer = get_optimizer(self.cfg, agent, solution_optimizer_vals)
        if cpkt_loaded is not None:
            optimizer.load_state_dict(cpkt_loaded['optimizer_state_dict'])  # this will overwrite solution_optimizer_vals
        optimizer = adjust_optimizer_settings(optimizer, solution_optimizer_vals)

        # scheduler
        scheduler = self.scheduler
        if scheduler is not None:
            # a scheduler is not compatible with PBTs, only with other HPO algos with fixed step size
            scheduler_vals = self.get_scheduler_vals(solution)
            match scheduler:
                case 'cosine':
                    max_steps = self.cfg.algo.t_max // self.cfg.algo.t_step # use t_step because wanna update lr once per train_and_eval() call
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
                case 'cosine_restart':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=scheduler_vals['T_0'],
                        T_mult=scheduler_vals['T_mult'],
                        eta_min=scheduler_vals['eta_min']
                    )
                case _:
                    raise ValueError(f'Unknown scheduler: {scheduler}')

            if cpkt_loaded is not None:
                scheduler.load_state_dict(cpkt_loaded['scheduler_state_dict'])

        return {
            'agent': agent, 'env': env, 'env_eval': env_eval, 'batch_size': batch_size,
            'num_update_epochs': num_update_epochs, 'unroll_length': unroll_length,
            'tb_writer': tb_writer, 'observation': observation,
            'optimizer': optimizer, 'scheduler': scheduler,
            'teacher': teacher, 'num_already_done_distill_steps': num_already_done_distill_steps,
            'num_total_distill_steps': num_total_distill_steps
        }

    def prepare_with_new_seed(self, kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # create env - within __call__ so that the seed would influence it.
        env = self.env
        env.seed(kwargs['seed'])
        env = torch_wrapper.TorchWrapper(env, device=device)
        # Warm-start
        observation = env.reset()
        action = torch.zeros(env.action_space.shape).to(device)
        env.step(action)
        env_eval = torch_wrapper.TorchWrapper(self.env_eval, device=device)
        env_eval.seed(random.randint(0, 2**32 - 1))
        kwargs['env'] = env
        kwargs['env_eval'] = env_eval
        kwargs['observation'] = observation
        return kwargs

    def eval(self, vars_dict, evaluate_targets):
        env_eval, agent, seed, t = [
            vars_dict[k]
            for k in ['env_eval', 'agent', 'seed', 't']
        ]
        assert type(evaluate_targets) == list
        out = {}

        for target in evaluate_targets:
            if target == 'test':
                env_eval.seed(seed)
            eval_res = self._eval(env_eval, agent, [], t, None)
            out[target] = eval_res[0]
            if target == 'val':
                out['fitness'] = eval_res[0]
            if target == 'test':
                gif = self.visualize_policy(agent)
                out['policy_gif'] = gif

        return out

    def train_and_eval(self, vars_dict):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        (batch_size, unroll_length, num_update_epochs, env,
         env_eval, agent, optimizer, scheduler,
         tb_writer, observation, t, t_step) = [
            vars_dict[k]
            for k in ['batch_size', 'unroll_length', 'num_update_epochs', 'env',
                      'env_eval', 'agent', 'optimizer', 'scheduler',
                      'tb_writer', 'observation', 't', 't_step']
        ]
        teacher = vars_dict.get('teacher', None)
        if teacher is not None:
            num_already_done_distill_steps = vars_dict['num_already_done_distill_steps']
            num_total_distill_steps = vars_dict['num_total_distill_steps']
            # default values copied from the BG-PBT codebase
            DISTILLATION_PARAMS = {
                'policy_reg_coef': 5.,  # FLOAT
                'value_reg_coef': 0.,  # FLOAT
                'rl_coef': 1.,
                'distill_num_epochs': 4,  # INT
            }
            num_update_epochs = DISTILLATION_PARAMS['distill_num_epochs'] #hardcoded in BG-PBT
            DISTILLATION_SCHEDULE = {
                'distill_anneal_frac': 0.8,
                'distill_anneal_init': 1,
                'distill_anneal_final': 0.05
            }
            alpha_schedule = LinearScheduleFromBGPBT(
                burnin=0, # BG paper doesn't mention burn-in, the code uses total distillation steps but that is equivalent to not having a schedule => go with the paper.
                initial_value=DISTILLATION_SCHEDULE['distill_anneal_init'],
                final_value=DISTILLATION_SCHEDULE['distill_anneal_final'],
                decay_time=int(DISTILLATION_SCHEDULE['distill_anneal_frac'] * num_total_distill_steps),
            )

        losses = {'train': [], 'test': []}
        curve = []

        # Should do t_step environment steps (and potentially more training steps).
        # To have the exact number of env steps, generate more and discard the excess. This is efficient thanks to brax.
        # Note that in order to have the desired number of intermediate evaluations,
        # we need to base them on training steps, not env steps.
        num_steps = batch_size * self.num_minibatches * unroll_length
        assert t_step >= num_steps
        num_epochs = t_step // num_steps
        num_unrolls = batch_size * self.num_minibatches // env.num_envs
        # print(f'{num_unrolls=}')
        extra_unrolls_last_epoch = math.ceil((t_step % num_steps) / (env.num_envs * unroll_length))  # to go over
        excess_individual_unrolls = extra_unrolls_last_epoch * env.num_envs - math.ceil(
            ((t_step % num_steps) / unroll_length))  # to go back to target

        total_training_batches = (num_epochs - 1) * num_update_epochs * self.num_minibatches + \
                                 1 * num_update_epochs * \
                                 (
                                    (
                                        (num_unrolls + extra_unrolls_last_epoch) * env.num_envs - excess_individual_unrolls
                                    ) // batch_size
                                 )  # important to first divide by batch_size

        eval_every_batches = total_training_batches // self.num_evals_per_step
        batches_since_last_eval = 0
        batches_total = 0
        num_evals_done = 0

        try:
            for i_inner_epoch in tqdm(range(num_epochs), desc=f'train+val'):
                agent.train()

                # train_unroll makes (num_unrolls * unroll_length * num_envs) steps of the environment.
                # num_unrolls is controlled by me. unroll_length is a searchable HP.
                # num_envs is controlled by me but should probably not be changed to keep efficiency high.

                # this optimization uses less VRAM but more time.
                observation, td = train_unroll_efficient(agent, env, observation,
                                               num_unrolls + (
                                                   0 if i_inner_epoch != num_epochs - 1 else extra_unrolls_last_epoch),
                                               unroll_length, env.num_envs, device)
                td = sd_map(unroll_first, td)

                if i_inner_epoch == num_epochs - 1 and extra_unrolls_last_epoch > 0:
                    if excess_individual_unrolls > 0:
                        def remove_extra(data):
                            return data[:, :-excess_individual_unrolls, ...]

                        td = sd_map(remove_extra, td)

                agent.update_normalization(td.observation)

                for i_update_epoch in range(num_update_epochs):
                    batch_iter = shuffle_and_batch(td, batch_size, device)
                    for td_minibatch in batch_iter:
                        td_minibatch_dict = td_minibatch._asdict()
                        if teacher is None:
                            loss = agent.loss(td_minibatch_dict)
                        else:
                            with torch.no_grad():
                                teacher_v, teacher_loc, teacher_scale = teacher.compute_policy_and_value(td_minibatch_dict)
                            rl_loss = agent.loss(td_minibatch_dict)
                            student_v, student_loc, student_scale = agent.compute_policy_and_value(td_minibatch_dict)

                            # Value loss is L2 loss - since the coeff is hardcoded as 0, economize compute
                            # val_loss = (student_v - teacher_v).pow(2).mean()

                            # Policy loss is KL divergence
                            te = torch.distributions.Normal(teacher_loc, teacher_scale)
                            st = torch.distributions.Normal(student_loc, student_scale)
                            policy_loss = torch.distributions.kl_divergence(te, st).mean()

                            # BG code uses "total_steps" as the argument, which makes little sense,
                            # as that correspond to using the final value of the schedule
                            cur_step_distill = num_already_done_distill_steps + batches_total * batch_size * unroll_length
                            alpha = alpha_schedule(cur_step_distill)

                            # if cur_step_distill % 1000 == 0:
                            #     print(f'{cur_step_distill=} {alpha=:.5f}')

                            loss = (DISTILLATION_PARAMS['rl_coef'] * rl_loss +
                                    alpha * (#DISTILLATION_PARAMS['value_reg_coef'] * val_loss +
                                             DISTILLATION_PARAMS['policy_reg_coef'] * policy_loss))

                        losses['train'].append(loss.item())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        batches_total += 1
                        batches_since_last_eval += 1
                        if batches_since_last_eval >= eval_every_batches:
                            # print('Before eval')
                            # time.sleep(5)
                            num_evals_done += 1
                            timestamp = int(t + num_evals_done * (
                                        eval_every_batches / total_training_batches) * t_step)  # fake timestamp, due to diff between env and training steps, see reasoning above
                            val_reward, curve = self._eval(env_eval, agent, curve, timestamp, tb_writer)
                            batches_since_last_eval = 0

                            if tb_writer is not None:
                                tb_writer.add_scalar('loss/train', np.mean(losses['train']), timestamp)
                            losses['train'] = []

                del td
                gc.collect() # experimentally confirmed to reduce VRAM usage in this case.

            if scheduler is not None:
                scheduler.step()
                # print(f'{scheduler.get_last_lr()=}')

            viz_weights = False
            if viz_weights:
                for name, param in agent.named_parameters():
                    param_data = param.clone().cpu().data.numpy()
                    try:
                        if param_data.min() != param_data.max():
                            tb_writer.add_histogram(name, param_data, t + t_step)
                    except:
                        print(f'Failed to viz {name=} {param=}')

        except Exception as e:
            print(f'Exception in RL training: {e}')
            print(traceback.format_exc())
            val_reward = -1000
            curve = []
            for i in range(self.num_evals_per_step):
                timestamp = int(t + i * (eval_every_batches / total_training_batches) * t_step)
                curve.append((timestamp, val_reward))

        if tb_writer is not None:
            tb_writer.close()
        print(f'{curve=}')

        # if self.viz:
        #     gif = self.visualize_policy(agent)
        #     with open(Path(self.cfg.path.dir_exp) / f'policy_{t:09d}.webp', 'wb') as f:
        #         f.write(gif)

        out = {'fitness': curve[-1][1],
                'curve': curve,
                'metrics': {'val': val_reward, 'test': None}}

        updated_state_dicts = self._get_dict_to_save(vars_dict)

        return out, updated_state_dicts

    def _get_dict_to_save(self, vars_dict):
        agent = vars_dict['agent']
        if vars_dict.get('to_cpu', True):
            agent = agent.cpu()
        out = {'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': vars_dict['optimizer'].state_dict()
               }
        if vars_dict['scheduler'] is not None:
            out['scheduler_state_dict'] = vars_dict['scheduler'].state_dict()
        return out

    def __call__(self, seed, solution, t, t_step, cpkt_loaded, tensorboard_dir, only_evaluate, distill_kwargs=None):
        st = time.time()
        prepped = self.prepare(seed, solution, cpkt_loaded, tensorboard_dir, only_evaluate, distill_kwargs)
        print(f'Preparation time: {time.time() - st:.2f} seconds')
        del cpkt_loaded

        prepped['t'], prepped['t_step'] = t, t_step

        if only_evaluate is not None:
            prepped['seed'] = seed
            out = self.eval(prepped, only_evaluate)
            return out

        out, dict_to_save = self.train_and_eval(prepped)

        out['dict_to_save'] = dict_to_save
        return out

    def create_agent(self, env, rl_config, device='cuda'):
        policy_layers = [env.observation_space.shape[-1], 64, 64, env.action_space.shape[-1] * 2]
        value_layers = [env.observation_space.shape[-1], 64, 64, 1]
        agent = Agent(policy_layers, value_layers, rl_config['entropy_cost'], rl_config['discounting'],
                      rl_config['reward_scaling'], rl_config['lambda_'], rl_config['ppo_epsilon'],
                      device)
        return agent

    def _load_states(self, vars_dict, cpkt_loaded):
        agent = vars_dict['agent']
        device = next(agent.parameters()).device
        state_dict = cpkt_loaded['model_state_dict']
        device_sd = next(iter(state_dict.values())).device
        if device_sd != device:
            state_dict = {k: v.to(device) for k, v in state_dict.items()}
        agent.load_state_dict(state_dict)

        vars_dict['optimizer'].load_state_dict(cpkt_loaded['optimizer_state_dict'])
        if vars_dict['scheduler'] is not None:
            vars_dict['scheduler'].load_state_dict(cpkt_loaded['scheduler_state_dict'])
        return vars_dict

    def _eval(self, env_eval, agent, curve, timestamp, tb_writer):
        st = time.time()
        agent.eval()
        with torch.no_grad():
            episode_count, episode_reward = eval_unroll(agent, env_eval, self.episode_length)
            episode_reward = episode_reward.item()

        if tb_writer is not None:
            tb_writer.add_scalar('reward/val', episode_reward, timestamp)

        curve.append((timestamp, episode_reward))
        agent.train()
        print(f'Eval time: {time.time() - st:.2f} seconds')
        return episode_reward, curve

    def get_optimizer_vals(self, solution):
        config_dict = self._solution_to_dict(solution)
        config_dict = convert_config_from_logarithmic(config_dict)

        config_dict = {k:v for k, v in config_dict.items() if k in ['lr', 'weight_decay', 'momentum']}
        return config_dict

    def get_scheduler_vals(self, solution):
        config_dict = self._solution_to_dict(solution)
        config_dict = convert_config_from_logarithmic(config_dict)

        config_dict = {k:v for k, v in config_dict.items() if k in ['T_0', 'T_mult', 'eta_min']}

        return config_dict

    def get_rl_vals(self, solution):
        config = copy.deepcopy(PPO_DEFAULT_CONFIGS[self.env_name])

        config_dict = self._solution_to_dict(solution)
        config_dict = convert_config_from_logarithmic(config_dict)

        for k, v in config_dict.items():
            if k in config:
                config[k] = v

        return config

    def _solution_to_dict(self, solution):
        if type(solution) == dict:
            config_dict = solution
        else:
            config_dict = self.search_space.vector_to_dict(solution)
        return config_dict


    def prepare_initial_ckpt(self, solution):
        rl_config = self.get_rl_vals(solution)
        model = self.create_agent(self.env, rl_config)
        optimizer = get_optimizer(self.cfg, model, self.get_optimizer_vals(solution))
        return {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}

    def get_fresh_model(self, solution):
        rl_config = self.get_rl_vals(solution)
        model = self.create_agent(self.env, rl_config, 'cpu')
        return model

    def visualize_policy(self, agent):
        agent = agent.cuda()
        env = envs.get_environment(self.env_name, backend='spring')
        rollout = []
        state = env.reset(jax.random.PRNGKey(0))
        env_step_jit = jax.jit(env.step)

        import brax.io.torch as brax_torch
        for _ in tqdm(range(1000)):
            rollout.append(state.pipeline_state)

            _, act = agent.get_logits_action(brax_torch.jax_to_torch(state.obs))
            act = agent.dist_postprocess(act)

            state = env_step_jit(state, brax_torch.torch_to_jax(act))

            if state.done:
                break

        gif = brax_task_utils.render(env.sys, rollout, fmt='webp', camera_id=0 if self.env_name != 'pusher' else -1)
        return gif

# optimized values from brax (except for lambda_, ppo_epsilon)
# for hopper and those after it use the same unoptimized default values as BG-PBT
PPO_DEFAULT_CONFIGS = {
    'ant': {
        'discounting': 0.97,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'reward_scaling': 10,
        'batch_size': 1024,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'humanoid': {
        'discounting': 0.97,
        'entropy_cost': 0.001,
        'unroll_length': 10,
        'reward_scaling': 0.1,
        'batch_size': 1024,
        'num_update_epochs': 8,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'halfcheetah': {
        'discounting': 0.95,
        'entropy_cost': 0.001,
        'unroll_length': 20,
        'reward_scaling': 1,
        'batch_size': 512,
        'num_update_epochs': 8,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'fetch': {
        'discounting': 0.997,
        'entropy_cost': 0.001,
        'unroll_length': 20,
        'reward_scaling': 5,
        'batch_size': 256,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'grasp': {
        'discounting': 0.99,
        'entropy_cost': 0.001,
        'unroll_length': 20,
        'reward_scaling': 10,
        'batch_size': 256,
        'num_update_epochs': 2,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'ur5e': {
        'discounting': 0.95,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'reward_scaling': 10,
        'batch_size': 1024,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'reacher': {
        'discounting': 0.95,
        'entropy_cost': 0.001,
        'unroll_length': 50,
        'reward_scaling': 5,
        'batch_size': 256,
        'num_update_epochs': 8,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'hopper': {
        'discounting': 0.97,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'reward_scaling': 10,
        'batch_size': 1024,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'walker2d': {
        'discounting': 0.97,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'reward_scaling': 10,
        'batch_size': 1024,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'pusher': {
        'discounting': 0.97,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'reward_scaling': 10,
        'batch_size': 1024,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
}
