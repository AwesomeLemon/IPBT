from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yaml
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from rliable.library import StratifiedBootstrap
from scipy.stats import stats
from rliable import library as rly
from rliable import metrics as rlm

LOGS_DIR = '/export/scratch2/data/aleksand/ipbt/logs/'

algo_to_pretty_name = {
    'pbt': 'PBT',
    'pb2rand': 'PB2',
    'pb2mix': 'PB2-Mix',
    'bgpbt': 'BG-PBT',
    'firepbt': 'FIRE-PBT',
    'ipbt': 'IPBT',
    'ipbt5': 'IPBT',
    'ipbt6': 'IPBT',
    'rs': 'RS',
    'asha': 'ASHA',
    'smac': 'SMAC3'
}

def _get_metric(seed_path, metric_name):
    if not (seed_path / 'best_info.yaml').exists():
        raise ValueError(f'no best_info.yaml in {seed_path}')
    with open(seed_path / 'best_info.yaml', 'r') as f:
        best_info = yaml.safe_load(f)
        best_test = best_info[metric_name]
    return best_test

def get_metric_many_exps(exp_names, fitness_transform=lambda x: x, seeds=(),
                         pretty_names=False, add_extra_dim=False, metric_name='test'):
    res = {}
    for exp_name in exp_names:
        exp_path = Path(LOGS_DIR) / exp_name
        test_results = []
        for seed in seeds:
            seed_path = exp_path / str(seed)
            best_test = _get_metric(seed_path, metric_name)
            test_results.append(fitness_transform(best_test))
        out = np.array(test_results)
        if pretty_names:
            if type(pretty_names) is bool:
                exp_name = algo_to_pretty_name[exp_name.split('_')[0]]
            else:
                exp_name = pretty_names(exp_name)
        if add_extra_dim:
            out = out[:, None]
        if exp_name not in res:
            res[exp_name] = out
        else:
            if type(res[exp_name]) is not list:
                res[exp_name] = [res[exp_name]]
            res[exp_name].append(out)
    return res

def get_iqms_many_exps(exp_names, fitness_transform=lambda x: x, seeds=(), metric_name = 'fitness'):
    res = get_metric_many_exps(exp_names, fitness_transform, seeds, metric_name=metric_name)
    print(res)
    iqms = [stats.trim_mean(res[exp_name], proportiontocut=0.25) for exp_name in exp_names]
    return iqms

def get_exp_with_best_iqm(exp_names, fitness_transform=lambda x: x, seeds=()):
    iqms = get_iqms_many_exps(exp_names, fitness_transform, seeds)
    print(f'val {iqms=}')
    index_best = np.argmax(iqms)
    return index_best

def plot_interval_estimates_toptobottom(point_estimates,
                            interval_estimates,
                            metric_names,
                            algorithms=None,
                            colors=None,
                            color_palette='colorblind',
                            max_ticks=4,
                            subfigure_width=3.4,
                            row_height=0.37,
                            xlabel_x_coordinate=0.4,
                            xlabel_y_coordinate=-0.1,
                            xlabel='Normalized Score',
                            show_title=True,
                            **kwargs):
  """Plots various metrics with confidence intervals.

  Args:
    point_estimates: Dictionary mapping algorithm to a list or array of point
      estimates of the metrics to plot.
    interval_estimates: Dictionary mapping algorithms to interval estimates
      corresponding to the `point_estimates`. Typically, consists of stratified
      bootstrap CIs.
    metric_names: Names of the metrics corresponding to `point_estimates`.
    algorithms: List of methods used for plotting. If None, defaults to all the
      keys in `point_estimates`.
    colors: Maps each method to a color. If None, then this mapping is created
      based on `color_palette`.
    color_palette: `seaborn.color_palette` object for mapping each method to a
      color.
    max_ticks: Find nice tick locations with no more than `max_ticks`. Passed to
      `plt.MaxNLocator`.
    subfigure_width: Width of each subfigure.
    row_height: Height of each row in a subfigure.
    xlabel_y_coordinate: y-coordinate of the x-axis label.
    xlabel: Label for the x-axis.
    **kwargs: Arbitrary keyword arguments.

  Returns:
    fig: A matplotlib Figure.
    axes: `axes.Axes` or array of Axes.
  """

  if algorithms is None:
    algorithms = list(point_estimates.keys())
  num_metrics = len(point_estimates[algorithms[0]])
  figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
  fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
  if colors is None:
    color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
    colors = dict(zip(algorithms, color_palette))
  h = kwargs.pop('interval_height', 0.6)

  for idx, metric_name in enumerate(metric_names):
    for alg_idx, algorithm in enumerate(algorithms):
      ax = axes[idx] if num_metrics > 1 else axes
      # Plot interval estimates.
      lower, upper = interval_estimates[algorithm][:, idx]
      y_algo = len(algorithms) - alg_idx - 1
      ax.barh(
          y=y_algo,
          width=upper - lower,
          height=h,
          left=lower,
          color=colors[algorithm],
          alpha=0.75,
          label=algorithm)
      # Plot point estimates.
      ax.vlines(
          x=point_estimates[algorithm][idx],
          ymin=y_algo - (7.5 * h / 16),
          ymax=y_algo + (6 * h / 16),
          label=algorithm,
          color='k',
          alpha=0.5)

      # plot dashed line for IQM of IPBT that goes across all rows
      if algorithm == 'IPBT':
          ax.vlines(
              x=point_estimates[algorithm][idx],
              ymin=- (7.5 * h / 16),
              ymax=len(algorithms) - 1 - (7.5 * h / 16),
              color='k',
              linestyle='--',
              alpha=0.5
          )

    ax.set_yticks(list(range(len(algorithms))))
    ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    if idx != 0:
      ax.set_yticks([])
    else:
      ax.set_yticklabels(list(reversed(algorithms)), fontsize='x-large')
    if show_title:
        ax.set_title(metric_name, fontsize='xx-large')
    # ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='x', which='major', labelsize='x-large', length=0.1, width=0.1)
    ax.tick_params(axis='y', which='major', labelsize='xx-large', length=0.1, width=0.1)
    _decorate_axis(ax, wrect=5)
    ax.spines['left'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.25)
  fig.text(xlabel_x_coordinate, xlabel_y_coordinate, xlabel, ha='center', fontsize='xx-large')
  plt.subplots_adjust(wspace=kwargs.pop('wspace', 0.11), left=0.0)
  return fig, axes

def _decorate_axis(ax, wrect=10, hrect=10):
  """Helper function for decorating plots."""
  # Hide the right and top spines
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['left'].set_linewidth(2)
  ax.spines['bottom'].set_linewidth(2)
  # Deal with ticks and the blank space at the origin
  # ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
  ax.spines['left'].set_position(('outward', hrect))
  ax.spines['bottom'].set_position(('outward', wrect))
  return ax


def plot_rliable_iqm_vs_best(
        exp_names_tasks_algos_hps, seeds, tasks, task_bounds, fast=False,
        suffix='', format='png', pad_inches=0.0,
        pretty_names=True, estimates_min=None, estimates_max=None,
        show_title=True,
        xlabel='Normalized performance',
        figure_width=8,
        row_height=0.8,
        xlabel_x_coordinate=0.4,
        xlabel_y_coordinate=-0.05,
):
    exp_names_per_task = []
    for exp_names_algos_hps in exp_names_tasks_algos_hps:
        exp_names_per_task.append([])
        for exp_names_hps in exp_names_algos_hps:
            best_idx = get_exp_with_best_iqm(exp_names_hps, seeds=seeds)
            exp_names_per_task[-1].append(exp_names_hps[best_idx])
    print(f'{exp_names_per_task=}')

    # dictionary mapping algorithms to scores of shape `(num_runs x num_games)`.
    score_dict = get_metric_many_exps(exp_names_per_task[0],
                                      seeds=seeds, pretty_names=pretty_names, add_extra_dim=True, metric_name='test')
    algo_names = list(score_dict.keys())
    for exp_names_cur in exp_names_per_task[1:]:
        score_dict_cur = get_metric_many_exps(exp_names_cur,
                                              seeds=seeds, pretty_names=pretty_names, add_extra_dim=True, metric_name='test')
        for k, v in score_dict_cur.items():
            score_dict[k] = np.concatenate([score_dict[k], v], axis=1)

    # assure that all seeds were loaded
    for k in score_dict:
        assert score_dict[k].shape[0] == len(seeds)

    # normalize to 0-1 per task
    for i, task in enumerate(tasks):
        task_min, task_max = task_bounds[task]
        for k in algo_names:
            score_dict[k][:, i] = (score_dict[k][:, i] - task_min) / (task_max - task_min)

    # #############IQM:
    aggregate_func = lambda x: np.array([rlm.aggregate_iqm(x)])
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        score_dict, aggregate_func, reps=500 if fast else 50000)
    fig, axes = plot_interval_estimates_toptobottom(
        aggregate_scores, aggregate_score_cis,
        metric_names=['IQM (vs best)'],
        algorithms=algo_names, xlabel=xlabel,
        row_height=row_height,
        subfigure_width=figure_width,
        xlabel_x_coordinate=xlabel_x_coordinate,
        xlabel_y_coordinate=xlabel_y_coordinate,
        show_title=show_title
    )

    if estimates_min is not None and estimates_max is not None:
        plt.xlim(estimates_min * 0.95, estimates_max * 1.05)

    plt.tight_layout()
    plt.savefig(Path(LOGS_DIR) / exp_names_tasks_algos_hps[0][0][0] / f'iqm_vs_best{suffix}.{format}',
                bbox_inches='tight', pad_inches=pad_inches)
    plt.show()


def store_rliable_pvalues_vs_best(exp_names_tasks_algos_hps, seeds, tasks, task_bounds, fast=False,
                                  pretty_names=True, ):
    exp_names_per_task = []
    for exp_names_algos_hps in exp_names_tasks_algos_hps:
        exp_names_per_task.append([])
        for exp_names_hps in exp_names_algos_hps:
            best_idx = get_exp_with_best_iqm(exp_names_hps, seeds=seeds)
            exp_names_per_task[-1].append(exp_names_hps[best_idx])

    # dictionary mapping algorithms to scores of shape `(num_runs x num_games)`.
    score_dict = get_metric_many_exps(exp_names_per_task[0],
                                      seeds=seeds, pretty_names=pretty_names, add_extra_dim=True, metric_name='test')
    algo_names = list(score_dict.keys())
    for exp_names_cur in exp_names_per_task[1:]:
        score_dict_cur = get_metric_many_exps(exp_names_cur,
                                              seeds=seeds, pretty_names=pretty_names, add_extra_dim=True, metric_name='test')
        for k, v in score_dict_cur.items():
            score_dict[k] = np.concatenate([score_dict[k], v], axis=1)

    # assure that all seeds were loaded
    for k in score_dict:
        assert score_dict[k].shape[0] == len(seeds)

    # normalize to 0-1 per task
    for i, task in enumerate(tasks):
        task_min, task_max = task_bounds[task]
        for k in algo_names:
            score_dict[k][:, i] = (score_dict[k][:, i] - task_min) / (task_max - task_min)

    def paired_iqm_diff(a: np.ndarray, b: np.ndarray):
        return rlm.aggregate_iqm(a.ravel()) - rlm.aggregate_iqm(b.ravel())

    # bootstrap
    baseline = score_dict["IPBT"]
    reps = 500 if fast else 50000
    records = []
    for key, scores in score_dict.items():
        if key == "IPBT":
            continue
        obs = paired_iqm_diff(baseline, scores)
        stratified_bs = StratifiedBootstrap(baseline, scores)
        boot = stratified_bs.apply(paired_iqm_diff, reps)

        centred = boot - obs # for IQM, centering after bootstrapping is equivalent to centering before

        pval = ((np.abs(centred) >= abs(obs)).sum() + 1) / (reps + 1) # corrected by "+1" in accordance with Davison & Hinkley, 1997 ("Bootstrap Methods and Their Application")
        records.append({"algo": key, "pval": pval})

    df = pd.DataFrame(records)
    df.to_csv(Path(LOGS_DIR) / f'pvalues_ipbt_vs_best.csv', index=False)

def store_results_as_csv(exp_names_tasks_algos_hps, seeds, tasks, csv_path):
    records = []
    for i, (task, exp_names_task_algos_hps) in enumerate(zip(tasks, exp_names_tasks_algos_hps)):
        for exp_names_algo_hps in exp_names_task_algos_hps:
            for exp_name in exp_names_algo_hps:
                algo_name = algo_to_pretty_name[exp_name.split('_')[0]]
                config_number = exp_name.split('_')[-1]

                exp_path = Path(LOGS_DIR) / exp_name
                for seed in seeds:
                    seed_path = exp_path / str(seed)
                    best_test = _get_metric(seed_path, 'test')
                    records.append({
                        'task': task,
                        'algo': algo_name,
                        'config': config_number,
                        'seed': seed,
                        'test': best_test
                    })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

def compute_task_bounds(df_results):
    task_bounds = {}
    for task in list(df_results['task'].unique()):
        task_df = df_results[df_results['task'] == task]
        task_min = task_df['test'].min()
        task_max = task_df['test'].max()
        task_bounds[task] = (task_min, task_max)
    return task_bounds

def plot_rliable_iqm_vs_avg(exp_names_tasks_algos_hps, seeds, tasks, task_bounds,
                            fast=False, suffix='', format='png', pad_inches=0.0, show_title=True,
                            xlabel='Normalized performance',
                            figure_width=8,
                            row_height=0.8,
                            xlabel_x_coordinate=0.4,
                            xlabel_y_coordinate=-0.05,
                            ):
    # dictionary mapping algorithms to scores of shape `(num_runs x num_games)`.
    score_dict = {}
    for i, exp_names_task_algos_hps in enumerate(exp_names_tasks_algos_hps):
        for exp_names_algo_hps in exp_names_task_algos_hps:
            out = get_metric_many_exps(exp_names_algo_hps,
                                       seeds=seeds, pretty_names=True, add_extra_dim=True)
            assert len(list(out.items())) == 1
            algo_name, scores = list(out.items())[0]
            if type(scores) is list:
                scores = np.vstack(scores)

            if i == 0:
                score_dict[algo_name] = scores
            else:
                score_dict[algo_name] = np.concatenate([score_dict[algo_name], scores], axis=1)

    algo_names = list(score_dict.keys())

    # normalize to 0-1 per task
    for i_task, task in enumerate(tasks):
        task_min, task_max = task_bounds[task]
        for k in algo_names:
            score_dict[k][:, i_task] = (score_dict[k][:, i_task] - task_min) / (task_max - task_min)

    # #############IQM:
    aggregate_func = lambda x: np.array([rlm.aggregate_iqm(x)])
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        score_dict, aggregate_func, reps=500 if fast else 50000)
    fig, axes = plot_interval_estimates_toptobottom(
        aggregate_scores, aggregate_score_cis,
        metric_names=['IQM (vs avg)'],
        algorithms=algo_names, xlabel=xlabel,
        row_height=row_height,
        subfigure_width=figure_width,
        xlabel_x_coordinate=xlabel_x_coordinate,
        xlabel_y_coordinate=xlabel_y_coordinate,
        show_title=show_title
    )

    estimates_min = float('inf')
    estimates_max = float('-inf')
    for ci in aggregate_score_cis.values():
        lower, upper = ci[:, 0]
        estimates_min = min(lower, estimates_min)
        estimates_max = max(upper, estimates_max)

    plt.xlim(estimates_min * 0.95, estimates_max * 1.05)

    ############## PoI:
    # ipbt, others = algo_names[0], algo_names[1:]
    # algo_pairs = {}
    # for o in others:
    #     algo_pairs[f'{ipbt},{o}'] = (score_dict[ipbt], score_dict[o])
    #
    # average_probabilities, average_prob_cis = rly.get_interval_estimates(
    #     algo_pairs, metrics.probability_of_improvement, reps=200) # REMEMBER TO UPDATE TO 2000
    # plot_probability_of_improvement_toptobottom2(average_probabilities, average_prob_cis,
    #                                             figsize=(8, 4))

    plt.tight_layout()
    plt.savefig(Path(LOGS_DIR) / exp_names_tasks_algos_hps[0][0][0] / f'iqm_vs_avg{suffix}.{format}', bbox_inches='tight', pad_inches=pad_inches)
    plt.show()
    return estimates_min, estimates_max


def plot_rliable_iqm_ablations(exp_names_tasks_mods, seeds, tasks, task_bounds, ablation_names,
                               fast=False, suffix='', format='png', pad_inches=0.0,
                            figure_width=8,
                            row_height=0.8,
                            xlabel_x_coordinate=0.6,
                            xlabel_y_coordinate=0.0,
                               ):
    # dictionary mapping algorithms to scores of shape `(num_runs x num_games)`.
    score_dict_cur = get_metric_many_exps(exp_names_tasks_mods[0],
                                      seeds=seeds, pretty_names=False, add_extra_dim=True, metric_name='fitness')
    # rename
    score_dict = {}
    for i, (k, v) in enumerate(score_dict_cur.items()):
        score_dict[ablation_names[i]] = v

    algo_names = list(score_dict.keys())
    for exp_names_cur in exp_names_tasks_mods[1:]:
        score_dict_cur = get_metric_many_exps(exp_names_cur,
                                              seeds=seeds, pretty_names=False, add_extra_dim=True, metric_name='fitness')
        for i, (k, v) in enumerate(score_dict_cur.items()):
            score_dict[ablation_names[i]] = np.concatenate([score_dict[ablation_names[i]], v], axis=1)

    # assure that all seeds were loaded
    for k in score_dict:
        assert score_dict[k].shape[0] == len(seeds)

    # normalize to 0-1 per task
    for i, task in enumerate(tasks):
        task_min, task_max = task_bounds[task]
        for k in algo_names:
            score_dict[k][:, i] = (score_dict[k][:, i] - task_min) / (task_max - task_min)

    # #############IQM:
    aggregate_func = lambda x: np.array([rlm.aggregate_iqm(x)])
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        score_dict, aggregate_func, reps=500 if fast else 50000)  # REMEMBER TO UPDATE TO 50000
    colors = defaultdict(lambda: sns.color_palette("colorblind", 10)[-1])
    colors['IPBT'] = sns.color_palette("colorblind", 10)[0]
    colors['Step 1.0%'] = sns.color_palette("colorblind", 10)[0]
    colors['Shrink-perturb'] = sns.color_palette("colorblind", 10)[0]
    fig, axes = plot_interval_estimates_toptobottom(
        aggregate_scores, aggregate_score_cis,
        metric_names=['IQM'],
        algorithms=algo_names, xlabel='Normalized performance',
        row_height=row_height,
        subfigure_width=figure_width,
        xlabel_x_coordinate=xlabel_x_coordinate,
        xlabel_y_coordinate=xlabel_y_coordinate,
        colors=colors,
        show_title=False
    )


    plt.tight_layout()
    plt.savefig(Path(LOGS_DIR) / exp_names_tasks_mods[0][0] / f'iqm_ablations{suffix}.{format}',
                bbox_inches='tight', pad_inches=pad_inches)
    plt.show()
    
def plot_iqm_heatmap(exp_names_algos_hps, hp_values, hp_name, seeds,
                     n_last_rows_span_hps=0, show_y_ticklabels=True,
                     fitness_transform=lambda x: x,
                     suffix='', format='png', pad_inches=0.0,
                     fmt=".2f",
                     pretty_names=True,
                     pretty_names_override=None,
                     show_title=True,
                     figsize=(6, 4)
                     ):
    iqms_per_algo = {}
    for exp_names_hps in exp_names_algos_hps:
        iqms = get_iqms_many_exps(exp_names_hps, fitness_transform,
                                  seeds, metric_name='test')
        algo_name = exp_names_hps[0].split('_')[0]
        iqms_per_algo[algo_name] = iqms

    # Convert the dictionary to a 2D matrix (list of lists)
    algo_names = list(iqms_per_algo.keys())  # Algo names as rows
    heatmap_data = [iqms_per_algo[algo] for algo in algo_names]  # List of IQMs for each algo
    if pretty_names:
        if pretty_names_override:
            for name_raw, name_pretty in pretty_names_override.items():
                algo_to_pretty_name[name_raw] = name_pretty
        algo_names = [algo_to_pretty_name[algo] for algo in algo_names]

    annnotation = True
    if n_last_rows_span_hps > 0:
        banner_rows = heatmap_data[-n_last_rows_span_hps:]
        heatmap_data = heatmap_data[:-n_last_rows_span_hps]

        for row in banner_rows:
            banner_value = row[0]
            banner_row = np.full((1, len(hp_values)), banner_value)
            heatmap_data = np.vstack([heatmap_data, banner_row]) if len(heatmap_data) else banner_row

        annnotation = np.full(heatmap_data.shape, "", dtype=object)
        annnotation[:-n_last_rows_span_hps, :] = [[f"{v:{fmt}}" for v in row] for row in heatmap_data[:-n_last_rows_span_hps]]
        for i in range(n_last_rows_span_hps):
            annnotation[-(i + 1), 0] = f"{heatmap_data[-(i + 1), 0]:{fmt}}"
        fmt = ""

    # Generate the heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(heatmap_data, annot=annnotation, fmt=fmt, cmap="viridis",
                     xticklabels=hp_values, yticklabels=algo_names,
                     cbar=True, linewidths=0.01,
                     cbar_kws={'pad': 0.025})

    ax.collections[0].set_edgecolor('face')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    for tick in cbar.ax.get_yticklabels():     # use get_xticklabels() for a horizontal bar
        tick.set_rotation(90)
        tick.set_rotation_mode('anchor')       # <- key line
        tick.set_verticalalignment('center')   # ha/va both 'center' keeps it symmetric
        tick.set_horizontalalignment('center')
    cbar.locator = MaxNLocator(4)
    cbar.update_ticks()

    # Drawing red outlines for max values
    heatmap_data = np.array(heatmap_data)
    highlight_mask = (heatmap_data == np.array(heatmap_data).max(axis=1)[:, None])
    for y in range(heatmap_data.shape[0] - n_last_rows_span_hps):  # Iterate rows
        for x in range(heatmap_data.shape[1]):  # Iterate columns
            if highlight_mask[y, x]:  # If the cell is a max value
                offset = 0.03
                ax.add_patch(plt.Rectangle((x, y + offset), 1, 1 - offset * 2 - 0.02, # to also adjust for line width
                                           fill=False, edgecolor='red', lw=2))

    # Add labels and title
    task_name = exp_names_algos_hps[0][0].split('_')[1]
    if show_title:
        plt.title(f"{task_name} - IQM", fontsize=16)
    plt.xlabel(hp_name)
    if not show_y_ticklabels:
        ax.set_yticklabels([])
    else:
        plt.ylabel("Algorithms")

    plt.tight_layout()
    plt.savefig(Path(LOGS_DIR) / exp_names_algos_hps[0][0] / f'heatmap{suffix}.{format}',
                bbox_inches='tight', pad_inches=pad_inches)
    plt.show()