import itertools

import yaml
from matplotlib import pyplot as plt

import utils


def plot_pop_history(exp_path, exp_name):
    path = exp_path / 'history_population.yaml'
    with open(path, 'r') as f:
        pop_history = yaml.safe_load(f)

    path = exp_path / 'best_info.yaml'
    with open(path, 'r') as f:
        best_info = yaml.safe_load(f)
        best_solution_history = best_info['solution_history']

    ts = list(pop_history[1].keys())
    it_cycle = itertools.cycle(iter(plt.rcParams['axes.prop_cycle']))

    hyperparameters_ = yaml.safe_load(open(exp_path / 'config.yaml', 'r'))['search_space']['hyperparameters']
    var_names = list(sorted([h['name'] for h in hyperparameters_]))
    _plot_single_solution_history(best_solution_history, var_names, next(it_cycle)['color'])

    for pop_id in sorted(pop_history.keys()):
        _plot_single_pop_history(pop_history[pop_id], next(it_cycle)['color'])

    plt.yscale('log')
    plt.tight_layout()
    plt.title(f'{exp_name}: Population history + the best')
    plt.savefig(exp_path / f'history_population.png', bbox_inches='tight')
    plt.show()


def _plot_single_solution_history(solution_history, var_names, c, linewidth=6):
    num_vars = len(solution_history[0][1])

    # Plot each variable
    for var_index in range(num_vars):
        # Select subplot for current variable
        if num_vars > 1:
            plt.subplot(num_vars, 1, var_index + 1)

        prev = None
        prev_t = 0
        for t, s in solution_history:
            if var_names[var_index].startswith('log'):
                s[var_index] = utils.convert_from_logarithmic(var_names[var_index], s[var_index])
                plt.yscale('log')
            plt.plot([prev_t, t], [s[var_index], s[var_index]],
                     color=c, linewidth=linewidth, marker="none")
            # set y axis name to var name:
            name_to_show = var_names[var_index]
            if name_to_show.startswith('log'):
                name_to_show = name_to_show[name_to_show.index('_') + 1:]
            plt.ylabel(name_to_show)
            if prev is not None:
                # vertical line
                if prev != s[var_index]:
                    plt.plot([prev_t, prev_t], [prev, s[var_index]],
                             color=c, linewidth=linewidth, marker="none")
            prev = s[var_index]
            prev_t = t


def _plot_single_pop_history(one_pop_history, c):
    times_and_solutions = list(one_pop_history.items())
    prev_t = 0
    for i in range(len(times_and_solutions)):
        for sol in times_and_solutions[i][1]:
            # horizontal line for each solution from t to next t (which can be variable)
            plt.plot([prev_t, times_and_solutions[i][0]], [sol[0], sol[0]], color=c)
        prev_t = times_and_solutions[i][0]