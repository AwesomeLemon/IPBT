from plot.plot_fns import *
import utils

def plot_ipbt_ALL_vs_avg_and_best(
        tasks=None,
        nonpbt_baselines=None,
        plot_best=True,
        store_pvalues=True,
        fast=False,
        suffix='',
        format='png',
        iqm_plot_kwargs=None,
):
    if tasks is None:
        tasks = [
            "Cifar10", "Humanoid", "Cifar100", "Hopper", "TinyImagenet", "FashionMnist", "Walker", "Pusher"
        ]
    if nonpbt_baselines is None:
        nonpbt_baselines = ["rs", "asha", "smac"]
    if iqm_plot_kwargs is None:
        iqm_plot_kwargs = {}

    exp_names_tasks_algos_hps = []
    for task in tasks:
        exp_names_cur = []
        exp_names_cur.append([f'ipbt6_{task}_0001'])
        exp_names_cur.extend(
            [
                [f'{algo}_{task}_000{i}' for i in [0, 1, 2, 3]]
                for algo in ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt']
            ]
        )
        exp_names_cur.extend(
            [
                [f'{algo_base}_{task}CosineRestart_0001']
                for algo_base in nonpbt_baselines
            ]
        )
        exp_names_tasks_algos_hps.append(exp_names_cur)

    seeds = list(range(8))
    csv_path = Path(LOGS_DIR) / 'results.csv'
    if not csv_path.exists():
        store_results_as_csv(exp_names_tasks_algos_hps, seeds, tasks, csv_path)
    df = pd.read_csv(csv_path)
    df_tasks = set(df['task'])
    for t in tasks:
        if t not in df_tasks:
            raise ValueError(f"Task {t} not found in results.csv")
    task_bounds = compute_task_bounds(df)

    estimates_min, estimates_max = plot_rliable_iqm_vs_avg(exp_names_tasks_algos_hps, seeds,
                                                           tasks, task_bounds,
                                                           fast=fast, suffix=suffix, format=format,
                                                           **iqm_plot_kwargs)
    if plot_best:
        plot_rliable_iqm_vs_best(exp_names_tasks_algos_hps, seeds, tasks, task_bounds,
                                 fast=fast, suffix=suffix, format=format,
                                 estimates_min=estimates_min, estimates_max=estimates_max,
                                 **iqm_plot_kwargs
                                 )
    if store_pvalues:
        store_rliable_pvalues_vs_best(exp_names_tasks_algos_hps, seeds, tasks, task_bounds,
                                      fast=fast,
                                      )

def plot_ipbt_all_ablations(
    fast = False,
    format='png',
    figure_width=8,
    row_height=0.8,
    xlabel_x_coordinate=0.6,
    xlabel_y_coordinate=0.0,
):
    tasks = ['Cifar10', 'Humanoid', 'Cifar100', 'Hopper']
    exp_names_tasks_mods = [
        [
            f'ipbt6_{task}_{i}'
            for i in ['0000',
                      '0016',
                      '0008', '0009',
                      '0006', '0007',
                      '0014',
                      '0017',
                      '0013', '0012',
                      '0010', '0011',
                      '0005', '0004',
                      ]
        ]
        for task in tasks
    ]
    names = [
        'IPBT',
        'Stagnation detection of BG-PBT',
        'Shrink-perturb (1.0, 0.0)',
        'Shrink-perturb (0.0, 1.0)',
        'Shrink-perturb (0.4, 0.1)',
        'Shrink-perturb (0.1, 0.1)',
        'Prob. of random weight reinit. 0.0',
        'Meta BO of BG-PBT',
        'Prob. of random HPs reinit. 1.0',
        'Prob. of random HPs reinit. 0.0',
        'Step size: linear increase',
        'Step size: constant',
        'Population multiple 1',
        'Population multiple 3',
    ]
    seeds = list(range(8))

    csv_path = Path(LOGS_DIR) / 'results.csv'
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    df_tasks = set(df['task'])
    for t in tasks:
        if t not in df_tasks:
            raise ValueError(f"Task {t} not found in results.csv")
    task_bounds = compute_task_bounds(df)

    plot_rliable_iqm_ablations(exp_names_tasks_mods, seeds, tasks, task_bounds,
                               names, fast=fast, suffix='_all_ablations',
                               format=format,
                               figure_width=figure_width,
                               row_height=row_height,
                               xlabel_x_coordinate=xlabel_x_coordinate,
                               xlabel_y_coordinate=xlabel_y_coordinate
                               )

def plot_ipbt_extra_ablations(
    fast = False,
    format='png',
    figure_width=8,
    row_height=0.8,
    xlabel_x_coordinate=0.6,
    xlabel_y_coordinate=0.0,
):
    tasks = ['Cifar10', 'Humanoid', 'Cifar100', 'Hopper']
    exp_names_tasks_mods = [
        [
            f'ipbt6_{task}_{i}'
            for i in ['0000',
                      '0022',
                      '0019',
                      ]
        ]
        for task in tasks
    ]
    names = [
        'Step 1.0%',
        'Step 0.5%',
        'Step 3.0%',
    ]
    seeds = list(range(8))

    csv_path = Path(LOGS_DIR) / 'results.csv'
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    df_tasks = set(df['task'])
    for t in tasks:
        if t not in df_tasks:
            raise ValueError(f"Task {t} not found in results.csv")
    task_bounds = compute_task_bounds(df)

    plot_rliable_iqm_ablations(exp_names_tasks_mods, seeds, tasks, task_bounds,
                               names, fast=fast, suffix='_extra_ablations',
                               format=format,
                               figure_width=figure_width,
                               row_height=row_height,
                               xlabel_x_coordinate=xlabel_x_coordinate,
                               xlabel_y_coordinate=xlabel_y_coordinate
                               )

def plot_ipbt_distillation_ablation(
    fast = False,
    format='png',
    figure_width=8,
    row_height=0.8,
    xlabel_x_coordinate=0.6,
    xlabel_y_coordinate=0.0,
):
    tasks = ['Humanoid', 'Hopper']
    exp_names_tasks_mods = [
        [
            f'ipbt6_{task}_{i}'
            for i in ['0000',
                      '0018',
                      ]
        ]
        for task in tasks
    ]
    names = [
        'Shrink-perturb',
        'Distillation',
    ]
    seeds = list(range(8))

    csv_path = Path(LOGS_DIR) / 'results.csv'
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    df_tasks = set(df['task'])
    for t in tasks:
        if t not in df_tasks:
            raise ValueError(f"Task {t} not found in results.csv")
    task_bounds = compute_task_bounds(df)

    plot_rliable_iqm_ablations(exp_names_tasks_mods, seeds, tasks, task_bounds,
                               names, fast=fast, suffix='_distillation_ablation',
                               format=format,
                               figure_width=figure_width,
                               row_height=row_height,
                               xlabel_x_coordinate=xlabel_x_coordinate,
                               xlabel_y_coordinate=xlabel_y_coordinate
                               )

def plot_heatmap_over_steps(
        task,
        nonpbt_baselines=None,
        is_rl = True,
        format='png',
        show_y_ticklabels=True,
        show_title=True,
        figsize=(6, 4)
):
    if nonpbt_baselines is None:
        nonpbt_baselines = ["rs", "asha", "smac"]
    exp_names_algos_hps = [
        [f'{algo}_{task}_000{i}' for i in [3, 2, 1, 0]]
        for algo in ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt']
    ] + [
        [f'ipbt6_{task}_0001']
    ] + [
        [f'{algo_base}_{task}CosineRestart_0001']
        for algo_base in nonpbt_baselines
    ]
    hp_values = ['1%', '3.3%', '10%', '33.3%']
    hp_name = 'Step size'
    seeds = list(range(8))

    if is_rl:
        float_fmt = '.0f'
        fitness_transform = lambda x: x
    else:
        float_fmt = '.2f'
        fitness_transform=lambda x: 100 * x

    plot_iqm_heatmap(exp_names_algos_hps, hp_values, hp_name,
                     seeds, n_last_rows_span_hps=1 + len(nonpbt_baselines),
                     show_y_ticklabels=show_y_ticklabels,
                     fitness_transform=fitness_transform,
                     fmt=float_fmt,
                     format=format,
                     pretty_names_override={'rs': 'RS'},
                     show_title=show_title,
                     figsize=figsize,
                     suffix='_' + task
                     )

def print_significance_with_holm_correction(target_p=0.05):
    df = pd.read_csv(Path(LOGS_DIR) / "pvalues_ipbt_vs_best.csv")
    algos = list(df["algo"])
    pvals = list(df["pval"])
    pvals_order = np.argsort(pvals)
    m = len(algos)
    out = []
    max_p_adj = 0.0
    for i, i_sorted in enumerate(pvals_order):
        is_sig = False
        if pvals[i_sorted] <= target_p / (m - i):
            is_sig = True
        p_adjusted = max(max_p_adj, min(1, pvals[i_sorted] * (m - i)))
        print(f'{p_adjusted=}')
        max_p_adj = max(max_p_adj, p_adjusted)
        out.append({
            "algo": algos[i_sorted],
            "is_sig": is_sig,
            "pval": pvals[i_sorted],
            "pval_adj": p_adjusted,
        })
        print(out[-1])
    out = pd.DataFrame(out)
    out.to_csv(Path(LOGS_DIR) / f"pvalues_ipbt_vs_best_holm_corrected_{target_p}.csv", index=False)

if __name__ == '__main__':
    utils.set_plot_style()

    # ---- vs untuned PBTs
    plot_ipbt_ALL_vs_avg_and_best(
        nonpbt_baselines=[],
        plot_best=False,
        store_pvalues=False,
        fast=False,
        suffix='_untuned_PBTs',
        format='pdf',
        iqm_plot_kwargs=dict(
            xlabel='Normalized performance',
            show_title=False,
            xlabel_x_coordinate=0.6,
            xlabel_y_coordinate=-0.02,
            row_height=0.7
        )
    )

    # ---- vs tuned PBTs & baselines
    plot_ipbt_ALL_vs_avg_and_best(
        plot_best=True,
        store_pvalues=True,
        fast=False,
        suffix='_tuned_PBTs',
        format='pdf',
        iqm_plot_kwargs=dict(
            xlabel='Normalized performance',
            show_title=False,
            xlabel_x_coordinate=0.6,
            xlabel_y_coordinate=-0.02,
            row_height=0.7,
        )
    )

    # ---- print p-values
    print_significance_with_holm_correction()

    # ---- heatmaps: Humanoid, Hopper
    plot_heatmap_over_steps(
        task='Humanoid',
        is_rl=True,
        format='pdf',
        show_y_ticklabels=True,
        show_title=False,
        figsize=(4.4, 4.5)
    )
    plot_heatmap_over_steps(
        task='Hopper',
        is_rl=True,
        format='pdf',
        show_y_ticklabels=False,
        show_title=False,
        figsize=(3.4, 4.5)
    )

    # ---- ablations
    plot_ipbt_all_ablations(
        fast = False,
        format='pdf',
        figure_width=9,
        row_height=0.6,
        xlabel_x_coordinate=0.8,
        xlabel_y_coordinate=-0.01,
    )

    # ----- extra ablations
    plot_ipbt_extra_ablations(
        fast = False,
        format='pdf',
        figure_width=9,
        row_height=0.6,
        xlabel_x_coordinate=0.6,
        xlabel_y_coordinate=-0.01,
    )

    plot_ipbt_distillation_ablation(
        fast = False,
        format='pdf',
        figure_width=9,
        row_height=1.0,
        xlabel_x_coordinate=0.6,
        xlabel_y_coordinate=-0.02,
    )