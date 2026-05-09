from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plot import plot_fns


def collect_seed_dirs(exp_dir: Path) -> list[Path]:
    if (exp_dir / "best_info.yaml").exists():
        return [exp_dir]

    seed_dirs = []
    for child in sorted(exp_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir() and (child / "best_info.yaml").exists():
            seed_dirs.append(child)
    return seed_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print per-seed val/test and IQM aggregates from best_info.yaml files."
    )
    parser.add_argument("exp_dir", type=Path, help="Experiment dir or parent dir with seed subdirs.")
    args = parser.parse_args()

    exp_dir = args.exp_dir.expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)

    seed_dirs = collect_seed_dirs(exp_dir)
    if len(seed_dirs) == 0:
        raise FileNotFoundError(
            f"No best_info.yaml found in {exp_dir} or its immediate subdirectories."
        )

    scale = 1 if "Humanoid" in str(exp_dir) else 100

    # Single-seed mode.
    if len(seed_dirs) == 1 and seed_dirs[0] == exp_dir:
        val = float(plot_fns._get_metric(exp_dir, "fitness"))
        test = float(plot_fns._get_metric(exp_dir, "test"))
        print(f"Experiment: {exp_dir}")
        print("Seeds with metrics: 1")
        print("")
        print("seed\tval\ttest")
        print(f"{exp_dir.name}\t{val * scale:.2f}\t{test * scale:.2f}")
        print("")
        print(f"IQM val : {val * scale:.2f}")
        print(f"IQM test: {test * scale:.2f}")
        print(f"IQR val : ({val * scale:.2f}, {val * scale:.2f})")
        print(f"IQR test: ({test * scale:.2f}, {test * scale:.2f})")
        return

    seed_ids = []
    for seed_dir in seed_dirs:
        try:
            seed_ids.append(int(seed_dir.name))
        except ValueError as e:
            raise ValueError(
                "Seed directory names must be integers to reuse plot functions. "
                f"Got: {seed_dir.name}"
            ) from e
    seed_ids = sorted(seed_ids)

    exp_name = exp_dir.name
    logs_root = exp_dir.parent

    old_logs_dir = plot_fns.LOGS_DIR
    plot_fns.LOGS_DIR = str(logs_root)
    try:
        vals_map = plot_fns.get_metric_many_exps([exp_name], seeds=seed_ids, metric_name="fitness")
        tests_map = plot_fns.get_metric_many_exps([exp_name], seeds=seed_ids, metric_name="test")
        iqm_val = float(plot_fns.get_iqms_many_exps([exp_name], seeds=seed_ids, metric_name="fitness")[0])
        iqm_test = float(plot_fns.get_iqms_many_exps([exp_name], seeds=seed_ids, metric_name="test")[0])
    finally:
        plot_fns.LOGS_DIR = old_logs_dir

    vals = vals_map[exp_name]
    tests = tests_map[exp_name]
    q1_val = float(np.percentile(vals, 25))
    q3_val = float(np.percentile(vals, 75))
    q1_test = float(np.percentile(tests, 25))
    q3_test = float(np.percentile(tests, 75))

    print(f"Experiment: {exp_dir}")
    print(f"Seeds with metrics: {len(seed_ids)}")
    print("")
    print("seed\tval\ttest")
    for seed, val, test in zip(seed_ids, vals, tests):
        print(f"{seed}\t{float(val) * scale:.2f}\t{float(test) * scale:.2f}")

    print("")
    print(f"IQM val : {iqm_val * scale:.2f}")
    print(f"IQM test: {iqm_test * scale:.2f}")
    print(f"IQR val : ({q1_val * scale:.2f}, {q3_val * scale:.2f})")
    print(f"IQR test: ({q1_test * scale:.2f}, {q3_test * scale:.2f})")


if __name__ == "__main__":
    main()
