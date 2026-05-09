from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
import yaml

from algo.base import BaseAlgo
from algo.ipbt_utils import shrink_perturb_state_via_named_params
from algo.plot_utils import plot_pop_history
from replay.schedule import compute_best_schedule
from utils import convert_from_logarithmic, save_yaml


class ReplayAlgo(BaseAlgo):
    def _resolve_restart_shrink_perturb_pair(self) -> tuple[float, float]:
        ref_cfg_path = self.reference_run_dir / "config.yaml"
        if not ref_cfg_path.exists():
            raise FileNotFoundError(
                "Replay restart shrink-perturb requires reference config file: "
                f"{ref_cfg_path}"
            )

        with open(ref_cfg_path, "r") as f:
            ref_cfg = yaml.safe_load(f) or {}
        ref_pair = (ref_cfg.get("algo") or {}).get("shrink_perturb_pair")
        if ref_pair is None:
            raise KeyError(
                "Replay restart shrink-perturb requires 'algo.shrink_perturb_pair' "
                f"in reference config: {ref_cfg_path}"
            )
        if not isinstance(ref_pair, (list, tuple)) or len(ref_pair) != 2:
            raise ValueError(
                "Invalid algo.shrink_perturb_pair in reference config: "
                f"{ref_cfg_path} -> {ref_pair}"
            )

        resolved = tuple(float(x) for x in ref_pair)
        print(f"Loaded restart shrink-perturb pair from reference config: {resolved}")
        return resolved

    def __init__(self, cfg, search_space, task, **__):
        # Replay always runs a single lineage.
        self.pop_size = 1
        super().__init__(cfg, search_space, task)

        self.reference_run_dir = Path(cfg.algo.reference_run_dir)
        self.schedule_info = compute_best_schedule(self.reference_run_dir)
        self.train_segments = self.schedule_info["train_segments"]
        self.schedule_hp_names = self.schedule_info.get("hp_names")
        self.restart_ticks = set(int(t) for t in self.schedule_info.get("duplicate_ticks", []))
        self.enable_restart_shrink_perturb = bool(cfg.algo.get("enable_restart_shrink_perturb", False))
        self.restart_shrink_perturb_pair = None
        if self.enable_restart_shrink_perturb and len(self.restart_ticks) > 0:
            self.restart_shrink_perturb_pair = self._resolve_restart_shrink_perturb_pair()
        self.fraction_random_weights_on_restart = float(cfg.algo.get("fraction_random_weights_on_restart", 0.0))

        if len(self.train_segments) == 0:
            raise ValueError("No train segments extracted from reference run")
        if self.restart_shrink_perturb_pair is not None and len(self.restart_shrink_perturb_pair) != 2:
            raise ValueError(
                "Reference config algo.shrink_perturb_pair must have exactly 2 values: "
                "[shrink_coeff, perturb_coeff]"
            )

        # Keep algo state aligned with extracted schedule metadata.
        self.pop[0] = self.train_segments[0]["solution"]
        self.trial_ids[0] = "replay_0"
        self.t_max = int(self.schedule_info["best_t"])

        configured_hp_names = self.search_space.get_hp_names()
        expected_dims = len(configured_hp_names)
        actual_dims = len(self.train_segments[0]["solution"])
        if actual_dims != expected_dims:
            raise ValueError(
                "Replay schedule/search-space mismatch: "
                f"schedule has {actual_dims} HPs but configured search_space has {expected_dims}. "
                "Use a replay config with matching search_space."
            )
        if self.schedule_hp_names is not None:
            if list(self.schedule_hp_names) != list(configured_hp_names):
                raise ValueError(
                    "Replay schedule/search-space HP-name mismatch. "
                    f"schedule hp_names={self.schedule_hp_names}, "
                    f"configured hp_names={configured_hp_names}. "
                    "Use the exact search_space from the reference run."
                )

    def _build_helper_plot(self, out_path: Path) -> None:
        var_names = self.search_space.get_hp_names()
        solution_history = self.schedule_info["solution_history"]

        n_vars = len(var_names)
        fig, axes = plt.subplots(n_vars, 1, figsize=(10, 2.2 * n_vars), sharex=True)
        if n_vars == 1:
            axes = [axes]

        for i_var, hp_name in enumerate(var_names):
            ax = axes[i_var]
            prev = None
            prev_t = 0
            for t, solution in solution_history:
                val = solution[i_var]
                if hp_name.startswith("log"):
                    val = convert_from_logarithmic(hp_name, val)
                    ax.set_yscale("log")

                ax.plot([prev_t, t], [val, val], linewidth=3)
                if prev is not None and prev != val:
                    ax.plot([prev_t, prev_t], [prev, val], linewidth=3)
                prev = val
                prev_t = t

            hp_label = hp_name[hp_name.index("_") + 1 :] if hp_name.startswith("log") else hp_name
            ax.set_ylabel(hp_label)
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("t")
        fig.suptitle("Best schedule extracted by compute_best_schedule")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def run(self):
        schedule_out = {
            "reference_run_dir": str(self.reference_run_dir),
            "selected_source": self.schedule_info["selected_source"],
            "best_fitness": self.schedule_info["best_fitness"],
            "best_t": self.schedule_info["best_t"],
            "solution_id": self.schedule_info["solution_id"],
            "hp_names": self.schedule_info.get("hp_names"),
            "duplicate_ticks": self.schedule_info["duplicate_ticks"],
            "train_segments": self.train_segments,
        }
        save_yaml(schedule_out, self.exp_dir / "replay_schedule.yaml")

        if self.cfg.algo.get("save_helper_plot", True):
            plot_name = self.cfg.algo.get("helper_plot_name", "helper_extracted_best_schedule.png")
            plot_path = self.exp_dir / plot_name
            self._build_helper_plot(plot_path)

            if self.cfg.algo.get("helper_plot_copy_to_tmp", False):
                tmp_path = Path("/tmp") / plot_name
                tmp_path.write_bytes(plot_path.read_bytes())

        if not self.cfg.algo.get("run_training", True):
            return

        if self.cfg.general.continue_auto:
            resume_t = int(self.t_cur)
            pending = [(i, s) for i, s in enumerate(self.train_segments) if int(s["t_end"]) > resume_t]
            if len(pending) > 0 and int(pending[0][1]["t_start"]) != resume_t:
                raise ValueError(
                    "Cannot resume replay: last_finished_tick is not aligned with schedule boundaries. "
                    f"last_finished_tick={resume_t}, next_segment_start={pending[0][1]['t_start']}"
                )
            resume_ckpt = self.cpkt_dir / f"pop_0_t{resume_t}.pt"
            if len(pending) > 0 and not resume_ckpt.exists():
                raise FileNotFoundError(
                    f"Cannot resume replay: missing checkpoint for last finished tick: {resume_ckpt}"
                )
        else:
            first_t = int(self.train_segments[0]["t_start"])
            first_solution = self.train_segments[0]["solution"]
            first_ckpt_path = self.cpkt_dir / f"pop_0_t{first_t}.pt"
            self.prepare_initial_ckpt(first_ckpt_path, first_solution)
            pending = list(enumerate(self.train_segments))

        tb_dir = self.exp_dir / "tb" / "pop_0"
        tb_dir.mkdir(parents=True, exist_ok=True)

        for i_segment, segment in pending:
            t_start = int(segment["t_start"])
            t_end = int(segment["t_end"])
            t_step = int(segment["t_step"])
            solution = segment["solution"]

            self.t_cur = t_start
            self.t_step = t_step
            self.pop[0] = solution

            # Replay should mirror reference semantics where classification used t_eval == t_step.
            if hasattr(self.task, "t_eval"):
                self.task.t_eval = t_step
            if hasattr(self.task, "t_step"):
                self.task.t_step = t_step

            ckpt_path = self.cpkt_dir / f"pop_0_t{t_start}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint for replay segment start: {ckpt_path}")
            ckpt_loaded = torch.load(ckpt_path)

            if self.enable_restart_shrink_perturb and t_start in self.restart_ticks:
                if not hasattr(self.task, "get_fresh_model"):
                    raise AttributeError(
                        "Replay restart shrink-perturb requires task.get_fresh_model(solution), "
                        f"but task {type(self.task).__name__} does not implement it."
                    )
                if "model_state_dict" not in ckpt_loaded:
                    raise KeyError(
                        "Replay restart shrink-perturb requires checkpoint key 'model_state_dict' "
                        f"at restart tick t={t_start}, got keys: {sorted(ckpt_loaded.keys())}"
                    )

                shpe_cur = self.restart_shrink_perturb_pair
                if self.fraction_random_weights_on_restart > 0:
                    random_weights_indices = np.random.choice(
                        len(self.pop),
                        round(self.fraction_random_weights_on_restart * len(self.pop)),
                        replace=False,
                    )
                    if 0 in random_weights_indices:
                        shpe_cur = (0.0, 1.0)

                print(
                    "Applying restart shrink-perturb at "
                    f"t={t_start} with pair={tuple(float(x) for x in shpe_cur)}"
                )
                fresh_model = self.task.get_fresh_model(solution)
                ckpt_loaded["model_state_dict"] = shrink_perturb_state_via_named_params(
                    ckpt_loaded["model_state_dict"], shpe_cur, fresh_model
                )

            seed = self.seed_base * 100 + i_segment
            self.result_records = []
            self.extend_population_history()

            st_train = time.time()
            future = self._task_fn_ray.options(**self.ray_options).remote(
                self.task, seed, solution, t_start, t_step, ckpt_loaded, tb_dir, None
            )
            result = ray.get(future)
            train_eval_time = time.time() - st_train
            self.train_and_eval_times.loc[len(self.train_and_eval_times)] = {
                "t": t_start,
                "time": train_eval_time,
            }

            st_tick = time.time()
            ckpt_out_path = self.cpkt_dir / f"pop_0_t{t_end}.pt"
            torch.save(result["dict_to_save"], ckpt_out_path)

            self.extend_result_records(result, self.trial_ids[0], t_end, t_step)
            self.extend_fitness_and_solution_history([result["fitness"]])
            self.save_state()

            tick_time = time.time() - st_tick
            self.tick_times.loc[len(self.tick_times)] = {"t": t_start, "time": tick_time}

            self.t_cur = t_end
            self.save_fitnesses_at_tick()
            save_yaml(self.t_cur, self.exp_dir / "last_finished_tick.yaml")
            save_yaml(self.t_cum, self.exp_dir / "cumulative_ticks.yaml")
            self.tick_times.to_csv(self.exp_dir / "tick_times.csv", index=False)

            if self.delete_old_ckpts:
                prev_ckpt = self.cpkt_dir / f"pop_0_t{t_start}.pt"
                if prev_ckpt.exists():
                    prev_ckpt.unlink()

        try:
            self.save_best()
        except Exception as e:
            print(f"Error saving best in replay: {e}")
        if (self.exp_dir / "config.yaml").exists():
            try:
                plot_pop_history(self.exp_dir, self.exp_name)
            except Exception as e:
                print(f"Error plotting population history in replay: {e}")

        if self.delete_all_ckpts_at_the_end:
            for p in self.cpkt_dir.glob("*.pt"):
                p.unlink()

    def tick(self):
        raise NotImplementedError("ReplayAlgo does not use tick(); run() is schedule-export only")

    def _schedule_all_populations(self):
        raise NotImplementedError

    def _exploit_and_explore(self, fitnesses):
        return None
