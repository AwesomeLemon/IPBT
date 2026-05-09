import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from omegaconf import OmegaConf

from algo.replay import ReplayAlgo


class DummySearchSpace:
    def sample(self, *_, **__):
        return [0.0, 0.0]

    def get_hp_names(self):
        return ["log10_lr", "momentum"]


class DummyTask:
    def __init__(self):
        self.t_eval = 1
        self.t_step = 1

    def prepare_initial_ckpt(self, solution):
        return {"step": 0, "solution": solution}

    def __call__(self, seed, solution, t, t_step, cpkt_loaded, tensorboard_dir, only_evaluate):
        if only_evaluate is not None:
            out = {}
            if "test" in only_evaluate:
                out["test"] = 0.123
            if "val" in only_evaluate:
                out["fitness"] = 0.456
            return out

        step = int(cpkt_loaded.get("step", 0)) + int(t_step)
        return {
            "fitness": float(step),
            "curve": [[t + t_step, float(step)]],
            "metrics": {"val": float(step), "test": None},
            "dict_to_save": {"step": step, "solution": solution},
        }


class SingleParamModel(torch.nn.Module):
    def __init__(self, init_weight):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([init_weight], dtype=torch.float32))


class DummyTaskWithModelState:
    def __init__(self):
        self.t_eval = 1
        self.t_step = 1

    def prepare_initial_ckpt(self, solution):
        return {
            "step": 0,
            "solution": solution,
            "model_state_dict": {"w": torch.tensor([2.0], dtype=torch.float32)},
        }

    def get_fresh_model(self, solution):
        return SingleParamModel(init_weight=10.0)

    def __call__(self, seed, solution, t, t_step, cpkt_loaded, tensorboard_dir, only_evaluate):
        if only_evaluate is not None:
            out = {}
            if "test" in only_evaluate:
                out["test"] = 0.123
            if "val" in only_evaluate:
                out["fitness"] = 0.456
            return out

        step = int(cpkt_loaded.get("step", 0)) + int(t_step)
        return {
            "fitness": float(step),
            "curve": [[t + t_step, float(step)]],
            "metrics": {"val": float(step), "test": None},
            "dict_to_save": {
                "step": step,
                "solution": solution,
                "model_state_dict": cpkt_loaded["model_state_dict"],
            },
        }


class _LocalTaskRunner:
    def options(self, **kwargs):
        return self

    def remote(self, task, seed, *args):
        out = task(seed, *args)
        out["datetime"] = datetime.now()
        out["seed"] = seed
        return out


class TestReplayAlgoSkeleton(unittest.TestCase):
    def setUp(self):
        self._task_runner_patch = patch.object(ReplayAlgo, "_task_fn_ray", _LocalTaskRunner())
        self._ray_get_patch = patch("algo.replay.ray.get", lambda obj: obj)
        self._task_runner_patch.start()
        self._ray_get_patch.start()

    def tearDown(self):
        self._ray_get_patch.stop()
        self._task_runner_patch.stop()

    @staticmethod
    def _write_ref_config(ref_run):
        (ref_run / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "search_space": {
                        "hyperparameters": [
                            {"name": "log10_lr"},
                            {"name": "momentum"},
                        ]
                    }
                }
            )
        )

    @staticmethod
    def _build_cfg(exp_dir, ckpt_dir, ref_run, continue_auto=False, extra_algo=None):
        algo_cfg = {
            "pop_size": 1,
            "t_step": 1,
            "t_max": 1,
            "delete_old_ckpts": False,
            "delete_all_ckpts_at_the_end": False,
            "reference_run_dir": str(ref_run),
            "save_helper_plot": False,
            "helper_plot_copy_to_tmp": False,
            "run_training": True,
        }
        if extra_algo:
            algo_cfg.update(extra_algo)
        return OmegaConf.create(
            {
                "path": {
                    "dir_exp": str(exp_dir),
                    "dir_ckpt": str(ckpt_dir),
                },
                "general": {
                    "exp_name": "replay_test",
                    "continue_auto": continue_auto,
                    "seed_base": 123,
                    "num_cpus": 1,
                    "num_gpus": 0,
                },
                "algo": algo_cfg,
            }
        )

    def test_exports_schedule_and_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_run = root / "ref_run"
            exp_dir = root / "exp"
            ckpt_dir = exp_dir / "checkpoints"
            ref_run.mkdir(parents=True)
            self._write_ref_config(ref_run)

            history_solution = [
                [[0, [-3.0, 0.7]], [10, [-3.0, 0.7]], [20, [-2.5, 0.8]]],
                [[0, [-4.0, 0.5]], [10, [-3.8, 0.55]], [20, [-3.7, 0.56]]],
            ]
            history_fitness = [
                [[0, 0.1], [10, 0.2], [20, 0.3]],
                [[0, 0.1], [10, 0.15], [20, 0.25]],
            ]
            (ref_run / "history_solution.yaml").write_text(yaml.safe_dump(history_solution))
            (ref_run / "history_fitness.yaml").write_text(yaml.safe_dump(history_fitness))
            (ref_run / "last_finished_tick.yaml").write_text(yaml.safe_dump(20))

            cfg = self._build_cfg(
                exp_dir,
                ckpt_dir,
                ref_run,
                continue_auto=False,
                extra_algo={"save_helper_plot": True, "helper_plot_name": "helper_plot.png", "run_training": False},
            )

            algo = ReplayAlgo(cfg=cfg, search_space=DummySearchSpace(), task=None)
            algo.run()

            schedule_path = exp_dir / "replay_schedule.yaml"
            plot_path = exp_dir / "helper_plot.png"
            self.assertTrue(schedule_path.exists())
            self.assertTrue(plot_path.exists())

            schedule = yaml.safe_load(schedule_path.read_text())
            self.assertEqual(schedule["selected_source"], "current_population")
            self.assertEqual(schedule["best_t"], 20)
            self.assertEqual(len(schedule["train_segments"]), 2)

    def test_training_loop_creates_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_run = root / "ref_run"
            exp_dir = root / "exp"
            ckpt_dir = exp_dir / "checkpoints"
            ref_run.mkdir(parents=True)
            self._write_ref_config(ref_run)

            history_solution = [
                [[0, [-3.0, 0.6]], [5, [-2.9, 0.62]], [9, [-2.8, 0.64]]]
            ]
            history_fitness = [
                [[0, 0.1], [5, 0.2], [9, 0.3]]
            ]
            (ref_run / "history_solution.yaml").write_text(yaml.safe_dump(history_solution))
            (ref_run / "history_fitness.yaml").write_text(yaml.safe_dump(history_fitness))
            (ref_run / "last_finished_tick.yaml").write_text(yaml.safe_dump(9))

            cfg = self._build_cfg(exp_dir, ckpt_dir, ref_run, continue_auto=False)

            algo = ReplayAlgo(cfg=cfg, search_space=DummySearchSpace(), task=DummyTask())
            algo.run()

            self.assertTrue((exp_dir / "replay_schedule.yaml").exists())
            self.assertTrue((exp_dir / "best_info.yaml").exists())
            self.assertTrue((exp_dir / "best_model.pt").exists())
            self.assertTrue((ckpt_dir / "pop_0_t9.pt").exists())

            history_solution_out = yaml.safe_load((exp_dir / "history_solution.yaml").read_text())
            self.assertEqual([entry[0] for entry in history_solution_out[0]], [5, 9])

    def test_continue_mode_resumes_from_last_finished_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_run = root / "ref_run"
            exp_dir = root / "exp"
            ckpt_dir = exp_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            ref_run.mkdir(parents=True)
            self._write_ref_config(ref_run)

            history_solution = [
                [[0, [-3.0, 0.6]], [5, [-2.9, 0.62]], [9, [-2.8, 0.64]]]
            ]
            history_fitness = [
                [[0, 0.1], [5, 0.2], [9, 0.3]]
            ]
            (ref_run / "history_solution.yaml").write_text(yaml.safe_dump(history_solution))
            (ref_run / "history_fitness.yaml").write_text(yaml.safe_dump(history_fitness))
            (ref_run / "last_finished_tick.yaml").write_text(yaml.safe_dump(9))

            # Simulate interrupted replay after first segment (0->5).
            torch.save({"step": 5, "solution": [-3.0, 0.6]}, ckpt_dir / "pop_0_t5.pt")
            (exp_dir / "history_solution.yaml").write_text(yaml.safe_dump([[[5, [-3.0, 0.6]]]]))
            (exp_dir / "history_fitness.yaml").write_text(yaml.safe_dump([[[5, 5.0]]]))
            (exp_dir / "history_population.yaml").write_text(yaml.safe_dump({1: {5: [[-3.0, 0.6]]}}))
            (exp_dir / "population.yaml").write_text(yaml.safe_dump([[-3.0, 0.6]]))
            (exp_dir / "last_finished_tick.yaml").write_text(yaml.safe_dump(5))
            (exp_dir / "cumulative_ticks.yaml").write_text(yaml.safe_dump(5))
            (exp_dir / "results.csv").write_text(
                "obj,tick,trial_id,date,relative_time,tick_cum,seed\n"
                "5.0,5,replay_0,2026-03-25_00-00-00,1.0,5,12300\n"
            )
            (exp_dir / "update_times.csv").write_text("t,time\n")
            (exp_dir / "tick_times.csv").write_text("t,time\n")
            (exp_dir / "train_and_eval_times.csv").write_text("t,time\n")

            cfg = self._build_cfg(exp_dir, ckpt_dir, ref_run, continue_auto=True)

            algo = ReplayAlgo(cfg=cfg, search_space=DummySearchSpace(), task=DummyTask())
            algo.run()

            # Should only replay pending second segment (5->9), preserving existing history.
            history_solution_out = yaml.safe_load((exp_dir / "history_solution.yaml").read_text())
            self.assertEqual([entry[0] for entry in history_solution_out[0]], [5, 9])
            self.assertTrue((ckpt_dir / "pop_0_t5.pt").exists())
            self.assertTrue((ckpt_dir / "pop_0_t9.pt").exists())

    def test_delete_all_ckpts_at_the_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_run = root / "ref_run"
            exp_dir = root / "exp"
            ckpt_dir = exp_dir / "checkpoints"
            ref_run.mkdir(parents=True)
            self._write_ref_config(ref_run)

            history_solution = [
                [[0, [-3.0, 0.6]], [5, [-2.9, 0.62]], [9, [-2.8, 0.64]]]
            ]
            history_fitness = [
                [[0, 0.1], [5, 0.2], [9, 0.3]]
            ]
            (ref_run / "history_solution.yaml").write_text(yaml.safe_dump(history_solution))
            (ref_run / "history_fitness.yaml").write_text(yaml.safe_dump(history_fitness))
            (ref_run / "last_finished_tick.yaml").write_text(yaml.safe_dump(9))

            cfg = self._build_cfg(
                exp_dir,
                ckpt_dir,
                ref_run,
                continue_auto=False,
                extra_algo={"delete_all_ckpts_at_the_end": True},
            )

            algo = ReplayAlgo(cfg=cfg, search_space=DummySearchSpace(), task=DummyTask())
            algo.run()

            self.assertTrue((exp_dir / "best_model.pt").exists())
            self.assertEqual(list(ckpt_dir.glob("*.pt")), [])

    def test_restart_shrink_perturb_applied_on_duplicate_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_run = root / "ref_run"
            exp_dir = root / "exp"
            ckpt_dir = exp_dir / "checkpoints"
            ref_run.mkdir(parents=True)
            self._write_ref_config(ref_run)

            # Duplicate tick at t=5 marks a restart boundary.
            history_solution = [
                [[0, [-3.2, 0.6]], [5, [-3.2, 0.6]], [5, [-3.0, 0.62]], [9, [-2.9, 0.64]]]
            ]
            history_fitness = [
                [[0, 0.1], [5, 0.2], [5, 0.2], [9, 0.3]]
            ]
            (ref_run / "history_solution.yaml").write_text(yaml.safe_dump(history_solution))
            (ref_run / "history_fitness.yaml").write_text(yaml.safe_dump(history_fitness))
            (ref_run / "last_finished_tick.yaml").write_text(yaml.safe_dump(9))
            (ref_run / "config.yaml").write_text(
                yaml.safe_dump({"algo": {"shrink_perturb_pair": [0.2, 0.1]}})
            )
            # include search-space metadata expected by schedule extraction
            self._write_ref_config(ref_run)
            cfg_cur = yaml.safe_load((ref_run / "config.yaml").read_text())
            cfg_cur.setdefault("algo", {})["shrink_perturb_pair"] = [0.2, 0.1]
            (ref_run / "config.yaml").write_text(yaml.safe_dump(cfg_cur))

            cfg = self._build_cfg(
                exp_dir,
                ckpt_dir,
                ref_run,
                continue_auto=False,
                extra_algo={"enable_restart_shrink_perturb": True},
            )

            algo = ReplayAlgo(cfg=cfg, search_space=DummySearchSpace(), task=DummyTaskWithModelState())
            algo.run()

            # First segment keeps initial model state (2.0), second segment at restart t=5
            # applies shrink-perturb with fresh value 10.0: 0.2*2.0 + 0.1*10.0 = 1.4.
            final_ckpt = torch.load(ckpt_dir / "pop_0_t9.pt")
            final_weight = float(final_ckpt["model_state_dict"]["w"].item())
            self.assertAlmostEqual(final_weight, 1.4, places=6)

    def test_restart_shrink_perturb_pair_loaded_from_reference_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_run = root / "ref_run"
            exp_dir = root / "exp"
            ckpt_dir = exp_dir / "checkpoints"
            ref_run.mkdir(parents=True)

            history_solution = [
                [[0, [-3.2, 0.6]], [5, [-3.2, 0.6]], [5, [-3.0, 0.62]], [9, [-2.9, 0.64]]]
            ]
            history_fitness = [
                [[0, 0.1], [5, 0.2], [5, 0.2], [9, 0.3]]
            ]
            (ref_run / "history_solution.yaml").write_text(yaml.safe_dump(history_solution))
            (ref_run / "history_fitness.yaml").write_text(yaml.safe_dump(history_fitness))
            (ref_run / "last_finished_tick.yaml").write_text(yaml.safe_dump(9))
            self._write_ref_config(ref_run)
            (ref_run / "config.yaml").write_text(
                yaml.safe_dump({"algo": {"shrink_perturb_pair": [0.3, 0.4]}})
            )
            cfg_cur = yaml.safe_load((ref_run / "config.yaml").read_text())
            cfg_cur.setdefault("search_space", {}).setdefault("hyperparameters", [
                {"name": "log10_lr"},
                {"name": "momentum"},
            ])
            (ref_run / "config.yaml").write_text(yaml.safe_dump(cfg_cur))

            cfg = self._build_cfg(
                exp_dir,
                ckpt_dir,
                ref_run,
                continue_auto=False,
                extra_algo={"enable_restart_shrink_perturb": True},
            )

            algo = ReplayAlgo(cfg=cfg, search_space=DummySearchSpace(), task=DummyTaskWithModelState())
            algo.run()

            # Loaded pair [0.3, 0.4] from reference config => 0.3*2.0 + 0.4*10.0 = 4.6
            final_ckpt = torch.load(ckpt_dir / "pop_0_t9.pt")
            final_weight = float(final_ckpt["model_state_dict"]["w"].item())
            self.assertAlmostEqual(final_weight, 4.6, places=6)


if __name__ == "__main__":
    unittest.main()
