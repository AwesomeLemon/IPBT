import tempfile
import unittest
from pathlib import Path

import yaml

from replay.schedule import compute_best_schedule


class TestBestScheduleSynthetic(unittest.TestCase):
    def test_extracts_segments_and_duplicate_ticks(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "search_space": {
                            "hyperparameters": [
                                {"name": "hp1"},
                                {"name": "hp2"},
                            ]
                        }
                    }
                )
            )

            # Duplicate tick at 10 simulates restart-boundary marker.
            history_solution = [
                [[0, [1.0, 2.0]], [10, [1.1, 2.1]], [10, [1.2, 2.2]], [25, [1.3, 2.3]]],
                [[0, [3.0, 4.0]], [10, [3.1, 4.1]], [25, [3.2, 4.2]], [40, [3.3, 4.3]]],
            ]
            history_fitness = [
                [[0, 0.1], [10, 0.2], [10, 0.21], [25, 0.5]],
                [[0, 0.1], [10, 0.15], [25, 0.35], [40, 0.4]],
            ]
            (run_dir / "history_solution.yaml").write_text(yaml.safe_dump(history_solution))
            (run_dir / "history_fitness.yaml").write_text(yaml.safe_dump(history_fitness))
            (run_dir / "last_finished_tick.yaml").write_text(yaml.safe_dump(25))

            schedule = compute_best_schedule(run_dir)

            self.assertEqual(schedule["selected_source"], "current_population")
            self.assertEqual(schedule["best_t"], 25)
            self.assertAlmostEqual(schedule["best_fitness"], 0.5)
            self.assertEqual(schedule["duplicate_ticks"], [10])

            segments = schedule["train_segments"]
            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0]["t_start"], 0)
            self.assertEqual(segments[0]["t_end"], 10)
            self.assertEqual(segments[0]["t_step"], 10)
            self.assertEqual(segments[1]["t_start"], 10)
            self.assertEqual(segments[1]["t_end"], 25)
            self.assertEqual(segments[1]["t_step"], 15)


class TestBestScheduleSelectionRule(unittest.TestCase):
    def test_prefers_restart_snapshot_if_strictly_better(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "search_space": {
                            "hyperparameters": [
                                {"name": "hp1"},
                                {"name": "hp2"},
                            ]
                        }
                    }
                )
            )

            history_solution = [
                [[0, [1.0, 2.0]], [10, [1.0, 2.0]]],
                [[0, [3.0, 4.0]], [10, [3.0, 4.0]]],
            ]
            history_fitness = [
                [[0, 0.1], [10, 0.5]],
                [[0, 0.2], [10, 0.4]],
            ]
            (run_dir / "history_solution.yaml").write_text(yaml.safe_dump(history_solution))
            (run_dir / "history_fitness.yaml").write_text(yaml.safe_dump(history_fitness))
            (run_dir / "last_finished_tick.yaml").write_text(yaml.safe_dump(10))

            # Restart snapshot with strictly better fitness than current best (0.5).
            snapshot = {
                "fitness": 0.6,
                "solution_history": [[0, [9.0, 9.0]], [10, [9.0, 9.0]]],
                "fitness_history": [[0, 0.3], [10, 0.6]],
                "solution_id": 0,
                "t": 10,
            }
            (run_dir / "best_info_10.yaml").write_text(yaml.safe_dump(snapshot))

            schedule = compute_best_schedule(run_dir)

            self.assertEqual(schedule["selected_source"], "best_info_10.yaml")
            self.assertAlmostEqual(schedule["best_fitness"], 0.6)
            self.assertEqual(schedule["solution_history"], snapshot["solution_history"])


class TestBestScheduleNonLineageFallback(unittest.TestCase):
    def test_builds_fixed_schedule_from_best_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            # Non-lineage run only has best_info + config (no history_solution/history_fitness).
            (run_dir / "config.yaml").write_text(
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
            (run_dir / "best_info.yaml").write_text(
                yaml.safe_dump(
                    {
                        "fitness": 0.42,
                        "config": {"log10_lr": -3.0, "momentum": 0.8, "seed": 123},
                        "metrics": {"tick": 120},
                        "test": 0.4,
                    }
                )
            )

            schedule = compute_best_schedule(run_dir)

            self.assertEqual(schedule["selected_source"], "best_info.yaml")
            self.assertEqual(schedule["best_t"], 120)
            self.assertAlmostEqual(schedule["best_fitness"], 0.42)
            self.assertEqual(schedule["duplicate_ticks"], [])
            self.assertEqual(len(schedule["train_segments"]), 1)
            self.assertEqual(schedule["train_segments"][0]["t_start"], 0)
            self.assertEqual(schedule["train_segments"][0]["t_end"], 120)
            self.assertEqual(schedule["train_segments"][0]["solution"], [-3.0, 0.8])

    def test_uses_reference_t_step_cadence_for_non_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "algo": {"t_step": 50},
                        "search_space": {
                            "hyperparameters": [
                                {"name": "log10_lr"},
                                {"name": "momentum"},
                            ]
                        },
                    }
                )
            )
            (run_dir / "best_info.yaml").write_text(
                yaml.safe_dump(
                    {
                        "fitness": 0.42,
                        "config": {"log10_lr": -3.0, "momentum": 0.8},
                        "metrics": {"tick": 120},
                        "test": 0.4,
                    }
                )
            )

            schedule = compute_best_schedule(run_dir)
            segments = schedule["train_segments"]
            self.assertEqual(len(segments), 3)
            self.assertEqual((segments[0]["t_start"], segments[0]["t_end"]), (0, 50))
            self.assertEqual((segments[1]["t_start"], segments[1]["t_end"]), (50, 100))
            self.assertEqual((segments[2]["t_start"], segments[2]["t_end"]), (100, 120))
            for seg in segments:
                self.assertEqual(seg["solution"], [-3.0, 0.8])


if __name__ == "__main__":
    unittest.main()
