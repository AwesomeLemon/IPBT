from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _normalize_history(history: list[Any]) -> list[list[Any]]:
    out: list[list[Any]] = []
    for item in history:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid history entry: {item}")
        t_raw, solution = item
        t = int(t_raw)
        out.append([t, solution])
    return out


def _extract_duplicate_ticks(solution_history: list[list[Any]]) -> list[int]:
    counts: dict[int, int] = {}
    for t, _ in solution_history:
        counts[t] = counts.get(t, 0) + 1
    return sorted([t for t, c in counts.items() if c > 1])


def _build_train_segments(solution_history: list[list[Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for (t_start, solution), (t_end, _) in zip(solution_history, solution_history[1:]):
        t_step = t_end - t_start
        if t_step < 0:
            raise ValueError(f"History is not non-decreasing in time: {t_start} -> {t_end}")
        if t_step == 0:
            # Restart boundaries can insert a second entry at the same tick.
            continue
        segments.append(
            {
                "t_start": t_start,
                "t_end": t_end,
                "t_step": t_step,
                "solution": solution,
            }
        )
    return segments


def _select_current_best(run_dir: Path) -> dict[str, Any]:
    history_fitness = _load_yaml(run_dir / "history_fitness.yaml")
    history_solution = _load_yaml(run_dir / "history_solution.yaml")

    if not history_fitness or not history_solution:
        raise ValueError("Missing or empty history_fitness/history_solution")

    best_idx = max(range(len(history_fitness)), key=lambda i: float(history_fitness[i][-1][1]))
    best_fitness = float(history_fitness[best_idx][-1][1])
    best_solution_history = _normalize_history(history_solution[best_idx])
    best_fitness_history = _normalize_history(history_fitness[best_idx])

    last_finished_tick_path = run_dir / "last_finished_tick.yaml"
    if last_finished_tick_path.exists():
        t_best = int(_load_yaml(last_finished_tick_path))
    else:
        t_best = int(best_solution_history[-1][0])

    return {
        "fitness": best_fitness,
        "solution_history": best_solution_history,
        "fitness_history": best_fitness_history,
        "solution_id": int(best_idx),
        "t": t_best,
        "source": "current_population",
    }


def _config_hp_names_from_run_config(run_dir: Path) -> list[str]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml needed for non-lineage fallback: {cfg_path}")
    cfg = _load_yaml(cfg_path) or {}
    hps = (((cfg.get("search_space") or {}).get("hyperparameters")) or [])
    names = [hp.get("name") for hp in hps if isinstance(hp, dict) and hp.get("name") is not None]
    if len(names) == 0:
        raise ValueError(
            "Could not infer search-space hyperparameter names from run config "
            f"for non-lineage fallback: {cfg_path}"
        )
    # ConfigSpaceSearchSpace exposes keys in ConfigSpace order, which in practice
    # is deterministic name order (not the YAML declaration order). Replay expects
    # schedule vectors to follow that same order.
    return sorted(names)


def _reference_t_step_from_run_config(run_dir: Path) -> int | None:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = _load_yaml(cfg_path) or {}
    t_step = (cfg.get("algo") or {}).get("t_step")
    if t_step is None:
        return None
    t_step = int(t_step)
    if t_step <= 0:
        raise ValueError(f"Invalid non-positive algo.t_step in {cfg_path}: {t_step}")
    return t_step


def _infer_best_tick_from_metrics(metrics: dict[str, Any], best_info: dict[str, Any]) -> int:
    # ASHA/RayTune-style.
    if "tick" in metrics:
        return int(metrics["tick"])
    # SMAC-style.
    if "tick_best" in metrics:
        return int(metrics["tick_best"])
    # SMAC trajectory fallback.
    trajectory = metrics.get("trajectory")
    if isinstance(trajectory, list) and len(trajectory) > 0:
        try:
            return int(max(int(point[0]) for point in trajectory))
        except Exception:
            pass
    # Last resort: try top-level t.
    if "t" in best_info:
        return int(best_info["t"])
    raise ValueError(
        "Could not infer best tick for non-lineage fallback; expected metrics.tick, "
        "metrics.tick_best, metrics.trajectory, or top-level t in best_info.yaml"
    )


def _select_non_lineage_best(run_dir: Path) -> dict[str, Any]:
    best_info_path = run_dir / "best_info.yaml"
    if not best_info_path.exists():
        raise FileNotFoundError(
            "Missing lineage files (history_fitness/history_solution) and missing non-lineage fallback file "
            f"{best_info_path}"
        )
    info = _load_yaml(best_info_path) or {}

    best_fitness = float(info["fitness"])
    metrics = info.get("metrics") or {}
    t_best = _infer_best_tick_from_metrics(metrics, info)

    config = info.get("config") or {}
    hp_names = _config_hp_names_from_run_config(run_dir)
    missing = [name for name in hp_names if name not in config]
    if len(missing) > 0:
        raise ValueError(
            "Non-lineage fallback could not build solution vector from best_info.yaml config. "
            f"Missing keys: {missing}"
        )
    solution = [config[name] for name in hp_names]

    # Fixed-configuration schedule for non-lineage runs:
    # reconstruct segment boundaries at the original algo.t_step cadence when available,
    # so scheduler stepping in replay mirrors reference behavior.
    t_best = int(t_best)
    t_step_ref = _reference_t_step_from_run_config(run_dir)
    if t_step_ref is None:
        tick_points = [0, t_best]
    else:
        tick_points = [0]
        t_cur = 0
        while t_cur < t_best:
            t_next = min(t_cur + t_step_ref, t_best)
            tick_points.append(t_next)
            t_cur = t_next

    solution_history = [[t, solution] for t in tick_points]
    fitness_history = [[t, best_fitness] for t in tick_points]

    return {
        "fitness": best_fitness,
        "solution_history": solution_history,
        "fitness_history": fitness_history,
        "solution_id": 0,
        "t": int(t_best),
        "source": "best_info.yaml",
    }


def _select_best_restart_snapshot(run_dir: Path) -> dict[str, Any] | None:
    best_snapshot: dict[str, Any] | None = None
    best_fitness = float("-inf")

    for p in sorted(run_dir.glob("best_info_*.yaml")):
        info = _load_yaml(p)
        fitness = float(info["fitness"])
        if fitness > best_fitness:
            best_fitness = fitness
            best_snapshot = {
                "fitness": fitness,
                "solution_history": _normalize_history(info["solution_history"]),
                "fitness_history": _normalize_history(info["fitness_history"]),
                "solution_id": int(info["solution_id"]),
                "t": int(info["t"]),
                "source": p.name,
            }

    return best_snapshot


def compute_best_schedule(run_dir: str | Path) -> dict[str, Any]:
    """
    Compute the best schedule for a run directory using IPBT save_best semantics.

    Selection rule mirrors algo/ipbt.py:save_best:
    - best current lineage at run end from history_fitness/history_solution
    - compare with best among restart snapshots (best_info_*.yaml)
    - choose restart snapshot only if strictly better fitness
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    has_lineage = (run_dir / "history_fitness.yaml").exists() and (run_dir / "history_solution.yaml").exists()
    if has_lineage:
        current_best = _select_current_best(run_dir)
        best_snapshot = _select_best_restart_snapshot(run_dir)

        selected = current_best
        if best_snapshot is not None and best_snapshot["fitness"] > current_best["fitness"]:
            selected = best_snapshot
    else:
        selected = _select_non_lineage_best(run_dir)

    solution_history = selected["solution_history"]
    fitness_history = selected["fitness_history"]
    hp_names_ref = _config_hp_names_from_run_config(run_dir)

    if not solution_history:
        raise ValueError("Selected best schedule has empty solution_history")

    duplicate_ticks = _extract_duplicate_ticks(solution_history)
    segments = _build_train_segments(solution_history)

    return {
        "run_dir": str(run_dir),
        "selected_source": selected["source"],
        "best_fitness": float(selected["fitness"]),
        "best_t": int(selected["t"]),
        "solution_id": int(selected["solution_id"]),
        "hp_names": hp_names_ref,
        "solution_history": solution_history,
        "fitness_history": fitness_history,
        "duplicate_ticks": duplicate_ticks,
        "train_segments": segments,
    }
