from typing import Dict, Iterable, Tuple

import numpy as np


def compute_success_rate(success_flags: Iterable[bool]) -> float:
    flags = np.array(list(success_flags), dtype=np.float32)
    if flags.size == 0:
        return 0.0
    return float(flags.mean())


def summarize_episode(ep_rewards: Iterable[float]) -> Dict[str, float]:
    rewards = np.array(list(ep_rewards), dtype=np.float32)
    if rewards.size == 0:
        return {"return": 0.0, "length": 0, "reward_mean": 0.0}
    return {
        "return": float(rewards.sum()),
        "length": int(rewards.size),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
    }


def action_statistics(actions: Iterable[Iterable[float]]) -> Dict[str, float]:
    arr = np.asarray(list(actions), dtype=np.float32)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(np.abs(arr))),
    }


def hinge_metrics(info: Dict[str, float]) -> Dict[str, float]:
    keys = [
        "door_hinge_angle",
        "door_hinge_velocity",
        "handle_distance",
    ]
    return {k: float(info.get(k, 0.0)) for k in keys}


def sliding_window(values: Iterable[float], window: int) -> Tuple[float, float]:
    arr = np.array(list(values), dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0
    recent = arr[-window:]
    return float(recent.mean()), float(recent.std())
