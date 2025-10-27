from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

from src.agents.ddpg_agent import AgentConfig, DDPGAgent
from src.envs.robosuite_wrapper import RobosuiteWrapper
from src.utils.metrics import compute_success_rate, summarize_episode
from src.utils.normalizer import RunningNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Panda Door policy")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--render", action="store_true", help="Render environment (headless environments only)")
    return parser.parse_args()


def build_agent(config: Dict, state_dim: int, action_dim: int, device: torch.device, action_limit: np.ndarray) -> DDPGAgent:
    cfg = AgentConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_hidden=tuple(config.get("actor_hidden", [256, 128])),
        critic_hidden=tuple(config.get("critic_hidden", [256, 128])),
        actor_lr=float(config.get("actor_lr", 1e-4)),
        critic_lr=float(config.get("critic_lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        gamma=float(config.get("gamma", 0.99)),
        tau=float(config.get("tau", 0.005)),
        use_twin_critics=bool(config.get("use_twin_critics", False)),
        target_noise_std=float(config.get("target_noise", {}).get("std", 0.0)),
        target_noise_clip=float(config.get("target_noise", {}).get("clip", 0.2)),
        policy_update_delay=int(config.get("policy_update_delay", 1)),
        gradient_clip_norm=config.get("gradient_clip_norm"),
        optimizer=config.get("optimizer", "adam"),
    )
    agent = DDPGAgent(cfg, device=device, action_limit=action_limit)
    return agent


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    env_cfg = config["env"]
    env = RobosuiteWrapper(
        env_name=env_cfg.get("task", env_cfg.get("name", "Door")),
        robots=env_cfg.get("robots", "Panda"),
        horizon=env_cfg.get("horizon", env_cfg.get("max_episode_steps", 300)),
        use_camera_obs=env_cfg.get("use_camera_obs", False),
        has_renderer=args.render,
        observation_noise={"enabled": False},
        action_delay={"enabled": False},
        randomization={"enabled": env_cfg.get("randomize", False)},
    )

    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_dim

    normalizer = RunningNormalizer(size=state_dim)
    norm_path = Path(args.checkpoint)
    normalizer_path = norm_path.with_name(norm_path.stem + "_normalizer.npz")
    if normalizer_path.exists():
        data = np.load(normalizer_path, allow_pickle=True)
        state_dict = {key: data[key] for key in data.files}
        clip_val = state_dict.get("clip_range")
        if clip_val is not None:
            if hasattr(clip_val, "tolist"):
                clip_val = clip_val.tolist()
            state_dict["clip_range"] = clip_val
        if "count" in state_dict:
            state_dict["count"] = float(np.array(state_dict["count"]).item())
        if "size" in state_dict:
            state_dict["size"] = int(np.array(state_dict["size"]).item())
        normalizer.load_state_dict(state_dict)
    else:
        normalizer.update(state[None, :])

    agent_cfg = config["agent"]
    agent = build_agent(agent_cfg, state_dim, action_dim, device, env.action_limit)
    agent.load(args.checkpoint)

    successes = []
    returns = []
    lengths = []

    for ep in range(args.episodes):
        state = env.reset()
        episode_rewards = []
        for t in range(env.horizon):
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            mean = torch.as_tensor(normalizer.mean, dtype=torch.float32)
            std = torch.as_tensor(np.sqrt(normalizer.var + normalizer.epsilon), dtype=torch.float32)
            norm_state = (state_tensor - mean) / std
            if normalizer.clip_range is not None:
                norm_state = torch.clamp(norm_state, -normalizer.clip_range, normalizer.clip_range)
            action = agent.act(norm_state.cpu().numpy(), deterministic=True)
            env_action = agent.act_to_env(action)
            next_state, reward, done, info = env.step(env_action)
            episode_rewards.append(reward)
            state = next_state
            if done:
                break
        stats = summarize_episode(episode_rewards)
        returns.append(stats["return"])
        lengths.append(stats["length"])
        successes.append(bool(info.get("task_success", False) or info.get("door_hinge_angle", 0.0) > 0.2))
        print(f"Episode {ep}: return={stats['return']:.2f}, length={stats['length']}, success={successes[-1]}")

    print("==== Evaluation Summary ====")
    print(f"Average Return: {np.mean(returns):.2f}")
    print(f"Success Rate: {compute_success_rate(successes)*100:.1f}%")
    print(f"Average Length: {np.mean(lengths):.1f}")

    env.close()


if __name__ == "__main__":
    main()
