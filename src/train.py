from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import torch
import yaml

from src.agents.ddpg_agent import AgentConfig, DDPGAgent
from src.envs.robosuite_wrapper import RobosuiteWrapper
from src.replay.replay_buffer import TransitionBatch, ReplayBuffer
from src.utils.logger import TrainLogger
from src.utils.metrics import compute_success_rate, summarize_episode
from src.utils.normalizer import RunningNormalizer


def save_normalizer(normalizer: RunningNormalizer, path: str) -> None:
    state = normalizer.state_dict()
    np.savez(path, **state)


def load_normalizer_state(normalizer: RunningNormalizer, path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    state_dict = {key: data[key] for key in data.files}
    clip_val = state_dict.get("clip_range")
    if clip_val is not None and hasattr(clip_val, "tolist"):
        clip_val = clip_val.tolist()
    if "clip_range" in state_dict:
        state_dict["clip_range"] = clip_val
    if "count" in state_dict:
        state_dict["count"] = float(np.array(state_dict["count"]).item())
    if "size" in state_dict:
        state_dict["size"] = int(np.array(state_dict["size"]).item())
    normalizer.load_state_dict(state_dict)


def prune_checkpoints(directory: str, keep: int) -> None:
    if keep <= 0:
        return
    path = Path(directory)
    if not path.exists():
        return
    checkpoints = sorted(path.glob("checkpoint_*.pt"), key=os.path.getmtime)
    if len(checkpoints) <= keep:
        return
    for ckpt in checkpoints[:-keep]:
        ckpt.unlink(missing_ok=True)
        norm = ckpt.with_name(ckpt.stem + "_normalizer.npz")
        if norm.exists():
            norm.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Panda Door DDPG agent")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--device", type=str, default=None, help="Override torch device (cpu or cuda)")
    return parser.parse_args()


class LinearNoiseScheduler:
    def __init__(self, init_std: float, final_std: float, decay_steps: int) -> None:
        self.init_std = init_std
        self.final_std = final_std
        self.decay_steps = max(1, decay_steps)

    def value(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.final_std
        frac = step / self.decay_steps
        return self.init_std + (self.final_std - self.init_std) * frac


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_env(env_cfg: Dict, seed: int) -> RobosuiteWrapper:
    randomization_cfg = deepcopy(env_cfg.get("randomization", {}))
    randomization_cfg["enabled"] = env_cfg.get("randomize", True)

    observation_noise = env_cfg.get("observation_noise", {})
    observation_noise.setdefault("enabled", observation_noise.get("enabled", False))

    action_delay = env_cfg.get("action_delay", {})
    action_delay.setdefault("enabled", action_delay.get("enabled", False))

    env = RobosuiteWrapper(
        env_name=env_cfg.get("task", env_cfg.get("name", "Door")),
        robots=env_cfg.get("robots", "Panda"),
        horizon=env_cfg.get("horizon", env_cfg.get("max_episode_steps", 300)),
        seed=seed,
        has_renderer=env_cfg.get("has_renderer", False),
        use_camera_obs=env_cfg.get("use_camera_obs", False),
        observation_noise=observation_noise,
        action_delay=action_delay,
        randomization=randomization_cfg,
    )
    return env


def make_agent(config: Dict, state_dim: int, action_dim: int, device: torch.device, action_limit: np.ndarray) -> DDPGAgent:
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


def normalize_batch(batch: TransitionBatch, normalizer: RunningNormalizer, device: torch.device) -> TransitionBatch:
    mean = torch.as_tensor(normalizer.mean, device=device, dtype=torch.float32)
    std = torch.as_tensor(np.sqrt(normalizer.var + normalizer.epsilon), device=device, dtype=torch.float32)
    clip = normalizer.clip_range
    states = (batch.states - mean) / std
    next_states = (batch.next_states - mean) / std
    if clip is not None:
        states = torch.clamp(states, -clip, clip)
        next_states = torch.clamp(next_states, -clip, clip)
    return TransitionBatch(states, batch.actions, batch.rewards, next_states, batch.dones)


def evaluate(
    agent: DDPGAgent,
    env: RobosuiteWrapper,
    normalizer: RunningNormalizer,
    episodes: int,
    max_steps: int,
) -> Dict[str, float]:
    returns = []
    successes = []
    with torch.no_grad():
        for ep in range(episodes):
            state = env.reset()
            done = False
            ep_rewards = []
            clip = normalizer.clip_range
            for step in range(max_steps):
                state_t = torch.as_tensor(state, dtype=torch.float32)
                mean = torch.as_tensor(normalizer.mean, dtype=torch.float32)
                std = torch.as_tensor(np.sqrt(normalizer.var + normalizer.epsilon), dtype=torch.float32)
                norm_state = (state_t - mean) / std
                if clip is not None:
                    norm_state = torch.clamp(norm_state, -clip, clip)
                action = agent.act(norm_state.cpu().numpy(), deterministic=True)
                env_action = agent.act_to_env(action)
                next_state, reward, done, info = env.step(env_action)
                ep_rewards.append(reward)
                state = next_state
                if done:
                    break
            ep_stats = summarize_episode(ep_rewards)
            returns.append(ep_stats["return"])
            successes.append(bool(info.get("task_success", False) or info.get("door_hinge_angle", 0.0) > 0.2))
    return {
        "avg_return": float(np.mean(returns) if returns else 0.0),
        "success_rate": compute_success_rate(successes),
    }


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    seed = int(config.get("seed", 0))
    set_seed(seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    env_cfg = config["env"]
    train_env = build_env(env_cfg, seed)
    eval_env = build_env(env_cfg, seed + 1000)

    init_state = train_env.reset(seed)
    state_dim = int(init_state.shape[0])
    action_dim = int(train_env.action_dim)

    action_limit = train_env.action_limit

    normalizer = RunningNormalizer(size=state_dim)
    normalizer.update(init_state[None, :])

    replay_cfg = config.get("replay", {})
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=int(replay_cfg.get("size", 1_000_000)),
        device=device,
    )

    agent_cfg = config["agent"]
    agent_cfg["state_dim"] = state_dim
    agent_cfg["action_dim"] = action_dim
    agent = make_agent(agent_cfg, state_dim, action_dim, device, action_limit)

    checkpoint_cfg = config.get("checkpoint", {})
    checkpoint_dir = os.path.join(logger.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    keep_latest = int(checkpoint_cfg.get("keep_latest", 5))
    resume_path = args.resume or checkpoint_cfg.get("resume_path")
    if resume_path:
        agent.load(resume_path)
        norm_file = Path(resume_path).with_name(Path(resume_path).stem + "_normalizer.npz")
        if norm_file.exists():
            load_normalizer_state(normalizer, norm_file)

    logger_cfg = config.get("logging", {})
    output_dir = logger_cfg.get("output_dir", "experiments")
    run_name = logger_cfg.get("run_name", f"panda_door_seed{seed}")
    tensorboard_dir = logger_cfg.get("tensorboard_dir")
    logger = TrainLogger(
        base_dir=output_dir,
        run_name=run_name,
        tensorboard_dir=tensorboard_dir,
        wandb_cfg=logger_cfg.get("wandb"),
        config_dump=config,
    )

    exploration_cfg = config.get("exploration", {})
    noise_scheduler = LinearNoiseScheduler(
        exploration_cfg.get("noise_std_init", 0.3),
        exploration_cfg.get("noise_std_final", 0.05),
        exploration_cfg.get("noise_decay_steps", 300_000),
    )
    smoothing_coef = float(exploration_cfg.get("action_smoothing_coef", 0.0))

    training_cfg = config.get("training", {})
    total_steps = int(training_cfg.get("total_env_steps", 1_000_000))
    max_episode_steps = int(training_cfg.get("max_episode_steps", env_cfg.get("horizon", 300)))
    warmup_steps = int(replay_cfg.get("warmup_steps", 10_000))
    batch_size = int(replay_cfg.get("batch_size", 256))
    n_updates = int(replay_cfg.get("n_updates_per_step", 1))
    eval_every = int(training_cfg.get("eval_every_steps", 50_000))
    save_every = int(training_cfg.get("save_every_steps", 50_000))
    log_every = int(training_cfg.get("log_every_steps", 1000))
    num_eval_episodes = int(training_cfg.get("num_eval_episodes", 5))
    reward_scale = float(training_cfg.get("reward_scale", 1.0))
    success_angle = float(training_cfg.get("success_angle", 0.2))

    global_step = 0
    episode = 0
    best_success = 0.0
    prev_action = np.zeros(action_dim, dtype=np.float32)

    start_time = time.time()
    while global_step < total_steps:
        episode += 1
        state = train_env.reset(seed + episode)
        episode_rewards = []
        episode_actions = []
        normalizer.update(state[None, :])
        info: Dict = {}
        prev_action = np.zeros(action_dim, dtype=np.float32)

        for step in range(max_episode_steps):
            if global_step >= total_steps:
                break
            norm_state = normalizer.normalize(state)
            if global_step < warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=action_dim)
            else:
                noise_std = noise_scheduler.value(global_step)
                action = agent.act(norm_state, noise_std=noise_std, deterministic=False)
            if smoothing_coef > 0:
                action = (1.0 - smoothing_coef) * action + smoothing_coef * prev_action
            env_action = agent.act_to_env(action)
            next_state, reward, done, info = train_env.step(env_action)
            scaled_reward = reward * reward_scale

            replay_buffer.add(state, action, scaled_reward, next_state, float(done))
            normalizer.update(next_state[None, :])

            episode_rewards.append(scaled_reward)
            episode_actions.append(action)

            state = next_state
            prev_action = action
            global_step += 1

            if replay_buffer.size >= batch_size and global_step > warmup_steps:
                for _ in range(n_updates):
                    batch = replay_buffer.sample(batch_size)
                    batch = normalize_batch(batch, normalizer, device)
                    metrics = agent.update(batch)
                if global_step % log_every == 0:
                    logger.log_scalars("loss", metrics, global_step)

            if global_step % log_every == 0:
                logger.log_scalar("env/reward_raw", float(reward), global_step)
                logger.log_scalar("env/door_angle", float(info.get("door_hinge_angle", 0.0)), global_step)

            if done:
                break

        ep_stats = summarize_episode(episode_rewards)
        logger.log_scalars("episode", {"return": ep_stats["return"], "length": ep_stats["length"]}, global_step)
        success_flag = float(info.get("task_success", False) or info.get("door_hinge_angle", 0.0) > success_angle)
        logger.log_scalar("episode/success", success_flag, global_step)
        if episode_actions:
            action_arr = np.asarray(episode_actions)
            action_stats = {
                "mean": float(np.mean(action_arr)),
                "std": float(np.std(action_arr)),
                "max_abs": float(np.max(np.abs(action_arr))),
            }
        else:
            action_stats = {"mean": 0.0, "std": 0.0, "max_abs": 0.0}
        logger.log_scalars("action", action_stats, global_step)

        if global_step % eval_every == 0:
            eval_metrics = evaluate(agent, eval_env, normalizer, num_eval_episodes, max_episode_steps)
            logger.log_scalars("eval", eval_metrics, global_step)
            if eval_metrics["success_rate"] > best_success:
                best_success = eval_metrics["success_rate"]
                ckpt_path = agent.save(checkpoint_dir, global_step)
                save_normalizer(normalizer, os.path.splitext(ckpt_path)[0] + "_normalizer.npz")
                prune_checkpoints(checkpoint_dir, keep_latest)

        if global_step % save_every == 0:
            ckpt_path = agent.save(checkpoint_dir, global_step)
            save_normalizer(normalizer, os.path.splitext(ckpt_path)[0] + "_normalizer.npz")
            prune_checkpoints(checkpoint_dir, keep_latest)

    total_time = time.time() - start_time
    logger.log_scalar("time/total_hours", total_time / 3600.0, global_step)
    logger.flush()
    logger.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
