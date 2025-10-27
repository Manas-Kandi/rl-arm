from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from src.models.networks import MLPActor, QNetwork, TwinQNetwork, hard_update, soft_update


def get_optimizer(name: str, params, lr: float, weight_decay: float = 0.0):
    name = name.lower()
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    return Adam(params, lr=lr, weight_decay=weight_decay)


@dataclass
class AgentConfig:
    state_dim: int
    action_dim: int
    actor_hidden: tuple
    critic_hidden: tuple
    actor_lr: float
    critic_lr: float
    weight_decay: float
    gamma: float
    tau: float
    use_twin_critics: bool
    target_noise_std: float
    target_noise_clip: float
    policy_update_delay: int
    gradient_clip_norm: Optional[float]
    optimizer: str = "adam"


class DDPGAgent:
    def __init__(
        self,
        config: AgentConfig,
        device: torch.device,
        action_limit: Optional[np.ndarray] = None,
    ) -> None:
        self.cfg = config
        self.device = device
        self.actor = MLPActor(config.state_dim, config.action_dim, config.actor_hidden, act_limit=1.0).to(device)
        self.actor_target = MLPActor(config.state_dim, config.action_dim, config.actor_hidden, act_limit=1.0).to(device)
        hard_update(self.actor_target, self.actor)

        if config.use_twin_critics:
            self.critic = TwinQNetwork(config.state_dim, config.action_dim, config.critic_hidden).to(device)
            self.critic_target = TwinQNetwork(config.state_dim, config.action_dim, config.critic_hidden).to(device)
        else:
            self.critic = QNetwork(config.state_dim, config.action_dim, config.critic_hidden).to(device)
            self.critic_target = QNetwork(config.state_dim, config.action_dim, config.critic_hidden).to(device)
        hard_update(self.critic_target, self.critic)

        self.actor_optim = get_optimizer(config.optimizer, self.actor.parameters(), config.actor_lr)
        self.critic_optim = get_optimizer(config.optimizer, self.critic.parameters(), config.critic_lr, config.weight_decay)

        self.action_limit = (
            np.asarray(action_limit, dtype=np.float32)
            if action_limit is not None
            else np.ones(config.action_dim, dtype=np.float32)
        )

        self.total_it = 0

    def act(self, state: np.ndarray, noise_std: float = 0.0, deterministic: bool = False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        self.actor.train()
        if not deterministic and noise_std > 0.0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape)
        action = np.clip(action, -1.0, 1.0)
        return action

    def act_to_env(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_limit

    def target_action(self, next_states: torch.Tensor) -> torch.Tensor:
        action = self.actor_target(next_states)
        if self.cfg.target_noise_std > 0.0:
            noise = torch.randn_like(action) * self.cfg.target_noise_std
            noise = torch.clamp(noise, -self.cfg.target_noise_clip, self.cfg.target_noise_clip)
            action = torch.clamp(action + noise, -1.0, 1.0)
        return action

    def update(self, batch, tau: Optional[float] = None) -> Dict[str, float]:
        tau = tau or self.cfg.tau
        self.total_it += 1
        info: Dict[str, float] = {}

        states = batch.states
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states
        dones = batch.dones

        with torch.no_grad():
            next_actions = self.target_action(next_states)
            if self.cfg.use_twin_critics:
                target_q1, target_q2 = self.critic_target(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2)
            else:
                target_q = self.critic_target(next_states, next_actions)
            q_target = rewards + (1.0 - dones) * self.cfg.gamma * target_q

        if self.cfg.use_twin_critics:
            current_q1, current_q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)
        else:
            current_q = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.cfg.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.gradient_clip_norm)
        self.critic_optim.step()
        info["critic_loss"] = float(critic_loss.item())

        update_actor = self.total_it % max(1, self.cfg.policy_update_delay) == 0
        if update_actor:
            actor_actions = self.actor(states)
            if self.cfg.use_twin_critics:
                actor_loss = -self.critic.q1_forward(states, actor_actions).mean()
            else:
                actor_loss = -self.critic(states, actor_actions).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            if self.cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.gradient_clip_norm)
            self.actor_optim.step()
            info["actor_loss"] = float(actor_loss.item())

            soft_update(self.actor_target, self.actor, tau)
        soft_update(self.critic_target, self.critic, tau)

        info["q_target_mean"] = float(q_target.mean().item())
        return info

    def save(self, directory: str, step: int) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"checkpoint_{step}.pt")
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "action_limit": self.action_limit,
                "total_it": self.total_it,
                "config": self.cfg.__dict__,
            },
            path,
        )
        return path

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optim.load_state_dict(checkpoint["actor_optim"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim"])
        self.action_limit = checkpoint.get("action_limit", self.action_limit)
        self.total_it = checkpoint.get("total_it", 0)
