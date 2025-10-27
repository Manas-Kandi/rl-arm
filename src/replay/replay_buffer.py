from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-size replay buffer that stores transitions as numpy arrays."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = int(capacity)
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done) -> None:
        idx = self.ptr % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        states = torch.as_tensor(self.states[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device)
        dones = torch.as_tensor(self.dones[indices], device=self.device)
        return TransitionBatch(states, actions, rewards, next_states, dones)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "states": self.states[: self.size],
            "actions": self.actions[: self.size],
            "rewards": self.rewards[: self.size],
            "next_states": self.next_states[: self.size],
            "dones": self.dones[: self.size],
            "ptr": self.ptr,
            "size": self.size,
        }

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.reset()
        size = state.get("size", 0)
        for key in ("states", "actions", "rewards", "next_states", "dones"):
            arr = state.get(key)
            if arr is not None:
                getattr(self, key)[: arr.shape[0]] = arr
        self.size = min(int(size), self.capacity)
        self.ptr = int(state.get("ptr", 0)) % self.capacity

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **self.state_dict())

    def load(self, path: str) -> None:
        data = np.load(path)
        state = {k: data[k] for k in data.files}
        self.load_state_dict(state)
