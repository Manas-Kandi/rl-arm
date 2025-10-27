from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_HIDDEN = (256, 128)


def fanin_init(layer: nn.Linear, bound: float = None) -> None:
    if bound is None:
        bound = 1.0 / layer.weight.data.size(0) ** 0.5
    nn.init.uniform_(layer.weight.data, -bound, bound)
    nn.init.uniform_(layer.bias.data, -bound, bound)


def final_layer_init(layer: nn.Linear, scale: float = 3e-3) -> None:
    nn.init.uniform_(layer.weight.data, -scale, scale)
    nn.init.uniform_(layer.bias.data, -scale, scale)


class MLPActor(nn.Module):
    """Deterministic policy network with tanh-squashed outputs."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int] = DEFAULT_HIDDEN,
        act_limit: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes)

        layers = []
        last_dim = state_dim
        for h in hidden_sizes:
            layer = nn.Linear(last_dim, h)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            last_dim = h
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(last_dim, action_dim)
        final_layer_init(self.output_layer)
        self.act_limit = act_limit

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        action = torch.tanh(self.output_layer(x))
        return action * self.act_limit


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int] = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes)

        layers = []
        last_dim = state_dim + action_dim
        for h in hidden_sizes:
            layer = nn.Linear(last_dim, h)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            last_dim = h
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(last_dim, 1)
        final_layer_init(self.output_layer)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


class TwinQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int] = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_sizes)
        self.q2 = QNetwork(state_dim, action_dim, hidden_sizes)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(state, action)


@dataclass
class TargetNetworks:
    actor_target: nn.Module
    critic_target: nn.Module


def hard_update(target: nn.Module, source: nn.Module) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
