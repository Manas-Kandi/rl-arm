from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

try:
    import robosuite as suite
except ImportError as exc:  # pragma: no cover - optional dependency in CI
    raise ImportError("Robosuite is required for RobosuiteWrapper") from exc


class RobosuiteWrapper:
    """Wraps a Robosuite environment with deterministic seeding and vector obs."""

    DEFAULT_ACTION_LIMIT = 0.75  # rad/s safe default for Panda joints

    def __init__(
        self,
        env_name: str = "Door",
        robots: str = "Panda",
        horizon: int = 300,
        seed: Optional[int] = None,
        action_limit: Optional[np.ndarray] = None,
        observation_noise: Optional[Dict] = None,
        action_delay: Optional[Dict] = None,
        randomization: Optional[Dict] = None,
        use_camera_obs: bool = False,
        has_renderer: bool = False,
        **env_kwargs,
    ) -> None:
        self.env = suite.make(
            env_name,
            robots=robots,
            has_renderer=has_renderer,
            use_camera_obs=use_camera_obs,
            horizon=horizon,
            **env_kwargs,
        )
        self.horizon = horizon
        self.seed(seed)

        self._action_limit = (
            np.array(action_limit, dtype=np.float32)
            if action_limit is not None
            else np.full(self.env.action_dim, self.DEFAULT_ACTION_LIMIT, dtype=np.float32)
        )

        self.obs_noise_cfg = observation_noise or {"enabled": False}
        self.action_delay_cfg = action_delay or {"enabled": False}
        self.randomization_cfg = randomization or {"enabled": False}

        self._rng = np.random.default_rng(seed)
        self._delay_queue: Deque[np.ndarray] = deque(maxlen=self.action_delay_cfg.get("delay_steps", 1))
        self.prev_action = np.zeros(self.env.action_dim, dtype=np.float32)

        # Cache base physical properties for domain randomization resets.
        model = self.env.sim.model
        self._base_body_mass = model.body_mass.copy()
        self._base_geom_friction = model.geom_friction.copy()
        self._base_damping = model.dof_damping.copy()

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    @property
    def action_limit(self) -> np.ndarray:
        return self._action_limit.copy()

    def seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        np.random.seed(seed)
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        if hasattr(self.env, "np_random") and self.env.np_random is not None:
            self.env.np_random.seed(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed(seed)
        self._apply_domain_randomization()
        obs = self.env.reset()
        self.prev_action = np.zeros_like(self.prev_action)
        self._delay_queue.clear()
        return self._process_obs(obs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self._action_limit, self._action_limit)
        env_action = self._apply_action_delay(action)
        obs, reward, done, info = self.env.step(env_action)
        obs_vec = self._process_obs(obs)
        info = self._augment_info(info, obs)
        if done:
            info["episode"] = self.env.episode_steps
        self.prev_action = action
        return obs_vec, float(reward), bool(done), info

    def _apply_action_delay(self, action: np.ndarray) -> np.ndarray:
        if not self.action_delay_cfg.get("enabled", False):
            return action
        delay_steps = max(1, int(self.action_delay_cfg.get("delay_steps", 1)))
        if not self._delay_queue:
            for _ in range(delay_steps):
                self._delay_queue.append(np.copy(self.prev_action))
        self._delay_queue.append(np.copy(action))
        return self._delay_queue[0]

    def _apply_domain_randomization(self) -> None:
        if not self.randomization_cfg.get("enabled", True):
            self._restore_nominal()
            return
        scale = self.randomization_cfg
        mass_range = scale.get("mass_scale")
        friction_range = scale.get("friction_scale")
        damping_range = scale.get("joint_damping_scale")
        visual_randomization = scale.get("visual_randomization", False)

        sim = self.env.sim
        if mass_range:
            factor = self._rng.uniform(mass_range[0], mass_range[1])
            sim.model.body_mass[:] = self._base_body_mass * factor
        if friction_range:
            factor = self._rng.uniform(friction_range[0], friction_range[1])
            sim.model.geom_friction[:] = self._base_geom_friction * factor
        if damping_range:
            factor = self._rng.uniform(damping_range[0], damping_range[1])
            sim.model.dof_damping[:] = self._base_damping * factor
        if visual_randomization and hasattr(self.env, "randomize_visual"):
            self.env.randomize_visual()

    def _restore_nominal(self) -> None:
        sim = self.env.sim
        sim.model.body_mass[:] = self._base_body_mass
        sim.model.geom_friction[:] = self._base_geom_friction
        sim.model.dof_damping[:] = self._base_damping

    def _process_obs(self, obs: Dict) -> np.ndarray:
        def _extract(keys, length: int) -> np.ndarray:
            if not isinstance(keys, (list, tuple)):
                keys = [keys]
            raw = None
            for key in keys:
                candidate = obs.get(key)
                if candidate is not None:
                    raw = candidate
                    break
            if raw is None:
                return np.zeros(length, dtype=np.float32)
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return np.zeros(length, dtype=np.float32)
            if arr.size < length:
                arr = np.pad(arr, (0, length - arr.size))
            return arr[:length]

        joint_pos = _extract("robot0_joint_pos", 7)
        joint_vel = _extract("robot0_joint_vel", 7)
        eef_pos = _extract("robot0_eef_pos", 3)
        eef_quat = _extract("robot0_eef_quat", 4)
        door_qpos = _extract(["door_pos", "door_to_target", "hinge_qpos"], 1)
        door_qvel = _extract(["door_vel", "hinge_qvel"], 1)
        gripper_state = _extract("robot0_gripper_qpos", 1)

        components = [
            joint_pos,
            joint_vel,
            eef_pos,
            eef_quat,
            door_qpos,
            door_qvel,
            gripper_state,
            self.prev_action.astype(np.float32),
        ]

        obs_vec = np.concatenate(components, axis=0)
        if self.obs_noise_cfg.get("enabled", False):
            std = float(self.obs_noise_cfg.get("std", 0.0))
            obs_vec = obs_vec + self._rng.normal(0.0, std, size=obs_vec.shape)
        return obs_vec

    def _augment_info(self, info: Dict, obs: Dict) -> Dict:
        hinge_arr = obs.get("door_pos")
        if hinge_arr is None:
            hinge_arr = obs.get("door_to_target")
        if hinge_arr is None:
            hinge_arr = obs.get("hinge_qpos")
        hinge_vel_arr = obs.get("door_vel")
        if hinge_vel_arr is None:
            hinge_vel_arr = obs.get("hinge_qvel")
        hinge_angle = float(np.asarray(hinge_arr, dtype=np.float32).reshape(-1)[0]) if hinge_arr is not None else 0.0
        hinge_vel = float(np.asarray(hinge_vel_arr, dtype=np.float32).reshape(-1)[0]) if hinge_vel_arr is not None else 0.0
        info = dict(info)
        info.update(
            {
                "door_hinge_angle": hinge_angle,
                "door_hinge_velocity": hinge_vel,
                "prev_action_norm": float(np.linalg.norm(self.prev_action)),
            }
        )
        return info

    def render(self):  # pragma: no cover - visualization helper
        return self.env.render()

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
