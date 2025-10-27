import numpy as np


class RunningNormalizer:
    """Tracks running mean and variance for online normalization."""

    def __init__(self, size: int, epsilon: float = 1e-8, clip_range: float = 5.0):
        self.size = size
        self.epsilon = epsilon
        self.clip_range = clip_range
        self.reset()

    def reset(self) -> None:
        self.mean = np.zeros(self.size, dtype=np.float64)
        self.var = np.ones(self.size, dtype=np.float64)
        self.count = self.epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        if self.clip_range is not None:
            normalized = np.clip(normalized, -self.clip_range, self.clip_range)
        return normalized

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x * np.sqrt(self.var + self.epsilon) + self.mean

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
            "clip_range": float(self.clip_range) if self.clip_range is not None else None,
            "size": int(self.size),
        }

    def load_state_dict(self, state: dict) -> None:
        self.size = int(state.get("size", self.size))
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = float(state["count"])
        self.clip_range = state.get("clip_range", self.clip_range)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
