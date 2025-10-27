import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Protocol

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - TensorBoard is optional in tests
    SummaryWriter = None  # type: ignore

try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional dependency
    wandb = None  # type: ignore


class MetricsSink(Protocol):
    def write_scalars(self, step: int, scalars: Dict[str, float]) -> None:
        ...


class TrainLogger:
    """Handles TensorBoard and optional Weights & Biases logging."""

    def __init__(
        self,
        base_dir: str,
        run_name: str,
        tensorboard_dir: Optional[str] = None,
        wandb_cfg: Optional[Dict[str, Any]] = None,
        config_dump: Optional[Dict[str, Any]] = None,
        structured_writer: Optional[MetricsSink] = None,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{run_name}-{timestamp}"
        self.output_dir = os.path.join(base_dir, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.structured_writer = structured_writer

        tb_dir = tensorboard_dir or os.path.join(self.output_dir, "tb")
        if SummaryWriter is None:
            logging.warning("TensorBoard not available; metrics will not be written.")
            self.tb_writer = None
        else:
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

        self.wandb_run = None
        if wandb_cfg and wandb_cfg.get("enabled", False):
            if wandb is None:
                logging.warning("wandb requested but not installed; skipping init.")
            else:
                wandb_args = {
                    "project": wandb_cfg.get("project", "panda-door-rl"),
                    "entity": wandb_cfg.get("entity"),
                    "group": wandb_cfg.get("group"),
                    "name": self.run_id,
                    "config": config_dump,
                }
                self.wandb_run = wandb.init(**{k: v for k, v in wandb_args.items() if v is not None})

        if config_dump is not None:
            config_path = os.path.join(self.output_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(config_dump, fh, indent=2)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)
        if self.wandb_run is not None:
            wandb.log({tag: value, "global_step": step}, step=step)
        if self.structured_writer is not None:
            self.structured_writer.write_scalars(step, {tag: float(value)})

    def log_scalars(self, prefix: str, metrics: Dict[str, float], step: int) -> None:
        payload: Dict[str, float] = {}
        for key, val in metrics.items():
            tag = f"{prefix}/{key}"
            payload[tag] = float(val)
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(tag, float(val), step)
            if self.wandb_run is not None:
                wandb.log({tag: float(val), "global_step": step}, step=step)
        if self.structured_writer is not None and payload:
            self.structured_writer.write_scalars(step, payload)

    def log_histogram(self, tag: str, values, step: int) -> None:
        if self.tb_writer is not None:
            self.tb_writer.add_histogram(tag, values, step)
        if self.wandb_run is not None:
            wandb.log({tag: wandb.Histogram(values), "global_step": step}, step=step)

    def log_video(self, tag: str, video_path: str, step: int) -> None:
        if self.wandb_run is not None and os.path.exists(video_path):
            wandb.log({tag: wandb.Video(video_path)}, step=step)

    def flush(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.flush()
        if self.wandb_run is not None:
            wandb.log({}, commit=True)

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            wandb.finish()

    def set_structured_writer(self, writer: Optional[MetricsSink]) -> None:
        self.structured_writer = writer
