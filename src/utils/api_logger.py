from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

from pydantic import ValidationError

from src.api.models import MetricRecord


class APIMetricsWriter:
    """Writes training metrics to a JSONL feed consumed by the API server."""

    def __init__(self, path: Path, *, run_id: Optional[str] = None, max_history: int = 10_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.max_history = max_history
        self._lock = threading.Lock()
        self._buffer: Deque[MetricRecord] = deque(maxlen=max_history)
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = MetricRecord.model_validate_json(line)
                except (json.JSONDecodeError, ValidationError):
                    continue
                self._buffer.append(record)

    def write_scalars(self, step: int, scalars: Dict[str, float]) -> None:
        if not scalars:
            return
        record = MetricRecord(
            timestamp=time.time(),
            step=step,
            run_id=self.run_id,
            metrics={k: float(v) for k, v in scalars.items()},
        )
        payload = record.model_dump_json()
        with self._lock:
            self._buffer.append(record)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(payload + "\n")

    def latest(self) -> Optional[MetricRecord]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None
