from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from .models import StartTrainingRequest, TrainingStatus


class TrainingManager:
    """Manages the lifecycle of the training subprocess."""

    def __init__(self, project_root: Path, default_config: Path) -> None:
        self.project_root = project_root
        self.default_config = default_config
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None

    def status(self) -> TrainingStatus:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return TrainingStatus(
                running=running,
                pid=self._process.pid if self._process else None,
                command=self._process.args if self._process else None,
                start_time=self._start_time,
                return_code=None if running or not self._process else self._process.returncode,
            )

    def start(self, request: Optional[StartTrainingRequest] = None) -> TrainingStatus:
        with self._lock:
            if self._process and self._process.poll() is None:
                return self.status()
            config_path = request.config_path if request and request.config_path else self.default_config
            config_path = config_path.resolve()
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
            command: List[str] = [
                sys.executable,
                "-m",
                "src.train",
                "--config",
                str(config_path),
            ]
            if request:
                command.extend(request.extra_args)
            env = os.environ.copy()
            self._process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                env=env,
            )
            self._start_time = time.time()
            return self.status()

    def stop(self, timeout: float = 10.0) -> TrainingStatus:
        with self._lock:
            if not self._process or self._process.poll() is not None:
                return self.status()
            self._process.send_signal(signal.SIGINT)
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
            finally:
                self._process = None
                self._start_time = None
            return self.status()
