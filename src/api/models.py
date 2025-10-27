from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MetricRecord(BaseModel):
    """Structured metric record emitted by the training loop."""

    timestamp: float = Field(default_factory=lambda: dt.datetime.utcnow().timestamp())
    run_id: Optional[str] = None
    step: int
    metrics: Dict[str, float]


class MetricHistoryResponse(BaseModel):
    items: List[MetricRecord]
    total: int


class TrainingStatus(BaseModel):
    running: bool
    pid: Optional[int] = None
    command: Optional[List[str]] = None
    start_time: Optional[float] = None
    return_code: Optional[int] = None


class StartTrainingRequest(BaseModel):
    config_path: Optional[Path] = None
    extra_args: List[str] = Field(default_factory=list)


class TrainingResponse(BaseModel):
    status: TrainingStatus


class ConfigUpdateRequest(BaseModel):
    content: str


class ConfigResponse(BaseModel):
    path: Path
    content: str
