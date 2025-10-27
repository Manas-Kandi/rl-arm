from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set

from pydantic import ValidationError

from .models import MetricRecord


class MetricsManager:
    """Manages metric persistence and realtime subscriptions."""

    def __init__(
        self,
        jsonl_path: Path,
        *,
        max_history: int = 10_000,
        poll_interval: float = 1.0,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.poll_interval = poll_interval
        self._records: Deque[MetricRecord] = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._file_position = 0
        self._subscribers: Set[asyncio.Queue[MetricRecord]] = set()
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.jsonl_path.exists():
            return
        with self.jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    data = json.loads(line)
                    record = MetricRecord.model_validate(data)
                except (json.JSONDecodeError, ValidationError):
                    continue
                self._records.append(record)
            self._file_position = fh.tell()

    def add_record(self, record: MetricRecord) -> None:
        line = record.model_dump_json()
        with self._lock:
            self._records.append(record)
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._file_position = self.jsonl_path.stat().st_size
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._notify_subscribers([record]))
        except RuntimeError:
            # No running loop, subscribers will catch up on next poll.
            pass

    def latest(self) -> Optional[MetricRecord]:
        with self._lock:
            return self._records[-1] if self._records else None

    def history(self, limit: int = 100, offset: int = 0) -> List[MetricRecord]:
        with self._lock:
            records = list(self._records)
        if offset < 0:
            offset = 0
        start = max(len(records) - offset - limit, 0)
        end = len(records) - offset if offset else len(records)
        return records[start:end]

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def _read_new_records(self) -> List[MetricRecord]:
        if not self.jsonl_path.exists():
            return []
        new_records: List[MetricRecord] = []
        with self._lock:
            with self.jsonl_path.open("r", encoding="utf-8") as fh:
                fh.seek(self._file_position)
                for line in fh:
                    try:
                        data = json.loads(line)
                        record = MetricRecord.model_validate(data)
                    except (json.JSONDecodeError, ValidationError):
                        continue
                    self._records.append(record)
                    new_records.append(record)
                self._file_position = fh.tell()
        return new_records

    async def _poll_loop(self) -> None:
        while True:
            new_records = self._read_new_records()
            if new_records:
                await self._notify_subscribers(new_records)
            await asyncio.sleep(self.poll_interval)

    async def _notify_subscribers(self, records: Iterable[MetricRecord]) -> None:
        queues = list(self._subscribers)
        if not queues:
            return
        for record in records:
            for queue in queues:
                await queue.put(record)

    async def subscribe(self) -> asyncio.Queue[MetricRecord]:
        queue: asyncio.Queue[MetricRecord] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[MetricRecord]) -> None:
        self._subscribers.discard(queue)

    def start(self) -> None:
        if self._poll_task is None:
            loop = asyncio.get_event_loop()
            self._poll_task = loop.create_task(self._poll_loop())

    def stop(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            self._poll_task = None
