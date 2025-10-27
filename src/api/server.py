from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import yaml
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .metrics_manager import MetricsManager
from .models import (
    ConfigResponse,
    ConfigUpdateRequest,
    MetricHistoryResponse,
    MetricRecord,
    StartTrainingRequest,
    TrainingResponse,
    TrainingStatus,
)
from .training_manager import TrainingManager

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = BASE_DIR / "configs" / "train_panda_door.yaml"
DEFAULT_METRICS_PATH = BASE_DIR / "experiments" / "metrics.jsonl"
CHECKPOINT_DIR = BASE_DIR / "experiments" / "checkpoints"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"


def get_metrics_manager() -> MetricsManager:
    return metrics_manager


def get_training_manager() -> TrainingManager:
    return training_manager


def create_app() -> FastAPI:
    app = FastAPI(title="Panda Door RL API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        metrics_manager.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        metrics_manager.stop()

    @app.get("/api/status", response_model=TrainingStatus)
    async def get_status(manager: TrainingManager = Depends(get_training_manager)) -> TrainingStatus:
        return manager.status()

    @app.post("/api/training/start", response_model=TrainingResponse)
    async def start_training(
        request: Optional[StartTrainingRequest] = None,
        manager: TrainingManager = Depends(get_training_manager),
    ) -> TrainingResponse:
        status = manager.start(request)
        return TrainingResponse(status=status)

    @app.post("/api/training/stop", response_model=TrainingResponse)
    async def stop_training(manager: TrainingManager = Depends(get_training_manager)) -> TrainingResponse:
        status = manager.stop()
        return TrainingResponse(status=status)

    @app.get("/api/metrics/latest", response_model=Optional[MetricRecord])
    async def latest_metrics(manager: MetricsManager = Depends(get_metrics_manager)) -> Optional[MetricRecord]:
        return manager.latest()

    @app.get("/api/metrics/history", response_model=MetricHistoryResponse)
    async def metrics_history(
        limit: int = 200,
        offset: int = 0,
        manager: MetricsManager = Depends(get_metrics_manager),
    ) -> MetricHistoryResponse:
        items = manager.history(limit=limit, offset=offset)
        return MetricHistoryResponse(items=items, total=manager.count())

    @app.websocket("/ws/metrics")
    async def metrics_websocket(
        ws: WebSocket,
        manager: MetricsManager = Depends(get_metrics_manager),
    ) -> None:
        await ws.accept()
        queue = await manager.subscribe()
        try:
            while True:
                try:
                    record = await asyncio.wait_for(queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    record = manager.latest()
                    if record is None:
                        continue
                await ws.send_text(record.model_dump_json())
        except WebSocketDisconnect:
            manager.unsubscribe(queue)
        finally:
            manager.unsubscribe(queue)

    @app.get("/api/checkpoints")
    async def list_checkpoints() -> JSONResponse:
        if not CHECKPOINT_DIR.exists():
            return JSONResponse({"items": []})
        items = []
        for path in sorted(CHECKPOINT_DIR.glob("*.pt")):
            items.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "modified": path.stat().st_mtime,
                }
            )
        return JSONResponse({"items": items})

    @app.get("/api/config", response_model=ConfigResponse)
    async def get_config() -> ConfigResponse:
        if not DEFAULT_CONFIG.exists():
            raise HTTPException(status_code=404, detail="Config file not found")
        content = DEFAULT_CONFIG.read_text(encoding="utf-8")
        return ConfigResponse(path=DEFAULT_CONFIG, content=content)

    @app.put("/api/config", response_model=ConfigResponse)
    async def update_config(request: ConfigUpdateRequest) -> ConfigResponse:
        try:
            yaml.safe_load(request.content)
        except yaml.YAMLError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
        DEFAULT_CONFIG.write_text(request.content, encoding="utf-8")
        return ConfigResponse(path=DEFAULT_CONFIG, content=request.content)

    @app.get("/api/logs/metrics")
    async def download_metrics(manager: MetricsManager = Depends(get_metrics_manager)) -> StreamingResponse:
        path = manager.jsonl_path
        if not path.exists():
            raise HTTPException(status_code=404, detail="Metrics log not found")
        return StreamingResponse(path.open("rb"), media_type="application/jsonl")

    if FRONTEND_DIST.exists():
        app.mount(
            "/",
            StaticFiles(directory=FRONTEND_DIST, html=True),
            name="frontend",
        )
    else:

        @app.get("/")
        async def root() -> JSONResponse:
            return JSONResponse(
                {
                    "status": "panda-door-rl-api",
                    "message": "Frontend build not found. Run `npm run build` inside frontend/ to generate the dist folder.",
                }
            )

    return app


metrics_manager = MetricsManager(DEFAULT_METRICS_PATH)
training_manager = TrainingManager(BASE_DIR, DEFAULT_CONFIG)
app = create_app()
