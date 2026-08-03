from __future__ import annotations

import io
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.config import AppSettings
from app.receipts import CORDRepository, find_cord_root


class LazyReceiptKIEService:
    """Keep the HTTP server responsive while heavy ML modules initialize lazily."""

    def __init__(self, app_settings: AppSettings):
        self.settings = app_settings
        self._service: Any | None = None
        self._load_lock = threading.Lock()
        self.load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self._service is not None and self._service.ready

    @property
    def device(self) -> str:
        return str(self._service.device) if self._service is not None else "pending"

    def load(self) -> None:
        if self.ready:
            return
        with self._load_lock:
            if self.ready:
                return
            try:
                from app.inference import ReceiptKIEService

                if self._service is None:
                    self._service = ReceiptKIEService(self.settings)
                self._service.load()
                self.load_error = None
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                raise

    def extract(self, image: Image.Image) -> dict:
        self.load()
        return self._service.extract(image)

    def analyze_words(self, *args, **kwargs) -> dict:
        self.load()
        return self._service.analyze_words(*args, **kwargs)


settings = AppSettings.from_env()
service = LazyReceiptKIEService(settings)
static_dir = Path(__file__).resolve().parent / "static"
project_root = Path(__file__).resolve().parents[1]
cord_root_override = os.getenv("CORD_ROOT")
cord_repository = CORDRepository(
    Path(cord_root_override).resolve() if cord_root_override else find_cord_root(project_root)
)
research_results_path = Path(__file__).resolve().parent / "assets" / "research_results.json"


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.eager_load:
        await run_in_threadpool(service.load)
    yield


app = FastAPI(
    title="ReceiptGraph Explorer API",
    version="2.0.0",
    description="Visual Hybrid KIE and relational graph exploration API",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/health/live")
def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
def readiness():
    return {
        "ready": service.ready,
        "device": str(service.device),
        "model_path": str(settings.model_path),
        "error": service.load_error,
    }


@app.get("/api/v1/research-results")
def research_results():
    return json.loads(research_results_path.read_text(encoding="utf-8"))


@app.get("/api/v1/samples")
def list_samples(
    split: str = Query("dev", pattern="^(train|dev|test)$"),
    limit: int = Query(24, ge=1, le=100),
):
    return {
        "available": cord_repository.available,
        "root": str(cord_repository.root) if cord_repository.root else None,
        "samples": cord_repository.list_samples(split, limit),
    }


@app.get("/api/v1/samples/{split}/{receipt_id}/image")
def sample_image(split: str, receipt_id: str):
    try:
        _, _, _, image_path = cord_repository.load(split, receipt_id)
        return FileResponse(image_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def validate_image_upload(content: bytes, content_type: str | None) -> Image.Image:
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG and WebP are supported")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_mb} MB")
    try:
        image = Image.open(io.BytesIO(content))
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image") from exc
    if image.width * image.height > settings.max_image_pixels:
        raise HTTPException(status_code=413, detail="Image dimensions are too large")
    try:
        image.load()
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image") from exc
    return image


@app.post("/api/v1/load")
async def load_model():
    try:
        await run_in_threadpool(service.load)
        return {"ready": True, "device": str(service.device)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model initialization failed: {exc}") from exc


@app.post("/api/v1/extract")
async def extract_receipt(file: UploadFile = File(...)):
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    image = validate_image_upload(content, file.content_type)
    try:
        return await run_in_threadpool(service.extract, image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference failed: {exc}") from exc


@app.post("/api/v1/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
):
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    image = validate_image_upload(content, file.content_type)
    try:
        # OCR is intentionally used only for uploaded real receipts.
        return await run_in_threadpool(service.extract, image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference failed: {exc}") from exc
@app.post("/api/v1/samples/{split}/{receipt_id}/analyze")
async def analyze_sample(
    split: str,
    receipt_id: str,
):
    try:
        image, words, labels, _ = cord_repository.load(split, receipt_id)
        return await run_in_threadpool(
            service.analyze_words,
            image,
            words,
            labels,
            f"cord_{split}",
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference failed: {exc}") from exc
