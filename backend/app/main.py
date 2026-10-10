import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import HistoryResponse, PredictRequest, PredictionResponse
from app.services.database_service import SupabaseService
from app.services.model_service import ABSAModelService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = ABSAModelService(settings)
    app.state.database_service = SupabaseService(settings)
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/")
async def root():
    return {"service": settings.app_name, "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health():
    model_service: ABSAModelService = app.state.model_service
    database_service: SupabaseService = app.state.database_service
    return {
        "status": "ok",
        "model_loaded": not model_service.demo_mode,
        "demo_mode": model_service.demo_mode,
        "model_source": model_service.source,
        "model_format": model_service.model_format,
        "device": str(model_service.device),
        "database_enabled": database_service.enabled,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(payload: PredictRequest):
    start = time.perf_counter()
    model_service: ABSAModelService = app.state.model_service
    database_service: SupabaseService = app.state.database_service
    predictions = model_service.predict(payload.review_text)
    latency_ms = round((time.perf_counter() - start) * 1000)
    review_id = await database_service.save_prediction(
        review_text=payload.review_text,
        model_version=settings.model_version,
        latency_ms=latency_ms,
        predictions=predictions,
    )
    return PredictionResponse(
        id=review_id,
        review_text=payload.review_text,
        predictions=predictions,
        model_version=settings.model_version,
        latency_ms=latency_ms,
        stored=review_id is not None,
        demo_mode=model_service.demo_mode,
    )


@app.get("/api/v1/history", response_model=HistoryResponse)
async def history(limit: int = Query(default=8, ge=1, le=50)):
    database_service: SupabaseService = app.state.database_service
    return HistoryResponse(items=await database_service.get_history(limit))
