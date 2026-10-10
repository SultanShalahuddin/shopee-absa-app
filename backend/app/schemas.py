from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator


Sentiment = Literal["tidak_disebut", "negatif", "netral", "positif"]


class PredictRequest(BaseModel):
    review_text: str = Field(min_length=3, max_length=2000)

    @field_validator("review_text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 3:
            raise ValueError("Review minimal terdiri dari 3 karakter.")
        return value


class AspectPrediction(BaseModel):
    aspect: str
    aspect_label: str
    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    id: str | None = None
    review_text: str
    predictions: list[AspectPrediction]
    model_version: str
    latency_ms: int
    stored: bool
    demo_mode: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HistoryItem(BaseModel):
    id: str
    review_text: str
    model_version: str
    latency_ms: int
    created_at: datetime
    predictions: list[AspectPrediction]


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
