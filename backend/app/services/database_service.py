import logging
from typing import Any

import httpx

from app.config import Settings
from app.schemas import AspectPrediction, HistoryItem

logger = logging.getLogger(__name__)


class SupabaseService:
    def __init__(self, settings: Settings):
        self.url = settings.supabase_url.rstrip("/")
        self.key = settings.supabase_service_role_key
        self.enabled = bool(self.url and self.key)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

    async def save_prediction(
        self,
        review_text: str,
        model_version: str,
        latency_ms: int,
        predictions: list[AspectPrediction],
    ) -> str | None:
        if not self.enabled:
            return None
        review_payload = {
            "review_text": review_text,
            "model_version": model_version,
            "latency_ms": latency_ms,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{self.url}/rest/v1/reviews",
                    headers={**self.headers, "Prefer": "return=representation"},
                    json=review_payload,
                )
                response.raise_for_status()
                review_id = response.json()[0]["id"]
                aspect_rows = [
                    {
                        "review_id": review_id,
                        "aspect": item.aspect,
                        "aspect_label": item.aspect_label,
                        "sentiment": item.sentiment,
                        "confidence": item.confidence,
                    }
                    for item in predictions
                ]
                aspect_response = await client.post(
                    f"{self.url}/rest/v1/aspect_predictions",
                    headers=self.headers,
                    json=aspect_rows,
                )
                aspect_response.raise_for_status()
                return review_id
        except Exception:
            logger.exception("Prediksi berhasil, tetapi gagal disimpan ke Supabase.")
            return None

    async def get_history(self, limit: int) -> list[HistoryItem]:
        if not self.enabled:
            return []
        params = {
            "select": "id,review_text,model_version,latency_ms,created_at,aspect_predictions(aspect,aspect_label,sentiment,confidence)",
            "order": "created_at.desc",
            "limit": str(limit),
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.url}/rest/v1/reviews", headers=self.headers, params=params)
                response.raise_for_status()
                return [self._to_history_item(row) for row in response.json()]
        except Exception:
            logger.exception("Gagal mengambil history dari Supabase.")
            return []

    @staticmethod
    def _to_history_item(row: dict[str, Any]) -> HistoryItem:
        return HistoryItem(
            id=row["id"],
            review_text=row["review_text"],
            model_version=row["model_version"],
            latency_ms=row["latency_ms"],
            created_at=row["created_at"],
            predictions=row.get("aspect_predictions", []),
        )
