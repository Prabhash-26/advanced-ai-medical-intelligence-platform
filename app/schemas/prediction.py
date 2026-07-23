from datetime import datetime
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    report: str
    heatmap_url: str | None
    created_at: datetime


class PredictionHistoryItem(PredictionResponse):
    pass

