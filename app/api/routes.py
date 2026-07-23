from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import BASE_DIR, get_settings
from app.db.session import get_db
from app.models.prediction import PredictionRecord
from app.schemas.prediction import PredictionHistoryItem, PredictionResponse
from app.services.report_generator import generate_report

router = APIRouter()
settings = get_settings()


def get_predictor():
    from app.main import predictor

    return predictor


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    predictor=Depends(get_predictor),
) -> PredictionResponse:
    if image.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Upload a PNG or JPEG image.")
    payload = await image.read()
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds upload limit.")

    result = predictor.predict(payload, BASE_DIR / "artifacts" / "gradcam")
    report = await generate_report(result.predicted_class, result.confidence, result.probabilities)
    record = PredictionRecord(
        filename=image.filename or "uploaded_image",
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        probabilities_json=json.dumps(result.probabilities),
        report=report,
        heatmap_path=result.heatmap_path,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return _record_to_response(record)


@router.get("/history", response_model=list[PredictionHistoryItem])
def history(db: Session = Depends(get_db), limit: int = 25) -> list[PredictionHistoryItem]:
    rows = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .limit(min(max(limit, 1), 100))
        .all()
    )
    return [_record_to_response(row) for row in rows]


@router.get("/heatmap/{filename}")
def heatmap(filename: str) -> FileResponse:
    path = BASE_DIR / "artifacts" / "gradcam" / filename
    if not path.exists() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=404, detail="Heatmap not found.")
    return FileResponse(path)


def _record_to_response(record: PredictionRecord) -> PredictionResponse:
    heatmap_url = None
    if record.heatmap_path:
        heatmap_url = f"/api/v1/heatmap/{Path(record.heatmap_path).name}"
    return PredictionResponse(
        id=record.id,
        filename=record.filename,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        probabilities=json.loads(record.probabilities_json),
        report=record.report,
        heatmap_url=heatmap_url,
        created_at=record.created_at,
    )

