from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db.session import Base, engine
from app.ml.predictor import MedicalPredictor


Base.metadata.create_all(bind=engine)
predictor = MedicalPredictor()

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description="Medical image classification, Grad-CAM explainability, LLM reports, and prediction history APIs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["medical-ai"])

