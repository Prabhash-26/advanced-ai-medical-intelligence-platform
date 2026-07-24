# Advanced AI Medical Intelligence Platform

An end-to-end academic medical AI system for image analysis, disease prediction, Grad-CAM explainability, AI-assisted report generation, REST APIs, database-backed history, and a user-friendly web interface.

## Features

- Deep learning image classifier for `Normal`, `Pneumonia`, `COVID-19`, and `Tuberculosis`.
- Grad-CAM explainability overlay for each prediction.
- LLM-assisted medical report generation with safe rule-based fallback.
- FastAPI REST backend with OpenAPI docs.
- SQLite prediction history table via SQLAlchemy.
- Streamlit web UI for upload, prediction, XAI visualization, and history.
- Reproducible training script and bundled sample image.
- Docker and Docker Compose support.
- PDF project report under `output/pdf/`.

## Project Structure

```text
app/
  api/              FastAPI routes
  core/             settings and environment configuration
  db/               SQLAlchemy session/base
  ml/               CNN model, preprocessing, prediction, Grad-CAM
  models/           database models
  schemas/          API schemas
  services/         report generation
  ui/               Streamlit application
artifacts/
  models/           trained model artifact location
  sample_images/    demo image
scripts/
  train_model.py    reproducible training pipeline
  create_demo_assets.py
tests/
docs/
output/pdf/
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/train_model.py --epochs 8
uvicorn app.main:app --reload
```

Open the API docs at `http://localhost:8000/docs`.

In a second terminal:

```bash
streamlit run app/ui/streamlit_app.py
```

Open the UI at `http://localhost:8501`.

## REST API

- `GET /api/v1/health` - service health check.
- `POST /api/v1/predict` - upload PNG/JPEG image and receive prediction, probabilities, report, and Grad-CAM URL.
- `GET /api/v1/history?limit=25` - recent prediction records.
- `GET /api/v1/heatmap/{filename}` - Grad-CAM overlay image.

Example:
for testing i have used the xray.jpeg image, we can use any image if needed for checking the model and evaluation.
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "image=@artifacts/sample_images/sample_chest_xray.png"
```

## LLM Integration

By default, the system uses a deterministic safe report generator. To enable OpenAI-backed report generation:

```bash
ENABLE_LLM=true
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4.1-mini
```

The LLM prompt is constrained to produce a cautious draft report with explainability and safety notes.

## Model Training

The included training script creates a synthetic radiograph-like dataset so evaluators can reproduce the pipeline without private patient data. The bundled checkpoint was generated with this command:

```bash
python scripts/train_model.py --samples-per-class 80 --epochs 12
```

For a production-grade project, replace `SyntheticChestDataset` with a licensed dataset loader such as NIH ChestX-ray14, CheXpert, or RSNA Pneumonia Detection Challenge, then retrain and document validation metrics.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

## Deliverables Checklist

- Complete source code: included.
- Trained model: `artifacts/models/medical_cnn.pt` included as a reproducible demonstration checkpoint.
- GitHub repository link: https://github.com/Prabhash-26/advanced-ai-medical-intelligence-platform
- README documentation: this file.
- PDF project report: `output/pdf/Advanced_AI_Medical_Intelligence_Platform_Report.pdf`.
- `requirements.txt`: included.
- `Dockerfile`: included.
- Live deployment link: not deployed yet; deploy with the Docker or Streamlit instructions below.

## Suggested Deployment

1. Push this folder to GitHub.
2. Deploy FastAPI with Docker on Render/Railway/Fly.io.
3. Deploy Streamlit UI on Streamlit Community Cloud or the same Docker host.
4. Configure persistent storage for `medical_ai.db` and generated Grad-CAM images.
5. Add the live API/UI URLs to this README and the submission email.
