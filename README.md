# Advanced AI Medical Intelligence Platform

An end-to-end AI application that analyzes chest X-ray images to detect **Pneumonia**,
explains its predictions using **Grad-CAM**, generates an **AI-assisted medical report**
using an LLM, and exposes everything through a **REST API** and a **web interface**, with
all predictions logged to a database.


---

## Features

- **Deep Learning model** — ResNet18 (transfer learning) fine-tuned to classify chest X-rays as `NORMAL` or `PNEUMONIA`
- **Explainable AI** — Grad-CAM heatmaps show which regions of the X-ray influenced the prediction
- **LLM-generated reports** — OpenAI API turns raw predictions into a structured, readable preliminary report
- **REST API** — FastAPI backend with endpoints for prediction and history retrieval
- **Database** — SQLite (via SQLAlchemy) stores every prediction, confidence score, and report
- **Web UI** — Streamlit frontend for uploading images and reviewing results/history
- **Deployment-ready** — Dockerfile provided for containerized deployment

---

## Project Structure

```
medical-ai-platform/
├── model.py            # Model architecture (ResNet18) + model loader
├── train.py             # Training script (run on Colab/GPU)
├── gradcam.py            # Grad-CAM implementation
├── database.py            # SQLAlchemy models + DB helper functions
├── llm_report.py           # OpenAI-based report generation
├── api.py                # FastAPI REST API
├── app.py                 # Streamlit frontend
├── requirements.txt        # Python dependencies
├── Dockerfile               # Container build
├── entrypoint.sh              # Starts API + frontend inside container
├── .env.example                 # Environment variable template
├── models/                       # Trained model weights go here (model.pth)
└── static/gradcam/                # Generated Grad-CAM images saved here
```

---

## Dataset

This project uses the public **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** dataset from Kaggle.

Expected folder structure after download:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Model Training (Google Colab)

Training is designed to run on Colab's free GPU.

```bash
# In a Colab notebook:
!pip install -q torch torchvision kaggle

# Set up Kaggle API credentials, then:
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip -q chest-xray-pneumonia.zip -d data

!git clone <your-repo-url>
%cd medical-ai-platform

!python train.py --data_dir ../data/chest_xray --epochs 10 --out models/model.pth
```

The script prints train/validation accuracy per epoch, saves the best-performing model
to `models/model.pth`, and prints a final classification report + confusion matrix on
the test set.

**The trained `model.pth` file is too large for a normal git push.** It is hosted separately —
see [Trained Model](#trained-model) below.

---

## Trained Model

Download the trained weights and place them in the `models/` folder:

(Hosted via GitHub Releases / Hugging Face Hub / Google Drive — replace this link with yours.)

```bash
mkdir -p models
# place the downloaded model.pth here: models/model.pth
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd medical-ai-platform
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
```bash
cp .env.example .env
# then edit .env and add your OPENAI_API_KEY
```

### 5. Add the trained model
Place `model.pth` inside the `models/` folder (see [Trained Model](#trained-model)).

---

## Running Locally

**Start the API:**
```bash
uvicorn api:app --reload --port 8000
```

**Start the frontend (in a separate terminal):**
```bash
streamlit run app.py
```

Then open:
- Frontend: `http://localhost:8501`
- API docs (Swagger UI): `http://localhost:8000/docs`

---

## Running with Docker

```bash
docker build -t medical-ai-platform .
docker run -p 8000:8000 -p 8501:8501 --env-file .env medical-ai-platform
```

---

## API Endpoints

| Method | Endpoint            | Description                                  |
|--------|---------------------|-----------------------------------------------|
| GET    | `/health`            | Health check + model load status              |
| POST   | `/predict`            | Upload an X-ray image → prediction + Grad-CAM + report |
| GET    | `/history`             | List past predictions                         |
| GET    | `/history/{id}`         | Get a specific prediction record              |

Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_xray.jpg"
```

Example response:
```json
{
  "id": 1,
  "filename": "sample_xray.jpg",
  "predicted_class": "PNEUMONIA",
  "confidence": 0.9421,
  "gradcam_url": "/static/gradcam/abc123.jpg",
  "report": "1. Summary of Finding...\n2. Confidence Interpretation...",
  "created_at": "2026-07-24T10:00:00"
}
```

---

## Explainable AI (Grad-CAM)

Every prediction includes a Grad-CAM heatmap overlay generated from the final
convolutional layer of the ResNet18 backbone, highlighting the image regions that most
influenced the model's decision. This is returned alongside the prediction via the
`gradcam_url` field and displayed in the Streamlit UI.

---

## LLM-Generated Reports

Each prediction is passed to OpenAI's API (`gpt-4o-mini`) to generate a structured,
human-readable preliminary report covering:
1. Summary of finding
2. Confidence interpretation
3. Recommended next steps
4. Disclaimer (AI-assisted, requires physician review)

If the OpenAI API is unavailable, the system falls back to a basic templated report so
the application doesn't break.

---

## Tech Stack

| Layer            | Technology              |
|------------------|--------------------------|
| Deep Learning     | PyTorch, torchvision (ResNet18) |
| Explainability     | Grad-CAM (custom implementation) |
| LLM Integration     | OpenAI API (gpt-4o-mini) |
| Backend API          | FastAPI                  |
| Database              | SQLite + SQLAlchemy       |
| Frontend                | Streamlit                  |
| Deployment                | Docker                      |

---

## Model Performance

_Fill this in after training — example format below:_

| Metric      | Value  |
|-------------|--------|
| Test Accuracy | 0.9X   |
| Precision      | 0.9X   |
| Recall          | 0.9X   |
| F1-score          | 0.9X   |

Confusion matrix and full classification report are printed at the end of `train.py`.

---

## Live Deployment

- **Live App:** _add your deployed URL here (Hugging Face Spaces / Render / etc.)_
- **API Docs:** `<your-deployed-url>/docs`

---

## Future Improvements

- Support additional disease categories beyond pneumonia
- Add user authentication for the API
- Add PDF export of individual reports
- Model versioning and A/B comparison of checkpoints

---

## License

This project is submitted as part of a technical assessment and is intended for
educational/demonstration purposes only.- `GET /api/v1/history?limit=25` - recent prediction records.
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
