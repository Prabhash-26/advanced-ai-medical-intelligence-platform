# System Architecture

## Flow

1. User uploads a medical image through Streamlit or `POST /predict`.
2. FastAPI validates file type and size.
3. Preprocessing converts the image to grayscale, resizes it, and normalizes tensors.
4. The CNN produces class probabilities.
5. Grad-CAM computes a heatmap from the final convolutional layer.
6. The report service creates a cautious AI-assisted draft report.
7. SQLAlchemy stores prediction metadata, probabilities, report text, and heatmap path.
8. The UI and API expose results and history.

## Database Design

`prediction_history`

| Column | Type | Purpose |
| --- | --- | --- |
| id | integer | Primary key |
| filename | string | Uploaded file name |
| predicted_class | string | Top class |
| confidence | float | Top probability |
| probabilities_json | text | Full class distribution |
| report | text | AI-assisted report |
| heatmap_path | string | Saved Grad-CAM overlay |
| created_at | datetime | Audit timestamp |

## Best Practices Included

- Separation of API, ML, service, schema, and persistence layers.
- Environment-driven configuration.
- Safe LLM fallback when API keys are unavailable.
- Upload validation and response schemas.
- Reproducible model-training entry point.
- Clinical safety disclaimers throughout the user-facing surfaces.

