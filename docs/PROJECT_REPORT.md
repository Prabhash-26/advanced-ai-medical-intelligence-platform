# Advanced AI Medical Intelligence Platform - Project Report

## Abstract

This project implements an end-to-end AI medical intelligence platform for academic evaluation. It analyzes medical images, predicts likely disease classes using a convolutional neural network, explains predictions with Grad-CAM, generates cautious AI-assisted medical reports, exposes REST APIs, stores prediction history in a relational database, and provides a user-friendly Streamlit interface.

## Problem Statement

Medical image triage systems can help organize radiology workflows by highlighting potentially abnormal studies and drafting structured summaries. The goal of this project is to demonstrate a complete software and machine learning architecture that combines deep learning, explainability, LLM-assisted reporting, API engineering, persistence, and deployment readiness.

## System Modules

- Image preprocessing: grayscale conversion, resizing to 224 x 224, tensor normalization.
- Deep learning model: compact CNN with convolution, batch normalization, pooling, dropout, and linear classification head.
- Explainable AI: Grad-CAM heatmap from the final convolutional feature layer.
- LLM reporting: optional OpenAI report generation with deterministic rule-based fallback.
- Backend: FastAPI service with health, prediction, history, and heatmap endpoints.
- Database: SQLite prediction history using SQLAlchemy ORM.
- Frontend: Streamlit upload dashboard with probabilities, report, heatmap, and history.
- Deployment: Dockerfile and docker-compose configuration.

## Classes

The demonstration classifier supports four labels: Normal, Pneumonia, COVID-19, and Tuberculosis. The included training script uses a synthetic radiograph-like dataset so the pipeline is reproducible without patient data. For clinical-quality evaluation, the dataset layer should be replaced with a licensed source such as NIH ChestX-ray14, CheXpert, MIMIC-CXR, or RSNA Pneumonia Detection Challenge.

## API Design

- `GET /api/v1/health`: returns service status.
- `POST /api/v1/predict`: accepts PNG/JPEG upload and returns prediction, confidence, class probabilities, report, Grad-CAM URL, and timestamp.
- `GET /api/v1/history`: returns recent prediction records.
- `GET /api/v1/heatmap/{filename}`: returns generated Grad-CAM image.

## Database Design

The `prediction_history` table stores filename, predicted class, confidence, probability distribution, AI report, heatmap path, and creation timestamp. This provides traceability for model outputs and supports audit-oriented review.

## Explainability

Grad-CAM identifies influential image regions by combining gradients from the target class with activations from the last convolutional layer. The resulting heatmap is overlaid on the original image to help users inspect whether the model attended to clinically plausible regions.

## LLM Safety

The report generator is intentionally conservative. It produces a draft report, includes the probability distribution, names the explainability method, and states that the system is not a medical device. If no LLM API key is configured, the application still works with deterministic report generation.

## Evaluation Plan

Model performance should be evaluated with accuracy, macro F1-score, sensitivity, specificity, confusion matrix, calibration, and class-wise ROC-AUC. A real submission should document train/validation/test splits and avoid patient leakage across splits.

## Limitations

The bundled artifact is a reproducible educational demonstration, not a clinically validated model. Real-world deployment requires approved datasets, external validation, security review, medical governance, privacy controls, monitoring, and regulatory assessment.

## Conclusion

The project satisfies the requested end-to-end system requirements: source code, model artifact path, training script, XAI, LLM integration, REST APIs, database storage, web UI, documentation, Docker support, and PDF report.

