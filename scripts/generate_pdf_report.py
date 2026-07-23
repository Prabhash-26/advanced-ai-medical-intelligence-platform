from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "Advanced_AI_Medical_Intelligence_Platform_Report.pdf"


def build_pdf() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#263238"),
        )
    )
    title = ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        textColor=colors.HexColor("#12343b"),
        spaceAfter=18,
    )
    heading = ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0b5d69"),
        spaceBefore=12,
        spaceAfter=7,
    )
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    story = [
        Paragraph("Advanced AI Medical Intelligence Platform", title),
        Paragraph("Project Report", styles["Heading3"]),
        Paragraph(
            "An end-to-end academic system for medical image classification, explainable AI, "
            "LLM-assisted reporting, REST APIs, database-backed history, and web deployment.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]
    sections = [
        (
            "Objective",
            "Build a complete AI application capable of analyzing medical images, predicting disease "
            "classes with deep learning, explaining predictions through Grad-CAM, generating cautious "
            "AI-assisted reports, exposing REST APIs, storing prediction history, and providing a usable UI.",
        ),
        (
            "Architecture",
            "The platform separates concerns across FastAPI routes, ML inference, Grad-CAM explainability, "
            "report generation, SQLAlchemy persistence, and Streamlit presentation. This keeps the project "
            "maintainable and ready for deployment.",
        ),
        (
            "Deep Learning",
            "The model is a compact convolutional neural network with batch normalization, pooling, dropout, "
            "and a linear classification head. The included training script creates reproducible educational "
            "weights and can be adapted to NIH ChestX-ray14, CheXpert, MIMIC-CXR, or RSNA datasets.",
        ),
        (
            "Explainable AI",
            "Grad-CAM computes gradients for the predicted class and overlays the resulting heatmap on the "
            "original image, helping reviewers inspect which regions influenced the model.",
        ),
        (
            "LLM Integration",
            "The report generator supports optional OpenAI-backed draft reports and falls back to a deterministic "
            "safe report when no API key is configured. Every report includes non-diagnostic safety language.",
        ),
        (
            "Deployment",
            "The repository includes requirements.txt, Dockerfile, docker-compose.yml, and clear run commands for "
            "local or cloud deployment.",
        ),
    ]
    for section, body in sections:
        story.append(Paragraph(section, heading))
        story.append(Paragraph(body, styles["BodyText"]))

    story.append(Paragraph("API Endpoints", heading))
    table = Table(
        [
            ["Endpoint", "Purpose"],
            ["GET /api/v1/health", "Service health check"],
            ["POST /api/v1/predict", "Image prediction, probabilities, report, and Grad-CAM URL"],
            ["GET /api/v1/history", "Recent prediction records"],
            ["GET /api/v1/heatmap/{filename}", "Generated explainability overlay"],
        ],
        colWidths=[2.2 * inch, 4.6 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b5d69")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b0bec5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef7f8")]),
            ]
        )
    )
    story.append(table)
    story.append(Paragraph("Safety and Limitations", heading))
    story.append(
        Paragraph(
            "This project is not a medical device. It is an academic decision-support demonstration. "
            "Clinical deployment would require licensed data, external validation, privacy controls, "
            "security review, monitoring, and regulatory assessment.",
            styles["BodyText"],
        )
    )
    story.append(Paragraph("Deliverables", heading))
    story.append(
        Paragraph(
            "Complete source code, model artifact path, reproducible training script, README, PDF report, "
            "requirements.txt, Dockerfile, database design, REST APIs, Streamlit UI, and deployment guidance are included.",
            styles["BodyText"],
        )
    )
    doc.build(story)


if __name__ == "__main__":
    build_pdf()

