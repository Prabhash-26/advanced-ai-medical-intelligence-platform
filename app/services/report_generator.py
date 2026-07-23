from __future__ import annotations

from app.core.config import get_settings


def build_rule_based_report(predicted_class: str, confidence: float, probabilities: dict[str, float]) -> str:
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    probability_text = ", ".join(f"{label}: {score:.1%}" for label, score in ranked)
    return (
        "AI-assisted radiology draft report\n\n"
        f"Primary model impression: {predicted_class} with {confidence:.1%} confidence.\n"
        f"Differential probability distribution: {probability_text}.\n\n"
        "Explainability note: the Grad-CAM overlay highlights image regions that most influenced "
        "the model output. A clinician should compare highlighted regions with the original image "
        "before using this draft.\n\n"
        "Clinical safety note: this system is for academic decision-support demonstration only. "
        "It is not a medical device and must not replace professional diagnosis, patient history, "
        "laboratory findings, or local clinical protocols."
    )


async def generate_report(predicted_class: str, confidence: float, probabilities: dict[str, float]) -> str:
    settings = get_settings()
    if not settings.enable_llm or not settings.openai_api_key:
        return build_rule_based_report(predicted_class, confidence, probabilities)

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        prompt = (
            "Create a concise, cautious AI-assisted radiology draft report from the following "
            "classification output. Include findings, impression, explainability note, and a "
            "clear non-diagnostic safety disclaimer.\n"
            f"Prediction: {predicted_class}\n"
            f"Confidence: {confidence:.4f}\n"
            f"Probabilities: {probabilities}"
        )
        response = await client.responses.create(
            model=settings.openai_model,
            input=prompt,
            temperature=0.2,
        )
        return response.output_text
    except Exception:
        return build_rule_based_report(predicted_class, confidence, probabilities)

