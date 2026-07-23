from app.services.report_generator import build_rule_based_report


def test_rule_based_report_contains_safety_disclaimer():
    report = build_rule_based_report(
        "Pneumonia",
        0.87,
        {"Normal": 0.05, "Pneumonia": 0.87, "COVID-19": 0.04, "Tuberculosis": 0.04},
    )

    assert "Pneumonia" in report
    assert "Grad-CAM" in report
    assert "not a medical device" in report

