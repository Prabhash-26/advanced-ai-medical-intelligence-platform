from app.api.routes import health


def test_health_contract():
    response = health()

    assert response["status"] == "ok"
    assert "service" in response

