from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_liveness_does_not_load_model():
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rejects_unsupported_upload_type():
    response = client.post(
        "/api/v1/extract",
        files={"file": ("receipt.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_web_ui_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Graph Explorer" in response.text


def test_research_results_are_explicit_about_metric_scope():
    response = client.get("/api/v1/research-results")
    assert response.status_code == 200
    payload = response.json()
    assert payload["hybrid"]["test_macro_f1_mean"] == 0.9398
    assert "Word-level" in payload["metric_scope"]


def test_cord_samples_are_discoverable_when_dataset_exists():
    response = client.get("/api/v1/samples?split=dev&limit=2")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert len(payload["samples"]) == 2
