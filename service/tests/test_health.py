import pytest
from fastapi.testclient import TestClient

from conftest import FakeStoryGenerator
from tinygpt_service.config import ServiceSettings
from tinygpt_service.main import create_app


def test_healthz_ok_without_model(make_client):
    client = make_client(generator=None)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_reports_provenance_when_loaded(make_client):
    client = make_client(generator=FakeStoryGenerator())
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_version"] == "xlarge-plus"
    assert body["run_id"] == "test-run-001"
    assert body["tokenizer_id"] == "tinystories_bpe_5000"


def test_readyz_503_when_model_missing(make_client):
    client = make_client(generator=None)
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_index_serves_chat_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "TinyGPT" in response.text


def test_default_app_fails_startup_without_bundle():
    app = create_app(settings=ServiceSettings())
    with pytest.raises(RuntimeError, match="TINYGPT_BUNDLE_DIR is required"):
        with TestClient(app):
            pass
